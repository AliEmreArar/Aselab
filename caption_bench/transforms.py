from __future__ import annotations

import random
import re
from collections.abc import Callable

from .data import Caption, Variant

Transform = Callable[[str, random.Random, dict], str]
_REGISTRY: dict[str, Transform] = {}

_DEFAULT_RELATIONS = {
    "identity": "same",
    "compact": "information_reduced",
    "head_words": "information_reduced",
    "synonym": "semantic_preserving",
    "clause_exchange": "semantic_preserving",
    "clause_reverse": "semantic_preserving",
    "word_dropout": "information_reduced",
    "color_change": "semantic_changed",
    "attribute_exchange": "semantic_changed",
    "negation": "semantic_changed",
}


def register(name: str):
    def decorator(func: Transform) -> Transform:
        _REGISTRY[name] = func
        return func

    return decorator


def available_transforms() -> list[str]:
    return sorted(_REGISTRY)


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


@register("identity")
def identity(text: str, rng: random.Random, options: dict) -> str:
    return text


@register("compact")
def compact(text: str, rng: random.Random, options: dict) -> str:
    """Deterministic extractive shortening; never invents visual attributes."""
    max_words = int(options.get("max_words", 24))
    if len(_words(text)) <= max_words:
        return text
    clauses = [part.strip(" ;,.") for part in re.split(r"[;.]|,(?=\s)", text) if part.strip()]
    chosen: list[str] = []
    count = 0
    for clause in clauses:
        clause_words = _words(clause)
        if chosen and count + len(clause_words) > max_words:
            continue
        if len(clause_words) > max_words and not chosen:
            chosen.append(" ".join(clause_words[:max_words]))
            break
        chosen.append(clause)
        count += len(clause_words)
        if count >= max_words:
            break
    result = ", ".join(chosen).strip()
    return result + ("." if result and result[-1] not in ".!?" else "")


@register("head_words")
def head_words(text: str, rng: random.Random, options: dict) -> str:
    """Keep the first N whitespace tokens to expose position/length effects."""
    max_words = int(options.get("max_words", 20))
    words = _words(text)
    return " ".join(words[:max_words])


_SYNONYMS = {
    "wearing": "dressed in",
    "wears": "has on",
    "carrying": "holding",
    "captured": "shown",
    "average": "medium",
    "moderate": "medium",
    "dark": "deep-colored",
    "light": "pale",
    "short": "cropped",
    "large": "big",
    "small": "little",
    "rounded": "round",
    "straight": "linear",
    "visible": "noticeable",
}


@register("synonym")
def synonym(text: str, rng: random.Random, options: dict) -> str:
    """Conservative lexical substitutions, capped to limit semantic drift."""
    probability = float(options.get("probability", 0.35))
    max_replacements = int(options.get("max_replacements", 4))
    replaced = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replaced
        word = match.group(0)
        candidate = _SYNONYMS.get(word.lower())
        if candidate is None or replaced >= max_replacements or rng.random() > probability:
            return word
        replaced += 1
        return candidate.capitalize() if word[0].isupper() else candidate

    return re.sub(r"\b[A-Za-z]+\b", replace, text)


@register("clause_exchange")
def clause_exchange(text: str, rng: random.Random, options: dict) -> str:
    """Rotate comma/semicolon-delimited clauses without changing their content."""
    clauses = [part.strip(" ;,.") for part in re.split(r"[;,]", text) if part.strip()]
    if len(clauses) < 3:
        return text
    shift = int(options.get("shift", max(1, len(clauses) // 2))) % len(clauses)
    clauses = clauses[shift:] + clauses[:shift]
    result = ", ".join(clauses)
    return result[0].upper() + result[1:] + "."


@register("clause_reverse")
def clause_reverse(text: str, rng: random.Random, options: dict) -> str:
    clauses = [part.strip(" ;,.") for part in re.split(r"[;,]", text) if part.strip()]
    if len(clauses) < 2:
        return text
    result = ", ".join(reversed(clauses))
    return result[0].upper() + result[1:] + "."


@register("word_dropout")
def word_dropout(text: str, rng: random.Random, options: dict) -> str:
    """Stress test only: this may remove identity-bearing evidence."""
    probability = float(options.get("probability", 0.1))
    words = _words(text)
    kept = [word for word in words if rng.random() >= probability]
    if len(kept) < max(1, len(words) // 2):
        kept = words[: max(1, len(words) // 2)]
    return " ".join(kept)


_COLORS = [
    "light blue", "dark blue", "light brown", "dark brown", "light gray", "dark gray",
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "brown", "gray", "grey", "silver", "gold", "beige",
]
_COLOR_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, _COLORS)) + r")\b", re.IGNORECASE)
_COLOR_CHANGE = {
    "black": "white", "white": "black", "red": "blue", "blue": "red", "green": "purple",
    "yellow": "blue", "orange": "green", "purple": "green", "pink": "blue", "brown": "gray",
    "gray": "brown", "grey": "brown", "silver": "black", "gold": "silver", "beige": "black",
    "light blue": "dark red", "dark blue": "light red", "light brown": "dark gray",
    "dark brown": "light gray", "light gray": "dark brown", "dark gray": "light brown",
}


@register("color_change")
def color_change(text: str, rng: random.Random, options: dict) -> str:
    """Change one explicit color; a meaning-changing hard-negative perturbation."""
    match = _COLOR_PATTERN.search(text)
    if not match:
        return text
    old = match.group(0)
    new = _COLOR_CHANGE[old.lower()]
    if old[0].isupper():
        new = new.capitalize()
    return text[: match.start()] + new + text[match.end() :]


@register("attribute_exchange")
def attribute_exchange(text: str, rng: random.Random, options: dict) -> str:
    """Swap the first two explicit color values while retaining all color tokens."""
    matches = list(_COLOR_PATTERN.finditer(text))
    if len(matches) < 2:
        return text
    first, second = matches[0], matches[1]
    a, b = first.group(0), second.group(0)
    if a.lower() == b.lower():
        return text
    return text[: first.start()] + b + text[first.end() : second.start()] + a + text[second.end() :]


@register("negation")
def negation(text: str, rng: random.Random, options: dict) -> str:
    """Negate the first supported attribute construction."""
    rules = [
        (r"\bwearing\b", "not wearing"),
        (r"\bwears\b", "does not wear"),
        (r"\bwith\b", "without"),
        (r"\bhas\b", "does not have"),
    ]
    for pattern, replacement in rules:
        changed, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if count:
            return changed
    return text


def generate_variants(captions: list[Caption], specs: list[dict], seed: int) -> list[Variant]:
    output: list[Variant] = []
    for caption_idx, caption in enumerate(captions):
        for spec_idx, spec in enumerate(specs):
            name = str(spec["name"])
            if name not in _REGISTRY:
                raise ValueError(f"Unknown transform '{name}'. Available: {available_transforms()}")
            label = str(spec.get("label", name))
            domains = set(map(str, spec.get("domains", [])))
            if domains and caption.domain not in domains:
                continue
            rng = random.Random(seed + caption_idx * 10_007 + spec_idx)
            text = _REGISTRY[name](caption.text, rng, dict(spec))
            if bool(spec.get("skip_unchanged", False)) and text == caption.text:
                continue
            output.append(
                Variant(
                    sample_id=caption.sample_id,
                    variant_type=label,
                    text=text,
                    generator=f"builtin:{name}",
                    relation=str(spec.get("relation", _DEFAULT_RELATIONS.get(name, "unspecified"))),
                )
            )
    return output

