"""Create reproducible single-span controls from existing balanced pilot captions."""
import hashlib
import re
from pathlib import Path

import yaml

from caption_bench.data import load_captions, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def make_controls(text, domain):
    # Whole-phrase replacements preserve entity attachment and avoid broad negation.
    preserving = [(r"\bwearing\b", "dressed in"), (r"\beyeglasses\b", "glasses"),
                  (r"\balloy rims\b", "alloy wheels"), (r"\brounded tip\b", "round tip"),
                  (r"\btrousers\b", "pants"), (r"\bfeaturing\b", "with")]
    if domain == "face":
        # Accessories are worn, not 'dressed in'. Keep the same explicit fact.
        preserving = [(r"\bis wearing\b", "wears")] + preserving[1:]
    changes = ([(r"\boval face\b", "angular face"), (r"\boblong face\b", "oval face")]
               if domain == "face" else
               [(r"\blight blue\b", "dark red"), (r"\blight brown\b", "dark gray"),
                (r"\bblack\b", "white"), (r"\bwhite\b", "black"),
                (r"\bsilver\b", "black"), (r"\bdark gray\b", "light brown"),
                (r"\blight-colored\b", "dark-colored")])
    output = []
    for label, relation, rules in [("control_synonym", "semantic_preserving", preserving),
                                   ("control_one_fact", "semantic_changed", changes)]:
        for pattern, replacement in rules:
            match = re.search(pattern, text, re.I)
            if match:
                output.append((label, relation, text[:match.start()] + replacement + text[match.end():],
                               f"{match.group()} → {replacement}"))
                break
    return output


def main():
    originals = {r.sample_id: r.text for r in load_captions(ROOT / "caption_samples_50/all_samples.jsonl")}
    rows = read_jsonl(ROOT / "variants/gemini_pilot_v2/variants.jsonl")
    controls = []
    for row in rows:
        if row["variant_type"] != "llm_balanced":
            continue
        domain = row["sample_id"].split("::")[0]
        for label, relation, text, note in make_controls(row["text"], domain):
            controls.append({"sample_id": row["sample_id"], "variant_type": label, "text": text,
                             "relation": relation, "generator": "controlled:single-span",
                             "change_note": note, "control_base": "llm_balanced",
                             "source_sha256": hashlib.sha256(originals[row["sample_id"]].encode()).hexdigest()})
    write_jsonl(ROOT / "variants/controlled_pilot/variants.jsonl", rows + controls)
    for source, target, output in [
        ("encoder_expansion_pilot.yaml", "controlled_pilot.yaml", "controlled_pilot"),
        ("mamba3_siso_pilot.yaml", "controlled_mamba3.yaml", "controlled_mamba3")]:
        config = yaml.safe_load((ROOT / "configs" / source).read_text())
        config["name"] = "controlled-encoder-expansion-pilot"
        config["output_dir"] = f"../runs/{output}"
        config["variant_files"] = ["../variants/controlled_pilot/variants.jsonl"]
        (ROOT / "configs" / target).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Added {len(controls)} controls to {len(rows)} existing variants")


if __name__ == "__main__":
    main()
