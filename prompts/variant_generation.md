# Caption variant generation contract

Generate variants from the supplied caption only. Do not inspect the image and do not add, infer, correct, or remove identity-bearing facts unless the requested operation explicitly requires removal.

Return JSONL with exactly these fields:

```json
{"sample_id":"...","variant_type":"summary_20w","text":"...","generator":"model-and-prompt-version","relation":"information_reduced"}
```

Recommended controlled variants:

1. `summary_10w`, `summary_20w`, `summary_40w`: obey the word limit; retain the most discriminative visible attributes.
2. `paraphrase`: preserve every visual fact while changing syntax and wording.
3. `synonym`: preserve syntax where possible and replace only safe lexical equivalents.
4. `attribute_order`: preserve every fact but change the order of attribute groups.
5. `swap_color`, `swap_attribute`, `negation`: deliberately change exactly one stated relation/fact and set `relation` to `semantic_changed`.

Reject a generated pair during review if it introduces a color, gender/presentation, age, garment, vehicle make/model, facial attribute, accessory, or condition that is absent from the source.

