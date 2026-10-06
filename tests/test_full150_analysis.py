import pandas as pd

from scripts.analyze_full150_study import query_split, select_recipes, summarize, proportion_interval


def test_paired_comparison_matches_sources_despite_row_order():
    frame = pd.DataFrame([
        dict(model='m', domain='face', sample_id='a', variant_type='identity',
             recall_at_1=1, recall_at_5=1, reciprocal_rank=1, margin=.2),
        dict(model='m', domain='face', sample_id='b', variant_type='identity',
             recall_at_1=0, recall_at_5=1, reciprocal_rank=.5, margin=-.1),
        dict(model='m', domain='face', sample_id='b', variant_type='short',
             recall_at_1=1, recall_at_5=1, reciprocal_rank=1, margin=.3),
        dict(model='m', domain='face', sample_id='a', variant_type='short',
             recall_at_1=1, recall_at_5=1, reciprocal_rank=1, margin=.2),
    ])
    row = summarize(frame, False).set_index('variant_type').loc['short']
    assert row['delta_R1'] == .5
    assert abs(row['delta_margin'] - .2) < 1e-9


def test_recipe_selection_cannot_use_held_out_performance():
    rows = []
    for partition, scores in [('development', {'a': 1, 'b': 0}),
                              ('held_out', {'a': 0, 'b': 1})]:
        for profile, score in scores.items():
            rows.append(dict(domain='face', model='m', variant_type=profile,
                             query_partition=partition, recall_at_1=score,
                             reciprocal_rank=score, recall_at_5=score))
    result = select_recipes(pd.DataFrame(rows), False)
    assert set(result.variant_type) == {'a'}
    assert set(result.held_out_R1) == {0}


def test_text_recipe_does_not_select_trivial_identity_self_match():
    rows = [dict(domain='face', model='m', variant_type=profile,
                 query_partition=partition, recall_at_1=score,
                 reciprocal_rank=score, recall_at_5=score)
            for profile, score in [('identity', 1), ('summary', .5)]
            for partition in ['development', 'held_out']]
    result = select_recipes(pd.DataFrame(rows), False)
    assert set(result.variant_type) == {'summary'}


def test_perfect_sample_accuracy_does_not_imply_perfect_population_accuracy():
    low, high = proportion_interval([1] * 25)
    assert .86 < low < .88
    assert abs(high - 1) < 1e-12


def test_partition_balanced_disjoint_and_pilot_in_development():
    from caption_bench.data import load_captions, read_jsonl
    from scripts.analyze_full150_study import ROOT
    captions = load_captions(ROOT / 'caption_samples_50/all_samples.jsonl')
    split = query_split(captions)
    assert split.sample_id.nunique() == 150
    assert set(split.groupby(['domain', 'query_partition']).size()) == {25}
    pilot = {r['sample_id'] for r in read_jsonl(ROOT / 'variants/gemini_pilot_v2/variants.jsonl')}
    assert pilot <= set(split[split.query_partition == 'development'].sample_id)
    pd.testing.assert_frame_equal(split, query_split(captions))
