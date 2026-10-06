"""Describe full coverage and evaluate development-selected recipes on held-out queries."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caption_bench.data import load_captions, read_jsonl
from scripts.run_full150_study import PROFILES, STUDY


def interval(values):
    values = np.asarray(values, dtype=float)
    means = np.random.default_rng(42).choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [.025, .975])


def proportion_interval(values):
    values = np.asarray(values, dtype=float)
    n, p, z = len(values), values.mean(), 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return np.clip([center-radius, center+radius], 0, 1)


def query_split(captions):
    pilot = {r['sample_id'] for r in read_jsonl(ROOT / 'variants/gemini_pilot_v2/variants.jsonl')}
    dev = set()
    for domain in sorted({c.domain for c in captions}):
        ids = [c.sample_id for c in captions if c.domain == domain]
        existing = sorted(set(ids) & pilot)
        remaining = sorted(set(ids) - pilot, key=lambda s: hashlib.sha256(('42:' + s).encode()).hexdigest())
        dev.update(existing + remaining[:25 - len(existing)])
    return pd.DataFrame([dict(sample_id=c.sample_id, domain=c.domain, original_split=c.split,
                             query_partition='development' if c.sample_id in dev else 'held_out') for c in captions])


def summarize(frame, image):
    p = 'image_' if image else ''
    groups = ['model', 'domain', 'variant_type']
    identity = frame[frame.variant_type == 'identity'].set_index(['model', 'sample_id'])
    rows = []
    for (model, domain, typ), group in frame.groupby(groups):
        base = identity.loc[[(model, s) for s in group.sample_id]]
        row = dict(model=model, domain=domain, variant_type=typ, n=len(group))
        for output, column in [('R1', p+'recall_at_1'), ('R5', p+'recall_at_5'),
                               ('MRR', p+'reciprocal_rank'), ('margin', p+'margin')]:
            a, b = group[column].to_numpy(), base[column].to_numpy()
            row[output] = float(a.mean())
            row[output+'_ci_low'], row[output+'_ci_high'] = (proportion_interval(a)
                if output in ['R1', 'R5'] else interval(a))
            row['delta_'+output] = float((a-b).mean())
            row['delta_'+output+'_ci_low'], row['delta_'+output+'_ci_high'] = interval(a-b)
        rows.append(row)
    return pd.DataFrame(rows)


def select_recipes(frame, image):
    p = 'image_' if image else ''
    dev = frame[frame.query_partition == 'development']
    if not image:
        # Searching for an unchanged caption among originals is a trivial self-match.
        # Keep identity in baseline tables, but select text recipes from generated captions.
        dev = dev[dev.variant_type != 'identity']
    scores = dev.groupby(['domain', 'model', 'variant_type']).agg(
        R1=(p+'recall_at_1', 'mean'), MRR=(p+'reciprocal_rank', 'mean'),
        R5=(p+'recall_at_5', 'mean')).reset_index()
    # Primary metric fixed in advance: R@1; MRR and R@5 break development ties.
    scores = scores.sort_values(['R1','MRR','R5','model','variant_type'],
                               ascending=[False,False,False,True,True], kind='stable')
    choices = []
    for (domain, model), group in scores.groupby(['domain', 'model']):
        choices.append(('per_model', group.iloc[0]))
    for domain, group in scores.groupby('domain'):
        choices.append(('overall', group.iloc[0]))
    rows = []
    for scope, chosen in choices:
        group = frame[(frame.domain == chosen.domain) & (frame.model == chosen.model)
                      & (frame.variant_type == chosen.variant_type) & (frame.query_partition == 'held_out')]
        row = dict(selection_scope=scope, domain=chosen.domain, model=chosen.model,
                   variant_type=chosen.variant_type, development_R1=chosen.R1, n=len(group))
        for metric, col in [('R1',p+'recall_at_1'),('R5',p+'recall_at_5'),('MRR',p+'reciprocal_rank')]:
            row['held_out_'+metric] = group[col].mean()
            row['held_out_'+metric+'_ci_low'], row['held_out_'+metric+'_ci_high'] = (
                proportion_interval(group[col]) if metric in ['R1', 'R5'] else interval(group[col]))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    design = json.loads((STUDY / 'design.json').read_text())
    captions = load_captions(ROOT / 'caption_samples_50/all_samples.jsonl')
    partition = query_split(captions)
    partition.to_csv(STUDY / 'query_partitions.csv', index=False)
    frames, images = [], []
    missing = []
    for name in design['models']:
        run = STUDY / name
        if not (run / 'completed.json').exists():
            missing.append(name)
            continue
        detail = pd.read_csv(run / 'details.csv')
        expected = {(c.sample_id, p) for c in captions for p in ['identity'] + PROFILES}
        core = detail[detail.variant_type.isin(['identity'] + PROFILES)]
        if set(zip(core.sample_id, core.variant_type)) != expected or len(core) != len(expected):
            raise ValueError(f'Incomplete query coverage for {name}')
        frames.append(core.merge(partition[['sample_id','query_partition']], on='sample_id', validate='many_to_one'))
        if (run / 'image_retrieval_details.csv').exists():
            img = pd.read_csv(run / 'image_retrieval_details.csv')
            img = img[img.variant_type.isin(['identity'] + PROFILES)]
            if set(zip(img.sample_id, img.variant_type)) != expected or len(img) != len(expected):
                raise ValueError(f'Incomplete image coverage for {name}')
            images.append(img.merge(partition[['sample_id','query_partition']], on='sample_id', validate='many_to_one'))
    if not frames:
        raise ValueError('No completed encoder runs yet')
    recipes = {}
    for name, group, image in [('text', frames, False), ('image', images, True)]:
        if not group:
            continue
        detail = pd.concat(group, ignore_index=True)
        detail.to_csv(STUDY / f'{name}_details.csv', index=False)
        full = summarize(detail, image)
        full.to_csv(STUDY / f'{name}_summary.csv', index=False)
        overall = summarize(detail.assign(domain='all'), image)
        overall.to_csv(STUDY / f'{name}_overall_summary.csv', index=False)
        recipes[name] = select_recipes(detail, image)
        recipes[name].to_csv(STUDY / f'{name}_held_out_recipes.csv', index=False)
        comparison = []
        chosen = recipes[name][recipes[name].selection_scope == 'per_model']
        p = 'image_' if image else ''
        for domain, choices in chosen.groupby('domain'):
            choices = list(choices.itertuples())
            for i, a in enumerate(choices):
                left = detail[(detail.domain == domain) & (detail.model == a.model)
                              & (detail.variant_type == a.variant_type)
                              & (detail.query_partition == 'held_out')].set_index('sample_id')
                for b in choices[i+1:]:
                    right = detail[(detail.domain == domain) & (detail.model == b.model)
                                   & (detail.variant_type == b.variant_type)
                                   & (detail.query_partition == 'held_out')].set_index('sample_id').loc[left.index]
                    row = dict(domain=domain, model_a=a.model, profile_a=a.variant_type,
                               model_b=b.model, profile_b=b.variant_type, n=len(left))
                    for metric, col in [('R1', p+'recall_at_1'), ('R5', p+'recall_at_5'),
                                        ('MRR', p+'reciprocal_rank')]:
                        delta = left[col].to_numpy() - right[col].to_numpy()
                        row['delta_'+metric] = delta.mean()
                        row['delta_'+metric+'_ci_low'], row['delta_'+metric+'_ci_high'] = interval(delta)
                    comparison.append(row)
        pd.DataFrame(comparison).to_csv(STUDY / f'{name}_held_out_model_comparisons.csv', index=False)
    variants = read_jsonl(ROOT / 'variants/full150/variants.jsonl')
    audit = [dict(sample_id=r['sample_id'], variant_type=r['variant_type'],
                  words=len(r['text'].split()), warnings='; '.join(r.get('review_warnings', [])))
             for r in variants if r['variant_type'] in PROFILES]
    pd.DataFrame(audit).to_csv(STUDY / 'caption_audit.csv', index=False)
    lines = ['# 150 görsel üzerinde caption ve encoder karşılaştırması', '',
             f'Durum: {"TAMAMLANDI" if not missing else "KISMİ; eksik modeller: " + ", ".join(missing)}.', '',
             '50 yüz, 50 kişi, 50 araç; altı Gemini biçimi ve orijinal açıklama. '
             'Aday havuzu her kategorinin 50 kaynak dosyasıdır. Görsel testi aynı kaynak fotoğrafı bulur; '
             'farklı kamera/kimlik ReID ölçümü değildir.', '',
             'Gemini mevcut kaynak metni kullanır; fotoğrafı görmez. Pilotun 15 örneği yeniden kullanılmıştır. '
             'Diğer biçimler aynı kaynak metinlerden üretilmiştir. Tek üretim/seed kullanıldığı için '
             'farklar prompt paketi, korunan bilgi ve üretim varyasyonunu birlikte içerir.', '',
             'Her kategoride 25 sorgu geliştirme, 25 sorgu ayrılmış değerlendirme içindir. '
             'Pilot örnekleri geliştirme grubundadır. Tarif/model seçimi geliştirme R@1 ile yapılır; '
             'MRR ve R@5 eşitlikleri çözer. Aday havuzu her iki aşamada da 50 görseldir. '
             'Metin tarif seçimi altı üretilmiş profilden yapılır; orijinalin kendisini '
             'bulması yalnızca kontrol olarak tutulur. Bu, bağımsız bir dış veri seti değildir.', '',
             '## Geliştirme grubunda seçilen tariflerin ayrılmış sorgulardaki başarısı', '']
    lines += ['', 'Caption biçimleri: `llm_brief` kısa; `llm_balanced` dengeli; '
              '`llm_detailed` ayrıntılı; `llm_human_description` doğal insan tarifi; '
              '`llm_attribute_list` özellik listesi; `llm_distinctive_first` ayırt edici '
              'özellikleri önce veren metin. `identity` orijinal kaynak açıklamadır.', '',
              'R@1 doğru kaynağın ilk sırada, R@5 ilk beşte bulunma oranıdır. '
              'Text testi üretilmiş metinle kaynak açıklamayı, image testi metinle '
              'doğru görseli arar. BGE-M3 ve DeBERTa yalnızca text testine katılır.', '',
              '| Test | Kategori | Model | Tarif | İlk sıra | İlk 5 | R@1 %95 aralık |',
              '|---|---|---|---|---:|---:|---|']
    for name, selected in recipes.items():
        for r in selected[selected.selection_scope == 'overall'].itertuples():
            lines.append(f'| {name} | {r.domain} | {r.model} | {r.variant_type} | '
                         f'{r.held_out_R1:.0%} | {r.held_out_R5:.0%} | '
                         f'{r.held_out_R1_ci_low:.0%}–{r.held_out_R1_ci_high:.0%} |')
    for name, selected in recipes.items():
        lines += ['', f'## {name}: her model için geliştirmede seçilen tarif', '',
                  '| Kategori | Model | Tarif | Ayrılmış ilk sıra | Ayrılmış ilk 5 |',
                  '|---|---|---|---:|---:|']
        for r in selected[selected.selection_scope == 'per_model'].itertuples():
            lines.append(f'| {r.domain} | {r.model} | {r.variant_type} | '
                         f'{r.held_out_R1:.0%} | {r.held_out_R5:.0%} |')
    lines += ['', 'CSV dosyaları tüm model/tarif kombinasyonlarını ve aynı sorgulardaki '
              'orijinal tarife göre eşleştirilmiş bootstrap farklarını içerir (2.000 örnekleme). '
              'R@1/R@5 oranlarının mutlak güven aralıkları Wilson yöntemiyle hesaplanır. '
              '150 sorgunun tamamındaki sıralamalar betimseldir; seçimden bağımsız sonuçlar '
              'held_out_recipes dosyalarındadır.', '',
              'Ham cosine modeller arasında kalibre değildir. Küçük farklar, geniş güven '
              'aralıkları ve çoklu karşılaştırmalar kesin üstünlük iddiasını desteklemez. '
              'Caption doğruluğu otomatik uyarılarla izlenir; insan tarafından onaylanmış '
              'görsel etiket olarak yorumlanmamalıdır.', '',
              'Mamba-3 CUDA/Triton olmadığı için bu çalıştırmada dışarıda bırakılmıştır. '
              'CPU float32 kullanılmıştır; önceki pilotun sayısal hassasiyetiyle aynı değildir.']
    accounting_path = STUDY / 'generation_accounting.json'
    if accounting_path.exists():
        accounting = json.loads(accounting_path.read_text())
        flagged = sum(bool(row['warnings']) for row in audit)
        lines += ['', '## Üretim ve doğrulama', '',
                  f"{accounting['variant_count']} Gemini caption tamamlandı; her altı biçimde "
                  '150 kayıt var. Kalan kaynak sayısı sıfır. '
                  f"Son devam işleminde {accounting['requests_in_final_resume']} istekten "
                  '69 başarılı yanıt ve 9 geçici HTTP 503 yanıtı alındı; HTTP 429 oluşmadı. '
                  f"Tüm süreçte kayıtlı toplam istek sayısı {accounting['total_recorded_requests']}. "
                  'Önceki aşamada ikinci prompt ailesi eklenmesi ve tekrarlanan başarısız '
                  'istekler ilk 135 istek beklentisini aşmıştır.', '',
                  f'{flagged}/900 caption otomatik içerik denetiminde uyarı aldı; '
                  'uyarılar `caption_audit.csv` dosyasındadır. Bu metinler sonuçlarda '
                  'tutulmuştur; görsele bakılarak doğrulanmış etiket sayılmamalıdır.']
    (STUDY / 'BULGULAR.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(STUDY / 'BULGULAR.md')
    if missing:
        print('Incomplete encoders:', ', '.join(missing))


if __name__ == '__main__':
    main()
