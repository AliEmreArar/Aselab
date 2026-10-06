"""Build a shareable Turkish report from all eight completed full150 runs.

Uses source rows for paired comparisons, not a ranking of raw cosine scales.
Run after analyze_full150_study.py and plot_full150_study.py.
"""
from pathlib import Path
import base64
import html
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'runs/full150'
PROFILES = ['llm_brief', 'llm_balanced', 'llm_detailed', 'llm_human_description',
            'llm_attribute_list', 'llm_distinctive_first']
LABELS = dict(identity='Orijinal', llm_brief='Kısa', llm_balanced='Dengeli',
              llm_detailed='Detaylı', llm_human_description='Gündelik anlatım',
              llm_attribute_list='Özellik listesi', llm_distinctive_first='Ayırt edici önce')
DOMAINS = dict(face='Yüz', person='İnsan', vehicle='Taşıt')
MODELS = dict(clip_b16='CLIP B/16', clip_l14='CLIP L/14', siglip2_b16_224='SigLIP2',
              jina_clip_v2='Jina-CLIP-v2', eva02_clip_l14='EVA02-CLIP', altclip='AltCLIP',
              bge_m3='BGE-M3', deberta_v3_base_mean_pool='DeBERTa (mean pool)')


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    means = np.random.default_rng(42).choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [.025, .975])


def pct(v):
    return f'{100*v:.0f}%'


class Report:
    def __init__(self):
        self.md, self.web = [], []

    def heading(self, title, level=2):
        self.md += ['#'*level + ' ' + title, '']
        self.web.append(f'<h{level}>{html.escape(title)}</h{level}>')

    def paragraph(self, text):
        self.md += [text, '']
        self.web.append('<p>' + html.escape(text) + '</p>')

    def table(self, headers, rows):
        self.md += ['| ' + ' | '.join(headers) + ' |',
                    '| ' + ' | '.join(['---']*len(headers)) + ' |']
        self.md += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows] + ['']
        self.web.append('<div class="table"><table><thead><tr>' + ''.join(
            '<th>'+html.escape(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
            '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>'
            for row in rows) + '</tbody></table></div>')

    def figure(self, filename, caption):
        path = STUDY / filename
        self.md += [f'![{caption}]({filename})', '']
        encoded = base64.b64encode(path.read_bytes()).decode()
        self.web.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
                        f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def validate_coverage(text, image, design):
    ids = [json.loads(line) for line in (ROOT / 'caption_samples_50/all_samples.jsonl').read_text().splitlines()]
    expected_ids = {f'{r["domain"]}::{r["split"]}::{r["filename"]}' for r in ids}
    required = {(sid, p) for sid in expected_ids for p in ['identity'] + PROFILES}
    assert set(text.model) == set(design['models'])
    assert len(set(image.model)) == 6
    for frame in [text, image]:
        for model, group in frame.groupby('model'):
            assert len(group) == 1050, (model, len(group))
            assert set(zip(group.sample_id, group.variant_type)) == required
            assert not group.duplicated(['sample_id', 'variant_type']).any()
            for domain, part in group.groupby('domain'):
                assert len(part.sample_id.unique()) == 50
                assert part[part.query_partition == 'held_out'].sample_id.nunique() == 25


def main():
    design = json.loads((STUDY / 'design.json').read_text())
    text = pd.read_csv(STUDY / 'text_details.csv')
    image = pd.read_csv(STUDY / 'image_details.csv')
    validate_coverage(text, image, design)
    report = Report()
    report.heading('Caption değişiklikleri encoder ve retrieval sonucunu nasıl etkiliyor?', 1)
    report.paragraph('150 görsel, 900 Gemini caption, 8 text encoder ve 6 hizalanmış vision–text çifti. '
                     'Tamamlanan çalışma • 6 Ekim 2026. Ölçülen görev aynı kaynak metin/fotoğrafı bulmadır; '
                     'farklı kamera veya pozlar arasında kimlik ReID değildir.')
    report.heading('1. Bu çalışmadan elde ettiğimiz çıktı')
    report.paragraph('Caption dönüşümlerini iki ayrı ölçümle karşılaştırdık: orijinal–varyant cosine ve '
                     'kaynak caption ayrımı, ayrıca doğru görselin retrieval sırası. '
                     'Böylece metin yakınlığının tek başına başarı olmadığını ve hangi caption profilinin '
                     'hangi encoder/kategoride işe yaradığını sayısal olarak raporlayabiliyoruz. '
                     'BGE-M3 ve DeBERTa text-only sonuç verir; bu iki model için görsel başarısı iddia etmiyoruz.')
    focus = image[(image.model == 'clip_l14') & (image.domain == 'face')
                  & (image.query_partition == 'held_out')]
    original_r1 = focus[focus.variant_type == 'identity'].image_recall_at_1.mean()
    human_r1 = focus[focus.variant_type == 'llm_human_description'].image_recall_at_1.mean()
    report.paragraph(f'En açık örneklerden biri yüz caption’ları: CLIP L/14 için geliştirmede '
                     f'seçilen gündelik anlatım, ayrılmış sorgularda doğru görseli ilk sırada '
                     f'bulmayı {pct(original_r1)}’ten {pct(human_r1)}’e taşıyor. '
                     'İnsan ve taşıtta her dönüşüm aynı kazanımı getirmiyor. '
                     'Dolayısıyla özetlemenin etkisi var, fakat evrensel bir en iyi uzunluk veya profil yok.')
    diagnostic = text[(text.model == 'deberta_v3_base_mean_pool') & (text.domain == 'face')
                      & (text.variant_type == 'llm_balanced')]
    report.paragraph(f'Yüksek cosine’ın sınırı: DeBERTa’nın dengeli yüz caption’larında '
                     f'orijinale ortalama yakınlığı {diagnostic.cosine.mean():.3f}, '
                     f'fakat kaynak caption’ı ilk sırada bulma oranı {pct(diagnostic.recall_at_1.mean())}. '
                     'Bu ham mean-pooling kullanımında yakınlık skoru ayırt edicilik yerine geçmiyor; '
                     'bulgu bütün DeBERTa tabanlı embedding modellerine genellenmemelidir.')
    report.heading('2. Deney tasarımı ve okunacak ölçüler')
    report.table(['Bileşen', 'Uygulama'], [
        ['Veri', '50 yüz + 50 insan + 50 taşıt; kategori başına 50 aday'],
        ['Caption', 'Orijinal + 6 Gemini profili; kesin kelime sınırı yok; Gemini görüntüyü görmez'],
        ['Kontrol', f'{design["controls"]} tek-ifade kontrolü; dengeli caption üzerinden üretilir'],
        ['Seçim', 'Kategori başına 25 geliştirme sorgusu; pilot örnekleri bu grupta'],
        ['Değerlendirme', 'Kategori başına diğer 25 sorgu; galeri yine aynı 50 kaynak dosyası'],
        ['R@1 / R@5', 'Doğru kaynak/görselin ilk sırada / ilk beşte bulunma oranı'],
        ['Margin', 'Doğru aday skoru − en güçlü rakip skoru; pozitif ise doğru aday öndedir'],
        ['Cosine', 'Metin temsilindeki değişim; modeller arasında ortak başarı ölçeği değildir'],
    ])
    report.paragraph('Tek Gemini üretimi/seed var. Profiller yalnızca uzunluğu değil, bilgi seçimini, '
                     'ifade biçimini ve özellik sırasını da değiştiriyor. Sonuçları salt uzunluk etkisi '
                     'veya promptun bağımsız nedensel etkisi olarak yorumlayamayız. '
                     'Ayrılmış sorgular aynı veri havuzunun içindedir; bağımsız dış veri veya kimlik split’i değildir.')
    report.heading('3. Caption görseli bulmayı ne kadar değiştirdi?')
    selected = pd.read_csv(STUDY / 'image_held_out_recipes.csv')
    comparison, rows = [], []
    for r in selected[selected.selection_scope == 'per_model'].itertuples():
        group = image[(image.model == r.model) & (image.domain == r.domain)
                      & (image.query_partition == 'held_out')]
        base = group[group.variant_type == 'identity'].set_index('sample_id')
        variant = group[group.variant_type == r.variant_type].set_index('sample_id').loc[base.index]
        delta = variant.image_recall_at_1.to_numpy() - base.image_recall_at_1.to_numpy()
        low, high = paired_interval(delta)
        record = dict(model=r.model, domain=r.domain, profile=r.variant_type, n=len(delta),
                      baseline_R1=base.image_recall_at_1.mean(), selected_R1=variant.image_recall_at_1.mean(),
                      delta_R1=delta.mean(), ci_low=low, ci_high=high,
                      gained=int((delta > 0).sum()), lost=int((delta < 0).sum()))
        comparison.append(record)
        rows.append([DOMAINS[r.domain], MODELS[r.model], LABELS[r.variant_type],
                     f'{int(base.image_recall_at_1.sum())}/25', f'{int(variant.image_recall_at_1.sum())}/25',
                     f'{100*delta.mean():+.0f} pp', f'{100*low:+.0f}…{100*high:+.0f} pp'])
    pd.DataFrame(comparison).to_csv(STUDY / 'selected_image_vs_original_held_out.csv', index=False)
    report.paragraph('Her model için caption tarifi geliştirme sorgularında seçildi. Aşağıda aynı '
                     '25 ayrılmış sorguda orijinalle eşleştirilmiş karşılaştırması var. '
                     'pp yüzde puandır; fark aralığı 2.000 eşleştirilmiş bootstrap örneklemesidir. '
                     'Aralık sıfırı kapsıyorsa net iyileşme iddiası kurmayın. '
                     'Çoklu karşılaştırma düzeltmesi yapılmamıştır.')
    report.table(['Kategori', 'Model', 'Geliştirmede seçilen tarif', 'Orijinal ilk sıra',
                  'Seçilen ilk sıra', 'Fark', 'Fark %95 aralık'], rows)
    report.heading('4. Kısa, dengeli ve uzun caption: aynı modeli sabit tutunca')
    summary = pd.read_csv(STUDY / 'image_summary.csv')
    rows = []
    for domain in DOMAINS:
        part = summary[(summary.model == 'clip_l14') & (summary.domain == domain)].set_index('variant_type')
        rows.append([DOMAINS[domain]] + [pct(part.loc[p, 'R1']) for p in ['identity'] + PROFILES])
    report.table(['CLIP L/14 · 50 sorgu', 'Orijinal', 'Kısa', 'Dengeli', 'Detaylı', 'Gündelik',
                  'Özellik listesi', 'Ayırt edici önce'], rows)
    report.paragraph('Bu tablo tüm 50 sorgunun betimsel karşılaştırmasıdır; en yüksek hücreyi seçip '
                     'bağımsız test başarısı diye sunmuyoruz. Yukarıdaki ayrılmış sorgu tablosu seçim sonrası değerlendirmedir.')
    for domain in DOMAINS:
        part = summary[(summary.model == 'clip_l14') & (summary.domain == domain)]
        base = part[part.variant_type == 'identity'].iloc[0]
        best = part[part.variant_type.isin(PROFILES)].sort_values(
            ['R1', 'variant_type'], ascending=[False, True]).iloc[0]
        report.paragraph(f'CLIP L/14 ile betimsel {DOMAINS[domain].lower()} örneği: '
                         f'orijinal R@1 {pct(base.R1)}; en yüksek gözlenen üretilmiş profil '
                         f'{LABELS[best.variant_type]} ile {pct(best.R1)}. '
                         f'Eşleştirilmiş fark {100*best.delta_R1:+.0f} yüzde puan; '
                         f'%95 bootstrap aralığı {100*best.delta_R1_ci_low:+.0f}…'
                         f'{100*best.delta_R1_ci_high:+.0f} yüzde puan. '
                         'Bu profil bu tablonun sonucuna bakılarak seçildiği için keşifsel bir örnektir.')
    report.figure('image_recall_at_1.png', 'Tüm multimodal modellerde caption profili ve görsel R@1 — betimsel sonuç')
    report.heading('5. Kısa ve uzun caption’ların vektörleri ne kadar yakın?')
    rows, vector_stats = [], []
    for (model, domain), group in text.groupby(['model', 'domain']):
        item = dict(model=model, domain=domain)
        values = []
        for profile in ['llm_brief', 'llm_balanced', 'llm_detailed']:
            part = group[group.variant_type == profile]
            item[profile+'_cosine'] = part.cosine.mean()
            item[profile+'_text_R1'] = part.recall_at_1.mean()
            values.append(f'{part.cosine.mean():.3f} / {pct(part.recall_at_1.mean())}')
        vector_stats.append(item)
        rows.append([DOMAINS[domain], MODELS[model]] + values)
    pd.DataFrame(vector_stats).to_csv(STUDY / 'short_long_vector_comparison.csv', index=False)
    report.paragraph('Hücreler: orijinal–varyant ortalama cosine / kaynak caption R@1. '
                     '50 sorgu üzerindeki metin sonuçlarıdır. Yüksek cosine ile düşük kaynak R@1 '
                     'birlikte görülebilir: rakip metinler de aynı derecede yakın olabilir. '
                     'Detaylı metnin daha yakın olması retrieval üstünlüğünü tek başına kanıtlamaz.')
    report.table(['Kategori', 'Encoder', 'Kısa cosine / kaynak R@1', 'Dengeli cosine / kaynak R@1',
                  'Detaylı cosine / kaynak R@1'], rows)
    report.figure('text_recall_at_1.png', 'Sekiz encoder ile kaynak caption ayrımı — görsel ReID değil')
    report.heading('6. Token sınırı ve tek-olgu kontrolleri')
    truncation = text.groupby(['model', 'domain', 'variant_type']).agg(
        n=('sample_id', 'size'), mean_tokens=('variant_tokens', 'mean'),
        truncation_rate=('variant_truncated', 'mean')).reset_index()
    truncation.to_csv(STUDY / 'token_truncation_summary.csv', index=False)
    rows = []
    for model in ['clip_b16', 'clip_l14', 'siglip2_b16_224', 'eva02_clip_l14']:
        for domain in DOMAINS:
            part = truncation[(truncation.model == model) & (truncation.domain == domain)].set_index('variant_type')
            rows.append([MODELS[model], DOMAINS[domain]] + [pct(part.loc[p, 'truncation_rate'])
                        for p in ['identity', 'llm_brief', 'llm_balanced', 'llm_detailed']])
    report.table(['Model', 'Kategori', 'Orijinal kesilme', 'Kısa kesilme', 'Dengeli kesilme', 'Detaylı kesilme'], rows)
    report.paragraph('Full150 koşusunda yerel token sınırını aşan metinler kesilerek işlendi. '
                     'Bu modellerde uzun caption sonuçları bütün metne ait değildir. '
                     'Bilgi kaybı, ifade değişikliği ve truncation etkileri birlikte bulunur. '
                     'Önceki chunking pilotu ayrıdır; full150 için truncation/chunking ablation’ı yapılmış sayılmaz.')
    controls = pd.concat([pd.read_csv(STUDY / model / 'control_sensitivity_details.csv')
                          for model in design['models']], ignore_index=True)
    agg = controls.groupby('model').agg(n=('sample_id', 'size'),
        gap=('preserving_minus_changed', 'mean'), win_rate=('preserving_wins', 'mean'),
        tie_rate=('tie', 'mean')).reset_index()
    agg.to_csv(STUDY / 'control_sensitivity_all_models.csv', index=False)
    report.table(['Model', 'Eşleştirilmiş kontrol', 'Ortalama cosine farkı', 'Koruyan değişiklik daha yakın', 'Eşitlik'],
                 [[MODELS[r.model], r.n, f'{r.gap:+.4f}', pct(r.win_rate), pct(r.tie_rate)]
                  for r in agg.itertuples()])
    report.paragraph('Kontrol referansı aynı dengeli caption’dır. Fark = eş anlamlı dönüşüm cosine’ı − '
                     'tek-olgu değişikliği cosine’ı. Pozitif değer, o dönüşüm çiftine duyarlılığı gösterir. '
                     'Dönüşümler sözcük sayısı/token konumu bakımından eşleştirilmiş değildir; '
                     'bu test tam bir anlam doğruluğu testi değildir. Görseldeki olgunun doğruluğu ayrıca denetlenmelidir.')
    report.heading('7. 170623.jpg örneğinin güncel sonucu')
    sample = image[image.filename == '170623.jpg']
    report.table(['Model', 'Orijinal sıra', 'Kısa sıra', 'Dengeli sıra', 'Detaylı sıra',
                  'Gündelik sıra', 'Liste sıra', 'Ayırt edici önce sıra'],
                 [[MODELS[m]] + [int(g.set_index('variant_type').loc[p, 'image_rank'])
                                 for p in ['identity'] + PROFILES] for m, g in sample.groupby('model')])
    report.paragraph('Küçük sıra daha iyidir. Bu tek örnek tüm veri setini temsil etmez. '
                     'Bu koşu CPU float32’dır; geçmiş fp16 pilotundan sayısal olarak farklı olabilir.')
    report.heading('8. Sunumda söyleyeceğimiz sonuç ve sonraki prompt kararı')
    for domain in DOMAINS:
        choice = selected[(selected.selection_scope == 'overall') & (selected.domain == domain)].iloc[0]
        report.paragraph(f'{DOMAINS[domain]}: geliştirmede seçilen model/tarif '
                         f'{MODELS[choice.model]} + {LABELS[choice.variant_type]}. '
                         f'Ayrılmış 25 sorguda R@1 {pct(choice.held_out_R1)}, '
                         f'R@5 {pct(choice.held_out_R5)}. Bu çift, tüm modeller arasında kesin üstün '
                         'ilan edilen değil, geliştirme grubunun seçtiği adaydır.')
    report.paragraph('Paylaşılacak sonuç: Caption biçimi metin temsilini ve kaynak görsel retrieval sırasını '
                     'değiştiriyor. Salt orijinale yüksek cosine veya salt kısa/uzun olma, '
                     'başarı seçimi için yeterli değil. Prompt seçimini hedef encoder ve kategori için '
                     'doğru görsel R@1/R@5, eşleştirilmiş fark ve görsel doğruluk denetimiyle yapmalıyız. '
                     'Tek bir evrensel caption profili belirlediğimizi söylemiyoruz.')
    report.paragraph('Bir sonraki üretim için: kaynakta açıkça bulunan nesne–özellik ilişkilerini koru; '
                     'ayırt edici kombinasyonu metnin başına al; genel anatomi, ışık/arka plan ve '
                     'desteksiz çıkarımları çıkar. Bu, yeni prompt adayıdır, burada ayrıca doğrulanmış '
                     'yeni bir sonuç değildir. Seçilen mevcut profili referans tutarak ayrı görsellerde '
                     'birkaç üretim tekrarıyla kıyasla. CLIP/SigLIP kullanılıyorsa token kesilmesini '
                     'ayrı ölç; değişen bilgileri yalnızca başa taşımayla karıştırma.')
    audit = pd.read_csv(STUDY / 'caption_audit.csv')
    flagged = int(audit.warnings.fillna('').ne('').sum())
    report.paragraph(f'Sınırlar: {flagged}/900 metin otomatik denetimde uyarılı; uyarısız olmak doğruluk '
                     'onayı değildir. Aynı kimliğe ait birden fazla aday varsa mevcut dosya-eşleşmesi '
                     'metriği bunları pozitif saymaz. Kimlik etiketli, kamera/poz ayrılmış sorgu–galeri '
                     'testi olmadan gerçek ReID sonucuna genelleme yapamayız. '
                     'Modellerin vision tower’ları da farklıdır; aralarındaki görsel performans '
                     'farkını yalnızca text tower’a yükleyemeyiz. Dört koşu başka hostta, kalan dört '
                     'koşu yerelde tamamlandı; her ikisinde CPU float32 fakat runtime sürümleri farklıdır. '
                     'AltCLIP/Jina tokenizer regex uyarısının etkisi ayrı izole edilmedi; '
                     'kesin model üstünlüğünden önce aynı runtime ve tokenizer denetimiyle tekrarlama gerekir.')
    report.heading('Kaynaklar ve yeniden üretim')
    report.paragraph('BULGULAR.md ayrılmış sorgu sonuçlarını; image/text_summary.csv tüm kombinasyonları; '
                     'selected_image_vs_original_held_out.csv eşleştirilmiş farkları; '
                     'short_long_vector_comparison.csv cosine ve kaynak R@1 karşılaştırmasını içerir. '
                     'Model bazlı klasörler caption snapshot’ını, embedding cache’ini ve çalışma kaydını korur. '
                     'Full150 model listesi sekiz encoder içerir; Mamba-3 önceki pilotta ayrı değerlendirilmiştir.')
    report.web.append('<p><a href="BULGULAR.md">Sayısal bulgular</a> · '
                      '<a href="selected_image_vs_original_held_out.csv">Eşleştirilmiş değişim CSV</a> · '
                      '<a href="short_long_vector_comparison.csv">Kısa/uzun vektör CSV</a></p>')
    (STUDY / 'SUNUM_RAPORU.md').write_text('\n'.join(report.md), encoding='utf-8')
    page = '<!doctype html><html lang="tr"><meta charset="utf-8"><title>Caption ve encoder deney sonuçları</title>'
    page += '''<style>body{font:16px/1.6 Arial,sans-serif;color:#203040;margin:0;background:#f3f5f8}
    main{max-width:1200px;margin:auto;padding:40px;background:white}h1{font-size:32px;line-height:1.25}
    h2{font-size:23px;margin-top:40px;border-top:1px solid #ccd5df;padding-top:20px}
    .table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}
    th{background:#20364f;color:white}td,th{padding:9px;text-align:left;border-bottom:1px solid #d9e0e7}
    tr:nth-child(even){background:#f5f7fa}figure{margin:24px 0}img{max-width:100%}figcaption{font-size:13px;color:#596879}
    a{color:#176591}@media print{body{background:white}main{padding:0;max-width:none}h2{break-after:avoid}
    tr{break-inside:avoid}figure{break-inside:avoid}table{font-size:10px}.table{overflow:visible}}
    </style><main>'''
    (STUDY / 'SUNUM_RAPORU.html').write_text(page + '\n'.join(report.web) + '</main></html>', encoding='utf-8')
    print(STUDY / 'SUNUM_RAPORU.html')


if __name__ == '__main__':
    main()
