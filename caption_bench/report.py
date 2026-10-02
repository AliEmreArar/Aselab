from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _records(frame: pd.DataFrame) -> str:
    clean = frame.where(pd.notna(frame), None)
    return json.dumps(clean.to_dict(orient="records"), ensure_ascii=False)


def build_report(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    details = pd.read_csv(run_path / "details.csv")
    summary = pd.read_csv(run_path / "summary.csv")
    contrast_path = run_path / "contrast_summary.csv"
    agreement_path = run_path / "encoder_agreement.csv"
    diagnostics_path = run_path / "embedding_diagnostics.csv"
    image_summary_path = run_path / "image_retrieval_summary.csv"
    contrast = pd.read_csv(contrast_path) if contrast_path.exists() else pd.DataFrame()
    agreement = pd.read_csv(agreement_path) if agreement_path.exists() else pd.DataFrame()
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    image_summary = pd.read_csv(image_summary_path) if image_summary_path.exists() else pd.DataFrame()
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    payload_summary = _records(summary)
    payload_details = _records(details)
    payload_contrast = _records(contrast)
    payload_agreement = _records(agreement)
    payload_diagnostics = _records(diagnostics)
    payload_image_summary = _records(image_summary)
    title = html.escape(str(manifest.get("name", run_path.name)))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Caption Encoder Benchmark</title>
<style>
:root{{--bg:#f5f3ee;--panel:#fff;--ink:#172026;--muted:#647078;--accent:#145c55;--line:#ddd8cd;--good:#0b7a53;--bad:#b34233}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{background:#173b3a;color:white;padding:28px clamp(18px,5vw,70px)}} h1{{margin:0 0 6px;font-size:28px}} header p{{margin:0;color:#cfe2de}}
main{{padding:22px clamp(12px,4vw,56px)}} .controls,.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}}
.controls{{display:flex;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:2}} label{{font-size:12px;color:var(--muted);display:grid;gap:4px}} select{{min-width:170px;padding:7px;border:1px solid var(--line);border-radius:6px;background:white}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}} .metric{{background:#edf3f1;border-radius:8px;padding:12px}} .metric b{{display:block;font-size:22px;color:var(--accent)}}
.barrow{{display:grid;grid-template-columns:minmax(150px,260px) 1fr 58px;gap:9px;align-items:center;margin:8px 0}} .bar{{height:15px;background:#e8e6df;border-radius:10px;overflow:hidden}} .fill{{height:100%;background:var(--accent)}}
.tablewrap{{overflow:auto;max-height:620px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border-bottom:1px solid #ece8df;padding:7px;text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#f6f8f7}} .text{{min-width:280px;max-width:520px}} .warn{{color:var(--bad)}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}.barrow{{grid-template-columns:120px 1fr 48px}}}}
</style></head><body>
<header><h1>{title}</h1><p>Caption invariance, within-domain retrieval and tokenizer diagnostics</p></header>
<main><section class="controls"><label>Model<select id="model"></select></label><label>Domain<select id="domain"></select></label><label>Variant<select id="variant"></select></label></section>
<section class="card"><div class="grid" id="metrics"></div></section>
<section class="card"><h2>Variant comparison</h2><div id="bars"></div></section>
<section class="card"><h2>Image retrieval (multimodal ReID)</h2><p>Each caption ranks its matching image among images in the same domain.</p><div id="imagebars"></div></section>
<section class="card"><h2>Semantic-change sensitivity</h2><p>Positive Δ cosine means the encoder kept the meaning-preserving control closer than the changed fact/swap.</p><div id="contrasts"></div></section>
<section class="card"><h2>Encoder agreement</h2><p>Pairwise comparison of the full original-caption geometry. This is rank-based and therefore safer than comparing raw cosine scales.</p><div class="tablewrap"><table><thead><tr><th>Model A</th><th>Model B</th><th>Spearman</th><th>kNN overlap</th></tr></thead><tbody id="agreement"></tbody></table></div></section>
<section class="card"><h2>Embedding diagnostics</h2><div class="tablewrap"><table><thead><tr><th>Model</th><th>Domain</th><th>Within-domain cosine</th><th>Domain silhouette</th></tr></thead><tbody id="diagnostics"></tbody></table></div></section>
<section class="card"><h2>Hard cases</h2><p>Lowest target-vs-hardest-negative margin first. Retrieval candidates are restricted to the same domain.</p><div class="tablewrap"><table><thead><tr><th>Model / domain / variant</th><th>Margin</th><th>Cosine</th><th>Rank</th><th>Original</th><th>Variant</th></tr></thead><tbody id="cases"></tbody></table></div></section>
</main><script>
const summary={payload_summary}, details={payload_details}, contrast={payload_contrast}, agreement={payload_agreement}, diagnostics={payload_diagnostics}, imageSummary={payload_image_summary};
const $=id=>document.getElementById(id); const choices=(id,values)=>{{$(id).innerHTML='<option value="*">All</option>'+[...new Set(values)].sort().map(x=>`<option>${{x}}</option>`).join('')}};
choices('model',summary.map(x=>x.model)); choices('domain',summary.map(x=>x.domain)); choices('variant',summary.map(x=>x.variant_type));
const keep=x=>(!$('model').value||$('model').value==='*'||x.model===$('model').value)&&($('domain').value==='*'||x.domain===$('domain').value)&&($('variant').value==='*'||x.variant_type===$('variant').value);
const value=(x,k)=>x[k]??(k.endsWith('chunks')?1:k.endsWith('overflowed')?x[k.replace('overflowed','truncated')]:0); const mean=(a,k)=>a.length?a.reduce((s,x)=>s+(+value(x,k)||0),0)/a.length:0; const pct=x=>(100*x).toFixed(1)+'%'; const num=x=>(+x).toFixed(3);
function render(){{const d=details.filter(keep),s=summary.filter(keep); $('metrics').innerHTML=[['Pairs',d.length],['Mean cosine',num(mean(d,'cosine'))],['Recall@1',pct(mean(d,'recall_at_1'))],['MRR',num(mean(d,'reciprocal_rank'))],['Mean margin',num(mean(d,'margin'))],['Recall@5',pct(mean(d,'recall_at_5'))],['Original over limit',pct(mean(d,'original_overflowed'))],['Variant over limit',pct(mean(d,'variant_overflowed'))],['Actually truncated',pct(mean(d,'original_truncated'))],['Mean original chunks',num(mean(d,'original_chunks'))],['Mean variant chunks',num(mean(d,'variant_chunks'))],['Variant truncated',pct(mean(d,'variant_truncated'))]].map(x=>`<div class="metric"><span>${{x[0]}}</span><b>${{x[1]}}</b></div>`).join('');
const grouped={{}}; s.forEach(x=>{{const key=`${{x.model}} · ${{x.domain}} · ${{x.variant_type}}`;grouped[key]=+x.recall_at_1}}); $('bars').innerHTML=Object.entries(grouped).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<div class="barrow"><span>${{k}}</span><div class="bar"><div class="fill" style="width:${{100*v}}%"></div></div><b>${{pct(v)}}</b></div>`).join('')||'No rows';
const imageRows=imageSummary.filter(keep); $('imagebars').innerHTML=imageRows.sort((a,b)=>b.delta_image_recall_at_1-a.delta_image_recall_at_1).map(x=>`<div class="barrow"><span>${{x.model}} · ${{x.domain}} · ${{x.variant_type}}</span><div class="bar"><div class="fill" style="width:${{100*x.image_recall_at_1}}%"></div></div><b>${{pct(x.image_recall_at_1)}} · Δ${{x.delta_image_recall_at_1>=0?'+':''}}${{(100*x.delta_image_recall_at_1).toFixed(0)}}pp</b></div>`).join('')||'No multimodal model enabled';
const c=contrast.filter(keep); $('contrasts').innerHTML=c.sort((a,b)=>b.delta_cosine_mean-a.delta_cosine_mean).map(x=>`<div class="barrow"><span>${{x.model}} · ${{x.domain}} · ${{x.contrast}}</span><div class="bar"><div class="fill" style="width:${{Math.max(0,Math.min(100,50+200*x.delta_cosine_mean))}}%"></div></div><b>${{num(x.delta_cosine_mean)}}</b></div>`).join('')||'No configured matched contrasts';
$('agreement').innerHTML=agreement.map(x=>`<tr><td>${{x.model_a}}</td><td>${{x.model_b}}</td><td>${{num(x.spearman)}}</td><td>${{num(x.knn_overlap)}} @ ${{x.knn_k}}</td></tr>`).join('')||'<tr><td colspan="4">Run at least two encoders to calculate agreement.</td></tr>';
$('diagnostics').innerHTML=diagnostics.filter(x=>($('model').value==='*'||x.model===$('model').value)&&($('domain').value==='*'||x.domain===$('domain').value)).map(x=>`<tr><td>${{x.model}}</td><td>${{x.domain}}</td><td>${{num(x.within_domain_cosine_mean)}}</td><td>${{num(x.global_domain_silhouette)}}</td></tr>`).join('');
$('cases').innerHTML=d.sort((a,b)=>a.margin-b.margin).slice(0,100).map(x=>`<tr><td>${{x.model}}<br>${{x.domain}} · ${{x.variant_type}}</td><td class="${{x.margin<0?'warn':''}}">${{num(x.margin)}}</td><td>${{num(x.cosine)}}</td><td>${{x.rank}}/${{x.retrieval_pool_size}}</td><td class="text">${{x.original_text}}</td><td class="text">${{x.variant_text}}</td></tr>`).join('')}}
['model','domain','variant'].forEach(id=>$(id).onchange=render); render();
</script></body></html>"""
    target = run_path / "report.html"
    target.write_text(document, encoding="utf-8")
    return target

