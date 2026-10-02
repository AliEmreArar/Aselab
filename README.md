# Caption Encoder Benchmark

## Gemini ile gerçek özet ve varyant üretimi

Kelime hedefi olmadan, birini arkadaşına tarif eder gibi günlük dilde caption üretmek için
[prompts/gemini_human_description.md](prompts/gemini_human_description.md) ve
`configs/gemini_human.yaml` kullanılır. Bu config tek API isteği yapar:

```powershell
python -m caption_bench generate --config configs/gemini_human.yaml --sample 170623.jpg
```

Çıktı `variants/gemini_human/variants.jsonl` içindeki `llm_human_description` varyantıdır.
Gemini benchmark'ı ve HTML üreticisi bu varyantı diğer üç özetle birlikte gösterir.

Her çalıştırmada caption başına en fazla **3 API isteği** gönderilir; HTTP tekrarları ve
içerik düzeltme tekrarları aynı sayaca dahildir. Sınırı `max_api_requests_per_caption`
ayarından değiştirebilirsiniz. Terminal her isteği ve HTTP durumunu ayrı gösterir.
Yeni çalıştırma yeni bir istek sayacı başlatır; başarısız komutu tekrar çalıştırmak yeni
API istekleri gönderebilir.

Gemini'nin yanıtları ve yerel doğrulama sonucu `variants/gemini/diagnostics` altında
kaydedilir. `api_requests` altındaki dosyalar HTTP durumunu ve görünür model yanıtını;
`*-validation-*.json` dosyaları kabul/ret gerekçesini gösterir. Anahtar ve HTTP başlıkları
bu kayıtlara yazılmaz. 10/20/40 kelime yaklaşık hedeflerdir; daha kısa veya daha uzun
yanıtlar aynen kabul edilir. Gerçek kelime sayıları sonuçlarda ve HTML'de gösterilir.
Alıntılar, korunan/çıkarılan bilgi listeleri ve açıklama notları isteğe bağlıdır; metni
alıntı formatı veya kelime sayısı yüzünden yeniden üretmeyiz. Sadece eksik/boş caption
ve okunamayan yanıt gibi sonuç kaydetmeyi engelleyen hatalar yeniden denenir.

Gemini entegrasyonu `gemini-3.1-flash-lite` modelini kullanır. API anahtarını proje kökündeki
`.env` dosyasında `GEMINI_API_KEY=` satırına girin. Modeli `GEMINI_MODEL=` satırından
değiştirebilirsiniz. `.env` Git tarafından yok sayılır; anahtar HTML'e veya sonuçlara yazılmaz.
Bağlantı standart Python REST istemcisiyle çalışır; ek Gemini paketi gerekmez.

İlk olarak **170623.jpg için yalnızca caption üretip HTML'de inceleyin**:

```powershell
cd C:\Users\ali\Desktop\aselab\Aselab
python scripts/run_gemini.py --sample 170623.jpg --generate-only
```

Script caption üretir ve `caption_inceleme.html` dosyasını günceller. Üretilen caption'lar
"Gemini özetleri — skor bekleniyor" deneyinde görünür. Encoder çalıştırılmadan hiçbir
benzerlik skoru gösterilmez. Aynı görselin yeni caption'larını üç encoder ile ölçmek için:

```powershell
python scripts/run_gemini.py --sample 170623.jpg
```

Tüm 150 caption için üretim, CLIP/SigLIP2/MiniLM deneyi ve HTML güncellemesi:

```powershell
python scripts/run_gemini.py
```

Aşamaları ayrı ayrı çalıştırmak için:

```powershell
python -m caption_bench generate --config configs/gemini_variants.yaml --sample 170623.jpg
python scripts/build_caption_viewer.py
python -m caption_bench run --config configs/gemini_benchmark.yaml
python scripts/build_caption_viewer.py
```

`--sample` kaldırılınca tüm veri seti üretilir. `--limit 3` küçük bir deneme yapar.
Anahtarsız ve API isteği göndermeden prompt/istek önizlemesi:

```powershell
python -m caption_bench generate --config configs/gemini_variants.yaml --sample 170623.jpg --dry-run
```

Prompt: [prompts/gemini_caption_variants.md](prompts/gemini_caption_variants.md).
Gemini, görseli yeniden yorumlamak yerine **caption'ın tamamından** ayırt edici bilgileri
seçerek yaklaşık 10/20/40 kelimelik doğal özetler yazar. Yeni bilgi eklememesi ve
özellikleri doğru nesneye bağlaması istenir. Sayısal hedefler için kelime sayması,
birebir alıntı yapması veya açıklama yazması zorunlu değildir. Hedefi aşan yanıt da
değiştirilmeden kaydedilir.

Önceden alınmış bir Gemini yanıtını yeni kurallarla API çağrısı yapmadan kullanabilirsiniz:

```powershell
python scripts/run_gemini.py --sample 170623.jpg --generate-only --reuse-response variants/gemini/diagnostics/RESPONSE-validation-2.json
```

`RESPONSE-validation-2.json` yerine kayıt dosyasının gerçek adını yazın. Eski yanıtın
kaynak caption'ı ve modeli eşleşen API kaydından doğrulanır; eski üretimin prompt bilgisi
korunur ve sonuçta `recovered_from` alanı gösterir.

[configs/gemini_variants.yaml](configs/gemini_variants.yaml) içinde `enabled: true` yaparak
paraphrase, bağlama uygun synonym, attribute_order, tek olgunun negation'ı, color_change
ve attribute_exchange deneylerini açabilirsiniz. Uygulanamayan değişiklikler skorlanmaz;
gerekçeleri `generation_manifest.json` içinde saklanır.

Çıktılar `variants/gemini/variants.jsonl`, `checkpoint.json` ve `generation_manifest.json`
dosyalarıdır. Her başarılı caption sonrasında kaydedilir; aynı komut kaldığı yerden devam eder.
Prompt, model veya etkin deneyler değişirse eski ve yeni üretimleri karıştırmamak için config'de
yeni `output_dir` seçin ve benchmark config'inin `variant_files` yolunu güncelleyin. Bu durumda
HTML üreticisindeki `--generated-variants` ve `--gemini-run-dir` seçenekleriyle yeni yolları belirtin.

Yeni encoder sonuçları `runs/gemini` altına gider; eski `compact_*` sonuçları geçmiş deney
olarak korunur. Kelime uzunluğu ile CLIP'in 77 token sınırı farklıdır; encoder kesilme
bilgisini raporda görebilirsiniz. Deney, LLM özetlerinin gerçek uzunluklarıyla encoder
benzerliğini ölçer.

Resmi kaynaklar: [model](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite),
[structured output](https://ai.google.dev/gemini-api/docs/structured-output).

Bu proje, aynı görsel açıklamasının özetlenmiş, yeniden sıralanmış, eş anlamlılarla değiştirilmiş veya paraphrase edilmiş biçimlerinin farklı text encoder'larda ne kadar kararlı kaldığını ölçer. Veri setindeki 150 caption (`person`, `face`, `vehicle`) doğrudan desteklenir.

## Ne ölçülür?

- `cosine`: varyant ile kendi orijinal caption'ı arasındaki benzerlik.
- `Recall@1`, `Recall@5`, `MRR`: varyant, aynı domain'deki 50 orijinal arasından kendi kaynağını geri bulabiliyor mu?
- `margin`: doğru caption skoru eksi en benzer yanlış caption skoru. Pozitif ve büyük olması iyidir.
- `delta_cosine`: anlam-koruyan kontrolün cosine'ı eksi anlam-değiştiren swap/negation cosine'ı.
- encoder agreement: 150x150 matrisler arasında Spearman ve kNN-overlap@10.
- tokenizer uzunluğu ve truncation oranı: özellikle uzun `face` caption'larında model limitinin etkisi.

Ham cosine değerlerini farklı model aileleri arasında doğrudan sıralamak doğru değildir; ölçekler kalibre değildir. Model kıyasında Recall/MRR/margin ve aynı dönüşümün `identity` satırına göre düşüşü daha anlamlıdır.

## Hızlı başlangıç (internet/model indirmesi gerektirmez)

```powershell
python -m pip install -e .
python -m caption_bench run --config configs/quick.yaml
```

Sonuçlar `runs/quick/variants.jsonl`, `details.csv`, `summary.csv`, `contrasts.csv`, `encoder_agreement.csv`, embedding cache'leri, `manifest.json` ve filtrelenebilir `report.html` olarak oluşur. `quick` deneyi yalnız lexical hash baseline'dır; pipeline doğrulaması ve lexical-overlap referansı içindir, gerçek semantic encoder değildir.

Gerçek model deneyi:

```powershell
python -m pip install -e ".[sentence-transformers]"
python -m caption_bench run --config configs/models.example.yaml
```

Öncelikli ilk gerçek koşu için yalnız CLIP B/16:

```powershell
python -m caption_bench run --config configs/clip.yaml
```

Bu config, PyTorch 2.5 ortamlarında eski pickle `.bin` ağırlığını açmamak için model deposundaki doğrulanmış safetensors dönüşüm ref'ini sabitler. PyTorch 2.6+ ve ana dalda safetensors bulunan bir checkpoint kullanıyorsanız `revision` satırı kaldırılabilir.

PDF'deki öncelikli D1-D4 karşılaştırmasını CLIP B/16, SigLIP2 ve MiniLM ile birlikte çalıştırmak için:

```powershell
python -m caption_bench run --config configs/priority.yaml
```

İlk çalıştırma Hugging Face ağırlıklarını indirir. GPU belleği yetmezse config içindeki `batch_size` değerlerini düşürün veya modelleri tek tek çalıştırın.

## Encoder seçimi

İki ayrı soru vardır ve sonuçlarda karıştırılmamalıdır:

1. **Text–image ReID:** CLIP ve SigLIP2 gibi beraber eğitilmiş vision/text çiftleri kullanılmalıdır. Bir CLIP vision encoder ile bağımsız DeBERTa/BGE/E5 embedding'i doğrudan aynı uzayda değildir.
2. **Caption–caption robustness:** BGE-M3 ve multilingual-E5 gibi retrieval encoder'ları burada güçlü karşılaştırma adaylarıdır. Bu deney, metin varyasyonuna duyarlılığı ölçer; tek başına image retrieval başarısını kanıtlamaz.

`configs/models.example.yaml` şu başlangıç matrisini içerir:

| Model | Rol | Not |
|---|---|---|
| CLIP ViT-B/32 | multimodal baseline | Kısa context; uzun caption truncation kritik |
| SigLIP2 Base | güncel multimodal aday | Eşleşen vision branch ile kullanılmalı |
| BGE-M3 | multilingual text retrieval | Caption–caption robustness |
| multilingual-E5-large | multilingual text retrieval | Config'de simetrik `query:` prefix'i uygulanır |
| mDeBERTa-v3-base | diagnostic control | Sentence retrieval için özel eğitilmemiştir |

## Caption varyantları

Yerleşik dönüşümler `identity`, `compact`, `head_words`, `synonym`, `clause_exchange`, `clause_reverse`, `color_change`, `attribute_exchange`, `negation` ve `word_dropout`'tır. `compact` extractive ve tekrarlanabilirdir; gerçek abstractive summary/paraphrase kalitesini temsil etmez. `word_dropout` semantik eşdeğerlik testi değil, kontrollü bilgi kaybı/stres testidir.

LLM veya insan tarafından üretilen varyantlar için [prompts/variant_generation.md](prompts/variant_generation.md) sözleşmesini kullanın. JSONL satırları:

```json
{"sample_id":"person::train::00197_4.png","variant_type":"summary_20w","text":"...","generator":"model-v1","relation":"information_reduced"}
```

Dosyayı config'e ekleyin:

```yaml
variant_files:
  - ../variants/my_curated_variants.jsonl
```

Yeni encoder eklemek için `caption_bench/encoders.py` içindeki `TextEncoder` arayüzünü, yeni dönüşüm için `caption_bench/transforms.py` içindeki registry'yi kullanın.

## Önerilen deney sırası

1. Hash baseline ile veri/rapor akışını doğrulayın.
2. CLIP ve SigLIP2'yi ayrı ayrı çalıştırın; `face` truncation oranını kontrol edin.
3. BGE-M3 ve E5'i caption–caption referansı olarak ekleyin.
4. Her caption için insan/LLM kontrollü `summary`, `paraphrase`, `synonym`, `attribute_order` varyantları üretin.
5. Varyantların gerçekten eşdeğer olduğunu küçük bir manuel denetimle doğrulayın.
6. Son aşamada eşleşen vision encoder ile image→text ve text→image Recall@K ölçümünü ayrı bir deney olarak ekleyin.

