# Caption Encoder Benchmark

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

