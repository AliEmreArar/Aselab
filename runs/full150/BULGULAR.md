# 150 görsel üzerinde caption ve encoder karşılaştırması

Durum: TAMAMLANDI.

50 yüz, 50 kişi, 50 araç; altı Gemini biçimi ve orijinal açıklama. Aday havuzu her kategorinin 50 kaynak dosyasıdır. Görsel testi aynı kaynak fotoğrafı bulur; farklı kamera/kimlik ReID ölçümü değildir.

Gemini mevcut kaynak metni kullanır; fotoğrafı görmez. Pilotun 15 örneği yeniden kullanılmıştır. Diğer biçimler aynı kaynak metinlerden üretilmiştir. Tek üretim/seed kullanıldığı için farklar prompt paketi, korunan bilgi ve üretim varyasyonunu birlikte içerir.

Her kategoride 25 sorgu geliştirme, 25 sorgu ayrılmış değerlendirme içindir. Pilot örnekleri geliştirme grubundadır. Tarif/model seçimi geliştirme R@1 ile yapılır; MRR ve R@5 eşitlikleri çözer. Aday havuzu her iki aşamada da 50 görseldir. Metin tarif seçimi altı üretilmiş profilden yapılır; orijinalin kendisini bulması yalnızca kontrol olarak tutulur. Bu, bağımsız bir dış veri seti değildir.

## Geliştirme grubunda seçilen tariflerin ayrılmış sorgulardaki başarısı


Caption biçimleri: `llm_brief` kısa; `llm_balanced` dengeli; `llm_detailed` ayrıntılı; `llm_human_description` doğal insan tarifi; `llm_attribute_list` özellik listesi; `llm_distinctive_first` ayırt edici özellikleri önce veren metin. `identity` orijinal kaynak açıklamadır.

R@1 doğru kaynağın ilk sırada, R@5 ilk beşte bulunma oranıdır. Text testi üretilmiş metinle kaynak açıklamayı, image testi metinle doğru görseli arar. BGE-M3 ve DeBERTa yalnızca text testine katılır.

| Test | Kategori | Model | Tarif | İlk sıra | İlk 5 | R@1 %95 aralık |
|---|---|---|---|---:|---:|---|
| text | face | jina_clip_v2 | llm_distinctive_first | 72% | 80% | 52%–86% |
| text | person | altclip | llm_attribute_list | 100% | 100% | 87%–100% |
| text | vehicle | altclip | llm_attribute_list | 100% | 100% | 87%–100% |
| image | face | clip_b16 | llm_human_description | 36% | 72% | 20%–55% |
| image | person | siglip2_b16_224 | llm_balanced | 60% | 76% | 41%–77% |
| image | vehicle | eva02_clip_l14 | llm_human_description | 60% | 92% | 41%–77% |

## text: her model için geliştirmede seçilen tarif

| Kategori | Model | Tarif | Ayrılmış ilk sıra | Ayrılmış ilk 5 |
|---|---|---|---:|---:|
| face | altclip | llm_attribute_list | 48% | 60% |
| face | bge_m3 | llm_detailed | 44% | 92% |
| face | clip_b16 | llm_detailed | 16% | 60% |
| face | clip_l14 | llm_detailed | 44% | 64% |
| face | deberta_v3_base_mean_pool | llm_distinctive_first | 0% | 12% |
| face | eva02_clip_l14 | llm_detailed | 48% | 88% |
| face | jina_clip_v2 | llm_distinctive_first | 72% | 80% |
| face | siglip2_b16_224 | llm_attribute_list | 24% | 32% |
| person | altclip | llm_attribute_list | 100% | 100% |
| person | bge_m3 | llm_attribute_list | 100% | 100% |
| person | clip_b16 | llm_distinctive_first | 100% | 100% |
| person | clip_l14 | llm_attribute_list | 80% | 96% |
| person | deberta_v3_base_mean_pool | llm_detailed | 44% | 60% |
| person | eva02_clip_l14 | llm_attribute_list | 100% | 100% |
| person | jina_clip_v2 | llm_attribute_list | 100% | 100% |
| person | siglip2_b16_224 | llm_detailed | 88% | 96% |
| vehicle | altclip | llm_attribute_list | 100% | 100% |
| vehicle | bge_m3 | llm_attribute_list | 100% | 100% |
| vehicle | clip_b16 | llm_distinctive_first | 88% | 100% |
| vehicle | clip_l14 | llm_detailed | 88% | 100% |
| vehicle | deberta_v3_base_mean_pool | llm_detailed | 56% | 76% |
| vehicle | eva02_clip_l14 | llm_attribute_list | 100% | 100% |
| vehicle | jina_clip_v2 | llm_attribute_list | 100% | 100% |
| vehicle | siglip2_b16_224 | llm_attribute_list | 68% | 92% |

## image: her model için geliştirmede seçilen tarif

| Kategori | Model | Tarif | Ayrılmış ilk sıra | Ayrılmış ilk 5 |
|---|---|---|---:|---:|
| face | altclip | llm_human_description | 40% | 88% |
| face | clip_b16 | llm_human_description | 36% | 72% |
| face | clip_l14 | llm_human_description | 44% | 76% |
| face | eva02_clip_l14 | llm_human_description | 64% | 88% |
| face | jina_clip_v2 | llm_human_description | 32% | 52% |
| face | siglip2_b16_224 | llm_human_description | 32% | 60% |
| person | altclip | llm_brief | 36% | 68% |
| person | clip_b16 | llm_distinctive_first | 28% | 48% |
| person | clip_l14 | llm_detailed | 32% | 52% |
| person | eva02_clip_l14 | llm_distinctive_first | 52% | 76% |
| person | jina_clip_v2 | identity | 24% | 64% |
| person | siglip2_b16_224 | llm_balanced | 60% | 76% |
| vehicle | altclip | llm_balanced | 48% | 88% |
| vehicle | clip_b16 | llm_balanced | 44% | 60% |
| vehicle | clip_l14 | llm_balanced | 40% | 84% |
| vehicle | eva02_clip_l14 | llm_human_description | 60% | 92% |
| vehicle | jina_clip_v2 | llm_balanced | 60% | 88% |
| vehicle | siglip2_b16_224 | llm_human_description | 56% | 88% |

CSV dosyaları tüm model/tarif kombinasyonlarını ve aynı sorgulardaki orijinal tarife göre eşleştirilmiş bootstrap farklarını içerir (2.000 örnekleme). R@1/R@5 oranlarının mutlak güven aralıkları Wilson yöntemiyle hesaplanır. 150 sorgunun tamamındaki sıralamalar betimseldir; seçimden bağımsız sonuçlar held_out_recipes dosyalarındadır.

Ham cosine modeller arasında kalibre değildir. Küçük farklar, geniş güven aralıkları ve çoklu karşılaştırmalar kesin üstünlük iddiasını desteklemez. Caption doğruluğu otomatik uyarılarla izlenir; insan tarafından onaylanmış görsel etiket olarak yorumlanmamalıdır.

Mamba-3, full150 çalışmasının önceden tanımlı sekiz-model setine dahil değildir; önceki pilotta ayrı bir ham causal-LM baseline olarak değerlendirilmiştir. CPU float32 kullanılmıştır; önceki pilotun sayısal hassasiyetiyle aynı değildir.

## Üretim ve doğrulama

900 Gemini caption tamamlandı; her altı biçimde 150 kayıt var. Kalan kaynak sayısı sıfır. Son devam işleminde 78 istekten 69 başarılı yanıt ve 9 geçici HTTP 503 yanıtı alındı; HTTP 429 oluşmadı. Tüm süreçte kayıtlı toplam istek sayısı 387. Önceki aşamada ikinci prompt ailesi eklenmesi ve tekrarlanan başarısız istekler ilk 135 istek beklentisini aşmıştır.

6/900 caption otomatik içerik denetiminde uyarı aldı; uyarılar `caption_audit.csv` dosyasındadır. Bu metinler sonuçlarda tutulmuştur; görsele bakılarak doğrulanmış etiket sayılmamalıdır.
