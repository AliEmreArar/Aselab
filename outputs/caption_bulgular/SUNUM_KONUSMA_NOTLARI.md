# 4 slayt için kısa konuşma rehberi

## 1 — Deneyi nasıl yaptık?

“Caption’ı kısaltınca veya yeniden yazınca doğru görseli bulmak kolaylaşıyor mu diye baktık. Yüz, insan ve taşıttan 50’şer görsel kullandık. Gemini mevcut caption’lardan altı farklı anlatım üretti; görseli görmedi. İki test yaptık: yeni metin kaynak caption’ı buluyor mu, ve yeni metin kaynak fotoğrafı buluyor mu? Görsel testinde gerçekten fotoğrafların vektörlerini kullandık.”

## 2 — Aynı görselin farklı tarifleri

“Bu aynı yüzün orijinal, kısa, dengeli ve gündelik tarifleri. Slaytta orijinal caption’ın tamamı görünüyor. Uzun teknik anlatımdan daha az özellik seçen anlatıma geçiyoruz. Caption’ların görsel doğruluğunu ayrıca denetlemedik; burada metin değişikliğinin etkisini ölçüyoruz.”

## 3 — Cosine neden yetmiyor?

“Cosine iki metin vektörünün yakınlığı. Bu örnekte kısa ve dengeli tarif orijinale yakın. Ama başka bir yüzün orijinal caption’ıyla benzerlik de yüksek: 0,893. Bu nedenle yalnız yüksek cosine’a bakıp başarılı diyemiyoruz. Doğru fotoğrafın diğer fotoğraflar arasında kaçıncı sıraya geldiğine bakıyoruz. Burada dengeli tarif, fotoğrafı 6. sıradan 4. sıraya getiriyor.”

## 4 — Sonuç ne?

“Yüzlerde bazı modellerde belirgin iyileşme var. CLIP L/14 aynı 25 test yüzünde orijinal tarifle yalnız bir fotoğrafı ilk sırada bulurken gündelik tarifle 11’ini buluyor. EVA02’de de artış var. İnsan örneğinde toplam başarı değişmiyor, taşıt örneğinde artıyor. Dolayısıyla her zaman kısa olan iyi değil; yararlı bilgiyi seçen anlatım bazı modellerde daha iyi. Evrensel olarak en iyi tarif biçimini bulduk demiyoruz.”

## Hocaların sorabileceği sorular

**Görsel vektörleriyle kıyaslamanız gerekmiyor muydu?**

Evet, kıyasladık. Fotoğraflar modelin vision encoder’ından, tarif aynı modelin eşleşen text encoder’ından geçti. Vektörler normalize edildi, cosine ile 50 aday sıralandı. Kaynak fotoğrafın ilk sırada veya ilk beşte olması başarı olarak sayıldı. Metin–metin testi ayrı bir yardımcı ölçüm.

**Farklı yüzlerin caption’ları neden benzer çıkıyor?**

Ortak özellikler ve benzer anlatım şeması katkıda bulunabilir. Modelin vektör uzayının yapısı ve uzun metnin kesilmesi de olası etkenler. Hangisinin ne kadar etkili olduğunu bu deney ayrı ayrı kanıtlamıyor. Yüksek cosine aynı kimlik anlamına gelmiyor.

**Özetleme ayırt ediciliği artırdı mı?**

Bazı modellerde bu görev için evet: aynı kaynak görseli diğer 49 aday arasından ilk sırada bulma sayısı arttı. Ama her model ve kategoride artmadı. Örneğin yüzlerde Jina-CLIP-v2, ayrı tutulan testte 9/25’ten 8/25’e indi.

**Bu gerçek ReID başarısı mı?**

Hayır. Caption’ın üretildiği kaynak dosyayı bulmayı ölçtük. Aynı kimliği başka fotoğraf veya kamerada bulmayı göstermek için kimlik etiketli, çoklu görüntülü ayrı bir ReID testi gerekir.

**Kısaltmanın etkisi mi, CLIP’in token sınırının etkisi mi?**

İkisi birbirinden tam ayrılmadı. CLIP uzun metnin tamamını göremiyor. Ayrıca seçilen özellikler, sıraları ve anlatım biçimi de değişiyor. İyileşmeyi yalnız kısalığa bağlamıyoruz. Aynı uzunlukta farklı içerikler ve kesilmeyen metinlerle ayrı kontrol deneyi gerekir.

**Neden 25 test örneği?**

Her kategoride 25 örneği tarif biçimini seçmekte, kalan 25’ini karşılaştırmada kullandık. Böylece testte en iyi görünen tarifi sonradan seçmiyoruz. Arama galerisi yine 50 görsel. Bu küçük, iç veri ayrımıdır; bağımsız dış veri doğrulaması değildir.

**Kaç model vardı?**

Sekiz model metin testinde, bunlardan eşleşen görsel encoder’ı olan altısı görsel testinde. Sunumda anlaşılır olması için birkaç temsilî sonucu gösterdik; bu bir model şampiyonluğu tablosu değil.

## Kaynaklar

Veriler `runs/full150/selected_image_vs_original_held_out.csv`, `runs/full150/combined/details.csv`, `runs/full150/combined/image_retrieval_details.csv` ve `runs/full150/combined/different_face_original_pair_summary.csv` dosyalarından. Caption’lar `caption_samples_50/all_samples.jsonl` ve `variants/full150/variants.jsonl` dosyalarından. Hesaplama mantığı `caption_bench/metrics.py` içinde.
