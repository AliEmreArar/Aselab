import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';
const root='C:/Users/emre/Desktop/Asellab/text_encoder';
const build=path.join(root,'.presentation-build/caption_bulgular');
const out=path.join(root,'outputs/caption_bulgular');
const skill='C:/Users/emre/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations';
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont();
const p=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#172F40', accent='#007F7A', muted='#526774';
function text(s,txt,x,y,w,h,size=26,color=ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=txt;t.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};return t;
}
function slide(title,sub){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,58,38,1164,65,40,ink,true);text(s,sub,58,111,1164,55,24,muted);text(s,`Caption deneyleri · full150`,58,671,1000,28,16,muted);text(s,String(p.slides.items.length),1170,671,45,28,16,muted);return s;}
async function photo(s,file,x,y,w,h){s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,'caption_samples_50',file))),contentType:'image/jpeg',alt:file,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function table(s,values,x,y,width,height,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:widths,values});t.borders.assign({fill:'#DCE5E8',width:1,style:'solid'});for(let r=0;r<values.length;r++){t.rows[r].height=height/values.length;for(let c=0;c<values[0].length;c++){const q=t.getCell(r,c);q.fill=r===0?ink:r%2?'#F0F6F6':'#FFFFFF';q.text.style={typeface:font,fontSize:r===0?23:25,color:r===0?'#FFFFFF':ink,bold:r===0};}}return t;}
const samples=(await fs.readFile(path.join(root,'caption_samples_50/all_samples.jsonl'),'utf8')).trim().split(/\r?\n/).map(JSON.parse);
const sample=samples.find(x=>x.filename==='170623.jpg');
const variants=(await fs.readFile(path.join(root,'variants/full150/variants.jsonl'),'utf8')).trim().split(/\r?\n/).map(JSON.parse).filter(x=>x.sample_id==='face::train::170623.jpg');
const v=type=>variants.find(x=>x.variant_type===type).text;
const sources='Kaynaklar: runs/full150/selected_image_vs_original_held_out.csv; runs/full150/combined/details.csv; runs/full150/combined/image_retrieval_details.csv; runs/full150/combined/different_face_original_pair_summary.csv; caption_bench/metrics.py; caption_samples_50/all_samples.jsonl; variants/full150/variants.jsonl.';
const notes=[];
let s=slide('Farklı caption’lar doğru görseli bulmayı etkiliyor mu?','Amaç: metni kısaltmanın ve yeniden yazmanın etkisini ölçmek.');
text(s,'150 görsel',58,183,350,55,34,accent,true);
text(s,'50 yüz · 50 insan · 50 taşıt\nHer caption için 6 farklı anlatım',58,240,1100,76,27);
text(s,'1  Metin → metin',58,348,540,48,30,accent,true);
text(s,'Yeni tarif, aynı kategorideki\n50 orijinal caption ile karşılaştırıldı.\nKaynak metin ayırt ediliyor mu?',58,408,545,144,27);
text(s,'2  Metin → görsel',680,348,540,48,30,accent,true);
text(s,'Tarif ve 50 fotoğraf, aynı modelin\neşleşen text / vision encoder’larından\ngeçirildi; cosine ile sıralandı.',680,408,545,144,27);
text(s,'Başarı: kaynak görsel ilk sırada mı, ilk 5 içinde mi?',58,574,1164,48,28,ink,true);
notes.push('Bu çalışmada Gemini mevcut caption’ı yeniden yazdı; görseli görmedi. 150 görselin her biri için 6 profil, toplam 900 Gemini varyantı var. Sekiz model metin testinde; eşleşen görsel encoder’ı bulunan altı model görsel testinde yer aldı.\nHoca sorarsa: Evet, gerçek görsel vektörleriyle karşılaştırdık. Her fotoğraf ilgili modelin vision encoder’ından, tarif aynı modelin text encoder’ından geçti. Normalize edilmiş vektörlerin cosine benzerliği hesaplandı. Aynı kategorideki 50 görsel sıralandı ve kaynak dosyanın sırası ölçüldü. İlk sıra Recall@1; ilk beş Recall@5.\nBGE-M3 ve DeBERTa yalnız metin testinde: hazır eşleşen vision tower yok. Modeller farklı vision tower’lar kullandığı için görsel sonuçları yalnız text encoder kalitesi olarak yorumlamıyoruz.\nBu aynı kaynak fotoğrafı bulma testidir; başka kamerada aynı kimliği bulma (gerçek ReID) testi değildir.\n'+sources);
s=slide('Bir yüz, farklı anlatımlar','170623.jpg · Caption’lar deneyde kullanılan İngilizce metinlerdir.');
await photo(s,sample.image,58,189,245,292);
text(s,'Orijinal caption (tam metin)',340,181,870,35,23,accent,true);
text(s,sample.caption,340,221,870,274,17);
text(s,'Kısa',58,509,245,30,23,accent,true);text(s,v('llm_brief'),58,548,245,112,20);
text(s,'Dengeli',340,509,350,30,23,accent,true);text(s,v('llm_balanced'),340,548,350,112,20);
text(s,'Gündelik anlatım',736,509,474,30,23,accent,true);text(s,v('llm_human_description'),736,548,474,112,19);
notes.push('Fotoğraftaki özelliklerin doğru anlatılıp anlatılmadığını ayrıca doğrulamadık; bunlar mevcut caption’dan üretilen varyantlar. Slaytta orijinal caption tamamen gösteriliyor.\nORİJİNAL:\n'+sample.caption+'\n\nALTI VARYANT:\n'+variants.map(x=>x.variant_type+'\n'+x.text).join('\n\n')+'\n'+sources);
s=slide('Yüksek cosine ≠ doğru eşleşme','Cosine metinlerin yakınlığını gösterir; tek başına başarı ölçüsü değildir.');
text(s,'Aynı yüz için · CLIP L/14',58,187,750,46,29,accent,true);
table(s,[['Tarif','Orijinalle cosine','Görsel sırası'],['Kısa','0,809','6'],['Dengeli','0,824','4'],['Gündelik','0,774','6']],58,248,750,236,[220,290,240]);
text(s,'Uzun orijinal tarifin görsel sırası: 6\nDengeli tarifte kaynak görsel ilk 5’e giriyor.',58,506,760,96,26);
await photo(s,'face/images/test/189457.jpg',898,189,235,214);
text(s,'Başka yüz · 189457.jpg',862,418,350,35,23,muted);
text(s,'İki orijinal caption\narasında cosine: 0,893',862,461,350,91,29,accent,true);
text(s,'Ortak özellikler ve benzer anlatım yüksek skora katkıda bulunabilir; nedeni ayrıca ayırmadık.',58,610,1150,55,23,muted);
notes.push('Bu tablo full150 koşusuna aittir. 170623.jpg için CLIP L/14: orijinal görsel sırası 6, kısa 6, dengeli 4, gündelik 6. Metin–metin cosine sırasıyla 0.8085154, 0.8243301 ve 0.7742162. Farklı yüzler 170623.jpg ile 189457.jpg orijinal caption cosine 0.8932774. Yani farklı bir kaynak metin bile kendi kısa varyantından daha yüksek ham cosine alabiliyor.\nCosine’ın yüksek olması kimliklerin aynı olduğunu veya doğru görselin bulunacağını kanıtlamaz. Caption’larda ortak yüz özellikleri ve benzer şema, encoder’ın vektör uzayı yapısı ve uzun metnin kesilmesi olası etkenlerdir; bu test bunların nedensel katkısını ayrı ayrı belirlemiyor. Ham cosine değerleriyle farklı modelleri en iyi/en kötü diye sıralamayız.\nBu tek örnekte gündelik tarif sıralamayı iyileştirmedi. Son slayttaki iyileşme 25 örneğin toplamıdır; her tekil fotoğrafın iyileşmesi beklenmez.\n'+sources);
s=slide('Bulgumuz: seçilmiş bilgi, bazı modellerde daha yararlı','Doğru görseli ilk sırada bulma · Her satır aynı 25 test örneğini karşılaştırıyor.');
table(s,[['Kategori / model','Orijinal tarif','Yeni tarif'],['Yüz · CLIP L/14','1 / 25','11 / 25'],['Yüz · EVA02-CLIP','2 / 25','16 / 25'],['İnsan · SigLIP2','15 / 25','15 / 25'],['Taşıt · SigLIP2','7 / 25','14 / 25']],58,187,1164,275,[550,307,307]);
text(s,'Yüz ve taşıt: gündelik anlatım. İnsan: dengeli tarif.\nTarif biçimi 25 geliştirme örneğinde seçildi; bu ayrı 25 örnekte ölçüldü.',58,480,1150,75,23,muted);
text(s,'Sonuç: her zaman daha kısa değil, daha yararlı ve ayırt edici bilgi.',58,565,1164,45,28,accent,true);
text(s,'Tek bir profil her yerde üstün değil. Bu test aynı fotoğrafı bulmayı ölçer;\nkimlik bazlı, farklı kamera ReID başarısı henüz gösterilmedi.',58,615,1150,55,23);
notes.push('Sunumda söyle: Yüzlerde uzun ve teknik tariften daha seçilmiş gündelik anlatıma geçmek bazı modellerde doğru fotoğrafı ayırt etmeyi belirgin artırdı. CLIP L/14 aynı 25 yüz test örneğinde 1 doğru ilk sıra eşleşmeden 11’e çıktı; EVA02 2’den 16’ya çıktı. İnsan SigLIP2’de toplam başarı aynı kaldı. Taşıt SigLIP2’de 7’den 14’e çıktı. Bunlar bütün model/kategori kombinasyonlarını temsil eden örneklerdir; evrensel kazanç iddiası değildir. Örneğin yüz Jina-CLIP-v2 9/25’ten 8/25’e indi.\nSeçim mantığı: her kategoride 25 geliştirme sorgusu ve 25 ayrı tutulan test sorgusu var. Her model için profil geliştirme sorgularında seçildi; orijinal ile fark test sorgularında ölçüldü. Galeri hâlâ kategori başına 50 görsel. Bu bağımsız dış veri testi ya da kimlik bazlı ayrım değildir.\nNeden iyileşti? Uzun CLIP caption’ları token sınırında kesiliyor. Yüzlerde CLIP L/14 orijinallerinin tümü kesilmiş; kısa ve dengeli olanların hiçbiri kesilmemiş. İçeriğin seçimi, sırası, dil biçimi ve kesilme birlikte değişti: iyileşmeyi yalnız kısalığa bağlayamayız. Bazı uzun bağlamlı modellerde de kazanım olduğundan tek açıklama kesilme değildir.\nAyırt edicilik var mı? Görsel testinde diğer 49 aday arasından kaynak dosyayı ilk sıraya getirme sayısı artınca bu görevde ayırt etme artmıştır. Bu, kaynak metnine yüksek cosine’dan daha anlamlı kanıttır. Ancak gerçek kimlik ReID için aynı kimliğin farklı görüntülerini içeren etiketli galeri ve uygun değerlendirme gerekir.\nSınırlılıklar: küçük ve sabit galeri; caption’ın görsel doğruluğu kontrol edilmedi; Gemini görseli görmedi. Koşuların dört modeli torch2.10, kalan dördü torch2.6 CPU float32; Jina/AltCLIP tokenizer uyarısı ayrıca izole edilmedi. Model üstünlüğünü yalnız text tower’a bağlamayın.\n'+sources);
for(let i=0;i<4;i++)p.slides.items[i].speakerNotes.textFrame.setText(notes[i]);
const candidate=path.join(build,'candidate-tam-caption.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const final=path.join(out,'Caption_Deneyi_Sunum_Tam_Caption.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,explicitTotalSlideCount:4,requiredNativeTableOwnerSlides:[3,4],requiredNativeChartOwnerSlides:[],pythonExecutable:'C:/Users/emre/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--validate-bullet-geometry','--require-native-table-slide','3','--require-native-table-slide','4'],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation-tam-caption.json')});
const checked=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<4;i++){const png=await checked.export({slide:checked.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
console.log(JSON.stringify({final,font,slides:4}));
