import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const port=9350+Math.floor(Math.random()*500);
const proc=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', ['--headless=new','--disable-gpu','--no-first-run',`--remote-debugging-port=${port}`,`--user-data-dir=${resolve('runs/viewer-browser-check-'+port)}`,'about:blank'],{windowsHide:true,stdio:'ignore'});
let socket;
const keepAlive=setInterval(()=>{},1000);
try {
  let targets;
  for(let i=0;i<80;i++){try{targets=await (await fetch(`http://127.0.0.1:${port}/json`)).json();break}catch{await new Promise(r=>setTimeout(r,250))}}
  if(!targets)throw Error('Browser did not start');
  socket=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j});
  let seq=0;const pending=new Map(),errors=[];
  socket.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.id){const p=pending.get(msg.id);pending.delete(msg.id);msg.error?p.reject(Error(JSON.stringify(msg.error))):p.resolve(msg.result)}if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params)};
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>reject(Error('Browser request timed out: '+method)),15000);pending.set(id,{resolve:r=>{clearTimeout(timer);resolve(r)},reject:e=>{clearTimeout(timer);reject(e)}});socket.send(JSON.stringify({id,method,params}))});
  const evaluate=async expression=>{const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value};
  await call('Runtime.enable');
  const htmlArg=process.argv.indexOf('--html');
  await call('Page.navigate',{url:pathToFileURL(resolve(htmlArg>=0?process.argv[htmlArg+1]:'caption_inceleme.html')).href});
  for(let i=0;i<80;i++){if(await evaluate("!!document.getElementById('pairScores')"))break;await new Promise(r=>setTimeout(r,250))}
  function check(ok,message){if(!ok)throw Error(message);console.log('PASS '+message)}
  if(process.argv.includes('--gemini-scored')){
    check(await evaluate("$('run').value==='gemini' && document.querySelectorAll('.score').length===3 && getSample().filename==='170623.jpg'"),'scored Gemini experiment and target image selected');
    check(await evaluate("$('a').value='identity'; $('a').onchange(); $('b').value='summary_10w'; $('b').onchange(); const vs=getVariants(),i=vs.findIndex(v=>v.type==='identity'),j=vs.findIndex(v=>v.type==='summary_10w'); getRun().models.every(m=>Math.abs(getRun().pairs[selected][m][i][j]-vs[j].scores[m].cosine)<0.00002) && Array.from(document.querySelectorAll('.score b')).every(el=>Number.isFinite(Number(el.textContent)))"),'all original-versus-summary cosine scores match measured CSV values');
    check(await evaluate("$('a').value='summary_20w'; $('a').onchange(); $('b').value='summary_40w'; $('b').onchange(); document.querySelectorAll('.score').length===3 && $('bText').textContent.includes('41 kelime') && getVariants().find(v=>v.type==='summary_40w').word_count===41"),'summary-versus-summary comparisons and actual word counts work');
    check(await evaluate("!$('main').textContent.includes('Encoder skorları henüz') && $('selectedScores').textContent.includes('SigLIP2') && $('selectedScores').textContent.includes('MiniLM')"),'three measured encoder scores replace the pending state');
    if(await evaluate("getVariants().some(v=>v.type==='llm_human_description')")){
      check(await evaluate("$('a').value='identity'; $('a').onchange(); $('b').value='llm_human_description'; $('b').onchange(); const v=getVariants().find(v=>v.type==='llm_human_description'); v.target_words===null && $('bText').textContent.includes('Kelime hedefi verilmedi') && document.querySelectorAll('.score').length===3 && getRun().models.every(m=>Number.isFinite(v.scores[m].cosine))"),'unconstrained human description and all three cosine scores shown');
    }
    check(errors.length===0,'no browser JavaScript exceptions in scored Gemini experiment');
    await call('Browser.close');
  } else if(process.argv.includes('--gemini-preview')){
    check(await evaluate("$('run').value==='gemini_pending' && document.querySelectorAll('.item').length===150"),'Gemini preview available and selected');
    check(await evaluate("choose(data.samples.find(s=>s.filename==='170623.jpg').id); $('b').value='summary_10w'; $('b').onchange(); $('bText').textContent.length>0 && document.querySelectorAll('.score').length===0 && $('pairScores').textContent.includes('benzerlik skoru yok')"),'LLM caption shown without fabricated scores');
    check(await evaluate("$('selectedScores').textContent.includes('Gemini') && $('selectedScores').textContent.includes('gemini-3.1-flash-lite')"),'LLM provenance shown');
    check(await evaluate("$('b').value='summary_40w'; $('b').onchange(); const v=getVariants().find(v=>v.type==='summary_40w'); $('bText').textContent.includes('41 kelime') && v.word_count===41 && v.target_words===40 && $('bText').textContent.includes(v.text.slice(0,40))"),'41-word summary preserved and shown with approximate 40-word target');
    check(errors.length===0,'no browser JavaScript exceptions in Gemini preview');
    await call('Browser.close');
    process.exitCode=0;
  } else {
  await evaluate("$('run').value='priority'; $('run').onchange(); choose(data.samples[0].id)");
  check(await evaluate("document.querySelectorAll('.item').length===150 && document.querySelectorAll('.score').length===3"),'150 images and three model scores render');
  check(await evaluate("Array.from(document.querySelectorAll('main img')).every(i=>i.complete&&i.naturalWidth>0)"),'embedded image loads');
  check(await evaluate("$('a').value='color_change'; $('a').onchange(); $('b').value='attribute_exchange'; $('b').onchange(); document.querySelectorAll('.score').length===3 && $('diff').textContent.length>0"),'variant-versus-variant selection and text diff');
  check(await evaluate("$('run').value='quick'; $('run').onchange(); $('b').value='dropout_10pct'; $('b').onchange(); document.querySelectorAll('.score').length===1 && $('bText').textContent===getVariants().find(v=>v.type==='dropout_10pct').text"),'quick run and dropout pair score');
  check(await evaluate("$('run').value='priority'; $('run').onchange(); $('domain').value='face'; $('domain').onchange(); document.querySelectorAll('.item').length===50 && getSample().domain==='face'"),'domain filter and valid selection on run change');
  check(await evaluate("$('search').value='not-a-real-caption-xyz'; $('search').oninput(); document.querySelectorAll('.item').length===0 && $('main').textContent.includes('bulunamadı')"),'empty search state');
  check(await evaluate("$('search').value=''; $('search').oninput(); $('next').click(); getSample().domain==='face' && document.querySelectorAll('.score').length===3"),'search reset and next-image navigation');
  check(await evaluate("choose(data.samples.find(s=>s.filename==='170623.jpg').id); $('a').value='identity'; $('a').onchange(); $('b').value='negation'; $('b').onchange(); $('aText').querySelector('.diff-removed').textContent==='with' && $('bText').querySelector('.diff-added').textContent==='without' && $('diff').textContent.includes('1 kelime çıkarıldı') && $('diff').closest('details').open"),'170623 negation visibly highlights with → without and opens change table');
  check(await evaluate("Math.abs(getRun().pairs[selected].clip_b16[getVariants().findIndex(v=>v.type==='identity')][getVariants().findIndex(v=>v.type==='negation')]-0.998136878)<0.000001 && $('main').textContent.includes('304 token')"),'170623 CLIP score and sample commentary');
  check(await evaluate("const i=getVariants().findIndex(v=>v.type==='negation'); document.querySelector('[data-compare=\"'+i+'\"]').click(); $('a').value==='identity' && $('b').value==='negation'"),'one-click original-versus-variant comparison');
  check(await evaluate("$('a').value='negation'; $('a').onchange(); $('diff').textContent.includes('birebir aynı')"),'identical caption selections show an explicit message');
  check(errors.length===0,'no browser JavaScript exceptions');
  await call('Browser.close');
  }
} finally {clearInterval(keepAlive);if(socket)socket.close();proc.kill()}
