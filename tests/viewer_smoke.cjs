// Runtime smoke test only: the mock DOM does not verify visual layout.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync('caption_inceleme.html', 'utf8');
const payload = html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = {};
const document = {
  getElementById(id) {
    return elements[id] ??= {
      value: id === 'domain' ? 'all' : '',
      textContent: id === 'payload' ? payload : '', innerHTML: '',
      add() {}, querySelectorAll() { return []; }, scrollIntoView() {},
      insertAdjacentHTML(_, content) { this.innerHTML += content; },
    };
  },
  querySelectorAll() { return []; },
};
const context = vm.createContext({ document, location: { hash: '' }, Option: function() {} });
vm.runInContext(js, context);
vm.runInContext(`
  for (const sample of data.samples) { selected=sample.id; renderMain(); }
  selected=data.samples.find(s=>getRun().variants[s.id].some(v=>v.type==='control_one_fact')
    &&getRun().variants[s.id].some(v=>v.type==='control_synonym')).id;
  aType='llm_balanced'; bType='control_one_fact'; renderMain();
`, context);
assert(elements.selectedScores.innerHTML.includes('Negatif percentile'));
assert(elements.selectedScores.innerHTML.includes('Tek ifade kontrolü'));
console.log('All 150 sample renders and control tables OK (mock DOM)');
