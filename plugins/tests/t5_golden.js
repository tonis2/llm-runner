// `llm-runner tests/t5_golden.js t5=<flan-t5 folder> golden=<dir from unimate_golden.py>`:
// the T5 tokenizer and encoder against transformers' numbers.
import { llm } from '../lib/llm.js';
import { T5Encoder } from '../lib/t5.js';

const cfg = globalThis.__llm_config ?? {};
const golden = cfg.golden;
const t5 = new T5Encoder(cfg.t5);
const { tests, ids } = JSON.parse(llm.readText(`${golden}/tokens.json`));
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
for (const [text, want] of Object.entries(ids)) {
	const got = t5.tokenizer.encode(text);
	check(JSON.stringify(got) === JSON.stringify(want), `tokens of ${JSON.stringify(text)}: ${got.join(',')}${JSON.stringify(got) === JSON.stringify(want) ? '' : ' want ' + want.join(',')}`);
}
const states = t5.encode(tests);
tests.forEach((text, i) => {
	const want = new Float32Array(llm.readBytes(`${golden}/t5_${i}.f32`).buffer.slice(0));
	const got = states[i];
	let maxDiff = 0, maxAbs = 0;
	for (let k = 0; k < want.length; k++) { maxDiff = Math.max(maxDiff, Math.abs(got[k] - want[k])); maxAbs = Math.max(maxAbs, Math.abs(want[k])); }
	check(got.length === want.length && maxDiff < 1e-3 * Math.max(1, maxAbs), `hidden of ${JSON.stringify(text)}: max |diff| ${maxDiff.toExponential(2)} (max |x| ${maxAbs.toFixed(2)})`);
});
llm.print(failures ? `${failures} FAILED` : 'all passed');
