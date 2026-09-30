// `llm-runner tests/unimate_names.js golden=<names_golden.json>`: the joint-name
// cleaner against UniMate's Python rules (a {raw: cleaned} map they made).
import { llm } from '../lib/llm.js';
import { modelJointName } from '../unimate/names.js';
const want = JSON.parse(llm.readText(globalThis.__llm_config.golden));
let bad = 0, n = 0;
for (const [raw, clean] of Object.entries(want)) {
	n++;
	const got = modelJointName(raw, 'x');
	if (got !== clean) { bad++; llm.print(`FAIL ${JSON.stringify(raw)} -> ${JSON.stringify(got)}, want ${JSON.stringify(clean)}`); }
}
llm.print(`${n - bad}/${n} names agree`);
