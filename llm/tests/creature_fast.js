// When a studio clip's turns go loose (fit.js fitsFaster): only when its reads
// fit markedly better for it. The shares are real clips', firm and loose. No GPU,
// no files.
//
//   llm-runner tests/creature_fast.js
import { llm } from '/core/llm.js';
import { fitsFaster } from '/plugins/creature/fit.js';

let failures = 0;
const check = (ok, what) => {
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
	if (!ok) failures++;
};
const report = (over30) => ({ reprojection_px: { over_30: over30 } });

check(fitsFaster(report(0.215), report(0.172)), 'a fast run goes loose: a fifth fewer reads missed');
check(!fitsFaster(report(0.121), report(0.116)), 'arms raised on the spot stay firm');
check(!fitsFaster(report(0.144), report(0.142)), 'a walk stays firm');
check(!fitsFaster(report(0.028), report(0.028)), 'a clip that gains nothing stays firm');
check(!fitsFaster(report(0), report(0)), 'nothing missed, nothing to gain');
check(!fitsFaster(report(0.1), report(0.2)), 'worse reads stay firm');

llm.print(failures ? `${failures} failed` : 'all passed');
if (failures) throw new Error(`${failures} checks failed`);
