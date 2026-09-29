// attention_gqa, causal and not, against attention worked out on the host.
// `llm-runner tests/attention_gqa.js`.
import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit } from '../lib/gpu.js';

const [n, qHeads, kvHeads, hd] = [23, 8, 2, 64];
let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5) * 2;
const fill = (count) => Float32Array.from({ length: count }, rnd);
const qs = fill(n * qHeads * hd), ks = fill(n * kvHeads * hd), vs = fill(n * kvHeads * hd);

function reference(causal) {
	const out = new Float32Array(n * qHeads * hd);
	for (let h = 0; h < qHeads; h++) {
		const kh = Math.floor(h / (qHeads / kvHeads));
		for (let i = 0; i < n; i++) {
			const seen = causal ? i + 1 : n;
			const s = new Float64Array(seen);
			for (let j = 0; j < seen; j++) {
				for (let d = 0; d < hd; d++) s[j] += qs[(i * qHeads + h) * hd + d] * ks[(j * kvHeads + kh) * hd + d];
				s[j] /= Math.sqrt(hd);
			}
			const mx = Math.max(...s);
			let sum = 0;
			for (let j = 0; j < seen; j++) sum += (s[j] = Math.exp(s[j] - mx));
			for (let d = 0; d < hd; d++) {
				let a = 0;
				for (let j = 0; j < seen; j++) a += s[j] * vs[(j * kvHeads + kh) * hd + d];
				out[(i * qHeads + h) * hd + d] = a / sum;
			}
		}
	}
	return out;
}

const upload = (data) => { const t = f32(data.length); t.buffer.write(data); return t; };
const q = upload(qs), k = upload(ks), v = upload(vs), out = f32(n * qHeads * hd);
let failed = false;
for (const causal of [true, false]) {
	op.attentionGqa(q, k, v, out, hd, kvHeads, qHeads, n, causal);
	submit();
	const got = new Float32Array(out.buffer.readBytes().buffer);
	const want = reference(causal);
	let maxErr = 0;
	for (let i = 0; i < want.length; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - want[i]));
	const ok = maxErr < 1e-4;
	failed ||= !ok;
	llm.print(`attention_gqa ${causal ? 'causal' : 'bidirectional'}: max |err| ${maxErr.toExponential(2)} ${ok ? 'ok' : 'FAIL'}`);
}
if (failed) throw new Error('attention_gqa does not match the host reference');
