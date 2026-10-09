// flash_attention_split over a prefix that is not packed (rows `stride` apart a
// head, keys from `skipAt` read `skipLen` rows further in) against the same keys
// packed: the two must agree exactly.
// `llm-runner tests/attn_layout.js`
import { llm } from '../lib/llm.js';
import { flashAttentionSplit } from '../lib/ops.js';
const C = three.compute;
const heads = 4, qLen = 200, M = 300, gap = 77, tail = 45, l2 = 150;
let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5) * 3;
const gen = (n) => Float32Array.from({ length: n }, rnd);
const up = (a) => { const b = C.f32(a.length); b.write(a); return b; };
const l1 = M + tail, stride = M + gap + tail;
// The packed prefix, and the same rows spread out with junk in the gap.
const pk = gen(heads * l1 * 128), pv = gen(heads * l1 * 128);
const sk = gen(heads * stride * 128), sv = gen(heads * stride * 128);
for (let h = 0; h < heads; h++) {
	for (let j = 0; j < l1; j++) {
		const to = j < M ? j : j + gap;
		sk.set(pk.subarray((h * l1 + j) * 128, (h * l1 + j + 1) * 128), (h * stride + to) * 128);
		sv.set(pv.subarray((h * l1 + j) * 128, (h * l1 + j + 1) * 128), (h * stride + to) * 128);
	}
}
const q = up(gen(heads * qLen * 128)), k2 = up(gen(heads * l2 * 128)), v2 = up(gen(heads * l2 * 128));
const lims = new Uint32Array(qLen);
for (let i = 0; i < qLen; i++) lims[i] = i % 3 === 0 ? l1 + l2 : M + 1 + i;
const lim = C.bytes(qLen * 4);
lim.write(new Uint8Array(lims.buffer));
let worst = 0;
for (const limits of [null, lim]) {
	const a = C.f32(qLen * heads * 128), b = C.f32(qLen * heads * 128);
	flashAttentionSplit(q, up(pk), up(pv), k2, v2, a, heads, qLen, l1, l2, limits);
	flashAttentionSplit(q, up(sk), up(sv), k2, v2, b, heads, qLen, l1, l2, limits, { stride, skipAt: M, skipLen: gap });
	C.submit();
	const x = new Float32Array(a.readBytes().buffer), y = new Float32Array(b.readBytes().buffer);
	let err = 0, mag = 0;
	for (let i = 0; i < x.length; i++) { err = Math.max(err, Math.abs(x[i] - y[i])); mag = Math.max(mag, Math.abs(x[i])); }
	llm.print(`${limits ? 'limits' : 'full'}: max |packed - strided| ${err} (max |out| ${mag.toFixed(3)})`);
	worst = Math.max(worst, err);
}
if (worst !== 0) throw new Error('the strided prefix does not match the packed one');
llm.print('ok');
