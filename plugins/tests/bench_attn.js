// Variants of flash_attention_split against a reference (`ref`, by default
// itself), timed.
// `llm-runner tests/bench_attn.js kernels=["flash_attention_split"] shapes=[[heads,qLen,l1,l2]] ref="..." limits=true`.
// A kernel named `*_h` reads K and V as halves.
import { llm } from '../lib/llm.js';
import { kernel, pc } from '../lib/gpu.js';
const C = three.compute;
const cfg = globalThis.__llm_config;
const shapes = cfg.shapes ?? [[32, 4096, 128, 4096]];
const names = cfg.kernels ?? ['flash_attention_split'];
const reps = cfg.reps ?? 10;
const ref = cfg.ref ?? 'flash_attention_split';
// Query rows a workgroup: flash_attention_split takes 128, anything else
// (an older kernel kept for comparison) 64 unless its name ends in 32.
const rows = (name) => (name === 'flash_attention_split' || name.endsWith('32') ? 128 : 64);
function f16(v) {
	const f = new Float32Array([v]), u = new Uint32Array(f.buffer)[0];
	const e = ((u >> 23) & 255) - 112, m = (u >> 13) & 1023, s = (u >> 16) & 32768;
	return e <= 0 ? s : s | (e << 10) | m;
}
function upload(xs, half) {
	if (!half) { const b = C.f32(xs.length); b.write(xs); return b; }
	const h = new Uint16Array(xs.length);
	for (let i = 0; i < xs.length; i++) h[i] = f16(xs[i]);
	const b = C.bytes(h.byteLength);
	b.write(new Uint8Array(h.buffer));
	return b;
}
for (const [heads, qLen, l1, l2] of shapes) {
	let seed = 1;
	const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
	const gen = (n, s) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = (rnd() - 0.5) * s; return a; };
	// Q.K around a few units, so the softmax is neither flat nor one-hot.
	const qs = gen(heads * qLen * 128, 3), k1s = gen(heads * Math.max(l1, 1) * 128, 3), v1s = gen(heads * Math.max(l1, 1) * 128, 2);
	const k2s = gen(heads * l2 * 128, 3), v2s = gen(heads * l2 * 128, 2);
	const q = upload(qs, false);
	const f = [k1s, v1s, k2s, v2s].map((a) => upload(a, false));
	const h = names.some((e) => e.endsWith('_h')) ? [k1s, v1s, k2s, v2s].map((a) => upload(a, true)) : null;
	// With `limits`, a block-causal staircase: row i sees the keys up to the
	// end of its block of 37 (a text-then-image prefix looks like this).
	const lim = C.bytes(qLen * 4);
	const lims = new Uint32Array(qLen);
	for (let i = 0; i < qLen; i++) lims[i] = Math.min(l1 + l2, (Math.floor(i / 37) + 1) * 37 + 5);
	lim.write(new Uint8Array(lims.buffer));
	const y1 = C.f32(qLen * heads * 128), y2 = C.f32(qLen * heads * 128);
	const push = pc('uuuufuuuu', heads, qLen, l1, l2, 1 / Math.sqrt(128), cfg.limits ? 1 : 0, l1, l1, 0);
	const grid = (name) => ({ workgroups: [Math.ceil(qLen / rows(name)), heads, 1], push });
	kernel(ref).dispatch([q, ...(ref.endsWith('_h') ? h : f), lim, y1], grid(ref));
	C.submit();
	const a = new Float32Array(y1.readBytes().buffer);
	const flops = 4 * heads * qLen * (l1 + l2) * 128;
	for (const name of names) {
		const kv = name.endsWith('_h') ? h : f;
		const k = kernel(name);
		const g2 = grid(name);
		C.zero(y2);
		k.dispatch([q, ...kv, lim, y2], g2);
		C.submit();
		const b = new Float32Array(y2.readBytes().buffer);
		let maxErr = 0, maxRef = 0;
		for (let i = 0; i < a.length; i++) { maxErr = Math.max(maxErr, Math.abs(a[i] - b[i])); maxRef = Math.max(maxRef, Math.abs(a[i])); }
		const t0 = llm.now();
		for (let i = 0; i < reps; i++) k.dispatch([q, ...kv, lim, y2], g2);
		C.submit();
		const t = (llm.now() - t0) / reps;
		llm.print(`[${heads}h ${qLen}q ${l1}+${l2}k] ${name.padEnd(28)} rel err ${(maxErr / maxRef).toExponential(2)}  ${t.toFixed(2)}ms  ${(flops / t / 1e9).toFixed(1)} TFLOPS`);
	}
	for (const b of [q, ...f, ...(h ?? []), lim, y1, y2]) b.dispose();
}
