// Variants of matmul_q8_coop against matmul_q8 (the reference), timed.
// `llm-runner tests/bench_q8.js kernels=["matmul_q8_coop"] shapes=[[seq,in,out,rowOffset]]`.
// A kernel is dispatched over 128x128 tiles, unless its entry is [name, tile].
import { llm } from '../lib/llm.js';
import { kernel, pc } from '../lib/gpu.js';
const C = three.compute;
const cfg = globalThis.__llm_config;
const shapes = cfg.shapes ?? [[4096, 4096, 12288, 0], [4096, 4096, 4096, 0], [4096, 12288, 4096, 0]];
const names = cfg.kernels ?? ['matmul_q8_coop'];
function f16(v) {
	const f = new Float32Array([v]), u = new Uint32Array(f.buffer)[0];
	const e = ((u >> 23) & 255) - 112, m = (u >> 13) & 1023, s = (u >> 16) & 32768;
	return e <= 0 ? s : s | (e << 10) | m;
}
for (const [seq, inDim, out, rowOffset] of shapes) {
	const rows = out + rowOffset, blocks = inDim / 32;
	const wb = new Uint8Array(rows * blocks * 34);
	let seed = 1;
	const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
	for (let b = 0; b < rows * blocks; b++) {
		const h = f16(0.001 + rnd() * 0.02);
		wb[b * 34] = h & 255; wb[b * 34 + 1] = h >> 8;
		for (let i = 0; i < 32; i++) wb[b * 34 + 2 + i] = (rnd() * 256) | 0;
	}
	const xs = new Float32Array(seq * inDim);
	for (let i = 0; i < xs.length; i++) xs[i] = (rnd() - 0.5) * 4;
	const w = C.bytes(wb.length), x = C.f32(xs.length), y1 = C.f32(seq * out), y2 = C.f32(seq * out);
	w.write(wb); x.write(xs);
	// The same x as halves, for the kernels that read it so (named `*_h`).
	let xh = null;
	if (names.some((e) => (Array.isArray(e) ? e[0] : e).endsWith('_h'))) {
		const xhBits = new Uint16Array(xs.length);
		for (let i = 0; i < xs.length; i++) xhBits[i] = f16(xs[i]);
		xh = C.bytes(xhBits.byteLength);
		xh.write(new Uint8Array(xhBits.buffer));
	}
	const push = pc('uuuu', out, inDim, seq, rowOffset);
	kernel('matmul_q8').dispatch([w, x, y1], { workgroups: [Math.ceil(seq / 64) * Math.ceil(out / 64), 1, 1], push });
	C.submit();
	const a = new Float32Array(y1.readBytes().buffer);
	const flops = 2 * seq * inDim * out;
	for (const entry of names) {
		const [name, tm, tn] = Array.isArray(entry) ? entry : [entry, 128, 128];
		const xin = name.endsWith('_h') ? xh : x;
		const k = kernel(name), g = [Math.ceil(seq / tm) * Math.ceil(out / (tn ?? tm)), 1, 1];
		C.zero(y2);
		k.dispatch([w, xin, y2], { workgroups: g, push });
		C.submit();
		const b = new Float32Array(y2.readBytes().buffer);
		let maxErr = 0, maxRef = 0;
		for (let i = 0; i < a.length; i++) { maxErr = Math.max(maxErr, Math.abs(a[i] - b[i])); maxRef = Math.max(maxRef, Math.abs(a[i])); }
		const reps = 20;
		const t0 = llm.now();
		for (let i = 0; i < reps; i++) k.dispatch([w, xin, y2], { workgroups: g, push });
		C.submit();
		const t = (llm.now() - t0) / reps;
		llm.print(`[${seq} x ${inDim}] -> ${out}: ${name.padEnd(24)} rel err ${(maxErr / maxRef).toExponential(2)}  ${t.toFixed(2)}ms  ${(flops / t / 1e9).toFixed(1)} TFLOPS`);
	}
	w.dispose(); x.dispose(); if (xh) xh.dispose(); y1.dispose(); y2.dispose();
}
