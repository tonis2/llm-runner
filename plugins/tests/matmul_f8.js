// matmul_f8 and matmul_f8_coop against matmul_f32 over the same weights decoded
// on the host, every E4M3 byte value among them, and the three timed.
// `llm-runner tests/matmul_f8.js shapes=[[seq,in,out,rowOffset]]`.
import { llm } from '../lib/llm.js';
import { kernel, pc } from '../lib/gpu.js';
const C = three.compute;
const shapes = globalThis.__llm_config.shapes ?? [[5, 64, 33, 0], [77, 256, 200, 0], [300, 512, 384, 128], [1088, 4096, 4096, 0], [2048, 4096, 12288, 0], [2560, 12288, 4096, 0]];

// E4M3 as the reference decodes it: no NaN in a weight, so 0x7f / 0xff are 480.
const LUT = new Float32Array(256);
for (let b = 0; b < 256; b++) {
	const e = (b >> 3) & 15, m = b & 7;
	const v = e === 0 ? (m / 8) * 2 ** -6 : (1 + m / 8) * 2 ** (e - 7);
	LUT[b] = b & 128 ? -v : v;
}

let failed = false;
for (const [seq, inDim, out, rowOffset] of shapes) {
	const rows = out + rowOffset;
	const wb = new Uint8Array(rows * inDim);
	let seed = 1;
	const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
	// Mostly weight-sized values (exponents 0..7, subnormals included), and the
	// first 256 bytes every value there is.
	for (let i = 0; i < wb.length; i++) wb[i] = i < 256 ? i : ((rnd() * 64) | 0) | (rnd() < 0.5 ? 128 : 0);
	const wf = new Float32Array(wb.length);
	for (let i = 0; i < wb.length; i++) wf[i] = LUT[wb[i]] / 64;
	const xs = new Float32Array(seq * inDim);
	for (let i = 0; i < xs.length; i++) xs[i] = (rnd() - 0.5) * 4;
	// The f8 kernels read the raw bytes (no /64): their output is 64x the reference's.
	const w8 = C.bytes(wb.length), w32 = C.f32(wf.length), x = C.f32(xs.length);
	const y0 = C.f32(seq * out), y1 = C.f32(seq * out), y2 = C.f32(seq * out);
	w8.write(wb); w32.write(wf); x.write(xs);
	const push = pc('uuuu', out, inDim, seq, rowOffset);
	const g64 = [Math.ceil(seq / 64) * Math.ceil(out / 64), 1, 1], g128 = [Math.ceil(seq / 128) * Math.ceil(out / 128), 1, 1];
	const kRef = kernel('matmul_f32'), k1 = kernel('matmul_f8'), k2 = kernel('matmul_f8_coop');
	kRef.dispatch([w32, x, y0], { workgroups: g64, push });
	k1.dispatch([w8, x, y1], { workgroups: g64, push });
	k2.dispatch([w8, x, y2], { workgroups: g128, push });
	C.submit();
	const r = new Float32Array(y0.readBytes().buffer);
	const a = new Float32Array(y1.readBytes().buffer), b = new Float32Array(y2.readBytes().buffer);
	let e1 = 0, e2 = 0, ref = 0;
	for (let i = 0; i < r.length; i++) {
		const v = r[i] * 64;
		ref = Math.max(ref, Math.abs(v));
		e1 = Math.max(e1, Math.abs(a[i] - v));
		e2 = Math.max(e2, Math.abs(b[i] - v));
	}
	// f32 math: only summation order differs. The matrix cores round x to half.
	const ok = e1 / ref < 1e-5 && e2 / ref < 2e-3;
	if (!ok) failed = true;
	const time = (k, w, g, reps = 10) => {
		const t0 = llm.now();
		for (let i = 0; i < reps; i++) k.dispatch([w, x, y1], { workgroups: g, push });
		C.submit();
		return (llm.now() - t0) / reps;
	};
	const flops = 2 * seq * inDim * out;
	const tf = (t) => (flops / t / 1e9).toFixed(1);
	llm.print(`[${seq} x ${inDim}] -> ${out} @${rowOffset}: f8 rel err ${(e1 / ref).toExponential(2)}, coop ${(e2 / ref).toExponential(2)} ${ok ? 'ok' : 'FAIL'}`
		+ ` | f32 ${tf(time(kRef, w32, g64))} f8 ${tf(time(k1, w8, g64))} coop ${tf(time(k2, w8, g128))} TFLOPS`);
	for (const t of [w8, w32, x, y0, y1, y2]) t.dispose();
}
if (failed) throw new Error('matmul_f8: outputs differ from the f32 reference');
