// matmul_q8_coop against matmul_q8 on random weights, and both timed.
// `llm-runner tests/matmul_coop.js shapes=[[seq,in,out,rowOffset]]`.
import { llm } from '../lib/llm.js';
import { kernel, pc } from '../lib/gpu.js';
const C = three.compute;
const shapes = globalThis.__llm_config.shapes ?? [[77, 256, 200, 0], [300, 512, 384, 128], [1088, 4096, 4096, 0], [1088, 4096, 12288, 0], [2048, 4096, 12288, 0], [2560, 12288, 4096, 0]];
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
	const push = pc('uuuu', out, inDim, seq, rowOffset);
	const g1 = [Math.ceil(seq / 64) * Math.ceil(out / 64), 1, 1], g2 = [Math.ceil(seq / 128) * Math.ceil(out / 128), 1, 1];
	const k1 = kernel('matmul_q8'), k2 = kernel('matmul_q8_coop');
	k1.dispatch([w, x, y1], { workgroups: g1, push });
	k2.dispatch([w, x, y2], { workgroups: g2, push });
	C.submit();
	const a = new Float32Array(y1.readBytes().buffer), b = new Float32Array(y2.readBytes().buffer);
	let maxErr = 0, maxRef = 0;
	for (let i = 0; i < a.length; i++) { maxErr = Math.max(maxErr, Math.abs(a[i] - b[i])); maxRef = Math.max(maxRef, Math.abs(a[i])); }
	const time = (k, g, reps = 10) => {
		const t0 = llm.now();
		for (let i = 0; i < reps; i++) k.dispatch([w, x, y2], { workgroups: g, push });
		C.submit();
		return (llm.now() - t0) / reps;
	};
	const flops = 2 * seq * inDim * out;
	const t1 = time(k1, g1), t2 = time(k2, g2);
	llm.print(`[${seq} x ${inDim}] -> ${out} @${rowOffset}: rel err ${(maxErr / maxRef).toExponential(2)} | q8 ${(flops / t1 / 1e9).toFixed(1)} TFLOPS | coop ${(flops / t2 / 1e9).toFixed(1)} TFLOPS`);
	w.dispose(); x.dispose(); y1.dispose(); y2.dispose();
}
