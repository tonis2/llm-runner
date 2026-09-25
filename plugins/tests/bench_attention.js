// Flash attention: the matrix-core kernel against the scalar one on random
// inputs, and both timed. [heads, seq, hd] Q, K, V.
// `llm-runner tests/bench_attention.js shapes=[[heads,seq,hd]]`.
import { llm } from '../lib/llm.js';
import { kernel, pc, flashDefines } from '../lib/gpu.js';
const C = three.compute;
const shapes = globalThis.__llm_config.shapes ?? [[4, 77, 128], [32, 2176, 128], [32, 1536, 128], [30, 1152, 128]];
for (const [heads, seq, hd] of shapes) {
	const n = heads * seq * hd;
	const q = C.f32(n), k = C.f32(n), v = C.f32(n), o1 = C.f32(n), o2 = C.f32(n);
	let seed = 7;
	const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5) * 4;
	for (const t of [q, k, v]) t.write(Float32Array.from({ length: n }, rnd));
	const push = pc('uuuf', hd, heads, seq, 1 / Math.sqrt(hd));
	const scalar = kernel('flash_attention', flashDefines(hd)), coop = kernel('flash_attention_coop');
	const g1 = [Math.ceil(seq / 16), heads, 1], g2 = [Math.ceil(seq / 64), heads, 1];
	scalar.dispatch([q, k, v, o1], { workgroups: g1, push });
	coop.dispatch([q, k, v, o2], { workgroups: g2, push });
	C.submit();
	const a = new Float32Array(o1.readBytes().buffer), b = new Float32Array(o2.readBytes().buffer);
	let err = 0, ref = 0;
	for (let i = 0; i < n; i++) { err = Math.max(err, Math.abs(a[i] - b[i])); ref = Math.max(ref, Math.abs(a[i])); }
	const time = (kern, g, out, reps = 5) => {
		const t0 = llm.now();
		for (let i = 0; i < reps; i++) kern.dispatch([q, k, v, out], { workgroups: g, push });
		C.submit();
		return (llm.now() - t0) / reps;
	};
	const flops = 4 * seq * seq * hd * heads;
	const t1 = time(scalar, g1, o1), t2 = time(coop, g2, o2);
	llm.print(`attention ${heads} x ${seq} x ${hd}: rel err ${(err / ref).toExponential(2)} | scalar ${t1.toFixed(2)}ms ${(flops / t1 / 1e9).toFixed(1)} TFLOPS | coop ${t2.toFixed(2)}ms ${(flops / t2 / 1e9).toFixed(1)} TFLOPS`);
	for (const t of [q, k, v, o1, o2]) t.dispose();
}
