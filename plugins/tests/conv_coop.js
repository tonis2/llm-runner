// conv2d_coop against conv2d_3x3 on random inputs, and both timed.
// `llm-runner tests/conv_coop.js shapes=[[inC,outC,h,w]]`.
import { llm } from '../lib/llm.js';
import { kernel, pc } from '../lib/gpu.js';
const C = three.compute;
const shapes = globalThis.__llm_config.shapes ?? [[32, 36, 13, 21], [512, 512, 64, 64], [512, 512, 128, 128], [256, 256, 256, 256], [128, 128, 512, 512], [256, 128, 512, 512]];
for (const [inC, outC, h, w, ks = 3] of shapes) {
	let seed = 3;
	const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5);
	const wt = C.f32(ks * ks * inC * outC), b = C.f32(outC), x = C.f32(inC * h * w), y1 = C.f32(outC * h * w), y2 = C.f32(outC * h * w);
	wt.write(Float32Array.from({ length: ks * ks * inC * outC }, () => rnd() * 0.05));
	b.write(Float32Array.from({ length: outC }, rnd));
	x.write(Float32Array.from({ length: inC * h * w }, () => rnd() * 4));
	const push = pc('uuuuuuuuuuuu', inC, outC, h, w, ks, ks, 1, (ks - 1) / 2, 1, h, w, 1);
	const k1 = kernel(ks === 3 ? 'conv2d_3x3' : 'conv2d'), k2 = kernel('conv2d_coop');
	const g1 = ks === 3 ? [Math.ceil(w / 16), Math.ceil(h / 16), Math.ceil(outC / 4)] : [Math.ceil(h * w / 256), outC, 1], g2 = [Math.ceil(h * w / 128) * Math.ceil(outC / 128), 1, 1];
	k1.dispatch([wt, b, x, y1], { workgroups: g1, push });
	k2.dispatch([wt, b, x, y2], { workgroups: g2, push });
	C.submit();
	const A = new Float32Array(y1.readBytes().buffer), B = new Float32Array(y2.readBytes().buffer);
	let err = 0, ref = 0;
	for (let i = 0; i < A.length; i++) { err = Math.max(err, Math.abs(A[i] - B[i])); ref = Math.max(ref, Math.abs(A[i])); }
	const time = (k, g, y, reps = 5) => { const t0 = llm.now(); for (let i = 0; i < reps; i++) k.dispatch([wt, b, x, y], { workgroups: g, push }); C.submit(); return (llm.now() - t0) / reps; };
	const flops = 2 * ks * ks * inC * outC * h * w, t1 = time(k1, g1, y1), t2 = time(k2, g2, y2);
	llm.print(`conv ${ks}x${ks} ${inC}->${outC} @${h}x${w}: rel err ${(err / ref).toExponential(2)} | scalar ${t1.toFixed(2)}ms ${(flops / t1 / 1e9).toFixed(1)} TFLOPS | coop ${t2.toFixed(2)}ms ${(flops / t2 / 1e9).toFixed(1)} TFLOPS`);
	for (const t of [wt, b, x, y1, y2]) t.dispose();
}
