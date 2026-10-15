// cfg_combine against the same guidance worked out on the host: plain, and with
// each pixel's channel vector rescaled to the conditional one's length.
// `llm-runner tests/cfg_combine.js`.
import { llm } from '../lib/llm.js';
import { kernel, pc } from '../lib/gpu.js';
const C = three.compute;

let seed = 1;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

let failed = false;
// [channels, spatial, s]: a partial last workgroup, Qwen-Image's 64 channels,
// and an s below 1.
for (const [channels, spatial, s] of [[3, 5, 4], [64, 300, 7.5], [16, 1024, 0.5]]) {
	const n = channels * spatial;
	const c = new Float32Array(n), u = new Float32Array(n);
	for (let i = 0; i < n; i++) {
		c[i] = (rnd() - 0.5) * 4;
		u[i] = c[i] + (rnd() - 0.5);
	}
	for (const rescale of [false, true]) {
		const want = new Float32Array(n);
		for (let p = 0; p < spatial; p++) {
			let cc = 0, gg = 0;
			for (let k = 0; k < channels; k++) {
				const i = k * spatial + p;
				want[i] = u[i] + s * (c[i] - u[i]);
				cc += c[i] * c[i];
				gg += want[i] * want[i];
			}
			if (rescale) for (let k = 0; k < channels; k++) want[k * spatial + p] *= Math.sqrt(cc / gg);
		}
		const cb = C.f32(n), vb = C.f32(n);
		cb.write(c);
		vb.write(u);
		kernel('cfg_combine').dispatch([cb, vb], { workgroups: [Math.ceil(spatial / 256), 1, 1], push: pc('uufu', channels, spatial, s, rescale ? 1 : 0) });
		C.submit();
		const got = new Float32Array(vb.readBytes().buffer);
		let err = 0, ref = 0;
		for (let i = 0; i < n; i++) {
			err = Math.max(err, Math.abs(got[i] - want[i]));
			ref = Math.max(ref, Math.abs(want[i]));
		}
		const ok = err <= 1e-5 * Math.max(1, ref);
		if (!ok) failed = true;
		llm.print(`cfg_combine: ${ok ? 'ok  ' : 'FAIL'} ${channels}x${spatial} s=${s} rescale=${rescale}: max error ${err.toExponential(2)} of ${ref.toFixed(2)}`);
		cb.dispose();
		vb.dispose();
	}
}
if (failed) throw new Error('cfg_combine: wrong results');
