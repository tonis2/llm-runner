// The joint-finder trainer's native loops (lib/host/finder_train.c3) against
// the JavaScript they port (train.js): orthonormalRows to float64 rounding,
// the composite and the photo crop byte for byte, on the same random draws.
//
//   llm-runner tests/creature_native.js
import { llm } from '/core/llm.js';
import { generator, composite, orthonormalRows, Backdrops } from '/plugins/creature/train.js';

let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

if (!llm.trainNative) throw new Error('this host has no orthonormalRows: build llm-runner with lib/host/finder_train.c3');

// orthonormalRows: rows of very different lengths and a nearly repeated one,
// what subspace iteration hands it.
{
	const n = 48, D = 300, r = generator(7);
	const Z = new Float64Array(n * D);
	for (let i = 0; i < n; i++) {
		const scale = 10 ** (6 * (i / n) - 3);
		for (let d = 0; d < D; d++) Z[i * D + d] = (r() - 0.5) * scale;
	}
	for (let d = 0; d < D; d++) Z[5 * D + d] = Z[4 * D + d] * 2 + 1e-9 * (r() - 0.5);
	const a = orthonormalRows(Z.slice(), n, D, false);
	let t = llm.now();
	const b = orthonormalRows(Z.slice(), n, D, true);
	t = llm.now() - t;
	let worst = 0;
	for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
	check(worst < 1e-9, `orthonormalRows ${n}x${D}: native matches JavaScript (worst ${worst.toExponential(2)})`);
	let off = 0;
	for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
		let s = 0;
		for (let d = 0; d < D; d++) s += b[i * D + d] * b[j * D + d];
		off = Math.max(off, Math.abs(s - (i === j ? 1 : 0)));
	}
	check(off < 1e-6, `and its rows are orthonormal (worst ${off.toExponential(2)})`);
	const big = new Float64Array(384 * 2560).map(() => r() - 0.5);
	t = llm.now();
	orthonormalRows(big, 384, 2560, true);
	llm.print(`     384x2560 natively in ${(llm.now() - t).toFixed(0)}ms`);
}

// composite: a picture with soft edges (alpha between 0 and 255) and grain,
// and one without grain, over the same backdrop.
{
	const S = 64, render = { width: S, height: S, channels: 4, pixels: new Uint8Array(S * S * 4) };
	const r = generator(3);
	for (let i = 0; i < S * S; i++) {
		const x = i % S, y = Math.floor(i / S), d = Math.hypot(x - S / 2, y - S / 2);
		render.pixels.set([r.int(256), r.int(256), r.int(256), d < 18 ? 255 : d < 26 ? Math.round(255 * (26 - d) / 8) : 0], i * 4);
	}
	const backdrop = new Uint8Array(S * S * 3).map(() => r.int(256));
	const noise = llm.randomNormal(1 << 16, 1);
	for (let seed = 1; seed <= 6; seed++) {
		const a = composite(render, backdrop, generator(seed), noise, false).pixels;
		const b = composite(render, backdrop, generator(seed), noise, true).pixels;
		let diff = 0;
		for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
		check(diff === 0, `composite, seed ${seed}: native matches JavaScript (${diff} bytes differ)`);
	}
	// And a whole render of the training set, if one is here.
	const dir = 'build/creature/zidane/renders2';
	if (llm.exists(`${dir}/000000_front.png`)) {
		const im = llm.image.load(`${dir}/000000_front.png`);
		const back = new Backdrops(im.width, null, generator(2));
		const big = llm.randomNormal(1 << 21, 1);
		const bd = back.next();
		let t = llm.now();
		const a = composite(im, bd, generator(9), big, false).pixels;
		const js = llm.now() - t;
		t = llm.now();
		const b = composite(im, bd, generator(9), big, true).pixels;
		const nat = llm.now() - t;
		let diff = 0;
		for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
		check(diff === 0, `composite of a ${im.width}px render: native matches (${diff} bytes differ; ${js.toFixed(1)}ms in JavaScript, ${nat.toFixed(1)}ms native)`);
	}
}

// The photo crop, flipped and not.
{
	const W = 97, H = 61, img = { width: W, height: H, channels: 3, pixels: new Uint8Array(W * H * 3).map((_, i) => (i * 37) & 255) };
	for (let seed = 1; seed <= 6; seed++) {
		const a = new Backdrops(48, null, generator(seed));
		a.photos = [img];
		const b = new Backdrops(48, null, generator(seed));
		b.photos = [img];
		const pa = a.photo(false), pb = b.photo(true);
		let diff = 0;
		for (let i = 0; i < pa.length; i++) if (pa[i] !== pb[i]) diff++;
		check(diff === 0, `photo crop, seed ${seed}: native matches JavaScript (${diff} bytes differ)`);
	}
}

// The made-up backgrounds, every kind, on the same draws.
{
	const a = new Backdrops(96, null, generator(5)), b = new Backdrops(96, null, generator(5));
	let diff = 0, t = 0, tn = 0;
	for (let i = 0; i < 40; i++) {
		let s = llm.now();
		const pa = a.made(false);
		t += llm.now() - s;
		s = llm.now();
		const pb = b.made(true);
		tn += llm.now() - s;
		for (let k = 0; k < pa.length; k++) if (pa[k] !== pb[k]) diff++;
	}
	check(diff === 0, `40 made-up backgrounds: native matches JavaScript (${diff} bytes differ; ${(t / 40).toFixed(1)}ms in JavaScript, ${(tn / 40).toFixed(2)}ms native at 96px)`);
}

llm.print(failures ? `${failures} failed` : 'all passed');
