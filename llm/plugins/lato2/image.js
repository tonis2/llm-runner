// The conditioning picture LATO.2 (and TRELLIS) take: a cutout's foreground,
// squared up around its alpha's bounding box with some margin, on black,
// resized to 518 - TRELLIS's preprocess_image / LATO.2's crop_foreground.
//
// The background is not removed here: the image must already be a cutout,
// its alpha the matte (LATO.2 itself mattes every input with BiRefNet).

// PIL's Lanczos (a = 3) weights for resampling `inSize` to `outSize`: per
// output pixel the first input pixel and the normalised weights.
function lanczosTaps(inSize, outSize) {
	const scale = inSize / outSize, fs = Math.max(scale, 1), support = 3 * fs;
	const sinc = (x) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));
	const lanczos = (x) => (x > -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0);
	const taps = [];
	for (let o = 0; o < outSize; o++) {
		const center = (o + 0.5) * scale;
		const lo = Math.max(Math.trunc(center - support + 0.5), 0), hi = Math.min(Math.trunc(center + support + 0.5), inSize);
		const w = new Float64Array(hi - lo);
		let sum = 0;
		for (let i = lo; i < hi; i++) sum += (w[i - lo] = lanczos((i - center + 0.5) / fs));
		for (let i = 0; i < w.length; i++) w[i] /= sum;
		taps.push({ lo, w });
	}
	return taps;
}

// planes [c][h * w] (Float32Array each) -> [c][oh * ow].
function resize(planes, w, h, ow, oh) {
	const tx = lanczosTaps(w, ow), ty = lanczosTaps(h, oh);
	return planes.map((src) => {
		const mid = new Float32Array(h * ow);
		for (let y = 0; y < h; y++) {
			const row = y * w;
			for (let x = 0; x < ow; x++) {
				const { lo, w: k } = tx[x];
				let s = 0;
				for (let i = 0; i < k.length; i++) s += src[row + lo + i] * k[i];
				mid[y * ow + x] = s;
			}
		}
		const out = new Float32Array(oh * ow);
		for (let y = 0; y < oh; y++) {
			const { lo, w: k } = ty[y];
			for (let x = 0; x < ow; x++) {
				let s = 0;
				for (let i = 0; i < k.length; i++) s += mid[(lo + i) * ow + x] * k[i];
				out[y * ow + x] = s;
			}
		}
		return out;
	});
}

// An 8-bit RGBA image ({ width, height, channels, pixels }) -> the
// conditioning picture { width, height, channels: 3, pixels } (`size` square).
export function conditioningImage(img, { padding = 1.2, size = 518 } = {}) {
	const { width: W, height: H, channels: C, pixels } = img;
	if (C !== 4) throw new Error(`LATO.2 needs a cutout with transparency (an RGBA image); this one has ${C} channels. Remove the background first.`);
	let x0 = W, y0 = H, x1 = -1, y1 = -1;
	for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
		if (pixels[(y * W + x) * 4 + 3] > 204) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
	}
	if (x1 < 0) throw new Error('the image has no foreground (no pixel is more than 80% opaque)');
	let opaque = 0;
	for (let i = 3; i < pixels.length; i += 4) if (pixels[i] === 255) opaque++;
	if (opaque === W * H) throw new Error('the image is fully opaque: LATO.2 needs a cutout whose background is transparent');
	const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
	const side = Math.trunc(Math.max(x1 - x0, y1 - y0) * padding);
	if (side < 2) throw new Error('the foreground is too small to crop');
	const half = Math.floor(side / 2);
	// PIL rounds the box; what falls outside the image is transparent black
	const bx = Math.round(cx - half), by = Math.round(cy - half), bw = Math.round(cx + half) - bx, bh = Math.round(cy + half) - by;
	const planes = [0, 1, 2].map(() => new Float32Array(bw * bh));
	for (let y = 0; y < bh; y++) {
		const sy = by + y;
		if (sy < 0 || sy >= H) continue;
		for (let x = 0; x < bw; x++) {
			const sx = bx + x;
			if (sx < 0 || sx >= W) continue;
			const p = (sy * W + sx) * 4, a = pixels[p + 3] / 255;
			for (let c = 0; c < 3; c++) planes[c][y * bw + x] = pixels[p + c] * a; // premultiplied, on black
		}
	}
	const out = resize(planes, bw, bh, size, size);
	const rgb = new Uint8Array(size * size * 3);
	for (let i = 0; i < size * size; i++) for (let c = 0; c < 3; c++) rgb[i * 3 + c] = Math.max(0, Math.min(255, Math.trunc(out[c][i])));
	return { width: size, height: size, channels: 3, pixels: rgb };
}

// The picture as DINOv2 reads it: [3, h, w] floats, ImageNet-normalised.
export function dinoPixels(img) {
	const { width: W, height: H, pixels } = img, n = W * H, out = new Float32Array(3 * n);
	const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
	for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[c * n + i] = (pixels[i * 3 + c] / 255 - mean[c]) / std[c];
	return out;
}
