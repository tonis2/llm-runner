// The picture Pixal3D conditions on, as its pipeline's preprocess_image makes
// it from a cutout: squared up around the alpha's bounding box (alpha over
// 80%) with a 10% margin, premultiplied onto black, left at that size. Each
// stage then resizes it (PIL Lanczos) to its own 512 or 1024.
//
// The background is not removed here: the image must already be a cutout.

import { resizeLanczos, roundHalfEven } from '/core/image/resize.js';

const f = Math.fround;

// PIL's RGBA -> RGBa (premultiplied) and back, which its resize of an RGBA
// picture goes through.
function premultiply(px) {
	const out = new Uint8Array(px.length);
	for (let i = 0; i < px.length; i += 4) {
		const a = px[i + 3];
		for (let c = 0; c < 3; c++) {
			const t = px[i + c] * a + 128;
			out[i + c] = ((t >> 8) + t) >> 8;
		}
		out[i + 3] = a;
	}
	return out;
}
function unpremultiply(px) {
	const out = new Uint8Array(px.length);
	for (let i = 0; i < px.length; i += 4) {
		const a = px[i + 3];
		for (let c = 0; c < 3; c++) out[i + c] = a === 255 || a === 0 ? px[i + c] : Math.min(255, Math.floor((255 * px[i + c]) / a));
		out[i + 3] = a;
	}
	return out;
}

// An 8-bit RGBA image ({ width, height, channels, pixels }) to the square RGB
// picture { width, height, channels: 3, pixels }.
export function preprocess(img) {
	let { width: W, height: H, channels: C, pixels } = img;
	if (C !== 4) throw new Error(`Pixal3D needs a cutout with transparency (an RGBA image); this one has ${C} channels. Remove the background first.`);
	if (Math.max(W, H) > 1024) {
		const s = 1024 / Math.max(W, H), w = Math.trunc(W * s), h = Math.trunc(H * s);
		pixels = unpremultiply(resizeLanczos(premultiply(pixels), W, H, 4, w, h));
		W = w;
		H = h;
	}
	let x0 = W, y0 = H, x1 = -1, y1 = -1, clear = 0;
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const a = pixels[(y * W + x) * 4 + 3];
			if (a > 204) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } else clear++;
		}
	}
	if (x1 < 0) throw new Error('the image has no foreground (no pixel is more than 80% opaque)');
	if (clear === 0) throw new Error('the image is opaque (every pixel is more than 80% opaque): Pixal3D needs a cutout whose background is transparent');
	const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
	const size = Math.trunc(Math.max(x1 - x0, y1 - y0) * 1.1);
	if (size < 2) throw new Error('the foreground is too small to crop');
	const half = Math.floor(size / 2);
	const bx = roundHalfEven(cx - half), by = roundHalfEven(cy - half);
	const bw = roundHalfEven(cx + half) - bx, bh = roundHalfEven(cy + half) - by;
	const rgb = new Uint8Array(bw * bh * 3);
	for (let y = 0; y < bh; y++) {
		const sy = by + y;
		if (sy < 0 || sy >= H) continue;
		for (let x = 0; x < bw; x++) {
			const sx = bx + x;
			if (sx < 0 || sx >= W) continue;
			const p = (sy * W + sx) * 4, a = f(pixels[p + 3] / 255);
			for (let c = 0; c < 3; c++) rgb[(y * bw + x) * 3 + c] = Math.trunc(f(Math.min(1, f(f(pixels[p + c] / 255) * a)) * 255));
		}
	}
	return { width: bw, height: bh, channels: 3, pixels: rgb };
}

// The square picture resized to `size` (PIL Lanczos), as [3, size, size]
// floats in [0, 1].
export function planes(pre, size) {
	const px = pre.width === size && pre.height === size ? pre.pixels : resizeLanczos(pre.pixels, pre.width, pre.height, 3, size, size);
	const n = size * size, out = new Float32Array(3 * n);
	for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[c * n + i] = px[i * 3 + c] / 255;
	return out;
}

// ImageNet normalisation of `planes` output, for DINOv3.
export function normalise(p, size) {
	const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
	const n = size * size, out = new Float32Array(3 * n);
	for (let c = 0; c < 3; c++) for (let i = c * n; i < (c + 1) * n; i++) out[i] = (p[i] - MEAN[c]) / STD[c];
	return out;
}
