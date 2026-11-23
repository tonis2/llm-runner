// The picture Hi3D conditions on, as its `prepare_image` makes it: the
// smallest square around the alpha's bounding box (any alpha above zero),
// composited onto white the way PIL's paste blends, then centred in a white
// square 1.2 times as wide. The encoders then take their own sizes of it:
// DINOv2 a torch bilinear resize to 756, CLIP a PIL Lanczos one to 224.
//
// Ported from sparc3d/utils/image.py and models/conditioners/image.py.

import { resizeLanczos } from '/core/image/resize.js';

const f = Math.fround;
export const PADDING = 1.2;
export const DINO_SIZE = 756;
export const CLIP_SIZE = 224;

// PIL's (a + 128 + ((a + 128) >> 8)) >> 8, its divide by 255.
const div255 = (a) => { const t = a + 128; return ((t >> 8) + t) >> 8; };

// An 8-bit image ({ width, height, channels, pixels }) to the square RGB
// picture { width, height, channels: 3, pixels }. An opaque image (or one
// without alpha) keeps its whole extent, squared.
export function prepare(img) {
	const { width: W, height: H, channels: C, pixels } = img;
	if (C !== 3 && C !== 4) throw new Error(`Hi3D: a ${C}-channel image; RGB or RGBA expected`);
	const alpha = (x, y) => (C === 4 ? pixels[(y * W + x) * 4 + 3] : 255);
	let x0 = W, y0 = H, x1 = -1, y1 = -1;
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			if (alpha(x, y) > 0) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
		}
	}
	if (x1 < 0) throw new Error('Hi3D: the image is fully transparent');
	const w = x1 + 1 - x0, h = y1 + 1 - y0, side = Math.max(w, h);
	const left = x0 - Math.floor((side - w) / 2), top = y0 - Math.floor((side - h) / 2);
	// The crop onto white; outside the image is transparent, so white.
	const outer = Math.trunc(side * PADDING);
	const off = Math.floor((outer - side) / 2);
	const out = new Uint8Array(outer * outer * 3).fill(255);
	for (let y = 0; y < side; y++) {
		const sy = top + y;
		if (sy < 0 || sy >= H) continue;
		for (let x = 0; x < side; x++) {
			const sx = left + x;
			if (sx < 0 || sx >= W) continue;
			const p = (sy * W + sx) * C, a = C === 4 ? pixels[p + 3] : 255, o = ((y + off) * outer + x + off) * 3;
			for (let c = 0; c < 3; c++) out[o + c] = div255(pixels[p + c] * a + 255 * (255 - a));
		}
	}
	return { width: outer, height: outer, channels: 3, pixels: out };
}

// torch's F.interpolate(mode='bilinear', align_corners=False), no antialias:
// per output index the two input indices and the second one's weight.
function linearTaps(inSize, outSize) {
	const scale = f(inSize / outSize), taps = [];
	for (let o = 0; o < outSize; o++) {
		let real = f(f(scale * f(o + 0.5)) - 0.5);
		if (real < 0) real = 0;
		const i = Math.min(Math.floor(real), inSize - 1);
		const l = Math.min(Math.max(f(real - i), 0), 1);
		taps.push({ i0: i, i1: i < inSize - 1 ? i + 1 : i, w0: f(1 - l), w1: l });
	}
	return taps;
}

// The prepared picture as DINOv2's input: [3, 756, 756], bilinear from the
// [0, 1] floats, then ImageNet's mean and std.
export function dinoPixels(pre, size = DINO_SIZE) {
	const MEAN = [0.485, 0.456, 0.406].map(f), STD = [0.229, 0.224, 0.225].map(f);
	const { width: W, height: H, pixels } = pre;
	const tx = linearTaps(W, size), ty = linearTaps(H, size), n = size * size, out = new Float32Array(3 * n);
	const v = (x, y, c) => f(pixels[(y * W + x) * 3 + c] / 255);
	for (let y = 0; y < size; y++) {
		const { i0: ya, i1: yb, w0: h0, w1: h1 } = ty[y];
		for (let x = 0; x < size; x++) {
			const { i0: xa, i1: xb, w0, w1 } = tx[x];
			for (let c = 0; c < 3; c++) {
				const top = f(f(v(xa, ya, c) * w0) + f(v(xb, ya, c) * w1));
				const bottom = f(f(v(xa, yb, c) * w0) + f(v(xb, yb, c) * w1));
				const p = f(f(top * h0) + f(bottom * h1));
				out[c * n + y * size + x] = f(f(p - MEAN[c]) / STD[c]);
			}
		}
	}
	return out;
}

// The prepared picture as CLIP's input: [3, 224, 224], PIL Lanczos on the
// bytes, then CLIP's mean and std.
export function clipPixels(pre, size = CLIP_SIZE) {
	const MEAN = [0.48145466, 0.4578275, 0.40821073].map(f), STD = [0.26862954, 0.26130258, 0.27577711].map(f);
	const px = resizeLanczos(pre.pixels, pre.width, pre.height, 3, size, size);
	const n = size * size, out = new Float32Array(3 * n);
	for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[c * n + i] = f(f(f(px[i * 3 + c] / 255) - MEAN[c]) / STD[c]);
	return out;
}
