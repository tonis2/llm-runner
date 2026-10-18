// What SAM 3D Body's estimator makes of a frame and a person's box before the
// network sees them (`sam_3d_body_estimator.py`, `data/transforms`,
// `SAM3DBody.get_ray_condition`, `CameraEncoder`):
//
//   - the box padded by 1.25 and widened to a square, the crop warped to
//     512 x 512 by OpenCV's warpAffine (bilinear, 8-bit, black outside);
//   - the camera: the intrinsics given, or SAM 3D Body's default with no FOV
//     estimator, focal length the image diagonal and the principal point the
//     centre;
//   - each crop pixel's camera ray (x, y at z = 1), filtered down to the
//     backbone's 32 x 32 grid and Fourier-encoded to 99 channels;
//   - the CLIFF condition: the box centre and size over the focal length.

import { f32 as f32Tensor } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit } from '../lib/gpu.js';

export const SIZE = 512;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const f32 = Math.fround;

// GetBBoxCenterScale and TopdownAffine's two fix_aspect_ratio calls (to 3:4,
// then to the square input), in float32 as numpy has them.
export function boxToCrop(box, { padding = 1.25, size = SIZE } = {}) {
	const [x1, y1, x2, y2] = box.map(f32);
	const center = [f32(f32(x1 + x2) * 0.5), f32(f32(y1 + y2) * 0.5)];
	let w = f32(f32(x2 - x1) * f32(padding)), h = f32(f32(y2 - y1) * f32(padding));
	const fix = (ratio) => {
		if (w > f32(h * ratio)) h = f32(w / ratio);
		else w = f32(h * ratio);
	};
	fix(0.75);
	fix(1);
	// get_warp_matrix with no rotation: the box onto the crop, scaled by the
	// width. Its three source points are float32 - the centre, half a width
	// above it, and that turned about the second - and cv2.getAffineTransform
	// solves through them in double, so x and y scale apart by their rounding.
	const above = f32(center[1] - w * 0.5);
	const left = f32(center[0] - f32(center[1] - above));
	const ax = size / 2 / (center[0] - left), ay = size / 2 / (center[1] - above);
	const affine = [ax, 0, -ax * left, 0, ay, -ay * above];
	return { center, scale: [w, h], affine, size };
}

// The default intrinsics [fx, fy, cx, cy].
export function defaultCamera(width, height) {
	const f = Math.sqrt(height * height + width * width);
	return [f, f, width / 2, height / 2];
}

// cv2.warpAffine(img, M, (size, size), INTER_LINEAR), BORDER_CONSTANT 0, for
// an M of scale and translation only, as OpenCV 5 (and 4.11 on) computes it:
// the inverse map in double, each pixel's source position and the bilinear
// blend in float32 with fused multiply-adds, rounded to nearest-even. (Older
// OpenCV cut positions to 1/32 pixel in fixed point; its crops differ by one
// level here and there.)
//
// The positions first: an M with no shear moves columns and rows apart, so a
// column's two source columns and blend are the same in every row, and a
// row's in every column. `geo` holds byte offsets [col0, col1, row0, row1] x
// size (-1 off the image), `wts` [alpha, beta] x size. `mirror` reads the
// image mirrored left to right (`img[:, ::-1]`) without making it.
export function cropGeometry(img, affine, { size = SIZE, mirror = false } = {}) {
	const { width, height, channels } = img;
	// cv2's invertAffineTransform, M having no shear.
	const [a, , tx, , d, ty] = affine;
	const D = 1 / (a * d);
	const M = [d * D, -(d * D) * tx, a * D, -(a * D) * ty];
	const m0 = f32(M[0]), m3 = f32(M[2]);
	const geo = new Int32Array(4 * size), wts = new Float32Array(2 * size);
	const column = (x) => (x < 0 || x >= width ? -1 : (mirror ? width - 1 - x : x) * channels);
	const row = (y) => (y < 0 || y >= height ? -1 : y * width * channels);
	for (let i = 0; i < size; i++) {
		// A product of two float32s is exact in a double: one rounding, an fma.
		const sx = f32(m0 * i + f32(M[1]));
		const ix = Math.floor(sx);
		wts[i] = sx - ix;
		geo[i] = column(ix);
		geo[size + i] = column(ix + 1);
		const sy = f32(m3 * i + f32(M[3]));
		const iy = Math.floor(sy);
		wts[size + i] = sy - iy;
		geo[2 * size + i] = row(iy);
		geo[3 * size + i] = row(iy + 1);
	}
	return { geo, wts, size };
}

// The warp on the host: RGB bytes [size, size, 3]. `img` is { width, height,
// channels, pixels }.
export function warpCrop(img, affine, { size = SIZE, mirror = false } = {}) {
	const { geo, wts } = cropGeometry(img, affine, { size, mirror });
	const pixels = img.pixels;
	const round = (v) => {
		const r = Math.round(v);
		return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
	};
	const out = new Uint8Array(size * size * 3);
	for (let y = 0; y < size; y++) {
		const row0 = geo[2 * size + y], row1 = geo[3 * size + y], beta = wts[size + y];
		for (let x = 0; x < size; x++) {
			const alpha = wts[x], c0 = geo[x], c1 = geo[size + x];
			const o = (y * size + x) * 3;
			for (let c = 0; c < 3; c++) {
				const p00 = row0 < 0 || c0 < 0 ? 0 : pixels[row0 + c0 + c], p01 = row0 < 0 || c1 < 0 ? 0 : pixels[row0 + c1 + c];
				const p10 = row1 < 0 || c0 < 0 ? 0 : pixels[row1 + c0 + c], p11 = row1 < 0 || c1 < 0 ? 0 : pixels[row1 + c1 + c];
				const v0 = f32(alpha * (p01 - p00) + p00), v1 = f32(alpha * (p11 - p10) + p10);
				out[o + c] = Math.max(0, Math.min(255, round(f32(beta * f32(v1 - v0) + v0))));
			}
		}
	}
	return out;
}

// The image's bytes on the GPU, once a frame, for warpCropGpu.
export function uploadImage(img) {
	const bytes = img.pixels;
	const words = Math.ceil(bytes.length / 4);
	const t = f32Tensor(words);
	const padded = new Uint8Array(words * 4);
	padded.set(bytes);
	t.buffer.write(padded);
	return t;
}

// The warp on the GPU from uploadImage's tensor: [3, size, size] floats,
// normalised for the backbone (normalize's numbers, to a float32 rounding),
// or with `normalize` false the bytes. Returns a Tensor the caller disposes.
export function warpCropGpu(image, img, affine, { size = SIZE, mirror = false, normalize: norm = true } = {}) {
	const { geo, wts } = cropGeometry(img, affine, { size, mirror });
	const g = f32Tensor(geo.length), w = f32Tensor(wts.length), out = f32Tensor(3 * size * size);
	g.buffer.write(geo);
	w.buffer.write(wts);
	op.warpCrop(image, g, w, out, size, img.channels, norm ? { mean: MEAN, std: STD } : null);
	submit();
	g.dispose();
	w.dispose();
	return out;
}

// ToTensor and data_preprocess: [3, size, size], ImageNet mean and std.
export function normalize(crop, size = SIZE) {
	const n = size * size, out = new Float32Array(3 * n);
	for (let c = 0; c < 3; c++) {
		const m = f32(MEAN[c]), s = f32(STD[c]);
		for (let i = 0; i < n; i++) out[c * n + i] = f32(f32(f32(crop[i * 3 + c]) / 255) - m) / s;
	}
	return out;
}

// F.interpolate(mode="bilinear", antialias=True) of one axis from `n` to
// `n / factor`: a triangle as wide as the factor each side, renormalised
// where it runs off the edge (PyTorch's _upsample_bilinear2d_aa).
function antialiasDown(values, factor) {
	const out = new Float64Array(values.length / factor);
	const support = factor;
	for (let i = 0; i < out.length; i++) {
		const center = factor * (i + 0.5);
		const lo = Math.max(Math.trunc(center - support + 0.5), 0);
		const hi = Math.min(Math.trunc(center + support + 0.5), values.length);
		let s = 0, wsum = 0;
		for (let j = lo; j < hi; j++) {
			const w = Math.max(0, 1 - Math.abs((j - center + 0.5) / factor));
			s += w * values[j];
			wsum += w;
		}
		out[i] = s / wsum;
	}
	return out;
}

// The ray features [gh * gw, 99]: (x, y, 1), then sin and cos of pi times
// each coordinate times 16 frequencies from 1 to 32 (coordinate-major), as
// FourierPositionEncoding(n=3, num_bands=16, max_resolution=64).
export function rayFeatures(affine, cam, { size = SIZE, patch = 16 } = {}) {
	const [fx, fy, cx, cy] = cam;
	// A crop pixel's full-image position, then its ray, in float32.
	const rx = new Float64Array(size), ry = new Float64Array(size);
	for (let i = 0; i < size; i++) {
		rx[i] = f32(f32(f32(f32(i / f32(affine[0])) - f32(f32(affine[2]) / f32(affine[0]))) - f32(cx)) / f32(fx));
		ry[i] = f32(f32(f32(f32(i / f32(affine[4])) - f32(f32(affine[5]) / f32(affine[4]))) - f32(cy)) / f32(fy));
	}
	const gx = antialiasDown(rx, patch), gy = antialiasDown(ry, patch);
	const g = size / patch, bands = 16;
	const freq = Array.from({ length: bands }, (_, b) => f32(1 + (b * (32 - 1)) / (bands - 1)));
	const out = new Float32Array(g * g * 99);
	for (let y = 0; y < g; y++) {
		for (let x = 0; x < g; x++) {
			const pos = [f32(gx[x]), f32(gy[y]), 1];
			const o = (y * g + x) * 99;
			out.set(pos, o);
			for (let d = 0; d < 3; d++) {
				for (let b = 0; b < bands; b++) {
					const v = f32(Math.PI * f32(pos[d] * freq[b]));
					out[o + 3 + d * bands + b] = Math.sin(v);
					out[o + 3 + 48 + d * bands + b] = Math.cos(v);
				}
			}
		}
	}
	return out;
}

// (cx - ppx) / f, (cy - ppy) / f, box size / f: `_get_decoder_condition`
// with USE_INTRIN_CENTER.
export function cliffCondition(crop, cam) {
	const [f, , ppx, ppy] = cam;
	return [(crop.center[0] - ppx) / f, (crop.center[1] - ppy) / f, crop.scale[0] / f];
}
