// A crop warped out of an 8-bit image by a scale-and-translate matrix, as
// cv2.warpAffine(img, M, (width, height), INTER_LINEAR) with BORDER_CONSTANT 0
// computes it - on the host or on the GPU - for a backbone's input.
//
//   const image = uploadImage(img);                  // once a frame
//   const input = warpCropGpu(image, img, affine, { width: 1280, height: 720 });
//
// `affine` is M [a, 0, tx, 0, d, ty]: a source pixel (x, y) lands at
// (a x + tx, d y + ty) in the crop.

import { f32 as f32Tensor } from './llm.js';
import * as op from './ops.js';
import { submit } from './gpu.js';

export const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };

const f32 = Math.fround;

// The positions first, as OpenCV 5 (and 4.11 on) computes them: the inverse
// map in double, each pixel's source position and the bilinear blend in
// float32 with fused multiply-adds. An M with no shear moves columns and rows
// apart, so a column's two source columns and blend are the same in every row,
// and a row's in every column. `geo` holds byte offsets [col0, col1] x width
// then [row0, row1] x height (-1 off the image), `wts` [alpha] x width then
// [beta] x height. `mirror` reads the image mirrored left to right
// (`img[:, ::-1]`) without making it. `size` is a square crop's side.
export function cropGeometry(img, affine, { size = 0, width = size, height = size, mirror = false } = {}) {
	const { channels } = img;
	const [a, , tx, , d, ty] = affine;
	// cv2's invertAffineTransform, M having no shear.
	const D = 1 / (a * d);
	const M = [d * D, -(d * D) * tx, a * D, -(a * D) * ty];
	const m0 = f32(M[0]), m3 = f32(M[2]);
	const geo = new Int32Array(2 * (width + height)), wts = new Float32Array(width + height);
	const column = (x) => (x < 0 || x >= img.width ? -1 : (mirror ? img.width - 1 - x : x) * channels);
	const row = (y) => (y < 0 || y >= img.height ? -1 : y * img.width * channels);
	for (let i = 0; i < width; i++) {
		// A product of two float32s is exact in a double: one rounding, an fma.
		const sx = f32(m0 * i + f32(M[1]));
		const ix = Math.floor(sx);
		wts[i] = sx - ix;
		geo[i] = column(ix);
		geo[width + i] = column(ix + 1);
	}
	for (let i = 0; i < height; i++) {
		const sy = f32(m3 * i + f32(M[3]));
		const iy = Math.floor(sy);
		wts[width + i] = sy - iy;
		geo[2 * width + i] = row(iy);
		geo[2 * width + height + i] = row(iy + 1);
	}
	return { geo, wts, width, height };
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

// The warp on the GPU from uploadImage's tensor: [3, height, width] floats,
// normalised by `norm` { mean, std } per channel (ImageNet's by default, to a
// float32 rounding of torchvision's normalize), or with `norm` null the bytes.
// Returns a Tensor the caller disposes.
export function warpCropGpu(image, img, affine, { size = 0, width = size, height = size, mirror = false, norm = IMAGENET } = {}) {
	const { geo, wts } = cropGeometry(img, affine, { width, height, mirror });
	const g = f32Tensor(geo.length), w = f32Tensor(wts.length), out = f32Tensor(3 * width * height);
	g.buffer.write(geo);
	w.buffer.write(wts);
	op.warpCrop(image, g, w, out, width, height, img.channels, norm);
	submit();
	g.dispose();
	w.dispose();
	return out;
}

// The matrix that puts `box` [x1, y1, x2, y2] (source pixels) onto a
// width x height crop, each axis scaled on its own.
export function boxAffine(box, width, height) {
	const [x1, y1, x2, y2] = box;
	const a = width / (x2 - x1), d = height / (y2 - y1);
	return [a, 0, -a * x1, 0, d, -d * y1];
}
