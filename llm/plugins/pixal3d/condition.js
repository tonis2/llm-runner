// Pixal3D's conditioning ("proj" mode): the picture's DINOv3 tokens, and for
// each voxel of the stage's grid the features under the point where the
// camera sees it.
//
//   global  [5, 1024]: CLS and the 4 register tokens, cross-attended by every block
//   proj    [n, 1024] (sparse structure) or [n, 2048] (shape, texture): per
//           voxel, the DINO patch map bilinearly sampled at its projection,
//           then (with NAF) the NAF-upsampled map sampled at the same point
//
// The camera is Blender's: the grid [-1, 1]^3, rotated z-up, scaled by
// 1 / (2 mesh_scale), seen from -y at `distance` with a horizontal field of
// view `fov`; Pixal3D picks the distance so that x = -1 at the object's
// depth lands on the picture's left edge.
//
// Ported from Pixal3D's trainers/flow_matching/mixins/image_conditioned_proj.py
// (DinoV3ProjFeatureExtractor, ProjGrid, project_points_to_image_batch,
// sample_features) and inference.py (distance_from_fov).

import { f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { DinoV3 } from '/core/models/dinov3.js';
import { Naf, sampleNaf, disposeEncoding } from './naf.js';
import { planes, normalise } from './image.js';

const f = Math.fround;
const DIM = 1024, PREFIX = 5, PATCH = 16;

export function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

// The camera for a horizontal field of view (radians), as inference.py's
// distance_from_fov with the grid point (-1, 0, 0) on pixel 0 of 512.
export function camera(fov, meshScale = 1) {
	const fPixels = ((16 / Math.tan(fov / 2)) * 512) / 32;
	const xw = -1 / meshScale / 2;
	return { fov, distance: (fPixels * xw) / (0 - 512 / 2), meshScale };
}

// torch.linspace(-1, 1, n) in float32.
function linspace(n) {
	const step = f(2 / (n - 1)), out = new Float64Array(n);
	for (let i = 0; i < n; i++) out[i] = i < n / 2 ? f(-1 + f(step * i)) : f(1 - f(step * (n - 1 - i)));
	return out;
}

// Where the camera sees the voxels of an r^3 grid in a `res` picture, as
// grid_sample's [-1, 1] coordinates (x right, y down) [n, 2]. `coords` [n, 3]
// are (i, j, k) cells, or null for the whole grid in its own order (i slowest).
export function projectGrid(r, cam, res, coords = null) {
	const l = linspace(r), n = coords ? coords.length / 3 : r * r * r;
	const fx = 16 / Math.tan(cam.fov / 2) / 32; // focal length over the picture's width
	const s = 2 * cam.meshScale, out = new Float64Array(n * 2);
	for (let t = 0; t < n; t++) {
		const i = coords ? coords[3 * t] : Math.floor(t / (r * r));
		const j = coords ? coords[3 * t + 1] : Math.floor(t / r) % r;
		const k = coords ? coords[3 * t + 2] : t % r;
		// world (x, -z, y) / s; the camera's frame is then (x, y, z - distance) / s
		const depth = cam.distance - l[k] / s;
		// pixel x = res (fx x / depth + 1/2), and grid_sample's (x + 0.5) / res * 2 - 1
		const px = (fx * (l[i] / s)) / (depth + 1e-8) + 0.5, py = 0.5 - (fx * (l[j] / s)) / (depth + 1e-8);
		out[2 * t] = 2 * (px + 0.5 / res) - 1;
		out[2 * t + 1] = 2 * (py + 0.5 / res) - 1;
	}
	return out;
}

// Those coordinates in a w x w map's pixel units (align_corners=False), clamped
// to the border, as float2 for bilinear_rows and naf_sample.
export function mapPoints(ndc, w) {
	const out = new Float32Array(ndc.length);
	for (let i = 0; i < ndc.length; i++) out[i] = Math.min(Math.max(((ndc[i] + 1) * w - 1) / 2, 0), w - 1);
	return out;
}

export class Conditioner {
	constructor(weights, { linear = 'f32' } = {}) {
		this.weights = weights;
		this.linear = linear;
	}

	async load() {
		this.dino = new DinoV3(openSafetensors(`${this.weights}/dinov3-vitl16/model.safetensors`), { prefix: '', linear: this.linear });
		await this.dino.load();
		this.naf = new Naf(`${this.weights}/naf/naf_release.pth`);
		this.naf.load();
	}

	// The weights; pictures already made stay usable for proj().
	dispose() {
		this.dino?.dispose();
		this.naf?.dispose();
		this.dino = this.naf = null;
	}

	// One picture size's DINOv3 tokens [5 + g * g, 1024] (normed without
	// affine) and, when `naf` is given, NAF's maps at that output size (the
	// shape-1024 stage's are 512 on its 1024 picture).
	async picture(pre, size, { naf = 0 } = {}) {
		const p = planes(pre, size);
		const tokens = await this.dino.encode(normalise(p, size), size, size, { all: true, norm: 'plain' });
		const g = size / PATCH;
		let enc = null;
		if (naf) {
			const img = upload(p);
			try {
				enc = this.naf.encode(img, size, g, naf);
			} finally {
				img.dispose();
			}
		}
		return { tokens, size, g, naf: enc };
	}

	disposePicture(pic) {
		pic?.tokens?.dispose();
		disposeEncoding(pic?.naf);
	}

	// CLS and registers [5, 1024], a Tensor of its own.
	global(pic) {
		const out = f32(PREFIX * DIM);
		op.copyRows(pic.tokens, out, PREFIX, DIM, DIM, 0);
		return out;
	}

	// The proj features of an r^3 grid's voxels (`coords` [n, 3], or null for
	// all of them), [n, 1024], or with `naf` (the picture's NAF maps) [n, 2048].
	proj(pic, r, cam, coords = null, { naf = !!pic.naf } = {}) {
		const ndc = projectGrid(r, cam, pic.size, coords), n = ndc.length / 2;
		const width = naf ? 2 * DIM : DIM;
		const out = f32(n * width, [width, n]);
		const patches = pic.tokens.view(PREFIX * DIM * 4, pic.g * pic.g * DIM * 4);
		const lr = upload(mapPoints(ndc, pic.g));
		let hr = null;
		try {
			op.bilinearRows(patches, lr, out, DIM, pic.g, pic.g, n, width, 0);
			if (naf) {
				hr = upload(mapPoints(ndc, pic.naf.s));
				sampleNaf(pic.naf, patches, hr, out, n, { stride: width, col: DIM });
			}
			submit();
		} finally {
			lr.dispose();
			hr?.dispose();
		}
		return out;
	}
}
