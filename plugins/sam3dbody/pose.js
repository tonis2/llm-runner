// SAM 3D Body on one image: a person's box (or the whole image) in, the
// posed MHR rig, its 70 keypoints in 3D and in the image, and the camera
// translation out. The body decoder only, as `inference_type="body"`: no hand
// crops or keypoint prompts yet, the hands are the body decoder's own.
//
//   const model = new Sam3dBody('sam-3d-body-dinov3');
//   await model.load();
//   const pose = await model.poseImage(llm.image.load('person.jpg'), { bbox: [x1, y1, x2, y2] });
//
// Coordinates are SAM 3D Body's camera frame (x right, y down, z forward),
// metres, the body's own: add `pred_cam_t` to place it in front of the camera.

import { llm } from '../lib/llm.js';
import { DinoV3 } from './dinov3.js';
import { openWeights, mhrPath } from './weights.js';
import { boxToCrop, warpCrop, normalize, defaultCamera, rayFeatures, cliffCondition, SIZE } from './crop.js';
import { MHR } from './mhr.js';
import { BodyHeads } from './heads.js';
import { BodyDecoder } from './decoder.js';

export class Sam3dBody {
	// `path` is the downloaded folder (model.ckpt and assets/mhr_model.pt).
	// `linear` is the backbone's weights ('q8', the default, or 'f32'),
	// `decoder` the decoder's ('f32', the default, or 'q8').
	constructor(path, { linear = 'q8', decoder = 'f32', mhr = null } = {}) {
		this.family = 'sam3dbody';
		this.path = path;
		this.linear = linear;
		this.decoderWeights = decoder;
		this.mhrFile = mhr ?? mhrPath(path);
	}

	async load() {
		const t0 = llm.now();
		this.m = openWeights(this.path);
		if (!llm.exists(this.mhrFile)) throw new Error(`${this.mhrFile} does not exist (the MHR rig, assets/mhr_model.pt in the download)`);
		this.mhr = new MHR(this.mhrFile);
		this.heads = new BodyHeads(this.m, this.mhr);
		this.encoder = new DinoV3(this.m, { linear: this.linear });
		await this.encoder.load();
		this.decoder = new BodyDecoder(this.m, { linear: this.decoderWeights });
		await this.decoder.load();
		llm.print(`  SAM 3D Body loaded in ${llm.since(t0)}`);
	}

	dispose() {
		this.encoder?.dispose();
		this.decoder?.dispose();
		this.heads?.dispose();
		this.mhr?.close();
		this.m?.close();
		this.encoder = this.decoder = this.heads = this.mhr = this.m = null;
	}

	// `img` is { width, height, channels, pixels }. `bbox` [x1, y1, x2, y2] in
	// pixels (the whole image without one, as SAM 3D Body with no detector);
	// `camera` [fx, fy, cx, cy] (the image diagonal as the focal length and the
	// centre without one), or `focal` alone. `mesh` skins all 18439 vertices
	// into `pred_vertices` (a few seconds more). `quiet` leaves out the line
	// this prints.
	async poseImage(img, { bbox = null, camera = null, focal = null, mesh = false, quiet = false } = {}) {
		const t0 = llm.now();
		const box = bbox ?? [0, 0, img.width, img.height];
		let cam = camera ?? defaultCamera(img.width, img.height);
		if (focal && !camera) cam = [focal, focal, cam[2], cam[3]];
		const crop = boxToCrop(box);
		const view = { crop, cam };
		const features = await this.encoder.encode(normalize(warpCrop(img, crop.affine)), SIZE, SIZE);
		let result;
		try {
			result = await this.decoder.run(features, rayFeatures(crop.affine, cam), cliffCondition(crop, cam),
				(token, layer) => this.heads.decode(token, view, { allVertices: mesh && layer === this.decoder.depth - 1 }));
		} finally {
			features.dispose();
		}
		const pose = result.pose;
		const hands = this.heads.handBoxes(result.tokens.subarray(3 * this.decoder.dim));
		pose.hand_box = hands.boxes;
		pose.hand_logits = hands.logits;
		pose.bbox = box;
		pose.camera = cam;
		if (!quiet) llm.print(`  SAM 3D Body: ${img.width}x${img.height}, box ${box.map((v) => Math.round(v)).join(' ')}, posed in ${llm.since(t0)}`);
		return pose;
	}
}
