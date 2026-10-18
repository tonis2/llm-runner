// SAM 3D Body on one image: a person's box (or the whole image) in, the
// posed MHR rig, its 70 keypoints in 3D and in the image, and the camera
// translation out. By default the body decoder only, as
// `inference_type="body"`: the hands are the body decoder's own. With `hands`,
// as `inference_type="full"` (hands.js): each hand's crop through the backbone
// and the hand decoder, the body decoder again prompted with the good hands'
// wrists and elbows, and their fingers and wrists put in; three backbone
// passes instead of one.
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
import { handBoxes, handInputs, unmirrorLeft, checkHands, handPrompts, previousEstimate, mergeHands } from './hands.js';

export class Sam3dBody {
	// `path` is the downloaded folder (model.ckpt and assets/mhr_model.pt).
	// `linear` is the backbone's weights ('q8', the default, or 'f32'),
	// `decoder` the decoders' ('f32', the default, or 'q8'). `hands` loads
	// the hand decoder as well, and poses hands unless poseImage says not to.
	constructor(path, { linear = 'q8', decoder = 'f32', mhr = null, hands = false } = {}) {
		this.family = 'sam3dbody';
		this.path = path;
		this.linear = linear;
		this.decoderWeights = decoder;
		this.mhrFile = mhr ?? mhrPath(path);
		this.hands = hands;
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
		if (this.hands) {
			this.handHeads = new BodyHeads(this.m, this.mhr, { hand: true });
			this.handDecoder = new BodyDecoder(this.m, { linear: this.decoderWeights, hand: true });
			await this.handDecoder.load();
			this.jointRotation = this.m.floats('head_pose.joint_rotation');
		}
		llm.print(`  SAM 3D Body loaded in ${llm.since(t0)}`);
	}

	dispose() {
		this.encoder?.dispose();
		this.decoder?.dispose();
		this.heads?.dispose();
		this.handDecoder?.dispose();
		this.handHeads?.dispose();
		this.mhr?.close();
		this.m?.close();
		this.encoder = this.decoder = this.heads = this.handDecoder = this.handHeads = this.mhr = this.m = null;
	}

	// `img` is { width, height, channels, pixels }. `bbox` [x1, y1, x2, y2] in
	// pixels (the whole image without one, as SAM 3D Body with no detector);
	// `camera` [fx, fy, cx, cy] (the image diagonal as the focal length and the
	// centre without one), or `focal` alone. `mesh` skins all 18439 vertices
	// into `pred_vertices` (a few seconds more). `hands` poses the hands with
	// the hand decoder (when loaded with it). `quiet` leaves out the line this
	// prints.
	async poseImage(img, { bbox = null, camera = null, focal = null, mesh = false, hands = this.hands, quiet = false } = {}) {
		const t0 = llm.now();
		const box = bbox ?? [0, 0, img.width, img.height];
		let cam = camera ?? defaultCamera(img.width, img.height);
		if (focal && !camera) cam = [focal, focal, cam[2], cam[3]];
		if (hands && !this.handDecoder) throw new Error('hands=true needs the model loaded with hands');
		const crop = boxToCrop(box);
		const view = { crop, cam };
		const rays = rayFeatures(crop.affine, cam), condition = cliffCondition(crop, cam);
		const features = await this.encoder.encode(normalize(warpCrop(img, crop.affine)), SIZE, SIZE);
		let pose;
		try {
			const result = await this.decoder.run(features, rays, condition,
				(token, layer) => this.heads.decode(token, view, { allVertices: mesh && !hands && layer === this.decoder.depth - 1 }));
			pose = result.pose;
			const boxes = this.heads.handBoxes(result.handTokens);
			pose.hand_box = boxes.boxes;
			pose.hand_logits = boxes.logits;
			if (hands) pose = await this.poseHands(img, features, pose, { view, rays, condition, mesh });
		} finally {
			features.dispose();
		}
		pose.bbox = box;
		pose.camera = cam;
		if (!quiet) llm.print(`  SAM 3D Body: ${img.width}x${img.height}, box ${box.map((v) => Math.round(v)).join(' ')}, posed in ${llm.since(t0)}`);
		return pose;
	}

	// One hand crop through the backbone and the hand decoder; `mirror` cuts
	// it from the image mirrored left to right.
	async poseHand(img, box, cam, mirror = false) {
		const input = handInputs(img, box, cam, { mirror });
		const features = await this.encoder.encode(input.pixels, SIZE, SIZE);
		try {
			const result = await this.handDecoder.run(features, input.rays, input.condition, (token) => this.handHeads.decode(token, input.view));
			return { pose: result.pose, crop: input.crop };
		} finally {
			features.dispose();
		}
	}

	// The hands of `body` (the body pass's pose on `features`), and the body
	// again with them (hands.js).
	async poseHands(img, features, body, { view, rays, condition, mesh }) {
		const [leftBox, rightBox] = handBoxes(body.hand_box, view.crop);
		const W = img.width;
		const left = await this.poseHand(img, [W - leftBox[2] - 1, leftBox[1], W - leftBox[0] - 1, leftBox[3]], view.cam, true);
		unmirrorLeft(left.pose, this.heads);
		const right = await this.poseHand(img, rightBox, view.cam);
		const check = checkHands(body, left.pose, right.pose, [left.crop, right.crop], W, this.jointRotation);
		const prompts = handPrompts(body, check, view);
		let prompted = body;
		if (prompts.length) {
			const result = await this.decoder.run(features, rays, condition, (token) => this.heads.decode(token, view), { prompts, prev: previousEstimate(body) });
			prompted = result.pose;
		}
		const out = mergeHands(this.heads, prompted, left.pose, right.pose, check, view, this.jointRotation, { allVertices: mesh });
		out.hand_box = body.hand_box;
		out.hand_logits = body.hand_logits;
		out.hand_boxes = [leftBox, rightBox];
		return out;
	}
}
