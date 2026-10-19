// A motion clip (motion.js) written out: a .glb (skeleton and animation, no
// mesh), BVH text in centimetres, and the frames' JSON - each frame's box,
// camera, camera translation, 2D keypoints and foot contacts, for an overlay,
// and its MHR model parameters, to make the clip again from (`readPosed`).

import { motionGlb } from '../kimodo/glb.js';
import { motionBvh } from '../lib/bvh.js';
import { KEYPOINT_NAMES } from './keypoints.js';

const round = (v, d = 5) => Number(v.toFixed(d));

export function clipGlb(clip, name = 'capture') {
	return motionGlb({
		name, skeleton: clip.skeleton, fps: clip.fps, frames: clip.frames, rotations: clip.rotations, root: clip.root,
		rest: clip.skeleton.restRoot, generator: 'llm-runner sam3dbody',
		extras: { source: 'sam3dbody', camera: clip.camera.map((v) => round(v)), camera_rotation: clip.cameraRotation.map((v) => round(v, 6)), fps: clip.fps },
	});
}

export function clipBvh(clip) {
	return motionBvh({
		skeleton: clip.skeleton, fps: clip.fps, frames: clip.frames, rotations: clip.rotations, root: clip.root,
		rest: clip.skeleton.restRoot, scale: 100,
	});
}

// `posed` the frames as frames.js posed them, one a clip frame. A frame a line.
export function framesJson(clip, posed) {
	const doc = {
		frames: clip.frames,
		fps: clip.fps,
		camera_position: clip.camera.map((v) => round(v)),
		camera_rotation: clip.cameraRotation.map((v) => round(v, 6)),
		keypoint_names: KEYPOINT_NAMES,
	};
	const poses = posed.map((p, f) => ({
		image: p.path,
		bbox: p.bbox.map((v) => round(v, 2)),
		camera: p.camera.map((v) => round(v, 3)),
		camera_translation: p.cam_t.map((v) => round(v)),
		keypoints_2d: Array.from({ length: p.keypoints_2d.length / 2 }, (_, k) => [round(p.keypoints_2d[k * 2], 2), round(p.keypoints_2d[k * 2 + 1], 2)]),
		in_crop: round(p.in_crop, 3),
		contacts: Array.from(clip.contacts.subarray(f * 4, f * 4 + 4)),
		...(p.hands ? { hands_used: p.hands, hand_pass: p.hand_pass } : {}),
		mhr_model_params: Array.from(p.params, (v) => round(v, 7)),
	}));
	const head = Object.entries(doc).map(([k, v]) => ` ${JSON.stringify(k)}: ${JSON.stringify(v)}`);
	return `{\n${head.join(',\n')},\n "poses": [\n${poses.map((p) => `  ${JSON.stringify(p)}`).join(',\n')}\n ]\n}\n`;
}

// framesJson's text back: the clip's fps, the posed frames as frames.js gave
// them, and the hand_stride they were posed with.
export function readPosed(text) {
	const doc = JSON.parse(text);
	const posed = doc.poses.map((p, f) => {
		if (!p.mhr_model_params) throw new Error(`frame ${f} has no mhr_model_params (written by an older version)`);
		return {
			path: p.image,
			bbox: p.bbox,
			camera: p.camera,
			params: Float64Array.from(p.mhr_model_params),
			cam_t: p.camera_translation,
			keypoints_2d: Float64Array.from(p.keypoints_2d.flat()),
			in_crop: p.in_crop,
			...(p.hands_used ? { hands: p.hands_used, hand_pass: p.hand_pass ?? true } : {}),
		};
	});
	// hand_stride: the spacing of the frames that had hand passes.
	const passes = posed.map((p, f) => (p.hand_pass ? f : -1)).filter((f) => f >= 0);
	const handStride = passes.length > 1 ? passes[1] - passes[0] : 1;
	return { fps: doc.fps, posed, handStride };
}
