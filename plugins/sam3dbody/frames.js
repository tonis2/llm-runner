// SAM 3D Body over a folder of video frames: one person, followed from frame
// to frame by a box around the keypoints the frame before found, in place of
// a person detector.
//
//   const paths = listFrames('frames/');           // ffmpeg -i clip.mp4 frames/%05d.png
//   const posed = await poseFrames(model, paths, { bbox, onFrame });
//
// Each posed frame keeps what a motion and an overlay need, not the whole
// pose: model parameters, camera translation, the 2D keypoints, the box.

import { llm } from '../lib/llm.js';

const IMAGE = /\.(png|jpe?g)$/i;

// The frame images in `dir`, in name order (zero-padded numbers sort right).
export function listFrames(dir) {
	const paths = llm.listDir(dir).filter((e) => !e.dir && IMAGE.test(e.name)).map((e) => `${dir.replace(/\/+$/, '')}/${e.name}`);
	if (paths.length === 0) throw new Error(`${dir} holds no .png or .jpg frames`);
	return paths;
}

// A person box [x1, y1, x2, y2] around 2D keypoints [70 * 2]: a margin on each
// side and more above, where the keypoints stop short of the top of the head.
export function keypointBox(kp, width, height) {
	let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
	for (let i = 0; i < kp.length; i += 2) {
		x1 = Math.min(x1, kp[i]);
		x2 = Math.max(x2, kp[i]);
		y1 = Math.min(y1, kp[i + 1]);
		y2 = Math.max(y2, kp[i + 1]);
	}
	const w = x2 - x1, h = y2 - y1;
	const box = [x1 - 0.1 * w, y1 - 0.15 * h, x2 + 0.1 * w, y2 + 0.05 * h];
	box[0] = Math.max(0, box[0]);
	box[1] = Math.max(0, box[1]);
	box[2] = Math.min(width, box[2]);
	box[3] = Math.min(height, box[3]);
	if (!(box[2] - box[0] >= 16 && box[3] - box[1] >= 16)) return null;
	return box;
}

// How much of a pose the crop saw: the fraction of its keypoints inside it.
function inCrop(pose) {
	const c = pose.pred_keypoints_2d_cropped;
	let n = 0;
	for (let i = 0; i < c.length; i += 2) if (Math.abs(c[i]) <= 0.5 && Math.abs(c[i + 1]) <= 0.5) n++;
	return n / (c.length / 2);
}

/**
 * Poses every frame of `paths` in turn.
 *
 *   bbox     the person's box in the first frame; without one the whole frame,
 *            posed twice - the second time in the box the first found
 *   focal    focal length in pixels (the image diagonal without one)
 *   hands    pose the hands as well (the model loaded with them)
 *   onFrame  (record, index, count) after each frame, awaited
 *
 * Returns a record a frame: { path, size, bbox, camera, params [204],
 * cam_t [3], keypoints_2d [140], in_crop }, and with hands `hands` [left,
 * right], whether each hand decoder's hand was used.
 */
export async function poseFrames(model, paths, { bbox = null, focal = null, hands = false, onFrame = null } = {}) {
	const out = [];
	let box = bbox;
	for (let i = 0; i < paths.length; i++) {
		const t0 = llm.now();
		const img = llm.image.load(paths[i]);
		// The first frame without a box is posed once for the box alone: no hands.
		const search = i === 0 && !bbox;
		let pose = await model.poseImage(img, { bbox: box, focal, hands: hands && !search, quiet: true });
		if (search) {
			const found = keypointBox(pose.pred_keypoints_2d, img.width, img.height);
			if (found || hands) pose = await model.poseImage(img, { bbox: found, focal, hands, quiet: true });
		}
		const record = {
			path: paths[i],
			size: [img.width, img.height],
			bbox: pose.bbox.slice(),
			camera: pose.camera.slice(),
			params: Float64Array.from(pose.mhr_model_params),
			cam_t: Array.from(pose.pred_cam_t),
			keypoints_2d: Float64Array.from(pose.pred_keypoints_2d),
			in_crop: inCrop(pose),
		};
		if (pose.hands_used) record.hands = pose.hands_used.slice();
		out.push(record);
		// The next frame's box: around this one's keypoints, or this box again
		// when they make none.
		box = keypointBox(record.keypoints_2d, img.width, img.height) ?? record.bbox;
		llm.print(`frame ${i + 1}/${paths.length} (${llm.since(t0)})`);
		await onFrame?.(record, i, paths.length);
	}
	return out;
}
