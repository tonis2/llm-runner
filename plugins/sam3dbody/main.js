// SAM 3D Body (Meta) as a plugin: a person in an image to a posed MHR rig, or
// a folder of video frames to a motion clip on the MHR skeleton.
//
//   llm-runner sam3dbody model=sam-3d-body-dinov3 input=person.jpg output=pose.json \
//       [bbox=x1,y1,x2,y2] [focal=1400] [preview=pose.png] [mesh=pose.obj]
//
//   llm-runner sam3dbody model=sam-3d-body-dinov3 frames=frames/ fps=24 glb=motion.glb \
//       [bvh=motion.bvh] [output=frames.json] [bbox=...] [first=0] [count=0] [stride=1] \
//       [smooth=7] [cleanup=true]
//
// `model` is the folder downloaded from huggingface.co/facebook/sam-3d-body-dinov3
// (gated): model.ckpt, and the MHR rig as assets/mhr_model.pt. Without `bbox`
// the whole image is the person's box, as SAM 3D Body with no detector; without
// `focal` the focal length is the image diagonal, as with no FOV estimator.
//
// One image: `output` is JSON: the 70 keypoints (names, 2D in the image, 3D),
// the 127 MHR joints' positions and world rotations, the camera translation and
// focal length, and the MHR parameters (model parameters, identity, scale, hand
// PCA). 3D is SAM 3D Body's camera frame (x right, y down, z forward, metres)
// about the body; add `camera_translation` to put it in front of the camera.
// `preview` draws the keypoints and their links over the image; `mesh` writes
// the posed mesh as an OBJ, in the camera frame turned y up (the skin is a
// few seconds more).
//
// Frames: `frames` is a folder of PNG/JPEG frames in name order
// (`ffmpeg -i clip.mp4 frames/%05d.png`); `first`, `count` (0: to the end) and
// `stride` pick which. The person is followed by a box around the keypoints of
// the frame before; `bbox` is their box in the first frame. The clip is the MHR
// skeleton (126 joints, y up, metres, standing on y = 0) with the body's
// proportions held to their median: `glb` (skeleton and animation, no mesh),
// `bvh` (centimetres). `smooth` is the smoothing window in frames (1: none);
// `cleanup` plants the feet where they touch the floor (Kimodo's clean-up).
// `output` writes each frame's box, camera, camera translation and 2D keypoints
// as JSON. Run in-process, the clip and that JSON are also handed over as the
// job's outputs `glb` (as Kimodo's) and `frames.json`.
//
// Settings: linear (q8, the backbone's weights; f32), decoder (f32; q8),
// matrix_cores (true).

import { llm } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { writeText, utf8 } from '../lib/graph/plugins.js';
import { Sam3dBody } from './pose.js';
import { KEYPOINT_NAMES, drawPreview } from './keypoints.js';
import { listFrames, poseFrames } from './frames.js';
import { assembleMotion } from './motion.js';
import { cleanUpMotion } from '../kimodo/cleanup.js';
import { clipGlb, clipBvh, framesJson } from './output.js';

const round = (v, d = 5) => Number(v.toFixed(d));
const rows = (a, n) => Array.from({ length: a.length / n }, (_, i) => Array.from(a.slice(i * n, i * n + n), (v) => round(v)));

function parseList(v, n, what) {
	if (v === undefined || v === null || v === '') return null;
	const list = Array.isArray(v) ? v.map(Number) : String(v).split(',').map(Number);
	if (list.length !== n || list.some((x) => !Number.isFinite(x))) throw new Error(`${what} wants ${n} numbers, got ${v}`);
	return list;
}

function writeObj(path, vertices, faces) {
	const lines = [`# SAM 3D Body: ${vertices.length / 3} vertices, ${faces.length / 3} triangles, metres, y up`];
	for (let i = 0; i < vertices.length; i += 3) lines.push(`v ${vertices[i].toFixed(5)} ${(-vertices[i + 1]).toFixed(5)} ${(-vertices[i + 2]).toFixed(5)}`);
	for (let i = 0; i < faces.length; i += 3) lines.push(`f ${faces[i] + 1} ${faces[i + 1] + 1} ${faces[i + 2] + 1}`);
	writeText(path, lines.join('\n') + '\n');
}

// A key a line, each value on one.
const jsonLines = (doc) => `{\n${Object.entries(doc).map(([k, v]) => ` ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(',\n')}\n}\n`;

async function poseOne(model, config, bbox, focal) {
	const img = llm.image.load(config.input);
	const pose = await model.poseImage(img, { bbox, focal, mesh: !!config.mesh });
	const out = config.output ?? 'pose.json';
	const doc = {
		image: config.input,
		size: [img.width, img.height],
		bbox: pose.bbox.map((v) => round(v, 2)),
		camera: { fx: round(pose.camera[0], 3), fy: round(pose.camera[1], 3), cx: round(pose.camera[2], 3), cy: round(pose.camera[3], 3) },
		camera_translation: Array.from(pose.pred_cam_t, (v) => round(v)),
		keypoint_names: KEYPOINT_NAMES,
		keypoints_2d: rows(pose.pred_keypoints_2d, 2),
		keypoints_3d: rows(pose.pred_keypoints_3d, 3),
		joints: rows(pose.pred_joint_coords, 3),
		joint_rotations: rows(pose.joint_global_rots, 9),
		mhr: {
			model_params: Array.from(pose.mhr_model_params, (v) => round(v, 6)),
			identity: Array.from(pose.shape, (v) => round(v, 6)),
			scale: Array.from(pose.scale, (v) => round(v, 6)),
			hand_pca: Array.from(pose.hand, (v) => round(v, 6)),
			global_rotation: Array.from(pose.global_rot, (v) => round(v, 6)),
			body_pose: Array.from(pose.body_pose, (v) => round(v, 6)),
		},
		hand_boxes: rows(pose.hand_box, 4),
	};
	writeText(out, jsonLines(doc));
	llm.print(`  wrote ${out}`);
	if (config.preview) {
		llm.image.savePng(config.preview, drawPreview(img, pose));
		llm.print(`  wrote ${config.preview}`);
	}
	if (config.mesh) {
		writeObj(config.mesh, pose.pred_vertices, model.mhr.faces());
		llm.print(`  wrote ${config.mesh}`);
	}
	return out;
}

async function poseClip(model, config, bbox, focal) {
	const fps = Number(config.fps ?? 24);
	const stride = Math.max(1, Math.trunc(Number(config.stride ?? 1)));
	const first = Math.max(0, Math.trunc(Number(config.first ?? 0)));
	const count = Math.max(0, Math.trunc(Number(config.count ?? 0)));
	let paths = listFrames(config.frames).slice(first).filter((_, i) => i % stride === 0);
	if (count) paths = paths.slice(0, count);
	llm.print(`  ${paths.length} frames from ${config.frames}`);
	const posed = await poseFrames(model, paths, { bbox, focal });
	const clip = assembleMotion(model.mhr, posed, { fps: fps / stride, smooth: Number(config.smooth ?? 7) });
	const cleanup = config.cleanup === undefined || config.cleanup === true || config.cleanup === 'true' || config.cleanup === '1';
	if (cleanup && clip.frames > 2) {
		const done = cleanUpMotion({ skeleton: clip.skeleton, key: 'mhr', frames: clip.frames, rotations: clip.rotations, root: clip.root, contacts: clip.contacts });
		llm.print(`  clean-up: ${done.plantedFrames} planted foot frames`);
	}
	const glb = clipGlb(clip, config.frames.replace(/\/+$/, '').split('/').pop() || 'capture');
	const glbPath = config.glb ?? (config.bvh ? null : 'motion.glb');
	if (glbPath) {
		llm.writeBytes(glbPath, glb);
		llm.print(`  wrote ${glbPath}`);
	}
	if (config.bvh) {
		writeText(config.bvh, clipBvh(clip));
		llm.print(`  wrote ${config.bvh}`);
	}
	const text = framesJson(clip, posed);
	if (config.output) {
		writeText(config.output, text);
		llm.print(`  wrote ${config.output}`);
	}
	if (llm.canOutput()) {
		llm.output('glb', glb);
		llm.output('frames.json', utf8(text));
	}
	return glbPath ?? config.bvh;
}

llm.plugin({
	name: 'sam3dbody',
	async generate(config) {
		const start = llm.now();
		if (!config.model) throw new Error('model= is the sam-3d-body-dinov3 folder');
		if (!config.input && !config.frames) throw new Error('input= is the image, or frames= a folder of video frames');
		op.useMatrixCores(config.matrix_cores ?? true);
		llm.print(`=== SAM 3D Body: ${config.input ?? config.frames} ===`);
		const bbox = parseList(config.bbox, 4, 'bbox');
		const focal = config.focal ? Number(config.focal) : null;
		const model = new Sam3dBody(config.model, { linear: config.linear ?? 'q8', decoder: config.decoder ?? 'f32' });
		await model.load();
		let result;
		try {
			result = config.frames ? await poseClip(model, config, bbox, focal) : await poseOne(model, config, bbox, focal);
		} finally {
			model.dispose();
		}
		llm.print(`=== done in ${llm.since(start)} ===`);
		return result;
	},
});
