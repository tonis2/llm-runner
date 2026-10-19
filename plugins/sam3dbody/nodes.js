// SAM 3D Body as nodes: load the model once, pose an image or a folder of
// video frames, then clean up and save the motion.
//
//   sam3dbody.load -> MODEL
//   sam3dbody.pose_image (MODEL, IMAGE) -> POSE, sam3dbody.draw_pose -> IMAGE
//   sam3dbody.pose_frames (MODEL, frames folder) -> MOTION
//   sam3dbody.fuse_views (MOTION front, MOTION side) -> MOTION
//   motion.clean_up, motion.save_glb, motion.save_bvh, motion.save_frames

import { llm } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { defineNodes } from '../lib/graph/registry.js';
import { writeText } from '../lib/graph/plugins.js';
import { cleanUpMotion } from '../kimodo/cleanup.js';
import { Sam3dBody } from './pose.js';
import { drawPreview } from './keypoints.js';
import { listFrames, poseFrames } from './frames.js';
import { assembleMotion } from './motion.js';
import { fuseViews } from './fuse.js';
import { clipGlb, clipBvh, framesJson } from './output.js';

// '' or 'x1,y1,x2,y2' to null or four numbers.
function parseBox(text) {
	if (!text || !String(text).trim()) return null;
	const v = String(text).split(',').map(Number);
	if (v.length !== 4 || v.some((x) => !Number.isFinite(x))) throw new Error(`bbox wants x1,y1,x2,y2, got ${text}`);
	return v;
}

function makeDirFor(path) {
	const slash = path.lastIndexOf('/');
	if (slash > 0) llm.makeDir(path.slice(0, slash));
}

defineNodes('sam3dbody', {
	'sam3dbody.load': {
		title: 'Load SAM 3D Body',
		category: 'motion',
		description: 'The sam-3d-body-dinov3 folder (model.ckpt, assets/mhr_model.pt). `linear` is the backbone\'s weights, `decoder` the decoders\'. `hands` poses the hands too, with the hand decoder: three backbone passes a frame instead of one.',
		inputs: { path: 'PATH(sam3dbody)', linear: 'ENUM(q8|f32)=q8', decoder: 'ENUM(f32|q8)=f32', hands: 'BOOL=false', matrix_cores: 'BOOL=true' },
		outputs: { model: 'MODEL' },
		async run(i) {
			op.useMatrixCores(i.matrix_cores);
			const model = new Sam3dBody(i.path, { linear: i.linear, decoder: i.decoder, hands: i.hands });
			await model.load();
			return { model };
		},
	},

	'sam3dbody.pose_image': {
		title: 'Pose image',
		category: 'motion',
		description: 'The person in `bbox` (x1,y1,x2,y2; the whole image when empty) as a posed MHR rig. `focal` in pixels, 0 for the image diagonal.',
		inputs: { model: 'MODEL', image: 'IMAGE', bbox: 'STRING=', focal: 'FLOAT=0' },
		outputs: { pose: 'POSE' },
		async run(i) {
			return { pose: await i.model.poseImage(i.image, { bbox: parseBox(i.bbox), focal: i.focal || null }) };
		},
	},

	'sam3dbody.draw_pose': {
		title: 'Draw pose',
		category: 'motion',
		description: 'The 70 keypoints and their links over the image.',
		inputs: { image: 'IMAGE', pose: 'POSE' },
		outputs: { image: 'IMAGE' },
		run(i) { return { image: drawPreview(i.image, i.pose) }; },
	},

	'sam3dbody.pose_frames': {
		title: 'Pose video frames',
		category: 'motion',
		description: 'A folder of PNG/JPEG frames in name order (ffmpeg -i clip.mp4 frames/%05d.png) to a motion on the MHR skeleton, y up, standing on y = 0. The person is followed from frame to frame; `bbox` is their box in the first frame. `count` 0 is to the end; `smooth` is the smoothing window in frames. With a model loaded with hands, `hand_stride` 2 or more runs the hand passes on every so many frames only, the fingers blended between them.',
		inputs: {
			model: 'MODEL', frames: 'PATH(dir)', fps: 'FLOAT=24', first: 'INT=0', count: 'INT=0', stride: 'INT=1',
			bbox: 'STRING=', focal: 'FLOAT=0', smooth: 'INT=7', hand_stride: 'INT=1', facing: 'ENUM(forward|camera)=forward',
		},
		outputs: { motion: 'MOTION' },
		async run(i, ctx) {
			const stride = Math.max(1, i.stride);
			let paths = listFrames(i.frames).slice(Math.max(0, i.first)).filter((_, k) => k % stride === 0);
			if (i.count > 0) paths = paths.slice(0, i.count);
			const posed = await poseFrames(i.model, paths, {
				bbox: parseBox(i.bbox), focal: i.focal || null, hands: i.model.hands, handStride: i.hand_stride,
				onFrame: (_, k, n) => ctx?.progress(k + 1, n),
			});
			const motion = assembleMotion(i.model.mhr, posed, { fps: i.fps / stride, smooth: i.smooth, facing: i.facing });
			motion.posed = posed;
			motion.smooth = i.smooth;
			motion.name = i.frames.replace(/\/+$/, '').split('/').pop() || 'capture';
			return { motion };
		},
	},

	'sam3dbody.fuse_views': {
		title: 'Fuse two cameras',
		category: 'motion',
		description: 'Two motions posed from the same moments, one filmed from the front and one from the side (frame N of each the same instant), as one: a lean to the side and the feet\'s spacing from the front camera, a lean forward and the arms\' swing from the side one. `facing` forward turns it to face +z.',
		inputs: { front: 'MOTION', side: 'MOTION', facing: 'ENUM(forward|camera)=forward' },
		outputs: { motion: 'MOTION' },
		run({ front, side, facing }) {
			if (!front.posed || !side.posed) throw new Error('both motions have to be posed from frames (sam3dbody.pose_frames)');
			const motion = fuseViews(front.skeleton.mhr, front.posed, side.posed, { fps: front.fps, smooth: front.smooth ?? 7, facing });
			motion.posed = front.posed.slice(0, motion.frames);
			motion.name = front.name;
			return { motion };
		},
	},

	'motion.clean_up': {
		title: 'Clean up motion',
		category: 'motion',
		description: 'Plants the feet where they touch the floor and smooths the hips (Kimodo\'s clean-up), from the motion\'s foot contacts.',
		inputs: { motion: 'MOTION' },
		outputs: { motion: 'MOTION' },
		run({ motion }) {
			if (!motion.contacts) throw new Error('this motion has no foot contacts to clean up from');
			const out = { ...motion, rotations: Float32Array.from(motion.rotations), root: Float32Array.from(motion.root) };
			const done = cleanUpMotion({ skeleton: out.skeleton, key: 'mhr', frames: out.frames, rotations: out.rotations, root: out.root, contacts: out.contacts });
			llm.print(`  clean-up: ${done.plantedFrames} planted foot frames`);
			return { motion: out };
		},
	},

	'motion.save_glb': {
		title: 'Save motion .glb',
		category: 'motion',
		description: 'The skeleton and its animation as a .glb, no mesh.',
		output: true,
		inputs: { motion: 'MOTION', path: 'STRING=output/motion.glb' },
		outputs: {},
		run({ motion, path }) {
			makeDirFor(path);
			llm.writeBytes(path, clipGlb(motion, motion.name));
			llm.print(`  saved ${path} (${motion.frames} frames)`);
			return { output: path, frames: motion.frames };
		},
	},

	'motion.save_bvh': {
		title: 'Save motion .bvh',
		category: 'motion',
		description: 'The motion as BVH, centimetres, Z-X-Y rotations.',
		output: true,
		inputs: { motion: 'MOTION', path: 'STRING=output/motion.bvh' },
		outputs: {},
		run({ motion, path }) {
			makeDirFor(path);
			writeText(path, clipBvh(motion));
			llm.print(`  saved ${path} (${motion.frames} frames)`);
			return { output: path, frames: motion.frames };
		},
	},

	'motion.save_frames': {
		title: 'Save frames JSON',
		category: 'motion',
		description: 'Each posed frame\'s box, camera, camera translation, 2D keypoints and foot contacts.',
		output: true,
		inputs: { motion: 'MOTION', path: 'STRING=output/frames.json' },
		outputs: {},
		run({ motion, path }) {
			if (!motion.posed) throw new Error('this motion was not posed from frames');
			makeDirFor(path);
			writeText(path, framesJson(motion, motion.posed));
			llm.print(`  saved ${path}`);
			return { output: path };
		},
	},
});
