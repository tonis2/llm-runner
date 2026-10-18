// The MHR motion skeleton and clip writers against MHR's own forward
// kinematics, with no network: only the rig file.
//
//   llm-runner tests/sam3dbody_motion.js mhr=build/sam-3d-body-dinov3/assets/mhr_model.pt
//
// 1. The rest skeleton (identity turns) stands where MHR's zero pose does.
// 2. A frame's parent-local turns, run down the skeleton, put every joint where
//    MHR's own FK puts it, for random poses and proportions.
// 3. BVH's Z-X-Y Euler angles turn back into the same rotations.
// 4. A clip of a standing pose stands on y = 0 with both feet down.
import { llm } from '../lib/llm.js';
import { forward, mul, axisAngle } from '../kimodo/rotation.js';
import { MHR, JOINTS, MODEL_PARAMS } from '../sam3dbody/mhr.js';
import { MotionSkeleton, PROPORTIONS } from '../sam3dbody/skeleton.js';
import { assembleMotion } from '../sam3dbody/motion.js';
import { cleanUpMotion } from '../kimodo/cleanup.js';
import { motionBvh } from '../lib/bvh.js';

const config = globalThis.__llm_config;
const mhr = new MHR(config.mhr ?? 'build/sam-3d-body-dinov3/assets/mhr_model.pt');
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

let seed = 7;
const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };

// MHR's own joint positions (metres, its frame), joint 0 left out.
function mhrJoints(params) {
	const world = mhr.skeleton(mhr.jointParameters(params));
	return world.slice(1).map((g) => g.t.map((v) => v / 100));
}

function worst(a, b) {
	let w = 0;
	for (let j = 0; j < a.length; j++) for (let d = 0; d < 3; d++) w = Math.max(w, Math.abs(a[j][d] - b[j][d]));
	return w;
}

// 1. Rest.
{
	const skeleton = new MotionSkeleton(mhr);
	check(skeleton.names[0] === 'root' && skeleton.jointCount === JOINTS - 1, `126 joints from '${skeleton.names[0]}'`);
	const rest = forward(skeleton, () => [0, 0, 0, 1], skeleton.restRoot).positions;
	const e = worst(rest, mhrJoints(new Float64Array(MODEL_PARAMS)));
	check(e < 1e-9, `rest pose matches MHR's zero pose (worst ${e.toExponential(2)} m)`);
	check(skeleton.aboveGround > 0 && skeleton.aboveGround < 0.05, `lowest rest joint ${skeleton.aboveGround.toFixed(4)} m over the soles`);
}

// 2. Posed frames.
{
	let e = 0;
	for (let trial = 0; trial < 5; trial++) {
		const params = new Float64Array(MODEL_PARAMS);
		for (let i = 3; i < PROPORTIONS; i++) params[i] = random() * 1.2;
		const proportions = new Float64Array(MODEL_PARAMS - PROPORTIONS);
		for (let i = 0; i < proportions.length; i++) proportions[i] = random() * 0.6;
		const skeleton = new MotionSkeleton(mhr, proportions);
		const { local, root } = skeleton.frame(params);
		const got = forward(skeleton, (j) => Array.from(local.subarray(j * 4, j * 4 + 4)), root).positions;
		e = Math.max(e, worst(got, mhrJoints(skeleton.params(params))));
	}
	check(e < 1e-9, `posed joints match MHR's FK (worst ${e.toExponential(2)} m)`);
}

// 3. BVH angles back to rotations.
{
	const skeleton = { names: ['a', 'b'], parents: [-1, 0], offsets: [[0, 0, 0], [0, 1, 0]] };
	const turns = [];
	for (let i = 0; i < 8; i++) {
		const axis = [random(), random(), random()];
		const n = Math.hypot(...axis);
		turns.push(axisAngle(axis.map((v) => v / n), random() * 5));
	}
	const rotations = Float32Array.from(turns.flat());
	const text = motionBvh({ skeleton, fps: 30, frames: 4, rotations, root: new Float32Array(12) });
	const rowsOut = text.trim().split('\n').slice(-4).map((l) => l.split(' ').map(Number));
	const R = (axis, deg) => axisAngle(axis, (deg * Math.PI) / 180);
	let e = 0;
	rowsOut.forEach((row, f) => {
		for (let j = 0; j < 2; j++) {
			const [z, x, y] = row.slice(3 + j * 3, 6 + j * 3);
			const q = mul(mul(R([0, 0, 1], z), R([1, 0, 0], x)), R([0, 1, 0], y));
			const want = turns[f * 2 + j];
			const dot = Math.abs(q[0] * want[0] + q[1] * want[1] + q[2] * want[2] + q[3] * want[3]);
			e = Math.max(e, 1 - dot);
		}
	});
	check(e < 1e-9, `BVH Z-X-Y angles give the rotations back (worst 1 - |dot| ${e.toExponential(2)})`);
	check(text.includes('ROOT a') && text.includes('JOINT b') && text.includes('End Site'), 'BVH hierarchy');
}

// 4. A standing clip, the camera 3 m away; Kimodo's clean-up plants its feet
//    where they stand.
{
	// MHR's zero pose, which SAM 3D Body's camera frame (y and z flipped) sees
	// upright and facing it.
	const params = new Float64Array(MODEL_PARAMS);
	const frames = Array.from({ length: 12 }, () => ({ params, cam_t: [0.1, 0.2, 3] }));
	const clip = assembleMotion(mhr, frames, { fps: 24, smooth: 5 });
	const J = clip.skeleton.jointCount;
	const posAt = (f) => forward(clip.skeleton, (j) => Array.from(clip.rotations.subarray((f * J + j) * 4, (f * J + j + 1) * 4)), clip.root.subarray(f * 3, f * 3 + 3)).positions;
	const p = posAt(5);
	const low = Math.min(...p.map((v) => v[1]));
	check(Math.abs(low - clip.skeleton.aboveGround) < 1e-6, `stands on y = 0 (lowest joint ${low.toFixed(4)} m)`);
	const head = p[clip.skeleton.index('c_head')];
	check(head[1] > 1.4, `head up (${head[1].toFixed(3)} m)`);
	check(Math.abs(clip.root[0]) < 1e-9 && Math.abs(clip.root[2]) < 1e-9, 'pelvis starts over the origin');
	check(Math.abs(clip.camera[2] - 3) < 0.05, `camera ~3 m in front (${clip.camera.map((v) => v.toFixed(3)).join(' ')})`);
	check(Array.from(clip.contacts).every((c) => c === 1), 'both feet down every frame');
	const before = posAt(5);
	const done = cleanUpMotion({ skeleton: clip.skeleton, key: 'mhr', frames: clip.frames, rotations: clip.rotations, root: clip.root, contacts: clip.contacts });
	let moved = 0;
	posAt(5).forEach((v, j) => { for (let d = 0; d < 3; d++) moved = Math.max(moved, Math.abs(v[d] - before[j][d])); });
	check(done.plantedFrames > 0 && moved < 0.01, `clean-up planted ${done.plantedFrames} foot frames, moved joints at most ${(moved * 1000).toFixed(2)} mm`);
}

mhr.close();
llm.print(failures ? `${failures} FAILED` : 'all passed');
