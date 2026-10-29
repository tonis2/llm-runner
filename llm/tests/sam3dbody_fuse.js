// Two cameras put together (sam3dbody/fuse.js), with no network: only the rig
// file.
//
//   llm-runner tests/sam3dbody_fuse.js mhr=build/sam-3d-body-dinov3/assets/mhr_model.pt
//
// 1. A camera draws every bone in the direction it really has and guesses only
//    how far towards or away from it the bone reaches. A known walk, each
//    bone's reach along the front camera's line of sight (z) guessed wrong in
//    one view and along the side one's (x) in the other: put together, the
//    joints come out at a quarter of either view's error or nearer.
// 2. The same walk posed as SAM 3D Body would see it from the front and from a
//    camera a quarter turn round, both right: the clip is the front camera's
//    own (the floor under the feet levels both alike), the side camera is found
//    90 degrees round, and the clip faces +z. One camera's clip faces +z too
//    unless `facing` is 'camera'.
import { llm } from '/core/llm.js';
import { forward, between, rotate, mul, conjugate, unit, normalized } from '/core/motion/rotation.js';
import { MHR, MODEL_PARAMS } from '/plugins/sam3dbody/mhr.js';
import { MotionSkeleton } from '/plugins/sam3dbody/skeleton.js';
import { assembleMotion, headings, meanAngle } from '/plugins/sam3dbody/motion.js';
import { fuseViews, turnFuser, boneChildren } from '/plugins/sam3dbody/fuse.js';

const config = globalThis.__llm_config;
const mhr = new MHR(config.mhr ?? 'build/sam-3d-body-dinov3/assets/mhr_model.pt');
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
const DEG = 180 / Math.PI;

let seed = 11;
const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };

// Model parameters (the rig's table): thighs forward 52 and 61, knees 53 and
// 62, upper arms 35 and 45, the spine forward 10, 16, 22.
const N = 32;
function walk(f) {
	const p = new Float64Array(MODEL_PARAMS), s = Math.sin((2 * Math.PI * f) / 16);
	p[52] = 0.45 * s; p[61] = -0.45 * s;
	p[53] = 0.25 + 0.35 * Math.max(0, -s); p[62] = 0.25 + 0.35 * Math.max(0, s);
	p[35] = 0.35 * s; p[45] = -0.35 * s;
	p[10] = p[16] = p[22] = 0.03;
	return p;
}
const truth = Array.from({ length: N }, (_, f) => walk(f));
const skeleton = new MotionSkeleton(mhr);
const J = skeleton.jointCount;

// Every joint about the pelvis, for world turns.
function placed(turns) {
	const p = forward(skeleton, (j) => {
		const q = skeleton.parents[j];
		return q < 0 ? turns[j] : unit(mul(conjugate(turns[q]), turns[j]));
	}, [0, 0, 0]).positions;
	return p.map((v) => [v[0] - p[0][0], v[1] - p[0][1], v[2] - p[0][2]]);
}
const meanOff = (a, b) => a.reduce((s, v, j) => s + Math.hypot(v[0] - b[j][0], v[1] - b[j][1], v[2] - b[j][2]), 0) / a.length;

// 1. Wrong only along each camera's line of sight.
{
	const bones = boneChildren(skeleton), fuse = turnFuser(skeleton);
	const wrongAlong = (turns, axis) => turns.map((q, j) => {
		const c = bones[j];
		if (j === 0 || c < 0) return q;
		const b = normalized(rotate(q, skeleton.offsets[c]));
		const reach = b.slice();
		reach[axis] += random() * 0.8;
		return unit(mul(between(b, normalized(reach)), q));
	});
	let front = 0, side = 0, both = 0;
	for (const t of truth) {
		const { local } = skeleton.frame(t);
		const turns = forward(skeleton, (j) => Array.from(local.subarray(j * 4, j * 4 + 4)), [0, 0, 0]).turns;
		const want = placed(turns);
		const a = wrongAlong(turns, 2), b = wrongAlong(turns, 0);
		front += meanOff(placed(a), want) / N;
		side += meanOff(placed(b), want) / N;
		both += meanOff(placed(fuse(a, b)), want) / N;
	}
	llm.print(`  joints off the walk: front camera ${(front * 1000).toFixed(1)} mm, side ${(side * 1000).toFixed(1)} mm, together ${(both * 1000).toFixed(1)} mm`);
	check(both <= 0.25 * Math.min(front, side), 'two cameras take each bone from where both draw it');
}

// 2. Posed frames from two cameras.
{
	const view = (yaw) => truth.map((t) => {
		const p = Float64Array.from(t);
		p[4] = yaw;
		return { params: p, cam_t: [0, 0.3, 4] };
	});
	const options = { fps: 24, smooth: 1 };
	const fused = fuseViews(mhr, view(0), view(Math.PI / 2), options);
	const front = assembleMotion(mhr, view(0), options);
	const turnsAt = (clip, f) => placed(forward(clip.skeleton, (j) => Array.from(clip.rotations.subarray((f * J + j) * 4, (f * J + j + 1) * 4)), [0, 0, 0]).turns);
	let off = 0;
	for (let f = 0; f < N; f++) off += meanOff(turnsAt(fused, f), turnsAt(front, f)) / N;
	const facing = (clip) => meanAngle(headings(clip.skeleton, clip.rotations, clip.frames)) * DEG;
	check(off < 0.002, `two right cameras give the front camera's walk back (${(off * 1000).toFixed(2)} mm off)`);
	check(Math.abs(fused.views.cameraTurn * DEG - 90) < 1, `the side camera found ${(fused.views.cameraTurn * DEG).toFixed(1)} degrees round`);
	check(Math.abs(facing(fused)) < 1, `the clip faces +z (${facing(fused).toFixed(2)} degrees)`);
	const one = assembleMotion(mhr, view(Math.PI / 2), options);
	check(Math.abs(facing(one)) < 1, `one camera's clip faces +z (${facing(one).toFixed(2)} degrees)`);
	const kept = assembleMotion(mhr, view(Math.PI / 2), { ...options, facing: 'camera' });
	check(Math.abs(facing(kept) - 90) < 1, `facing 'camera' keeps the side view's facing (${facing(kept).toFixed(1)} degrees)`);
}

mhr.close();
llm.print(failures ? `${failures} failed` : 'all passed');
