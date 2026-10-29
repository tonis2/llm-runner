// One motion from two cameras: the same moments filmed from the front and from
// the side, frame N of each the same instant, posed one by one (frames.js) and
// put together into one clip on the MHR skeleton.
//
// Each camera sees across its picture and guesses along its line of sight. The
// front one sees the body's left and right, a lean to the side, how far apart
// the feet are, and guesses how far forward an arm swings; the side one sees
// forward and back, a lean forward, the arms' swing, and guesses the rest. So
// every bone takes its direction from both: across the body (x) from the front,
// forward and back (z) from the side, up and down from both. A turn about the
// bone itself, which neither sees well, is the two views' halfway turn.
//
// Bones are put together where they point in the world, not joint to joint, so
// one view's error at the hips is not carried down into the feet. The side view
// is first turned about +y onto the front one's facing (its camera stands about
// a quarter turn round), each view stood level on its own floor.

import { forward, between, rotate, mul, conjugate, unit, slerp, normalized, fromAxes, cross, dot, add, scaled, axisAngle } from '../lib/motion/rotation.js';
import { MotionSkeleton, clipProportions } from './skeleton.js';
import { poseRows, smoothMotion, levelFloor, footPositions, finishMotion, faceForward, headings, meanAngle } from './motion.js';

// The two cameras' planes through a bone are trusted to cross where they meet
// at this sine of an angle or more; below it the bone's parts are taken instead.
const CROSSING = 0.35;

// Each joint's bone: the child that carries it on, or -1 for none (a leaf, or
// children all where the joint is, as a twist joint's). A joint on the body's
// midline (c_) carries on to the midline (the spine's top to the neck, not a
// collarbone); otherwise the child with the most joints below it, the longer
// bone when that ties.
export function boneChildren(skeleton) {
	const J = skeleton.jointCount, children = Array.from({ length: J }, () => []);
	const reach = (c) => Math.hypot(...skeleton.offsets[c]);
	for (let j = 1; j < J; j++) if (reach(j) > 1e-6) children[skeleton.parents[j]].push(j);
	const below = new Int32Array(J);
	for (let j = J - 1; j > 0; j--) below[skeleton.parents[j]] += below[j] + 1;
	return children.map((cs, j) => {
		if (cs.length === 0) return -1;
		const middle = skeleton.names[j].startsWith('c_') ? cs.filter((c) => skeleton.names[c].startsWith('c_')) : [];
		const pool = middle.length ? middle : cs;
		return pool.reduce((best, c) => (below[c] > below[best] || (below[c] === below[best] && reach(c) > reach(best)) ? c : best));
	});
}

// Two views' directions of one bone as one. Each camera has the bone's
// direction in its picture right however wrong it is along its line of sight
// (a knee bent towards it draws shorter, not turned), so the bone lies in the
// plane through that line of sight and the bone as drawn: the front camera's
// through z, the side one's through x. It is where the two planes cross.
// Where they are nearly one plane (a bone lying level, drawn level by both),
// it takes across from the front, forward and back from the side.
function fusedDirection(front, side) {
	const level = [front[0], (front[1] + side[1]) / 2, side[2]];
	const byParts = Math.hypot(...level) > 1e-9 ? normalized(level) : normalized(front);
	const planeFront = [front[1], -front[0], 0], planeSide = [0, side[2], -side[1]];
	const crossing = cross(planeFront, planeSide);
	const sine = Math.hypot(...crossing) / Math.max(1e-12, Math.hypot(...planeFront) * Math.hypot(...planeSide));
	const trust = Math.min(1, sine / CROSSING);
	if (trust <= 0) return byParts;
	const line = dot(crossing, byParts) < 0 ? normalized(scaled(crossing, -1)) : normalized(crossing);
	return normalized(add(scaled(line, trust), scaled(byParts, 1 - trust)));
}

// The orthonormal frame [x, y, z] with y along `up` and x as near `across` as
// that allows.
function frameOf(across, up) {
	const y = normalized(up);
	const x = normalized([across[0] - y[0] * dot(across, y), across[1] - y[1] * dot(across, y), across[2] - y[2] * dot(across, y)]);
	return fromAxes(x, y, cross(x, y));
}

/**
 * Two views' world turns of one frame (forward()'s `turns`, both facing the
 * same way) put together, joint by joint: a function of (front, side) on
 * `skeleton` answering world turns.
 */
export function turnFuser(skeleton) {
	const bones = boneChildren(skeleton);
	const off = skeleton.offsets;
	const left = skeleton.index('l_upleg'), right = skeleton.index('r_upleg');
	const across = [off[left][0] - off[right][0], off[left][1] - off[right][1], off[left][2] - off[right][2]];
	const restRoot = frameOf(across, off[bones[0]]);
	return (a, b) => {
		const turns = new Array(skeleton.jointCount);
		for (let j = 0; j < turns.length; j++) {
			const halfway = unit(slerp(a[j], b[j], 0.5));
			const c = bones[j];
			if (j === 0) {
				// The pelvis: its frame from the spine's direction and the line
				// across the hips, each put together.
				const up = fusedDirection(normalized(rotate(a[0], off[c])), normalized(rotate(b[0], off[c])));
				const hips = fusedDirection(normalized(rotate(a[0], across)), normalized(rotate(b[0], across)));
				turns[0] = unit(mul(frameOf(hips, up), conjugate(restRoot)));
			} else if (c < 0) {
				turns[j] = halfway;
			} else {
				const want = fusedDirection(normalized(rotate(a[j], off[c])), normalized(rotate(b[j], off[c])));
				turns[j] = unit(mul(between(normalized(rotate(halfway, off[c])), want), halfway));
			}
		}
		return turns;
	};
}

/**
 * The clip from two cameras' posed frames, `front` and `side`, as frames.js
 * posed them ([{ params, cam_t }], the same instants). Options as
 * assembleMotion's. The answer is assembleMotion's, with `views`: the side
 * camera's turn from the front one (radians about +y) and each view's floor
 * turn.
 */
export function fuseViews(mhr, front, side, { fps = 24, smooth = 7, facing = 'forward' } = {}) {
	const N = Math.min(front.length, side.length);
	if (N === 0) throw new Error('no frames to make a motion of');
	front = front.slice(0, N);
	side = side.slice(0, N);
	const skeleton = new MotionSkeleton(mhr, clipProportions([...front, ...side].map((f) => f.params)));
	const J = skeleton.jointCount;

	const views = [front, side].map((frames) => {
		const { rotations, root } = poseRows(skeleton, frames);
		smoothMotion(rotations, root, N, J, smooth);
		return { rotations, root, level: levelFloor(skeleton, rotations, root, N).level };
	});

	// The side view turned onto the front one's facing, about the camera.
	const facingFront = headings(skeleton, views[0].rotations, N), facingSide = headings(skeleton, views[1].rotations, N);
	const cameraTurn = meanAngle(facingSide.map((s, f) => s - facingFront[f]));
	const onto = axisAngle([0, 1, 0], -cameraTurn);
	{
		const { rotations, root } = views[1];
		for (let f = 0; f < N; f++) {
			const o = f * J * 4;
			rotations.set(mul(onto, Array.from(rotations.subarray(o, o + 4))), o);
			root.set(rotate(onto, [root[f * 3], root[f * 3 + 1], root[f * 3 + 2]]), f * 3);
		}
	}

	const fuse = turnFuser(skeleton);
	const rotations = new Float64Array(N * J * 4);
	const root = new Float64Array(N * 3);
	const local = (v, f) => (j) => Array.from(views[v].rotations.subarray((f * J + j) * 4, (f * J + j + 1) * 4));
	for (let f = 0; f < N; f++) {
		const turns = fuse(forward(skeleton, local(0, f), [0, 0, 0]).turns, forward(skeleton, local(1, f), [0, 0, 0]).turns);
		for (let j = 0; j < J; j++) {
			const p = skeleton.parents[j];
			rotations.set(p < 0 ? turns[j] : unit(mul(conjugate(turns[p]), turns[j])), (f * J + j) * 4);
		}
		// The pelvis' path from the first frame: across from the front, forward
		// and back from the side, up and down from both.
		const rf = views[0].root, rs = views[1].root;
		root[f * 3] = rf[f * 3];
		root[f * 3 + 1] = rf[1] + (rf[f * 3 + 1] - rf[1] + rs[f * 3 + 1] - rs[1]) / 2;
		root[f * 3 + 2] = rf[2] + (rs[f * 3 + 2] - rs[2]);
	}
	// Kept on one side of the 4D sphere frame to frame, as assembleMotion's.
	smoothMotion(rotations, new Float64Array(0), N, J, 1);

	const stood = footPositions(skeleton, rotations, root, N);
	stood.level = views[0].level;
	const clip = finishMotion(skeleton, rotations, root, fps, stood);
	if (facing !== 'camera') faceForward(clip);
	clip.views = { cameraTurn, levels: views.map((v) => v.level) };
	return clip;
}
