// The clean-up upstream runs on a generated motion before handing it out
// (kimodo/postprocess.py `post_process_motion`): feet that the model says are
// planted stop sliding, and the frames a guide pose is keyed on are snapped
// exactly onto it, with the frames around them bent to meet it without a pop.
//
// A port of NVIDIA's motion_correction package (MotionCorrection/src/cpp/
// AnimProcessing: Utility.cpp CorrectMotion, TrajectoryCorrector.cpp,
// InverseKinematics.cpp; Apache-2.0), in the same order and with the same
// weights:
//
//   1. the hips' height, pinned on whole-body keys;
//   2. the hips' floor position, pinned on whole-body keys and kept within
//      4 cm of it on hand and foot keys;
//   3. every joint's turn, pinned on whole-body keys;
//   4. hands and feet keyed on their own, reached by two-bone IK;
//   5. planted feet: each stretch of contact held on one spot (two-bone IK on
//      the leg), then the toes laid on the floor (one-bone IK on the foot).
//
// Every correction is a trajectory warp: the corrected value is pinned on the
// frames it applies to, and the frames between are refitted to keep the
// original curve's shape - its velocity and acceleration - rather than jumping.
// A joint's turn is warped as its forward and up axes, which do not flip the
// way quaternion components do.
//
// Upstream computes a per-frame weight from the hips' speed and passes it to the
// velocity term, but the weighting (`multVelWeights`) builds its result and
// drops it, so it never applies; it is left out here.

import {
	IDENTITY, sub, add, scaled, length, dot, cross, normalized, mul, conjugate, unit, rotate, axisAngle,
	between, slerp, lookRotation, forward,
} from './rotation.js';

const POS_WEIGHT = 0.001;
const VEL_WEIGHT = 1;
const ACC_WEIGHT = 10;
const ADMM_ITERATIONS = 100;
const CONTACT_THRESHOLD = 0.5;
const ROOT_MARGIN = 0.04;
// How far above the rest pose's lowest point the feet are held; upstream lifts
// SOMA's higher because the model sets it lower to the floor.
const ABOVE_GROUND = { soma30: 0.02 };
const DEFAULT_ABOVE_GROUND = 0.007;
// Two-bone IK bends the middle joint toward this point in its own frame: elbows
// back, knees forward.
const HAND_HINT = [0, 0, -0.1];
const FOOT_HINT = [0, 0, 0.1];

// ---------------------------------------------------------------------------
// Trajectory warping.

// One trajectory warp over N frames: `margins[i]` < 0 leaves frame i free, 0
// pins it to the observation, > 0 keeps it within that distance of it.
class Corrector {
	constructor(margins, posWeight, velWeight, accWeight) {
		const N = margins.length;
		this.N = N;
		this.marginAt = [];
		this.marginSize = [];
		this.pinned = new Uint8Array(N);
		this.free = [];
		for (let i = 0; i < N; i++) {
			if (margins[i] > 0) {
				this.marginAt.push(i);
				this.marginSize.push(margins[i]);
			}
			if (margins[i] === 0) this.pinned[i] = 1;
			else this.free.push(i);
		}
		this.anyPinned = this.free.length < N;

		// The energy's matrix pos I + vel VᵀV + acc AᵀA, which is banded: V takes
		// neighbouring frames' differences and A second differences, so nothing
		// reaches past two frames. band[3i + d] is entry (i, i + d).
		const band = new Float64Array(N * 3);
		for (let i = 0; i < N; i++) band[3 * i] += posWeight;
		for (let i = 0; i + 1 < N; i++) {
			band[3 * i] += velWeight;
			band[3 * (i + 1)] += velWeight;
			band[3 * i + 1] -= velWeight;
		}
		const second = [-1, 2, -1];
		for (let i = 0; i + 2 < N; i++) {
			for (let a = 0; a < 3; a++) {
				for (let b = a; b < 3; b++) band[3 * (i + a) + (b - a)] += accWeight * second[a] * second[b];
			}
		}
		this.band = band;
		let diagonal = 0;
		for (let i = 0; i < N; i++) diagonal = Math.max(diagonal, band[3 * i]);
		this.step = 0.5 * Math.sqrt(diagonal);

		// The system over the free frames, the margin frames' diagonal raised by
		// the ADMM step, Cholesky-factored. Dropping the pinned frames keeps it
		// banded: two free frames next to each other in the list are at least as
		// far apart as before.
		const n = this.free.length;
		const withMargins = band.slice();
		for (const i of this.marginAt) withMargins[3 * i] += this.step;
		const entry = (r, d) => {
			const i = this.free[r], k = this.free[r + d];
			return k - i <= 2 ? withMargins[3 * i + (k - i)] : 0;
		};
		const L = new Float64Array(n * 3); // L[3r + d] is (r, r - d)
		for (let r = 0; r < n; r++) {
			for (let d = 2; d >= 1; d--) {
				const k = r - d;
				if (k < 0) continue;
				let s = entry(k, d);
				if (d === 1 && k - 1 >= 0 && r - 2 >= 0) s -= L[3 * r + 2] * L[3 * k + 1];
				L[3 * r + d] = s / L[3 * k];
			}
			let s = entry(r, 0);
			for (let d = 1; d <= 2; d++) if (r - d >= 0) s -= L[3 * r + d] * L[3 * r + d];
			L[3 * r] = Math.sqrt(Math.max(s, 1e-300));
		}
		this.L = L;
	}

	solve(rhs) {
		const n = this.free.length, L = this.L;
		const y = new Float64Array(n);
		for (let r = 0; r < n; r++) {
			let s = rhs[r];
			for (let d = 1; d <= 2; d++) if (r - d >= 0) s -= L[3 * r + d] * y[r - d];
			y[r] = s / L[3 * r];
		}
		const x = new Float64Array(n);
		for (let r = n - 1; r >= 0; r--) {
			let s = y[r];
			for (let d = 1; d <= 2; d++) if (r + d < n) s -= L[3 * (r + d) + d] * x[r + d];
			x[r] = s / L[3 * r];
		}
		return x;
	}

	// The banded energy matrix times `v`.
	times(v) {
		const N = this.N, band = this.band, out = new Float64Array(N);
		for (let i = 0; i < N; i++) {
			let s = band[3 * i] * v[i];
			for (let d = 1; d <= 2; d++) {
				if (i + d < N) s += band[3 * i + d] * v[i + d];
				if (i - d >= 0) s += band[3 * (i - d) + d] * v[i - d];
			}
			out[i] = s;
		}
		return out;
	}

	// One ADMM x-step, into `x`, over columns of equal length.
	update(x, z, u, reference, observed) {
		const cols = x.length;
		for (let c = 0; c < cols; c++) {
			const diffs = Float64Array.from(reference[c]);
			for (let i = 0; i < this.N; i++) if (this.pinned[i]) diffs[i] -= observed[c][i];
			const r = this.times(diffs);
			for (let k = 0; k < this.marginAt.length; k++) r[this.marginAt[k]] += this.step * (z[k * cols + c] - u[k * cols + c]);
			const reduced = new Float64Array(this.free.length);
			this.free.forEach((i, row) => { reduced[row] = r[i]; });
			const solved = this.free.length ? this.solve(reduced) : reduced;
			this.free.forEach((i, row) => { x[c][i] = solved[row]; });
			for (let i = 0; i < this.N; i++) if (this.pinned[i]) x[c][i] = observed[c][i];
		}
	}

	// `reference` (columns of N values) warped onto `observed` where the margins
	// say so.
	interpolate(observed, reference) {
		const cols = reference.length;
		const x = reference.map((column) => Float64Array.from(column));
		if (!this.anyPinned && this.marginAt.length === 0) return x;
		if (this.marginAt.length === 0) {
			this.update(x, null, null, reference, observed);
			return x;
		}
		const m = this.marginAt.length;
		const z = new Float64Array(m * cols), target = new Float64Array(m * cols), u = new Float64Array(m * cols);
		for (let k = 0; k < m; k++) {
			for (let c = 0; c < cols; c++) {
				target[k * cols + c] = observed[c][this.marginAt[k]];
				z[k * cols + c] = reference[c][this.marginAt[k]];
			}
		}
		for (let iteration = 0; iteration < ADMM_ITERATIONS; iteration++) {
			this.update(x, z, u, reference, observed);
			// z: the frame pulled back inside its margin around the target.
			for (let k = 0; k < m; k++) {
				const at = this.marginAt[k];
				let norm = 0;
				for (let c = 0; c < cols; c++) {
					const d = x[c][at] + u[k * cols + c] - target[k * cols + c];
					z[k * cols + c] = d;
					norm += d * d;
				}
				norm = Math.sqrt(norm);
				const shrink = norm > this.marginSize[k] ? this.marginSize[k] / norm : 1;
				for (let c = 0; c < cols; c++) z[k * cols + c] = target[k * cols + c] + z[k * cols + c] * shrink;
			}
			for (let k = 0; k < m; k++) {
				for (let c = 0; c < cols; c++) u[k * cols + c] += x[c][this.marginAt[k]] - z[k * cols + c];
			}
		}
		return x;
	}
}

// Where a constrained stretch meets a free frame, the constrained end frame is
// replaced by its neighbours' average, so the warp does not start from a kink.
function smoothChannels(column, mask) {
	const n = mask.length;
	for (let i = 0; i < n; i++) {
		const prev = i === 0 ? 0 : i - 1, next = Math.min(i + 1, n - 1);
		if (i > 0 && mask[i] > 0 && mask[prev] === 0) column[i] = 0.5 * (column[prev] + column[next]);
		if (mask[i] > 0 && mask[next] === 0) column[i] = 0.5 * (column[prev] + column[next]);
	}
}

const marginsOf = (mask) => Array.from(mask, (m) => (m ? 0 : -1));

// ---------------------------------------------------------------------------
// Poses: parent-local turns [frames, J, 4] and the root's position [frames, 3].

class Motion {
	constructor(skeleton, rotations, root) {
		this.skeleton = skeleton;
		this.J = skeleton.parents.length;
		this.rotations = rotations;
		this.root = root;
	}

	copy() {
		return new Motion(this.skeleton, Float64Array.from(this.rotations), Float64Array.from(this.root));
	}

	local(f, j) {
		const o = (f * this.J + j) * 4, r = this.rotations;
		return [r[o], r[o + 1], r[o + 2], r[o + 3]];
	}

	setLocal(f, j, q) {
		const o = (f * this.J + j) * 4;
		for (let k = 0; k < 4; k++) this.rotations[o + k] = q[k];
	}

	pose(f) {
		const o = f * 3;
		return forward(this.skeleton, (j) => this.local(f, j), [this.root[o], this.root[o + 1], this.root[o + 2]]);
	}
}

// Warps one joint's turn in `motion` toward `targets` on the frames `mask`
// marks, as its forward and up axes.
function correctBone(motion, targets, mask, corrector, bone, smooth) {
	const N = mask.length;
	const axes = [];
	const observed = [];
	for (let d = 0; d < 6; d++) {
		axes.push(new Float64Array(N));
		observed.push(new Float64Array(N));
	}
	for (let f = 0; f < N; f++) {
		const q = motion.local(f, bone), t = targets.local(f, bone);
		const fwd = rotate(q, [0, 0, 1]), up = rotate(q, [0, 1, 0]);
		const tf = rotate(t, [0, 0, 1]), tu = rotate(t, [0, 1, 0]);
		for (let d = 0; d < 3; d++) {
			axes[d][f] = fwd[d];
			axes[3 + d][f] = up[d];
			observed[d][f] = mask[f] * tf[d];
			observed[3 + d][f] = mask[f] * tu[d];
		}
	}
	for (let d = 0; d < 6; d++) {
		if (smooth) smoothChannels(axes[d], mask);
		axes[d] = corrector.interpolate([observed[d]], [axes[d]])[0];
	}
	for (let f = 0; f < N; f++) {
		const fwd = normalized([axes[0][f], axes[1][f], axes[2][f]]);
		const up = normalized([axes[3][f], axes[4][f], axes[5][f]]);
		motion.setLocal(f, bone, lookRotation(fwd, up));
	}
}

// Every joint whose mask is in `masks` warped toward `targets` on its frames.
function correctBones(motion, targets, masks) {
	for (const bone of [...masks.keys()].sort((a, b) => a - b)) {
		const mask = masks.get(bone);
		correctBone(motion, targets, mask, new Corrector(marginsOf(mask), POS_WEIGHT * 10, VEL_WEIGHT, ACC_WEIGHT), bone, false);
	}
}

// ---------------------------------------------------------------------------
// IK, on frame `f` of `motion`.

function angleBetween(a, b) {
	const l = normalized(a), r = normalized(b);
	return Math.atan2(length(cross(l, r)), dot(l, r));
}

function cosineRule(left, right, across) {
	const v = (right * right + left * left - across * across) / (2 * left * right);
	return Math.acos(Math.min(1, Math.max(-1, v)));
}

// Bends joint c's parent and grandparent so c reaches `target`, the middle
// joint bending toward `hint` in its own frame.
function twoBoneIk(motion, f, c, target, hint) {
	const parents = motion.skeleton.parents;
	const b = parents[c];
	if (b < 0) return;
	const a = parents[b];
	if (a < 0) return;
	const { turns, positions } = motion.pose(f);
	const above = parents[a] < 0 ? IDENTITY : turns[parents[a]];
	const pa = positions[a], pb = positions[b], pc = positions[c];

	const eps = 0.0001;
	const ab = length(sub(pb, pa)), bc = length(sub(pc, pb));
	const at = Math.min(Math.max(length(sub(pa, target)), eps), (ab + bc) * 0.999);
	const bacNow = angleBetween(sub(pa, pb), sub(pa, pc));
	const abcNow = angleBetween(sub(pb, pa), sub(pb, pc));
	if (ab < eps || bc < eps || at < eps) return;
	const bacWanted = cosineRule(ab, at, bc);
	const abcWanted = cosineRule(ab, bc, at);

	let axis = cross(sub(pc, pa), sub(add(pb, rotate(turns[b], hint)), pa));
	const l = length(axis);
	axis = l === 0 ? [0, 0, 1] : scaled(axis, 1 / l);

	motion.setLocal(f, b, mul(motion.local(f, b), axisAngle(rotate(conjugate(turns[b]), axis), abcWanted - abcNow)));
	motion.setLocal(f, a, mul(motion.local(f, a), axisAngle(rotate(conjugate(turns[a]), axis), bacWanted - bacNow)));

	// The first stage kept a→c's direction; now swing it onto the target.
	const aTurn = mul(above, motion.local(f, a));
	const toC = rotate(conjugate(aTurn), sub(pc, pa));
	const toTarget = rotate(conjugate(aTurn), sub(target, pa));
	motion.setLocal(f, a, unit(mul(motion.local(f, a), between(toC, toTarget))));
}

// Turns joint b's parent so b lands on `target`.
function oneBoneIk(motion, f, b, target) {
	const a = motion.skeleton.parents[b];
	if (a < 0) return;
	const { turns, positions } = motion.pose(f);
	const inverse = conjugate(turns[a]);
	const toB = rotate(inverse, sub(positions[b], positions[a]));
	const toTarget = rotate(inverse, sub(target, positions[a]));
	motion.setLocal(f, a, unit(mul(motion.local(f, a), between(toB, toTarget))));
}

// 1 while `target` is inside the limb's reach from its base, falling to 0 just
// past it, so a leg is not yanked straight at a target it cannot reach.
function reachFalloff(motion, j, twoBone, target, f) {
	const { parents, offsets } = motion.skeleton;
	let reach = length(offsets[j]);
	if (twoBone) {
		j = parents[j];
		reach += length(offsets[j]);
	}
	j = parents[j];
	const distance = length(sub(target, motion.pose(f).positions[j]));
	let t = Math.max(distance / reach - 0.99, 0) / 0.01;
	t = t * t;
	return Math.exp(-2 * t * t);
}

// ---------------------------------------------------------------------------
// Contacts.

function contactIntervals(contacts, mask, threshold) {
	const out = [];
	let start = -1;
	for (let f = 0; f < mask.length; f++) {
		const touching = !mask[f] && contacts[f] > threshold;
		if (touching && start === -1) start = f;
		else if (!touching && start !== -1) {
			out.push([start, f]);
			start = -1;
		}
	}
	if (start !== -1) out.push([start, mask.length]);
	return out;
}

const startsKeyed = ([first], mask) => first !== 0 && mask[first - 1] > 0;
const endsKeyed = ([, end], mask) => end !== mask.length && mask[end] > 0;

// A contact between two keyed frames cannot be held without popping at one end,
// so it is dropped; a toe contact touching a keyed frame at all is too.
function filterIntervals(intervals, mask, oneBone) {
	return intervals.filter((interval) => {
		const s = startsKeyed(interval, mask), e = endsKeyed(interval, mask);
		return oneBone ? !(s || e) : !(s && e);
	});
}

// The spot each contact frame holds joint j on: where it is on the keyed frame
// the stretch touches, or at the stretch's middle, at the frame's own height
// but no lower than `floor`.
function contactPoints(motion, j, intervals, mask, floor) {
	const N = mask.length;
	const points = new Array(N).fill(null);
	for (const interval of intervals) {
		const [first, end] = interval;
		const s = startsKeyed(interval, mask), e = endsKeyed(interval, mask);
		const frame = s ? first - 1 : e ? end : Math.floor((first + end) / 2);
		const target = motion.pose(frame).positions[j];
		for (let f = first; f < end; f++) {
			const point = target.slice();
			if (!s && !e) point[1] = Math.max(motion.pose(f).positions[j][1], floor);
			points[f] = point;
		}
	}
	return points;
}

// ---------------------------------------------------------------------------

// The joints the clean-up works on: each hand and foot as the skeleton names
// them (the first joint of each effector chain), and each foot's toe.
function effectorJoints(skeleton) {
	const index = new Map(skeleton.names.map((n, j) => [n, j]));
	const at = (name) => index.get(skeleton.effectors?.[name]?.[0]) ?? -1;
	const out = { LeftHand: at('LeftHand'), RightHand: at('RightHand'), LeftFoot: at('LeftFoot'), RightFoot: at('RightFoot') };
	if (Object.values(out).some((j) => j < 0)) return null;
	const toe = (foot) => {
		let found = -1;
		skeleton.parents.forEach((p, j) => { if (p === foot) found = j; });
		return found;
	};
	out.LeftToe = toe(out.LeftFoot);
	out.RightToe = toe(out.RightFoot);
	return out;
}

// Whether `skeleton` can be cleaned up at all. Upstream turns it off for the
// G1 robot, whose double ankle the model does not keep well.
export function canCleanUp(skeleton, key) {
	return key !== 'g1skel34' && effectorJoints(skeleton) !== null;
}

/**
 * Cleans up a decoded motion in place.
 *
 *   rotations  Float32Array [frames, J, 4], parent-local xyzw
 *   root       Float32Array [frames, 3], the hips in metres
 *   contacts   Float32Array [frames, 4], left heel, left toe, right heel, right toe
 *   keys       [{ frame, turns, positions, effectors }] - the guide poses, each
 *              joint's global turn and position; `effectors` names the hands
 *              and feet a hand-and-foot key holds, null for a whole-body key
 *
 * Returns what it did: { plantedFrames, keyedFrames }.
 */
export function cleanUpMotion({ skeleton, key, frames, rotations, root, contacts, keys = [] }) {
	const joints = effectorJoints(skeleton);
	const { parents, offsets } = skeleton;
	const J = parents.length, N = frames;
	const motion = new Motion(skeleton, Float64Array.from(rotations), Float64Array.from(root));

	// The guide poses as targets: every joint's parent-local turn, and the hips.
	const targets = new Motion(skeleton, new Float64Array(N * J * 4), new Float64Array(N * 3));
	for (let f = 0; f < N; f++) for (let j = 0; j < J; j++) targets.setLocal(f, j, IDENTITY);
	const wholeBody = new Float64Array(N);
	const limbKeyed = { LeftHand: new Float64Array(N), RightHand: new Float64Array(N), LeftFoot: new Float64Array(N), RightFoot: new Float64Array(N) };
	for (const k of keys) {
		if (!(k.frame >= 0 && k.frame < N)) continue;
		if (!k.effectors) wholeBody[k.frame] = 1;
		for (const name of k.effectors ?? []) if (limbKeyed[name]) limbKeyed[name][k.frame] = 1;
		for (let j = 0; j < J; j++) {
			const p = parents[j];
			targets.setLocal(k.frame, j, p < 0 ? k.turns[j] : unit(mul(conjugate(k.turns[p]), k.turns[j])));
		}
		for (let d = 0; d < 3; d++) targets.root[k.frame * 3 + d] = k.positions[0][d];
	}

	// Hands and feet pinned where a hand-and-foot key holds them and no
	// whole-body key already does.
	const pins = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot'].map((name) => ({
		joint: joints[name],
		hint: name.endsWith('Hand') ? HAND_HINT : FOOT_HINT,
		mask: Float64Array.from(limbKeyed[name], (m, f) => (1 - wholeBody[f]) * m),
	}));

	// The rest pose standing on the floor, which says how high a planted foot
	// and toe sit.
	const rest = forward(skeleton, () => IDENTITY, [0, 0, 0]).positions;
	const lowest = Math.min(...rest.map((p) => p[1]));
	const lift = -lowest + (ABOVE_GROUND[key] ?? DEFAULT_ABOVE_GROUND);
	// A foot keyed on its own is left to its key, not planted.
	const touching = (limb, channel) => Float64Array.from({ length: N }, (_, f) => (limbKeyed[limb][f] ? 0 : contacts[4 * f + channel]));
	// The ankle is down when the heel or the toe is.
	const either = (a, b) => a.map((v, f) => Math.min(v + b[f], 1));
	const footContacts = [
		{ joint: joints.RightFoot, hint: FOOT_HINT, floor: rest[joints.RightFoot][1] + lift, mask: either(touching('RightFoot', 2), touching('RightFoot', 3)) },
		{ joint: joints.LeftFoot, hint: FOOT_HINT, floor: rest[joints.LeftFoot][1] + lift, mask: either(touching('LeftFoot', 0), touching('LeftFoot', 1)) },
	];
	const toeContacts = [];
	if (joints.LeftToe >= 0 && joints.RightToe >= 0) {
		// Upstream measures both toes' floor off the right one.
		const toeFloor = rest[joints.RightToe][1] + lift;
		toeContacts.push(
			{ joint: joints.RightToe, floor: toeFloor, mask: touching('RightFoot', 3) },
			{ joint: joints.LeftToe, floor: toeFloor, mask: touching('LeftFoot', 1) },
		);
	}

	// 1. The hips' height.
	{
		const corrector = new Corrector(marginsOf(wholeBody), POS_WEIGHT * 10, VEL_WEIGHT, ACC_WEIGHT * 0.1);
		const y = Float64Array.from({ length: N }, (_, f) => motion.root[f * 3 + 1]);
		const target = Float64Array.from({ length: N }, (_, f) => targets.root[f * 3 + 1]);
		const [fixed] = corrector.interpolate([target], [y]);
		for (let f = 0; f < N; f++) motion.root[f * 3 + 1] = fixed[f];
	}

	// 2. The hips over the floor.
	{
		const margins = marginsOf(wholeBody);
		for (let f = 0; f < N; f++) {
			if (margins[f] !== 0 && pins.some((p) => p.mask[f])) margins[f] = ROOT_MARGIN;
		}
		const corrector = new Corrector(margins, POS_WEIGHT, VEL_WEIGHT, ACC_WEIGHT);
		const x = [0, 2].map((d) => Float64Array.from({ length: N }, (_, f) => motion.root[f * 3 + d]));
		const target = [0, 2].map((d) => Float64Array.from({ length: N }, (_, f) => targets.root[f * 3 + d]));
		for (const column of x) smoothChannels(column, wholeBody);
		const fixed = corrector.interpolate(target, x);
		for (let f = 0; f < N; f++) {
			motion.root[f * 3] = fixed[0][f];
			motion.root[f * 3 + 2] = fixed[1][f];
		}
	}

	// 3. Every joint's turn onto the whole-body keys.
	{
		const corrector = new Corrector(marginsOf(wholeBody), POS_WEIGHT * 10, VEL_WEIGHT, ACC_WEIGHT);
		for (let j = 0; j < J; j++) correctBone(motion, targets, wholeBody, corrector, j, true);
	}

	// 4. Hands and feet onto their keys.
	{
		const masks = new Map();
		const maskOf = (j) => {
			if (!masks.has(j)) masks.set(j, Float64Array.from(wholeBody));
			return masks.get(j);
		};
		const fixed = motion.copy();
		for (const pin of pins) {
			const j = pin.joint, parent = parents[j], grand = parents[parent];
			maskOf(j);
			maskOf(parent);
			maskOf(grand);
			for (let f = 0; f < N; f++) {
				if (!pin.mask[f]) continue;
				const goal = targets.pose(f);
				twoBoneIk(fixed, f, j, goal.positions[j], pin.hint);
				maskOf(parent)[f] = 1;
				maskOf(grand)[f] = 1;
				maskOf(j)[f] = 1;
				// And the hand or foot turned as keyed.
				const above = fixed.pose(f).turns[parent];
				fixed.setLocal(f, j, unit(mul(conjugate(above), goal.turns[j])));
			}
		}
		correctBones(motion, fixed, masks);
	}

	// 5. Planted feet.
	let planted = 0;
	{
		const fixed = motion.copy();
		// A joint's own keyed frames, and its pin's if it or its parent has one.
		const keyedMask = (j) => {
			const mask = Float64Array.from(wholeBody);
			const pin = pins.find((p) => p.joint === j) ?? pins.find((p) => p.joint === parents[j]);
			if (pin) pin.mask.forEach((m, f) => { if (m) mask[f] = 1; });
			return mask;
		};
		let masks = new Map();
		for (const c of footContacts) {
			const j = c.joint, parent = parents[j], grand = parents[parent];
			const keyed = keyedMask(j);
			for (const k of [parent, grand, j]) if (!masks.has(k)) masks.set(k, Float64Array.from(keyed));
			const intervals = filterIntervals(contactIntervals(c.mask, keyed, CONTACT_THRESHOLD), keyed, false);
			const points = contactPoints(motion, j, intervals, keyed, c.floor);
			for (let f = 0; f < N; f++) {
				const target = points[f];
				if (!target) continue;
				planted++;
				for (const k of [parent, grand, j]) masks.get(k)[f] = 1;
				const footTurn = fixed.pose(f).turns[j];
				const w = reachFalloff(fixed, j, true, target, f);
				const parentBefore = fixed.local(f, parent), grandBefore = fixed.local(f, grand);
				twoBoneIk(fixed, f, j, target, c.hint);
				fixed.setLocal(f, parent, unit(slerp(parentBefore, fixed.local(f, parent), w)));
				fixed.setLocal(f, grand, unit(slerp(grandBefore, fixed.local(f, grand), w)));
				// The foot keeps the turn it had.
				fixed.setLocal(f, j, unit(mul(conjugate(fixed.pose(f).turns[parent]), footTurn)));
			}
		}
		correctBones(motion, fixed, masks);

		masks = new Map();
		for (const c of toeContacts) {
			const j = c.joint, parent = parents[j];
			const keyed = keyedMask(j);
			if (!masks.has(parent)) masks.set(parent, Float64Array.from(keyed));
			for (const [first, end] of filterIntervals(contactIntervals(c.mask, keyed, CONTACT_THRESHOLD), keyed, true)) {
				for (let f = first; f < end; f++) {
					// The toe laid on the floor, free to slide, at its bone's length.
					const { positions } = fixed.pose(f);
					const base = positions[parent];
					let target = positions[j];
					const bone = length(sub(target, base));
					for (let i = 0; i < 10; i++) {
						target = [target[0], c.floor, target[2]];
						target = add(base, scaled(normalized(sub(target, base)), bone));
					}
					oneBoneIk(fixed, f, j, target);
					masks.get(parent)[f] = 1;
				}
			}
		}
		correctBones(motion, fixed, masks);
	}

	for (let i = 0; i < rotations.length; i++) rotations[i] = motion.rotations[i];
	for (let i = 0; i < root.length; i++) root[i] = motion.root[i];
	return { plantedFrames: planted, keyedFrames: keys.length };
}
