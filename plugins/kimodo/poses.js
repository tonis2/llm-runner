// Guide poses: keyframes a generation has to pass through, as upstream's
// full-body and end-effector constraints (kimodo/constraints.py
// FullBodyConstraintSet and EndEffectorConstraintSet, encoded the way
// motion_rep/reps/kimodo_motionrep.py create_conditions does).
//
// A caller hands them over in its own rig's terms - joint positions in its own
// units and frame, keyed by the name of the skeleton joint each one stands for:
//
//   poses: {
//     rest: { Hips: [x, y, z], LeftLeg: [...], ... },   // the rig at rest
//     keys: [
//       { frame: 0, joints: { Hips: [...], ... } },                  // whole body
//       { frame: 60, joints: { ... }, effectors: ['RightHand'] },     // one hand
//     ],
//   }
//
// A key with `effectors` holds only those hands and feet (LeftHand, RightHand,
// LeftFoot, RightFoot): where each is and how it is turned, and where the hips
// stand and face - the rest of the body is the model's to choose. A key without
// holds the whole body. Either way `joints` is the whole pose, since the hands
// and feet are placed down the chains from the hips.
//
// Only the joints the rig has an answer for are named. `rest` gives the scale
// and the floor: the rig's hips height over its lowest named joint becomes the
// skeleton's, and the rig's rest hips stand over the origin. Each posed joint is
// then placed from its nearest named ancestor along the rig's direction but at
// the skeleton's own bone length, so a pose from a long-legged or short-armed
// character is still one this skeleton can reach.
//
// A joint the rig does not name is filled in rather than left free - the model
// holds a whole-body keyframe far better than one with holes (a head with its
// neck and eyes free drifts a quarter of a metre off its own constraint). It is
// carried rigidly by the nearest bone the rig does pose: the span from its
// placed ancestor to its first named descendant, or its parent's own bone for
// a leaf (jaw, eyes, finger ends). That bone's swing carries no twist, so a
// filled joint follows the bend and not the roll.
//
// The rig is taken as facing +Z, Y up - glTF's convention and Kimodo's. A
// caller whose rig faces elsewhere turns its points about the hips first.

import { IDENTITY, sub, add, length, cross, normalized, mul, conjugate, rotate, between, fromAxes } from './rotation.js';

const scale = (std) => Math.sqrt(std * std + 1e-5);

// Where each joint of `skeleton` stands at rest, the root at the origin.
export function restPositions(skeleton) {
	const out = [];
	skeleton.parents.forEach((p, j) => {
		const o = skeleton.offsets[j];
		out.push(p < 0 ? [0, 0, 0] : [out[p][0] + o[0], out[p][1] + o[1], out[p][2] + o[2]]);
	});
	return out;
}

// `v` turned by the shortest rotation carrying direction `from` onto `to`.
const swing = (v, from, to) => rotate(between(from, to), v);

// Upstream's compute_heading_angle: the facing, from the right hip to the left.
export function headingOf(right, left) {
	const d = sub(right, left);
	return Math.atan2(d[2], -d[0]);
}

// The first joint under `j` (depth first) the pose has placed, or -1.
function firstNamedBelow(skeleton, j, at) {
	for (let c = j + 1; c < skeleton.parents.length; c++) {
		let a = skeleton.parents[c];
		while (a > j) a = skeleton.parents[a];
		if (a === j && at[c]) return c;
	}
	return -1;
}

// The hands and feet a key holds on their own, as the skeleton's effector
// names; null for a whole-body key.
function effectorsOf(key, skeleton) {
	if (!Array.isArray(key.effectors)) return null;
	const known = key.effectors.filter((name) => skeleton.effectors?.[name]);
	if (known.length < key.effectors.length) {
		const unknown = key.effectors.filter((name) => !known.includes(name));
		throw new Error(`a guide pose holds ${unknown.join(', ')}; the effectors are ${Object.keys(skeleton.effectors ?? {}).join(', ')}`);
	}
	return known.length > 0 ? known : null;
}

// Each joint's global turn in a placed pose, from the positions alone: a joint
// with two or more children is turned so they point as placed, one with a
// single child swings its parent's turn onto that child's direction, and a leaf
// turns with its parent. At rest every joint is unturned.
export function globalTurns(positions, skeleton) {
	const { parents } = skeleton;
	const rest = restPositions(skeleton);
	const children = parents.map(() => []);
	parents.forEach((p, j) => { if (p >= 0) children[p].push(j); });
	const frame = (u, v) => {
		const z = cross(u, v);
		if (length(z) < 1e-6 * length(u) * length(v)) return null;
		const x = normalized(u), zz = normalized(z);
		return fromAxes(x, cross(zz, x), zz);
	};
	const turns = new Array(parents.length);
	parents.forEach((p, j) => {
		const above = p < 0 ? IDENTITY : turns[p];
		const kids = children[j];
		if (kids.length >= 2) {
			const [a, b] = kids;
			const posed = frame(sub(positions[a], positions[j]), sub(positions[b], positions[j]));
			const resting = frame(sub(rest[a], rest[j]), sub(rest[b], rest[j]));
			if (posed && resting) {
				turns[j] = mul(posed, conjugate(resting));
				return;
			}
		}
		if (kids.length >= 1) {
			const c = kids[0];
			turns[j] = mul(between(rotate(above, sub(rest[c], rest[j])), sub(positions[c], positions[j])), above);
			return;
		}
		turns[j] = above;
	});
	return turns;
}

// The guide poses in the skeleton's space: [{ frame, positions, turns,
// effectors }] with positions[j] an [x, y, z] in metres, turns[j] its global
// turn, and effectors the hands and feet a partial key holds (null: all of it).
export function placePoses(poses, skeleton, frames) {
	const names = skeleton.names;
	const index = new Map(names.map((n, j) => [n, j]));
	const root = names[0];
	const rest = poses.rest ?? {};
	if (!rest[root]) throw new Error(`the guide poses do not say where the ${root} joint rests`);
	const known = Object.keys(rest).filter((n) => index.has(n));
	if (known.length < 2) throw new Error(`the guide poses name no joints of this skeleton (${names.slice(0, 4).join(', ')}, …)`);

	const skeletonRest = restPositions(skeleton);
	const skeletonFloor = Math.min(...skeletonRest.map((p) => p[1]));
	const rigFloor = Math.min(...known.map((n) => rest[n][1]));
	const rigHips = rest[root][1] - rigFloor;
	if (!(rigHips > 0)) throw new Error('the guide rig has its hips at its lowest joint - no height to scale by');
	const s = -skeletonFloor / rigHips;
	const origin = rest[root];
	const toSkeleton = (p) => [(p[0] - origin[0]) * s, (p[1] - rigFloor) * s, (p[2] - origin[2]) * s];

	const placed = [];
	for (const key of poses.keys ?? []) {
		const frame = Math.round(key.frame);
		if (!(frame >= 0 && frame < frames)) continue;
		const at = new Array(names.length).fill(null);
		const raw = new Array(names.length).fill(null);
		for (const [name, p] of Object.entries(key.joints ?? {})) {
			const j = index.get(name);
			if (j !== undefined && rest[name]) raw[j] = toSkeleton(p);
		}
		if (!raw[0]) continue;
		at[0] = raw[0];
		for (let j = 1; j < names.length; j++) {
			if (!raw[j]) continue;
			let a = skeleton.parents[j];
			while (a > 0 && !at[a]) a = skeleton.parents[a];
			const bone = length(sub(skeletonRest[j], skeletonRest[a]));
			let d = sub(raw[j], raw[a]);
			let n = length(d);
			if (n < 1e-6) {
				d = sub(skeletonRest[j], skeletonRest[a]);
				n = length(d) || 1;
			}
			at[j] = [at[a][0] + (d[0] / n) * bone, at[a][1] + (d[1] / n) * bone, at[a][2] + (d[2] / n) * bone];
		}
		// The joints the rig has no word for, parents first.
		for (let j = 1; j < names.length; j++) {
			if (at[j]) continue;
			// The parent is placed by now: named, or filled on an earlier turn.
			const p = skeleton.parents[j], pp = skeleton.parents[p];
			const below = firstNamedBelow(skeleton, j, at);
			let from = [0, 0, 1], to = [0, 0, 1];
			if (below >= 0) {
				from = sub(skeletonRest[below], skeletonRest[p]);
				to = sub(at[below], at[p]);
			} else if (pp >= 0) {
				from = sub(skeletonRest[p], skeletonRest[pp]);
				to = sub(at[p], at[pp]);
			}
			at[j] = add(at[p], swing(sub(skeletonRest[j], skeletonRest[p]), from, to));
		}
		placed.push({ frame, positions: at, turns: globalTurns(at, skeleton), effectors: effectorsOf(key, skeleton) });
	}
	placed.sort((a, b) => a.frame - b.frame);
	return placed;
}

// The observed motion and its mask, [frames, D] each, for `placed` poses:
// normalised feature values where a pose says something and 1 in the mask
// there, zeros everywhere else. `heading` is the first frame's facing when a
// pose is keyed on it.
//
// Every key holds the hips: where they stand over the floor (the smooth root),
// how high, and which way the body faces. A whole-body key adds every joint's
// position; a hand-and-foot key adds each held effector's chain positions and
// the global turn of its first joint.
export function encodePoses(placed, skeleton, stats, frames) {
	const J = skeleton.parents.length, D = 9 + 12 * J;
	const observed = new Float32Array(frames * D);
	const mask = new Float32Array(frames * D);
	const { globalMean: gm, globalStd: gs, bodyMean: bm, bodyStd: bs } = stats;
	const put = (t, i, v) => {
		observed[t * D + i] = i < 5 ? (v - gm[i]) / scale(gs[i]) : (v - bm[i - 5]) / scale(bs[i - 5]);
		mask[t * D + i] = 1;
	};
	const index = new Map(skeleton.names.map((n, j) => [n, j]));
	const [right, left] = (skeleton.hips ?? []).map((n) => index.get(n) ?? -1);
	const rotationsAt = 5 + 3 * J;
	let heading = null;
	for (const { frame: t, positions, turns, effectors } of placed) {
		const hips = positions[0];
		// The smooth root is the hips on the floor plane; its height is the hips'.
		put(t, 0, hips[0]);
		put(t, 1, hips[1]);
		put(t, 2, hips[2]);
		if (right >= 0 && left >= 0 && positions[right] && positions[left]) {
			const angle = headingOf(positions[right], positions[left]);
			put(t, 3, Math.cos(angle));
			put(t, 4, Math.sin(angle));
			if (t === 0) heading = angle;
		}
		const putPosition = (j) => {
			put(t, 5 + 3 * j, positions[j][0] - hips[0]);
			put(t, 5 + 3 * j + 1, positions[j][1]);
			put(t, 5 + 3 * j + 2, positions[j][2] - hips[2]);
		};
		if (!effectors) {
			positions.forEach((p, j) => { if (p) putPosition(j); });
			continue;
		}
		for (const name of effectors) {
			const chain = skeleton.effectors[name].map((n) => index.get(n));
			chain.forEach(putPosition);
			// The 6D turn: the rotation matrix's first two columns.
			const j = chain[0];
			const columns = [...rotate(turns[j], [1, 0, 0]), ...rotate(turns[j], [0, 1, 0])];
			columns.forEach((v, k) => put(t, rotationsAt + 6 * j + k, v));
		}
	}
	return { observed, mask, heading };
}
