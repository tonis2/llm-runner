// Guide poses: keyframes a generation has to pass through, as upstream's
// full-body constraints (kimodo/constraints.py FullBodyConstraintSet, encoded
// the way motion_rep/reps/kimodo_motionrep.py create_conditions does).
//
// A caller hands them over in its own rig's terms - joint positions in its own
// units and frame, keyed by the name of the skeleton joint each one stands for:
//
//   poses: {
//     rest: { Hips: [x, y, z], LeftLeg: [...], ... },   // the rig at rest
//     keys: [{ frame: 0, joints: { Hips: [...], ... } }, ...],
//   }
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
// The rig is taken as facing +Z, Y up - glTF's convention and Kimodo's.

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

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const length = (v) => Math.hypot(v[0], v[1], v[2]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// `v` turned by the shortest rotation carrying direction `from` onto `to`.
function swing(v, from, to) {
	const a = length(from), b = length(to);
	if (a < 1e-9 || b < 1e-9) return v;
	const f = from.map((x) => x / a), t = to.map((x) => x / b);
	const axis = cross(f, t), sin = length(axis), cos = dot(f, t);
	if (sin < 1e-9) {
		if (cos > 0) return v;
		// Half a turn: about any axis square to the bone.
		const other = Math.abs(f[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
		const k = cross(f, other), n = length(k);
		const u = k.map((x) => x / n);
		return sub(u.map((x) => 2 * dot(u, v) * x), v);
	}
	const k = axis.map((x) => x / sin);
	// Rodrigues.
	const kv = cross(k, v), kd = dot(k, v);
	return [0, 1, 2].map((i) => v[i] * cos + kv[i] * sin + k[i] * kd * (1 - cos));
}

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

// The guide poses in the skeleton's space: [{ frame, positions }] with
// positions[j] an [x, y, z] in metres or null for a joint left free.
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
		placed.push({ frame, positions: at });
	}
	placed.sort((a, b) => a.frame - b.frame);
	return placed;
}

// The observed motion and its mask, [frames, D] each, for `placed` poses:
// normalised feature values where a pose says something and 1 in the mask
// there, zeros everywhere else. `heading` is the first frame's facing when a
// pose is keyed on it.
export function encodePoses(placed, skeleton, stats, frames) {
	const J = skeleton.parents.length, D = 9 + 12 * J;
	const observed = new Float32Array(frames * D);
	const mask = new Float32Array(frames * D);
	const { globalMean: gm, globalStd: gs, bodyMean: bm, bodyStd: bs } = stats;
	const put = (t, i, v) => {
		observed[t * D + i] = i < 5 ? (v - gm[i]) / scale(gs[i]) : (v - bm[i - 5]) / scale(bs[i - 5]);
		mask[t * D + i] = 1;
	};
	const right = skeleton.names.indexOf('RightLeg'), left = skeleton.names.indexOf('LeftLeg');
	let heading = null;
	for (const { frame: t, positions } of placed) {
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
		positions.forEach((p, j) => {
			if (!p) return;
			put(t, 5 + 3 * j, p[0] - hips[0]);
			put(t, 5 + 3 * j + 1, p[1]);
			put(t, 5 + 3 * j + 2, p[2] - hips[2]);
		});
	}
	return { observed, mask, heading };
}
