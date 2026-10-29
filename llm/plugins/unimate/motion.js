// UniMate's output features to a clip: per-joint local rotations and a root
// path, in the rig's own frame and units.
//
// A frame has 12 numbers a joint. Joint j's are its position relative to the
// root's facing, the 6D rotation of its *parent* (the HumanML3D layout: slot j
// holds bone parent(j)'s rotation), and its velocity; the root's are its height,
// its facing (6D) and its planar velocity in that facing. So each joint's
// rotation is read from a child's slot (a leaf's is identity), and the root
// path is the facing-rotated velocities summed up.
//
// Those are rotations of the canonical rest pose (identity rotations, the
// bones as world vectors), and positions in the canonical frame. They are
// taken back to the rig's frame by the inverse of the canonicalisation: turned
// back about Y, scaled back up, and put where the rig's root stands.

import { rotate } from './skeleton.js';

const FEAT = 12;

// The 6D rotation (first two matrix columns) as a quaternion [w, x, y, z].
function quatOf6d(r) {
	const n = (v) => { const l = Math.hypot(...v); return v.map((c) => c / l); };
	const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
	const x = n([r[0], r[1], r[2]]);
	const z = n(cross(x, [r[3], r[4], r[5]]));
	const y = cross(z, x);
	// Columns x, y, z: m[row][col].
	const m00 = x[0], m10 = x[1], m20 = x[2], m01 = y[0], m11 = y[1], m21 = y[2], m02 = z[0], m12 = z[1], m22 = z[2];
	const tr = m00 + m11 + m22;
	let q;
	if (tr > 0) {
		const s = Math.sqrt(tr + 1) * 2;
		q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s];
	} else if (m00 > m11 && m00 > m22) {
		const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
		q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s];
	} else if (m11 > m22) {
		const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
		q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s];
	} else {
		const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
		q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s];
	}
	return q;
}

export const qmul = (a, b) => [
	a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
	a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
	a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
	a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
];
const qinv = (q) => [q[0], -q[1], -q[2], -q[3]];

// Normalised [J, 12, 60] (the sampler's layout) to features [frames, J, 12].
export function denormalize(sample, sk, frames = 60) {
	const J = sk.J, out = new Float32Array(frames * J * FEAT);
	for (let j = 0; j < J; j++) {
		for (let d = 0; d < FEAT; d++) {
			const m = sk.mean[j * FEAT + d], s = sk.std[j * FEAT + d];
			for (let f = 0; f < frames; f++) out[(f * J + j) * FEAT + d] = sample[(j * FEAT + d) * frames + f] * s + m;
		}
	}
	return out;
}

// Features [frames, J, 12] to the canonical clip: `rotations[f][j]` [w, x, y, z]
// local rotations and `root[f]` the root's position.
export function decodeCanonical(feats, sk, frames = 60) {
	const J = sk.J, at = (f, j) => (f * J + j) * FEAT;
	const firstChild = new Array(J).fill(-1);
	for (let j = J - 1; j >= 1; j--) firstChild[sk.parents[j]] = j;
	const rotations = [], root = [];
	let x = 0, z = 0;
	for (let f = 0; f < frames; f++) {
		const rot = Array.from({ length: J }, () => [1, 0, 0, 0]);
		// Slot j holds its parent's rotation; with several children the last
		// written wins, as the reference's loop does.
		for (let j = 1; j < J; j++) rot[sk.parents[j]] = quatOf6d(feats.subarray(at(f, j) + 3, at(f, j) + 9));
		rotations.push(rot);
		const facing = quatOf6d(feats.subarray(at(f, 0) + 3, at(f, 0) + 9));
		if (f > 0) {
			const v = [feats[at(f - 1, 0) + 9], 0, feats[at(f - 1, 0) + 11]];
			const w = rotate(qinv(facing), v);
			x += w[0];
			z += w[2];
		}
		root.push([x, feats[at(f, 0) + 1], z]);
	}
	return { rotations, root };
}

// Joint positions of a canonical clip (forward kinematics on the rest bones).
export function positionsOf(clip, sk) {
	return clip.rotations.map((rot, f) => {
		const g = new Array(sk.J), p = new Array(sk.J);
		for (let j = 0; j < sk.J; j++) {
			const par = sk.parents[j];
			if (par === -1) {
				g[j] = rot[j];
				p[j] = clip.root[f];
			} else {
				const o = rotate(g[par], sk.offsets[j]);
				p[j] = [p[par][0] + o[0], p[par][1] + o[1], p[par][2] + o[2]];
				g[j] = qmul(g[par], rot[j]);
			}
		}
		return p;
	});
}

// The canonical clip in the rig's frame and units: `rotations` flat
// [frames, J, 4] xyzw, `root` flat [frames, 3], and the skeleton they apply to
// (UniMate's joint order, root first, the rig's names; rest rotations
// identity, bones as the rig's own vectors).
export function toRigFrame(clip, sk) {
	const J = sk.J, frames = clip.rotations.length;
	const back = qinv(sk.turn);
	// A canonical point to the rig's frame.
	const place = (p) => rotate(back, [p[0] / sk.scale + sk.rootXZ[0], (p[1] + sk.ground) / sk.scale, p[2] / sk.scale + sk.rootXZ[2]]);
	const offsets = sk.offsets.map((o) => rotate(back, o.map((c) => c / sk.scale)));
	const rotations = new Float32Array(frames * J * 4), root = new Float32Array(frames * 3);
	for (let f = 0; f < frames; f++) {
		for (let j = 0; j < J; j++) {
			// The same rotation seen from the rig's frame: back * q * turn.
			const q = qmul(qmul(back, clip.rotations[f][j]), sk.turn);
			rotations.set([q[1], q[2], q[3], q[0]], (f * J + j) * 4);
		}
		root.set(place(clip.root[f]), f * 3);
	}
	return { skeleton: { names: sk.names, parents: sk.parents, offsets }, rotations, root, frames };
}
