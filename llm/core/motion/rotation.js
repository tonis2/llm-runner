// Small vector and quaternion helpers for the host-side motion work: guide
// poses (poses.js) and the clean-up after generation (cleanup.js).
//
// Quaternions are [x, y, z, w] - the order the decoded motion and glTF use -
// multiplied the usual way: mul(a, b) turns by b, then by a. A joint's global
// turn is mul(parentGlobal, local).

export const IDENTITY = Object.freeze([0, 0, 0, 1]);

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scaled = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const length = (v) => Math.hypot(v[0], v[1], v[2]);
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function normalized(v) {
	const n = length(v);
	return n > 0 ? [v[0] / n, v[1] / n, v[2] / n] : [0, 0, 0];
}

export const mul = (a, b) => [
	a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
	a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
	a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
	a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];

export const conjugate = (q) => [-q[0], -q[1], -q[2], q[3]];

export function unit(q) {
	const n = Math.hypot(q[0], q[1], q[2], q[3]);
	return n > 0 ? [q[0] / n, q[1] / n, q[2] / n, q[3] / n] : [0, 0, 0, 1];
}

// `v` turned by `q`.
export function rotate(q, v) {
	const [x, y, z, w] = q;
	const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
	return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

export function axisAngle(axis, angle) {
	const s = Math.sin(angle / 2);
	return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

// The shortest turn carrying direction `from` onto `to`; half a turn about some
// axis square to both when they point opposite ways.
export function between(from, to) {
	const a = normalized(from), b = normalized(to);
	const d = dot(a, b);
	if (d >= 1 - 1e-7) return [0, 0, 0, 1];
	if (d <= -1 + 1e-7) {
		const other = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
		return axisAngle(normalized(cross(a, other)), Math.PI);
	}
	const c = cross(a, b);
	return unit([c[0], c[1], c[2], 1 + d]);
}

export function slerp(a, b, t) {
	let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
	const sign = d < 0 ? -1 : 1;
	d *= sign;
	let s0 = 1 - t, s1 = t;
	if (d < 1 - 1e-5) {
		const omega = Math.atan2(Math.sqrt(1 - d * d), d);
		const sin = Math.sin(omega);
		s0 = Math.sin((1 - t) * omega) / sin;
		s1 = Math.sin(t * omega) / sin;
	}
	s1 *= sign;
	return [a[0] * s0 + b[0] * s1, a[1] * s0 + b[1] * s1, a[2] * s0 + b[2] * s1, a[3] * s0 + b[3] * s1];
}

// The turn whose x, y and z axes are the given (orthonormal) columns.
export function fromAxes(x, y, z) {
	const m = [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]];
	let q;
	const t = m[0] + m[4] + m[8];
	if (t > 0) {
		const s = 2 * Math.sqrt(t + 1);
		q = [(m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s, 0.25 * s];
	} else if (m[0] > m[4] && m[0] > m[8]) {
		const s = 2 * Math.sqrt(1 + m[0] - m[4] - m[8]);
		q = [0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s, (m[7] - m[5]) / s];
	} else if (m[4] > m[8]) {
		const s = 2 * Math.sqrt(1 + m[4] - m[0] - m[8]);
		q = [(m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s, (m[2] - m[6]) / s];
	} else {
		const s = 2 * Math.sqrt(1 + m[8] - m[0] - m[4]);
		q = [(m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s, (m[3] - m[1]) / s];
	}
	return unit(q);
}

// The turn with its z axis on `forward` and its y axis as near `up` as that
// allows.
export function lookRotation(forward, up) {
	const x = normalized(cross(up, forward));
	return fromAxes(x, cross(forward, x), forward);
}

// Where every joint of `skeleton` stands and how it is turned, for parent-local
// turns `local(j)` and the root at `root`: { turns, positions }, both by joint.
// Parents come before their children in every skeleton here.
export function forward(skeleton, local, root) {
	const J = skeleton.parents.length;
	const turns = new Array(J), positions = new Array(J);
	for (let j = 0; j < J; j++) {
		const p = skeleton.parents[j];
		if (p < 0) {
			turns[j] = local(j);
			positions[j] = [root[0], root[1], root[2]];
		} else {
			turns[j] = mul(turns[p], local(j));
			positions[j] = add(positions[p], rotate(turns[p], skeleton.offsets[j]));
		}
	}
	return { turns, positions };
}
