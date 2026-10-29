// Frames posed one by one into a motion clip on the MHR skeleton
// (skeleton.js): parent-local turns and the pelvis' path, smoothed, stood on a
// floor at y = 0, with each foot's contacts read off the motion.
//
// The camera is taken as fixed. SAM 3D Body places the body in the camera's
// frame (`pred_cam_t`); turned y up, the camera sits at the origin looking down
// -z. The clip is then moved so the pelvis starts over the origin and the
// feet's lowest stretch is the floor; `camera` says where the camera ended up.
// A camera tilted up or down tilts the floor with it, so a person walking
// towards it sinks or climbs: the floor is fitted as a plane under the feet
// over the whole clip, and the clip turned about the camera to stand it level.

import { forward, between, rotate, mul, axisAngle } from '../lib/motion/rotation.js';
import { MotionSkeleton, clipProportions } from './skeleton.js';
import { smoothRows } from '../lib/smooth.js';

// Contacts: a foot joint within this of its standing height, moving slower
// than this, is down.
const CONTACT_HEIGHT = 0.05;
const CONTACT_SPEED = 0.6;
// The floor sits under this fraction of frames' lowest foot.
const FLOOR_PERCENTILE = 0.05;
// The floor plane's fit: a foot this far over the plane is off the floor and
// left out of the next pass; slopes are held towards level as though every
// frame also had a foot this far to each side on a level floor, so a clip
// that stays on one spot keeps a level floor; no more tilt than this.
const FLOOR_OFF = 0.03;
const FLOOR_SPREAD = 0.05;
const MAX_TILT = (20 * Math.PI) / 180;

// Each joint's turn kept on the same side of the 4D sphere as the frame before,
// so neither smoothing nor interpolation goes the long way round.
function continuous(rotations, frames, J) {
	for (let f = 1; f < frames; f++) {
		for (let j = 0; j < J; j++) {
			const a = ((f - 1) * J + j) * 4, b = (f * J + j) * 4;
			const d = rotations[a] * rotations[b] + rotations[a + 1] * rotations[b + 1] + rotations[a + 2] * rotations[b + 2] + rotations[a + 3] * rotations[b + 3];
			if (d < 0) for (let k = 0; k < 4; k++) rotations[b + k] = -rotations[b + k];
		}
	}
}

function normalizeAll(rotations) {
	for (let i = 0; i < rotations.length; i += 4) {
		const n = Math.hypot(rotations[i], rotations[i + 1], rotations[i + 2], rotations[i + 3]) || 1;
		for (let k = 0; k < 4; k++) rotations[i + k] /= n;
	}
}

function percentile(values, q) {
	const v = Array.from(values).sort((a, b) => a - b);
	return v.length ? v[Math.min(v.length - 1, Math.max(0, Math.round(q * (v.length - 1))))] : 0;
}

// The plane y = a x + b z + c under points [x, y, z]: least squares, again
// over the points not FLOOR_OFF over the last fit, slopes held towards zero.
function floorPlane(points) {
	let use = points;
	let a = 0, b = 0, c = 0;
	for (let pass = 0; pass < 4 && use.length >= 3; pass++) {
		// Centred, so the slopes come out of a 2 x 2 system.
		const n = use.length;
		const mx = use.reduce((s, p) => s + p[0], 0) / n, mz = use.reduce((s, p) => s + p[2], 0) / n;
		const my = use.reduce((s, p) => s + p[1], 0) / n;
		const ridge = n * FLOOR_SPREAD * FLOOR_SPREAD;
		let xx = ridge, xz = 0, zz = ridge, xy = 0, zy = 0;
		for (const p of use) {
			const x = p[0] - mx, z = p[2] - mz, y = p[1] - my;
			xx += x * x; xz += x * z; zz += z * z; xy += x * y; zy += z * y;
		}
		const det = xx * zz - xz * xz;
		a = (xy * zz - zy * xz) / det;
		b = (zy * xx - xy * xz) / det;
		c = my - a * mx - b * mz;
		const kept = points.filter((p) => p[1] - (a * p[0] + b * p[2] + c) < FLOOR_OFF);
		if (kept.length === use.length) break;
		use = kept;
	}
	return { a, b, c };
}

/**
 * The clip from posed frames.
 *
 *   frames   [{ params (model parameters [204]), cam_t [3] }], one a video frame
 *   fps      the video's frame rate
 *   smooth   the smoothing window in frames (odd; 1 or 0 for none)
 *   facing   'forward' turns the clip about +y so the figure faces +z on
 *            average, as the rigs it is put on do; 'camera' keeps the facing it
 *            had to the camera
 *
 * Returns { skeleton, fps, frames, rotations Float32Array [frames, J, 4],
 * root Float32Array [frames, 3], contacts Float32Array [frames, 4] (left heel,
 * left toe, right heel, right toe), camera [3] (where the camera
 * is), cameraRotation [x, y, z, w] }.
 */
export function assembleMotion(mhr, frames, { fps = 24, smooth = 7, facing = 'forward' } = {}) {
	const N = frames.length;
	if (N === 0) throw new Error('no frames to make a motion of');
	const skeleton = new MotionSkeleton(mhr, clipProportions(frames.map((f) => f.params)));
	const { rotations, root } = poseRows(skeleton, frames);
	smoothMotion(rotations, root, N, skeleton.jointCount, smooth);
	const clip = finishMotion(skeleton, rotations, root, fps, levelFloor(skeleton, rotations, root, N));
	if (facing !== 'camera') faceForward(clip);
	return clip;
}

// Each frame's parent-local turns [N, J, 4] and the pelvis' path [N, 3] on
// `skeleton`, in SAM 3D Body's camera frame turned y up.
export function poseRows(skeleton, frames) {
	const N = frames.length, J = skeleton.jointCount;
	const rotations = new Float64Array(N * J * 4);
	const root = new Float64Array(N * 3);
	frames.forEach((fr, f) => {
		const { local, root: r } = skeleton.frame(fr.params);
		rotations.set(local, f * J * 4);
		// SAM 3D Body's camera frame turned y up: y and z flip.
		const t = fr.cam_t;
		root[f * 3] = r[0] + t[0];
		root[f * 3 + 1] = r[1] - t[1];
		root[f * 3 + 2] = r[2] - t[2];
	});
	return { rotations, root };
}

export function smoothMotion(rotations, root, N, J, smooth) {
	continuous(rotations, N, J);
	smoothRows(rotations, N, J * 4, smooth);
	normalizeAll(rotations);
	smoothRows(root, N, 3, smooth);
}

// Each frame's foot joints (ankles and balls) where they stand, the joints, and
// how high each stands at rest over y = 0.
export function footPositions(skeleton, rotations, root, N) {
	const J = skeleton.jointCount;
	const feet = ['l_foot', 'l_ball', 'r_foot', 'r_ball'].map((n) => skeleton.index(n));
	const pose = (f) => forward(skeleton, (j) => Array.from(rotations.subarray((f * J + j) * 4, (f * J + j + 1) * 4)), [root[f * 3], root[f * 3 + 1], root[f * 3 + 2]]).positions;
	const rest = forward(skeleton, () => [0, 0, 0, 1], skeleton.restRoot).positions;
	const standing = feet.map((j) => rest[j][1]);
	const positions = Array.from({ length: N }, (_, f) => { const p = pose(f); return feet.map((j) => p[j]); });
	// Where the floor is under each frame: under its lowest foot joint, less how
	// high that joint stands.
	const under = positions.map((p) => {
		let best = 0;
		p.forEach((q, i) => { if (q[1] - standing[i] < p[best][1] - standing[best]) best = i; });
		return [p[best][0], p[best][1] - standing[best], p[best][2]];
	});
	return { feet, standing, positions, under, level: [0, 0, 0, 1] };
}

// The floor plane under the feet, turned level: the root's path and turn and
// the feet turn about the camera, at the origin. Returns footPositions'
// answer, levelled, with the turn as `level`.
export function levelFloor(skeleton, rotations, root, N) {
	const J = skeleton.jointCount;
	const stood = footPositions(skeleton, rotations, root, N);
	const { positions, under } = stood;
	const { a, b } = floorPlane(under);
	let level = between([-a, 1, -b], [0, 1, 0]);
	const tilt = 2 * Math.acos(Math.min(1, level[3]));
	if (tilt > MAX_TILT) level = [0, 0, 0, 1];
	for (let f = 0; f < N; f++) {
		const r = rotate(level, [root[f * 3], root[f * 3 + 1], root[f * 3 + 2]]);
		root.set(r, f * 3);
		const o = f * J * 4;
		rotations.set(mul(level, Array.from(rotations.subarray(o, o + 4))), o);
		positions[f] = positions[f].map((q) => rotate(level, q));
		under[f] = rotate(level, under[f]);
	}
	stood.level = level;
	return stood;
}

// Where the figure faces in each frame (radians about +y, 0 at +z): across its
// hips, left (+x at rest) to right.
export function headings(skeleton, rotations, N) {
	const J = skeleton.jointCount;
	const l = skeleton.offsets[skeleton.index('l_upleg')], r = skeleton.offsets[skeleton.index('r_upleg')];
	const across = [l[0] - r[0], l[1] - r[1], l[2] - r[2]];
	return Array.from({ length: N }, (_, f) => {
		const o = f * J * 4;
		const x = rotate(Array.from(rotations.subarray(o, o + 4)), across);
		// Facing is across x up: (x, ., z) x (0, 1, 0) = (-z, ., x).
		return Math.atan2(-x[2], x[0]);
	});
}

// The mean of angles, round the circle.
export function meanAngle(angles) {
	let s = 0, c = 0;
	for (const a of angles) { s += Math.sin(a); c += Math.cos(a); }
	return Math.atan2(s, c);
}

// The clip turned about +y through the origin (under the first frame's pelvis)
// so the figure faces +z on average: the root's turn and path, and the camera.
export function faceForward(clip) {
	const { skeleton, frames: N, rotations, root } = clip;
	const J = skeleton.jointCount;
	const turn = axisAngle([0, 1, 0], -meanAngle(headings(skeleton, rotations, N)));
	for (let f = 0; f < N; f++) {
		const o = f * J * 4;
		rotations.set(mul(turn, Array.from(rotations.subarray(o, o + 4))), o);
		root.set(rotate(turn, [root[f * 3], root[f * 3 + 1], root[f * 3 + 2]]), f * 3);
	}
	clip.camera = rotate(turn, clip.camera);
	clip.cameraRotation = mul(turn, clip.cameraRotation);
	return clip;
}

// The clip from levelled turns and path: stood on the floor with the first
// frame's pelvis over the origin, and each foot's contacts. `stood` is
// footPositions' (or levelFloor's) answer for them.
export function finishMotion(skeleton, rotations, root, fps, stood) {
	const N = root.length / 3;
	const { feet, standing, positions, under, level } = stood;

	// The floor: under the lowest of the frames' lowest foot joints.
	const floor = percentile(under.map((p) => p[1]), FLOOR_PERCENTILE);
	const shift = [root[0], floor, root[2]];
	for (let f = 0; f < N; f++) for (let d = 0; d < 3; d++) root[f * 3 + d] -= shift[d];
	for (const p of positions) for (const q of p) for (let d = 0; d < 3; d++) q[d] -= shift[d];

	// Contacts: the heel is the ankle ('foot'), the toe the ball.
	const contacts = new Float32Array(N * 4);
	for (let i = 0; i < feet.length; i++) {
		for (let f = 0; f < N; f++) {
			const a = positions[Math.max(f - 1, 0)][i], b = positions[Math.min(f + 1, N - 1)][i];
			const span = Math.min(f + 1, N - 1) - Math.max(f - 1, 0);
			const speed = span > 0 ? (Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) * fps) / span : 0;
			const low = positions[f][i][1] - standing[i] < CONTACT_HEIGHT;
			contacts[f * 4 + i] = low && speed < CONTACT_SPEED ? 1 : 0;
		}
	}

	return {
		skeleton,
		fps,
		frames: N,
		rotations: Float32Array.from(rotations),
		root: Float32Array.from(root),
		contacts,
		camera: [-shift[0], -shift[1], -shift[2]],
		// The camera's turn: it looks down its own -z, y up, turned by this.
		cameraRotation: level,
	};
}
