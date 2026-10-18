// Frames posed one by one into a motion clip on the MHR skeleton
// (skeleton.js): parent-local turns and the pelvis' path, smoothed, stood on a
// floor at y = 0, with each foot's contacts read off the motion.
//
// The camera is taken as fixed. SAM 3D Body places the body in the camera's
// frame (`pred_cam_t`); turned y up, the camera sits at the origin looking down
// -z. The clip is then moved so the pelvis starts over the origin and the
// feet's lowest stretch is the floor; `camera` says where the camera ended up.
// A tilted camera tilts the floor with it: nothing here levels it.

import { forward } from '../kimodo/rotation.js';
import { MotionSkeleton, clipProportions } from './skeleton.js';

// Contacts: a foot joint within this of its standing height, moving slower
// than this, is down.
const CONTACT_HEIGHT = 0.05;
const CONTACT_SPEED = 0.6;
// The floor sits under this fraction of frames' lowest foot.
const FLOOR_PERCENTILE = 0.05;

// Savitzky-Golay weights (quadratic) for a window of `size` frames, odd.
function savitzkyGolay(size) {
	const m = (size - 1) >> 1;
	const d = (2 * m - 1) * (2 * m + 1) * (2 * m + 3);
	return Array.from({ length: 2 * m + 1 }, (_, k) => {
		const i = k - m;
		return (3 * (3 * m * m + 3 * m - 1 - 5 * i * i)) / d;
	});
}

// `data` [frames, width] smoothed along frames. Past either end the frames
// are reflected through the end one (2 x[0] - x[k]), which keeps a steady
// drift steady where a mirror would bend it.
function smoothRows(data, frames, width, size) {
	if (size < 3 || frames < 3) return;
	size = Math.min(size | 1, frames % 2 ? frames : frames - 1);
	if (size < 3) return;
	const w = savitzkyGolay(size), m = (size - 1) >> 1;
	const src = Float64Array.from(data);
	const value = (f, c) => {
		if (f < 0) return 2 * src[c] - src[-f * width + c];
		if (f >= frames) return 2 * src[(frames - 1) * width + c] - src[(2 * (frames - 1) - f) * width + c];
		return src[f * width + c];
	};
	for (let f = 0; f < frames; f++) {
		for (let c = 0; c < width; c++) {
			let s = 0;
			for (let k = -m; k <= m; k++) s += w[k + m] * value(f + k, c);
			data[f * width + c] = s;
		}
	}
}

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

/**
 * The clip from posed frames.
 *
 *   frames   [{ params (model parameters [204]), cam_t [3] }], one a video frame
 *   fps      the video's frame rate
 *   smooth   the smoothing window in frames (odd; 1 or 0 for none)
 *
 * Returns { skeleton, fps, frames, rotations Float32Array [frames, J, 4],
 * root Float32Array [frames, 3], contacts Float32Array [frames, 4] (left heel,
 * left toe, right heel, right toe), camera [3] }.
 */
export function assembleMotion(mhr, frames, { fps = 24, smooth = 7 } = {}) {
	const N = frames.length;
	if (N === 0) throw new Error('no frames to make a motion of');
	const skeleton = new MotionSkeleton(mhr, clipProportions(frames.map((f) => f.params)));
	const J = skeleton.jointCount;
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
	continuous(rotations, N, J);
	smoothRows(rotations, N, J * 4, smooth);
	normalizeAll(rotations);
	smoothRows(root, N, 3, smooth);

	// The feet, every frame, and at rest standing on y = 0.
	const feet = ['l_foot', 'l_ball', 'r_foot', 'r_ball'].map((n) => skeleton.index(n));
	const pose = (f) => forward(skeleton, (j) => Array.from(rotations.subarray((f * J + j) * 4, (f * J + j + 1) * 4)), [root[f * 3], root[f * 3 + 1], root[f * 3 + 2]]).positions;
	const rest = forward(skeleton, () => [0, 0, 0, 1], skeleton.restRoot).positions;
	const standing = feet.map((j) => rest[j][1]);
	const positions = Array.from({ length: N }, (_, f) => { const p = pose(f); return feet.map((j) => p[j]); });

	// The floor: under the lowest foot joint, less how high that joint stands.
	const lowest = positions.map((p) => Math.min(...p.map((q, i) => q[1] - standing[i])));
	const floor = percentile(lowest, FLOOR_PERCENTILE);
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
	};
}
