// SAM 3D Body's hands (`SAM3DBody.run_inference`, inference_type="full"):
// after the body pass, a crop around each hand goes through the backbone and
// the hand decoder, which only knows right hands, so the left hand's crop is
// cut from the image mirrored left to right and its answer mirrored back.
// Each hand is then checked; the good ones' wrists (from the hand decoder)
// and elbows (from the body's) prompt the body decoder once more, and their
// finger angles, wrist turn, hand scale and hand shape go into its answer.
//
// A hand is good when
//   1. its wrist turn, as the hand decoder has it, is within 1.4 rad of the
//      body decoder's;
//   2. its box is over 64 pixels;
//   3. all its crop's keypoints are inside that crop;
//   4. its 2D wrist is within a quarter box of the body's.
//
// Sides are [left, right] throughout, as the reference has them.

import { boxToCrop, warpCrop, normalize, rayFeatures, cliffCondition, SIZE } from './crop.js';
import { matmul3, matmul3T, projectAt } from './heads.js';
import { quatToMatrix } from './mhr.js';

export const WRIST_ANGLE = 1.4;
const HAND_PADDING = 0.9;
const MIN_BOX = 64;
const WRIST_DISTANCE = 0.25;
// Keypoints: the right hand's wrist (in either hand's crop, which holds a
// right hand), the left's in the body's, the elbows.
const RIGHT_WRIST = 41, LEFT_WRIST = 62, RIGHT_ELBOW = 8, LEFT_ELBOW = 7;
// MHR joints [left, right]: lower arm, wrist twist, wrist.
const LOWARM = [76, 40], TWIST = [77, 41], WRIST = [78, 42];
// The wrist's body-pose angles [left, right], in X, Z, Y order.
const WRIST_POSE = [[41, 43, 42], [31, 33, 32]];
// The hand-only scales and identity components, shared by the two hands.
const HAND_SCALES = 18, HAND_SHAPE = 40;
const LEFT_SCALE = 9, RIGHT_SCALE = 8;

const rotX = (a) => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
const rotY = (a) => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
const rotZ = (a) => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];

// roma's "XZY" (intrinsic): R = Rx(a) Rz(b) Ry(c), from and to (a, b, c).
export function eulerXzyToMatrix([a, b, c]) {
	return matmul3(matmul3(rotX(a), rotZ(b)), rotY(c));
}
export function matrixToEulerXzy(R) {
	return [Math.atan2(R[7], R[4]), Math.atan2(-R[1], Math.hypot(R[0], R[2])), Math.atan2(R[2], R[0])];
}

// rotation_angle_difference: the angle of A B^T.
export function angleBetween(A, B) {
	let trace = 0;
	for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) trace += A[r * 3 + k] * B[r * 3 + k];
	return Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2)));
}

// fix_wrist_euler: the wrist's X, Z, Y angles, or their other solution
// (x + pi, -(z + pi), y + pi) where it breaks the joint's limits less.
export function fixWrist([x, z, y]) {
	const wrap = (v) => Math.atan2(Math.sin(v), Math.cos(v));
	const violation = (v, lo, hi) => Math.max(lo - v, 0) ** 2 + Math.max(v - hi, 0) ** 2;
	const cost = (a, b, c) => violation(a, -2.2, 1.0) + violation(b, -2.2, 1.5) + violation(c, -1.2, 1.5);
	const alt = [wrap(x + Math.PI), wrap(-(z + Math.PI)), wrap(y + Math.PI)];
	return cost(...alt) < cost(x, z, y) ? alt : [x, z, y];
}

const joint = (rots, j) => Array.from(rots.slice(j * 9, j * 9 + 9));

// The body pass's two hand boxes [left, right] in the image, x1 y1 x2 y2:
// the hand-box tokens' (centre, size in the crop, 0 .. 1) made square and
// taken back through the crop's warp (`_get_hand_box`).
export function handBoxes(handBox, crop) {
	const a = crop.affine[0];
	return [0, 1].map((h) => {
		const [cx, cy, w, ht] = handBox.slice(h * 4, h * 4 + 4).map((v) => v * SIZE);
		const s = Math.max(w, ht) / a;
		const x = (cx - crop.affine[2]) / a, y = (cy - crop.affine[5]) / a;
		return [x - s / 2, y - s / 2, x + s / 2, y + s / 2];
	});
}

// A hand crop's inputs: `box` in the image the hand is cut from, the image
// mirrored left to right with `mirror` (the box is in the mirrored image).
export function handInputs(img, box, cam, { mirror = false } = {}) {
	const crop = boxToCrop(box, { padding: HAND_PADDING });
	return { crop, view: { crop, cam }, pixels: normalize(warpCrop(img, crop.affine, { mirror })), rays: rayFeatures(crop.affine, cam), condition: cliffCondition(crop, cam) };
}

// The left hand's pose, decoded from the mirrored image as a right hand, as
// a left hand: the right hand's scale, turn and finger angles moved over.
// `heads` has the scale PCA. In place.
export function unmirrorLeft(pose, heads) {
	const S = heads.scaleMean.length;
	const mean = (i) => heads.scaleMean[i], std = (i) => heads.scaleComps[i * S + i];
	pose.scale[LEFT_SCALE] = (mean(RIGHT_SCALE) + std(RIGHT_SCALE) * pose.scale[RIGHT_SCALE] - mean(LEFT_SCALE)) / std(LEFT_SCALE);
	const R = joint(pose.joint_global_rots, WRIST[1]);
	for (let k = 3; k < 9; k++) R[k] = -R[k];
	pose.joint_global_rots.splice(WRIST[0] * 9, 9, ...R);
	pose.hand.set(pose.hand.slice(54, 108), 0);
	return pose;
}

// The wrists' turns relative to their twist joints [left, right] that would
// give the hands' global wrist turns, for a body posed with these joint
// rotations, and the zero-wrist frames they are relative to.
function wristTurns(bodyRots, jointRotation, left, right) {
	const hands = [joint(left.joint_global_rots, WRIST[0]), joint(right.joint_global_rots, WRIST[1])];
	return [0, 1].map((s) => {
		const zero = matmul3(joint(bodyRots, LOWARM[s]), Array.from(jointRotation.slice(TWIST[s] * 9, TWIST[s] * 9 + 9)));
		return matmul3T(zero, hands[s]);
	});
}

// The four checks for each hand [left, right]. `body` is the body pass's
// pose; `left` (already unmirrored) and `right` the hands', `crops` their
// crops [left, right]; `width` the image's.
export function checkHands(body, left, right, crops, width, jointRotation) {
	const original = WRIST_POSE.map((idx) => eulerXzyToMatrix(idx.map((i) => body.body_pose[i])));
	const fused = wristTurns(body.joint_global_rots, jointRotation, left, right);
	const angle = [0, 1].map((s) => angleBetween(original[s], fused[s]));
	const bigEnough = crops.map((c) => c.scale[0] > MIN_BOX && c.scale[1] > MIN_BOX);
	const inside = [left, right].map((p) => p.pred_keypoints_2d_cropped.every((v) => Math.abs(v) < 0.5));
	const at = (p, k) => [p.pred_keypoints_2d[k * 2], p.pred_keypoints_2d[k * 2 + 1]];
	const leftWrist = at(left, RIGHT_WRIST);
	leftWrist[0] = width - leftWrist[0] - 1;
	const rightWrist = at(right, RIGHT_WRIST);
	const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
	// Each measured against the other hand's box, as the reference does.
	const near = [
		dist(leftWrist, at(body, LEFT_WRIST)) / crops[1].scale[0] < WRIST_DISTANCE,
		dist(rightWrist, at(body, RIGHT_WRIST)) / crops[0].scale[0] < WRIST_DISTANCE,
	];
	const good = [0, 1].map((s) => angle[s] < WRIST_ANGLE && bigEnough[s] && inside[s] && near[s]);
	return { good, angle, bigEnough, inside, near, original, wrists: { left: leftWrist, right: rightWrist } };
}

// The second body pass's prompts [[x, y, label]] in the body crop (0 .. 1):
// the good hands' wrists and their elbows, those inside the crop.
export function handPrompts(body, check, view) {
	const A = view.crop.affine, size = view.crop.size;
	const toCrop = ([u, v]) => [(A[0] * u + A[1] * v + A[2]) / size - 0.5, (A[3] * u + A[4] * v + A[5]) / size - 0.5];
	const at = (k) => [body.pred_keypoints_2d[k * 2], body.pred_keypoints_2d[k * 2 + 1]];
	const points = [
		[check.wrists.right, RIGHT_WRIST, 1], [check.wrists.left, LEFT_WRIST, 0],
		[at(RIGHT_ELBOW), RIGHT_ELBOW, 1], [at(LEFT_ELBOW), LEFT_ELBOW, 0],
	];
	const out = [];
	for (const [p, label, side] of points) {
		const c = toCrop(p);
		if (!check.good[side] || !c.every((v) => v > -0.5 && v < 0.5)) continue;
		out.push([...c.map((v) => Math.min(1, Math.max(0, v + 0.5))), label]);
	}
	return out;
}

// The previous estimate a prompted pass refines [522]: pose raw, identity,
// scale, hands, face (zero), camera.
export function previousEstimate(pose) {
	return [...pose.pred_pose_raw, ...pose.shape, ...pose.scale, ...pose.hand, ...new Array(72).fill(0), ...pose.pred_cam];
}

// The good hands put into the body's pose `body` (the prompted pass's, or the
// first one's without prompts): finger angles, the wrist turn that gives the
// hand decoder's wrist, the hand's scale; the shared hand scales and hand
// identity averaged over the good hands. `original` is the first pass's
// wrist turns (checkHands). Returns the merged pose, posed again.
export function mergeHands(heads, body, left, right, check, view, jointRotation, { allVertices = false } = {}) {
	const hand = Float64Array.from([...left.hand.slice(0, 54), ...right.hand.slice(54, 108)]);
	const scale = Float64Array.from(body.scale), shape = Float64Array.from(body.shape);
	scale[LEFT_SCALE] = left.scale[LEFT_SCALE];
	scale[RIGHT_SCALE] = right.scale[RIGHT_SCALE];
	for (let i = HAND_SCALES; i < scale.length; i++) scale[i] = (left.scale[i] + right.scale[i]) / 2;
	for (let i = HAND_SHAPE; i < shape.length; i++) shape[i] = (left.shape[i] + right.shape[i]) / 2;
	// The wrist turns, on the body as it would be with these hands.
	const posed = heads.pose(heads.modelParams(body.global_rot, body.body_pose, hand, scale), shape);
	const rots = posed.rig.skeleton.flatMap((g) => quatToMatrix(g.q));
	const fused = wristTurns(rots, jointRotation, left, right);
	const good = [0, 1].map((s) => check.good[s] && angleBetween(check.original[s], fused[s]) < WRIST_ANGLE);

	const out = { ...body, body_pose: Float64Array.from(body.body_pose), hand: Float64Array.from(body.hand), scale: Float64Array.from(body.scale), shape: Float64Array.from(body.shape) };
	for (const s of [0, 1]) {
		if (!good[s]) continue;
		const xzy = fixWrist(matrixToEulerXzy(fused[s]));
		WRIST_POSE[s].forEach((i, k) => { out.body_pose[i] = xzy[k]; });
		out.hand.set(hand.subarray(s * 54, s * 54 + 54), s * 54);
		const i = s === 0 ? LEFT_SCALE : RIGHT_SCALE;
		out.scale[i] = scale[i];
	}
	const n = good.filter(Boolean).length;
	if (n) {
		const w = good.map((g) => (g ? 1 : 0));
		for (let i = HAND_SCALES; i < scale.length; i++) out.scale[i] = (left.scale[i] * w[0] + right.scale[i] * w[1]) / (n + 1e-8);
		for (let i = HAND_SHAPE; i < shape.length; i++) out.shape[i] = (left.shape[i] * w[0] + right.shape[i] * w[1]) / (n + 1e-8);
	}

	const params = heads.modelParams(out.global_rot, out.body_pose, out.hand, out.scale);
	const rig = heads.pose(params, out.shape, { allVertices });
	const flip = (a) => { for (let i = 0; i < a.length; i += 3) { a[i + 1] = -a[i + 1]; a[i + 2] = -a[i + 2]; } return a; };
	out.mhr_model_params = params;
	out.pred_keypoints_3d = flip(rig.keypoints);
	out.pred_joint_coords = flip(Float64Array.from(rig.rig.joints));
	out.joint_global_rots = rig.rig.skeleton.flatMap((g) => quatToMatrix(g.q));
	out.pred_vertices = rig.vertices ? flip(Float64Array.from(rig.vertices)) : null;
	// The reference zeroes the raw pose: it no longer matches.
	out.pred_pose_raw = new Array(body.pred_pose_raw.length).fill(0);
	Object.assign(out, projectAt(out.pred_keypoints_3d, out.pred_cam_t, view));
	out.hands_used = good;
	return out;
}
