// SAM 3D Body's body heads, on the host: the pose token in, a posed MHR rig,
// its 70 keypoints and their projection into the image out
// (`heads/mhr_head.py`, `heads/camera_head.py`, `modules/mhr_utils.py`,
// `SAM3DBody.camera_project`). Everything after the two MLPs is a few thousand
// numbers, in float64.
//
// The pose head's 519 outputs, added to the learned initial pose:
//   6 global rotation (6D), 260 body pose (23 3-DoF joints as 6D, 58 1-DoF
//   angles as sin/cos, 6 translations), 45 identity, 28 scale PCA, 2 x 54
//   hand pose PCA, 72 face (multiplied by zero).
// The camera head's 3 (s, tx, ty) give a perspective translation.
//
// The hand heads (`head_pose_hand`, `head_camera_hand`, MHRHead with
// enable_hand_model) are the same with their own weights, for a crop around a
// right hand: the root rotation is the right wrist's, turned to the body's
// root and moved so that wrist stays put; everything but the root and the
// hands is zeroed, and every keypoint but the right hand's 21; the camera's
// scale is ten times the body's (DEFAULT_SCALE_FACTOR_HAND).

import { f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';
import { JOINTS, sixDFromXyz, quatToMatrix } from './mhr.js';

export const KEYPOINTS = 70;
export const POSE_DIM = 519;
const BODY_CONT = 260;

// fmt: from mhr_utils.compact_cont_to_model_params_body.
const BODY_3DOF = [[0, 2, 4], [6, 8, 10], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26], [27, 28, 29], [34, 35, 36], [37, 38, 39], [44, 45, 46], [53, 54, 55], [64, 65, 66], [85, 69, 73], [86, 70, 79], [87, 71, 82], [88, 72, 76], [91, 92, 93], [112, 96, 100], [113, 97, 106], [114, 98, 109], [115, 99, 103], [130, 131, 132]];
const BODY_1DOF = [1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123];
const BODY_TRANS = [124, 125, 126, 127, 128, 129];
// The hand angles among the body's 133, which the body head leaves to the
// hand PCA (mhr_param_hand_idxs).
const HAND_PARAMS = Array.from({ length: 54 }, (_, i) => 62 + i);
// Degrees of freedom of a hand's 16 joints, in order (27 angles).
const HAND_DOFS = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1];

const normalize3 = (v) => {
	const n = Math.max(Math.hypot(v[0], v[1], v[2]), 1e-12);
	return [v[0] / n, v[1] / n, v[2] / n];
};
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// rot6d_to_rotmat: columns b1, b2, b3 (Gram-Schmidt), row-major [9].
export function rot6dToMatrix(p) {
	const b1 = normalize3([p[0], p[1], p[2]]);
	const d = b1[0] * p[3] + b1[1] * p[4] + b1[2] * p[5];
	const b2 = normalize3([p[3] - d * b1[0], p[4] - d * b1[1], p[5] - d * b1[2]]);
	const b3 = cross(b1, b2);
	return [b1[0], b2[0], b3[0], b1[1], b2[1], b3[1], b1[2], b2[2], b3[2]];
}

// roma.rotmat_to_euler("ZYX", R): R = Rz(a) Ry(b) Rx(c), returned (a, b, c).
export function matrixToEulerZYX(R) {
	return [Math.atan2(R[3], R[0]), Math.atan2(-R[6], Math.hypot(R[0], R[3])), Math.atan2(R[7], R[8])];
}

// roma's "xyz" (extrinsic, MHR's own order): R = Rz(c) Ry(b) Rx(a) from and
// to (a, b, c).
export function eulerXyzToMatrix([a, b, c]) {
	const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
	return [
		cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa,
		sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa,
		-sb, cb * sa, cb * ca,
	];
}
export function matrixToEulerXyz(R) {
	return [Math.atan2(R[7], R[8]), Math.atan2(-R[6], Math.hypot(R[0], R[3])), Math.atan2(R[3], R[0])];
}

// Row-major 3 x 3 products: A B, and A^T B.
export function matmul3(A, B) {
	const C = new Array(9);
	for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
	return C;
}
export function matmul3T(A, B) {
	const C = new Array(9);
	for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[r * 3 + c] = A[r] * B[c] + A[3 + r] * B[3 + c] + A[6 + r] * B[6 + c];
	return C;
}

// batchXYZfrom6D: the XYZ Euler angles (R = Rz Ry Rx) of a 6D rotation whose
// two columns are p[0..3] and p[3..6], Gram-Schmidt the other way round.
export function xyzFrom6d(p) {
	const x = normalize3([p[0], p[1], p[2]]);
	const z = normalize3(cross(x, [p[3], p[4], p[5]]));
	const y = cross(z, x);
	// Columns x, y, z: m[r][c].
	const m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
	const sy = Math.sqrt(m[0][0] * m[0][0] + m[1][0] * m[1][0]);
	if (sy < 1e-6) return [Math.atan2(-m[1][2], m[1][1]), Math.atan2(-m[2][0], sy), 0];
	return [Math.atan2(m[2][1], m[2][2]), Math.atan2(-m[2][0], sy), Math.atan2(m[1][0], m[0][0])];
}

// compact_cont_to_model_params_body: 260 -> 133.
export function bodyParamsFromCont(c) {
	const out = new Float64Array(133);
	let p = 0;
	for (const idx of BODY_3DOF) {
		const e = xyzFrom6d(c.slice(p, p + 6));
		p += 6;
		for (let k = 0; k < 3; k++) out[idx[k]] = e[k];
	}
	for (const i of BODY_1DOF) { out[i] = Math.atan2(c[p], c[p + 1]); p += 2; }
	for (const i of BODY_TRANS) out[i] = c[p++];
	return out;
}

// compact_model_params_to_cont_body: 133 -> 260, the inverse.
export function bodyContFromParams(q) {
	const out = [];
	for (const idx of BODY_3DOF) out.push(...sixDFromXyz(q[idx[0]], q[idx[1]], q[idx[2]]));
	for (const i of BODY_1DOF) out.push(Math.sin(q[i]), Math.cos(q[i]));
	for (const i of BODY_TRANS) out.push(q[i]);
	return out;
}

// compact_cont_to_model_params_hand: 54 -> 27, joint by joint.
export function handParamsFromCont(c) {
	const out = [];
	let p = 0;
	for (const k of HAND_DOFS) {
		if (k === 3) { out.push(...xyzFrom6d(c.slice(p, p + 6))); p += 6; }
		else for (let i = 0; i < k; i++) { out.push(Math.atan2(c[p], c[p + 1])); p += 2; }
	}
	return out;
}

// x @ W^T + b on the host; W [out, in] row-major.
function linear(W, b, x, out, inDim) {
	const y = new Float64Array(out);
	for (let o = 0; o < out; o++) {
		let s = b ? b[o] : 0;
		const r = o * inDim;
		for (let i = 0; i < inDim; i++) s += W[r + i] * x[i];
		y[o] = s;
	}
	return y;
}
const relu = (v) => v.map((x) => (x > 0 ? x : 0));

// An FFN(num_fcs=2) or MLP head: Linear, ReLU, Linear (and so on).
function mlp(layers, x) {
	let h = x;
	layers.forEach((l, i) => {
		h = linear(l.w, l.b, h, l.out, l.in);
		if (i < layers.length - 1) h = relu(h);
	});
	return h;
}

// MLPs over one input vector on the GPU, all of them in one submit: a pose
// head is a few million multiply-adds, which the host is slow at.
class GpuMlps {
	constructor(mlps) {
		const up = (data) => { const t = f32(data.length); t.buffer.write(data); return t; };
		const dim = mlps[0][0].in;
		this.x = f32(dim);
		this.mlps = mlps.map((layers) => layers.map((l) => ({ ...l, wt: up(l.w), bt: up(l.b), y: f32(l.out) })));
	}
	run(x) {
		this.x.buffer.write(x instanceof Float32Array ? x : Float32Array.from(x));
		for (const layers of this.mlps) {
			let h = this.x;
			layers.forEach((l, i) => {
				op.matmul(l.wt, h, l.y, l.out, l.in, 1);
				op.biasAdd(l.y, l.bt, l.out, l.out);
				if (i < layers.length - 1) op.relu(l.y, l.out);
				h = l.y;
			});
		}
		submit();
		return this.mlps.map((layers) => {
			const l = layers[layers.length - 1];
			return Float64Array.from(new Float32Array(l.y.buffer.readBytes(0, l.out * 4).buffer));
		});
	}
	dispose() {
		this.x.dispose();
		for (const layers of this.mlps) for (const l of layers) { l.wt.dispose(); l.bt.dispose(); l.y.dispose(); }
	}
}

export class BodyHeads {
	// `m` is the open checkpoint, `mhr` an MHR. `gpu` runs the head MLPs (the
	// hand boxes' too) and the keypoint vertices' rest shape on the GPU (the
	// host is ten times slower at them); the host path gives the same numbers
	// to float32.
	// `hand` makes the hand heads.
	constructor(m, mhr, { gpu = true, hand = false } = {}) {
		this.mhr = mhr;
		this.hand = hand;
		const f = (name) => m.floats(name);
		const layer = (p) => {
			const s = m.shape(`${p}.weight`);
			return { w: f(`${p}.weight`), b: f(`${p}.bias`), in: s[0], out: s[1] };
		};
		const own = hand ? '_hand' : '';
		this.poseMlp = [layer(`head_pose${own}.proj.layers.0.0`), layer(`head_pose${own}.proj.layers.1`)];
		this.cameraMlp = [layer(`head_camera${own}.proj.layers.0.0`), layer(`head_camera${own}.proj.layers.1`)];
		this.initPose = f(`init_pose${own}.weight`);
		this.initCamera = f(`init_camera${own}.weight`);
		this.cameraScale = hand ? 10 : 1;
		if (hand) {
			this.rightWrist = f('head_pose_hand.right_wrist_coords');
			this.root = f('head_pose_hand.root_coords');
			this.wristToWorld = Array.from(f('head_pose_hand.local_to_world_wrist'));
			this.nonHand = m.ints('head_pose_hand.nonhand_param_idxs');
		} else {
			this.handBox = [layer('bbox_embed.layers.0'), layer('bbox_embed.layers.1'), layer('bbox_embed.layers.2')];
			this.handCls = [layer('hand_cls_embed')];
		}
		this.scaleMean = f('head_pose.scale_mean');
		this.scaleComps = f('head_pose.scale_comps');
		this.handMean = f('head_pose.hand_pose_mean');
		this.handComps = f('head_pose.hand_pose_comps');
		this.handLeft = m.ints('head_pose.hand_joint_idxs_left');
		this.handRight = m.ints('head_pose.hand_joint_idxs_right');
		// The 70 keypoints SAM 3D Body uses, of the mapping's 308: each a
		// weighted sum of vertices and joints. Only those vertices are skinned.
		const mapping = f('head_pose.keypoint_mapping');
		const cols = m.shape('head_pose.keypoint_mapping')[0];
		const V = mhr.vertexCount;
		const used = new Set();
		this.keypointTerms = [];
		for (let k = 0; k < KEYPOINTS; k++) {
			const terms = [];
			for (let c = 0; c < cols; c++) {
				const w = mapping[k * cols + c];
				if (w === 0) continue;
				terms.push(c, w);
				if (c < V) used.add(c);
			}
			this.keypointTerms.push(terms);
		}
		this.keypointVertices = [...used].sort((a, b) => a - b);
		if (gpu) {
			this.gpu = new GpuMlps([this.poseMlp, this.cameraMlp]);
			if (!hand) this.handGpu = new GpuMlps([this.handBox, this.handCls]);
			// The body and hand heads skin the same vertices: one list on the GPU.
			const g = mhr.gpu?.list;
			if (g && g.length === this.keypointVertices.length && g.every((v, i) => v === this.keypointVertices[i])) this.keypointVertices = g;
			else mhr.offload(this.keypointVertices);
		}
		this.vertexSlot = new Map(this.keypointVertices.map((v, i) => [v, i]));
	}

	dispose() {
		this.gpu?.dispose();
		this.handGpu?.dispose();
		this.gpu = this.handGpu = null;
	}

	// mhr_forward's model parameters [204]: the root translation (none without
	// `trans`, metres, which MHR takes as tenths), the root rotation, the body
	// angles with the hands' put in from their PCA, and the 68 scales from
	// theirs.
	modelParams(globalRot, body, hand, scale, trans = null) {
		const scales = Float64Array.from(this.scaleMean);
		for (let c = 0; c < scale.length; c++) for (let i = 0; i < scales.length; i++) scales[i] += scale[c] * this.scaleComps[c * scales.length + i];
		const params = new Float64Array(204);
		if (trans) for (let k = 0; k < 3; k++) params[k] = trans[k] * 10;
		params.set(globalRot, 3);
		params.set(body.subarray(0, 130), 6);
		for (const [side, idx] of [[0, this.handLeft], [1, this.handRight]]) {
			const cont = Float64Array.from(this.handMean);
			for (let a = 0; a < 54; a++) {
				const h = hand[side * 54 + a];
				if (h !== 0) for (let b = 0; b < 54; b++) cont[b] += h * this.handComps[a * 54 + b];
			}
			handParamsFromCont(cont).forEach((v, i) => { params[idx[i]] = v; });
		}
		params.set(scales, 136);
		return params;
	}

	// The rig posed by model parameters, and its 70 keypoints [70 * 3] from
	// its vertices and joints (MHR's axes, metres). `allVertices` skins the
	// whole mesh as well.
	pose(params, shape, { allVertices = false } = {}) {
		const rig = this.mhr.forward(params, shape, { vertices: allVertices ? null : this.keypointVertices });
		const V = this.mhr.vertexCount;
		const at = (c, k) => {
			if (c >= V) return rig.joints[(c - V) * 3 + k];
			return rig.vertices[(allVertices ? c : this.vertexSlot.get(c)) * 3 + k];
		};
		const keypoints = new Float64Array(KEYPOINTS * 3);
		for (let n = 0; n < KEYPOINTS; n++) {
			const t = this.keypointTerms[n];
			for (let k = 0; k < 3; k++) {
				let s = 0;
				for (let i = 0; i < t.length; i += 2) s += t[i + 1] * at(t[i], k);
				keypoints[n * 3 + k] = s;
			}
		}
		return { rig, keypoints, vertices: allVertices ? rig.vertices : null };
	}

	// The pose token [1024] (after the decoder's final norm) to a pose: what
	// MHRHead.forward, PerspectiveHead.forward and camera_project return, with
	// SAM 3D Body's names. `view` is { crop, cam }.
	decode(token, view, { allVertices = false } = {}) {
		const [p0, c0] = this.gpu ? this.gpu.run(token) : [mlp(this.poseMlp, token), mlp(this.cameraMlp, token)];
		const pred = p0.map((v, i) => v + this.initPose[i]);
		const cam = c0.map((v, i) => v + this.initCamera[i]);
		const rot6d = Array.from(pred.subarray(0, 6));
		const globalRot = matrixToEulerZYX(rot6dToMatrix(rot6d));
		const cont = Array.from(pred.subarray(6, 6 + BODY_CONT));
		const body = bodyParamsFromCont(cont);
		for (const i of HAND_PARAMS) body[i] = 0;
		body[130] = body[131] = body[132] = 0;
		let p = 6 + BODY_CONT;
		const shape = pred.slice(p, p + 45); p += 45;
		const scale = pred.slice(p, p + 28); p += 28;
		const hand = pred.slice(p, p + 108);
		let params;
		if (this.hand) {
			// The right wrist's turn to the root's, about the wrist.
			const R = matmul3(eulerXyzToMatrix(globalRot), this.wristToWorld);
			const d = [0, 1, 2].map((k) => this.rightWrist[k] - this.root[k]);
			const trans = [0, 1, 2].map((r) => -(R[r * 3] * d[0] + R[r * 3 + 1] * d[1] + R[r * 3 + 2] * d[2] + this.root[r]));
			params = this.modelParams(matrixToEulerXyz(R), body, hand, scale, trans);
			for (const i of this.nonHand) params[i] = 0;
		} else {
			params = this.modelParams(globalRot, body, hand, scale);
		}
		const mhr = this.pose(params, shape, { allVertices });
		if (this.hand) {
			mhr.keypoints.fill(0, 0, 21 * 3);
			mhr.keypoints.fill(0, 42 * 3);
		}
		// The camera's y and z point the other way from MHR's.
		const flip = (a) => { for (let i = 0; i < a.length; i += 3) { a[i + 1] = -a[i + 1]; a[i + 2] = -a[i + 2]; } return a; };
		const kp3d = flip(mhr.keypoints);
		const joints = flip(Float64Array.from(mhr.rig.joints));
		const out = {
			pred_pose_raw: [...rot6d, ...cont],
			global_rot: globalRot,
			body_pose: body,
			shape, scale, hand,
			pred_cam: cam,
			mhr_model_params: params,
			pred_joint_coords: joints,
			joint_global_rots: mhr.rig.skeleton.flatMap((g) => quatToMatrix(g.q)),
			pred_keypoints_3d: kp3d,
			pred_vertices: mhr.vertices ? flip(Float64Array.from(mhr.vertices)) : null,
		};
		Object.assign(out, project(kp3d, cam, view, this.cameraScale));
		return out;
	}

	// The two hand-box tokens [2 * 1024]: boxes (x1, y1, w, h in the crop, 0..1)
	// and left/right logits.
	handBoxes(tokens) {
		const boxes = [], logits = [];
		for (let h = 0; h < 2; h++) {
			const t = tokens.subarray(h * 1024, (h + 1) * 1024);
			const [box, cls] = this.handGpu ? this.handGpu.run(t) : [mlp(this.handBox, t), mlp(this.handCls, t)];
			boxes.push(...box.map((v) => 1 / (1 + Math.exp(-v))));
			logits.push(...cls);
		}
		return { boxes, logits };
	}
}

// PerspectiveHead.perspective_projection (USE_INTRIN_CENTER) and
// _full_to_crop: the camera translation from (s, tx, ty) and the box, the
// keypoints in the image, in the crop ([-0.5, 0.5]) and their depths.
export function project(points, predCam, { crop, cam }, scaleFactor = 1) {
	const [f, fy, ppx, ppy] = cam;
	const s = -predCam[0], tx = predCam[1], ty = -predCam[2];
	const bs = crop.scale[0] * s * scaleFactor + 1e-8;
	const t = [tx + (2 * (crop.center[0] - ppx)) / bs, ty + (2 * (crop.center[1] - ppy)) / bs, (2 * f) / bs];
	return { ...projectAt(points, t, { crop, cam }), pred_cam_t: t };
}

// Points [n * 3] moved by the camera translation `t` into the image, the crop
// and their depths.
export function projectAt(points, t, { crop, cam }) {
	const [f, fy, ppx, ppy] = cam;
	const n = points.length / 3;
	const kp2d = new Float64Array(n * 2), cropped = new Float64Array(n * 2), depth = new Float64Array(n);
	const A = crop.affine;
	for (let i = 0; i < n; i++) {
		const x = points[i * 3] + t[0], y = points[i * 3 + 1] + t[1], z = points[i * 3 + 2] + t[2];
		const u = f * (x / z) + ppx, v = fy * (y / z) + ppy;
		kp2d[i * 2] = u;
		kp2d[i * 2 + 1] = v;
		cropped[i * 2] = (A[0] * u + A[1] * v + A[2]) / crop.size - 0.5;
		cropped[i * 2 + 1] = (A[3] * u + A[4] * v + A[5]) / crop.size - 0.5;
		depth[i] = z;
	}
	return { focal_length: f, pred_keypoints_2d: kp2d, pred_keypoints_2d_cropped: cropped, pred_keypoints_2d_depth: depth };
}

export { JOINTS };
