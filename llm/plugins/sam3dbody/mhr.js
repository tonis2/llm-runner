// MHR, Meta's Momentum Human Rig, as `assets/mhr_model.pt` (TorchScript,
// Apache-2.0) has it: 127 joints, 18439 vertices, 45 identity blend shapes and
// a pose-corrective MLP. Ported from that file's own code (pymomentum's
// `torch/character.py`, `skel_state.py`, `quaternion.py`, `mhr/mhr.py`), on
// the host in float64, but for the rest shape's large products, which go to
// the GPU in float32 (see offload and correctives).
//
//   model parameters [204]  (3 root translation, 3 root rotation, 130 body
//                            and hand angles, 68 scales)
//     -> joint parameters [127, 7] = parameter_transform @ [params, 0 x 45]
//     -> each joint's local (t, q, s): t = offset + jp[0..3], q = prerotation
//        * euler_xyz(jp[3..6]), s = 2^jp[6]; then down the parents
//     -> vertices: base shape + identity blend + pose correctives, skinned
//
// Lengths are in the file's centimetres until `forward` divides by 100, as
// SAM 3D Body's `mhr_forward` does. Face expressions are left out: SAM 3D Body
// multiplies its face prediction by zero.
//
// The skin can be computed for some vertices only (`vertices`): the keypoints
// SAM 3D Body feeds back between decoder layers read about 500 of them, and
// the corrective MLP's output layer (3000 x 55317, 660 MB) is read a row per
// vertex coordinate that is wanted: those 500's once, onto the GPU; the whole
// mesh's a chunk at a time.

import { llm, f32 } from '/core/llm.js';
import { openTorch } from '/core/weights/torch.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';

export const JOINTS = 127;
export const MODEL_PARAMS = 204;
export const IDENTITY = 45;

const CORRECTIVE_W1 = 'pose_correctives_model.pose_dirs_predictor.0.';
const CORRECTIVE_W2 = 'pose_correctives_model.pose_dirs_predictor.2.weight';

// Quaternions are (x, y, z, w), as pymomentum's.
function qmul(a, b, out, o = 0) {
	const [x1, y1, z1, w1] = a, [x2, y2, z2, w2] = b;
	out[o] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
	out[o + 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
	out[o + 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
	out[o + 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
	return out;
}

function qnormalize(q) {
	const n = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-12);
	return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

// v + 2 (w (a x v) + a x (a x v)), q normalised.
function rotate(q, v) {
	const [x, y, z, w] = q;
	const ax = y * v[2] - z * v[1], ay = z * v[0] - x * v[2], az = x * v[1] - y * v[0];
	const bx = y * az - z * ay, by = z * ax - x * az, bz = x * ay - y * ax;
	return [v[0] + 2 * (w * ax + bx), v[1] + 2 * (w * ay + by), v[2] + 2 * (w * az + bz)];
}

// R = Rz(z) Ry(y) Rx(x), as pymomentum's euler_xyz_to_quaternion.
function eulerXyzQuat(rx, ry, rz) {
	const cy = Math.cos(rz * 0.5), sy = Math.sin(rz * 0.5);
	const cp = Math.cos(ry * 0.5), sp = Math.sin(ry * 0.5);
	const cr = Math.cos(rx * 0.5), sr = Math.sin(rx * 0.5);
	return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy];
}

// A skeleton state (t, q, s) composed with another: s1 (s2 applied first).
function compose(t1, q1, s1, t2, q2, s2) {
	q1 = qnormalize(q1);
	q2 = qnormalize(q2);
	const r = rotate(q1, t2);
	return { t: [t1[0] + s1 * r[0], t1[1] + s1 * r[1], t1[2] + s1 * r[2]], q: qmul(q1, q2, [0, 0, 0, 0]), s: s1 * s2 };
}

// The first two columns of R = Rz Ry Rx, as batch6DFromXYZ returns them.
export function sixDFromXyz(x, y, z) {
	const cx = Math.cos(x), cy = Math.cos(y), cz = Math.cos(z);
	const sx = Math.sin(x), sy = Math.sin(y), sz = Math.sin(z);
	return [cy * cz, cy * sz, -sy, -cx * sz + sx * sy * cz, cx * cz + sx * sy * sz, sx * cy];
}

export function quatToMatrix(q) {
	const [x, y, z, w] = q;
	return [
		1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
		2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
		2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
	];
}

export class MHR {
	// `path` is assets/mhr_model.pt.
	constructor(path) {
		const t0 = llm.now();
		const m = openTorch(path, { select: (v) => v });
		this.m = m;
		const f = (name) => m.floats(name);
		this.parents = m.ints('character_torch.skeleton.joint_parents');
		// The names are the TorchScript module's own attribute, not a tensor.
		const at = (v, k) => (v instanceof Map ? v.get(k) : v?.[k]);
		const names = ['state', 'character_torch', 'state', 'skeleton', 'state', 'joint_names', 'args', 0].reduce(at, m.pickle);
		this.names = Array.isArray(names) && names.length === JOINTS ? names : Array.from({ length: JOINTS }, (_, j) => `joint${j}`);
		this.offsets = f('character_torch.skeleton.joint_translation_offsets');
		this.prerotations = f('character_torch.skeleton.joint_prerotations');
		// [889, 249], almost all zeros: kept as each row's (column, value) pairs.
		const pt = f('character_torch.parameter_transform.parameter_transform');
		const cols = pt.length / (JOINTS * 7);
		this.transform = [];
		for (let r = 0; r < JOINTS * 7; r++) {
			const row = [];
			for (let c = 0; c < cols; c++) if (pt[r * cols + c] !== 0) row.push(c, pt[r * cols + c]);
			this.transform.push(row);
		}
		this.baseShape = f('character_torch.blend_shape.base_shape');
		this.shapeVectors = f('character_torch.blend_shape.shape_vectors');
		this.vertexCount = this.baseShape.length / 3;
		this.inverseBind = f('character_torch.linear_blend_skinning.inverse_bind_pose');
		// Each vertex's (joint, weight) pairs.
		const si = m.ints('character_torch.linear_blend_skinning.skin_indices_flattened');
		const sw = f('character_torch.linear_blend_skinning.skin_weights_flattened');
		const vi = m.ints('character_torch.linear_blend_skinning.vert_indices_flattened');
		this.skin = Array.from({ length: this.vertexCount }, () => []);
		for (let i = 0; i < si.length; i++) this.skin[vi[i]].push(si[i], sw[i]);
		// The corrective MLP's first layer, sparse [3000, 750]: per hidden unit.
		const idx = m.ints(`${CORRECTIVE_W1}sparse_indices`);
		const w = f(`${CORRECTIVE_W1}sparse_weight`);
		const nnz = w.length;
		this.hidden = m.torch.get(CORRECTIVE_W2).shape[1];
		this.w1 = Array.from({ length: this.hidden }, () => []);
		for (let i = 0; i < nnz; i++) this.w1[idx[i]].push(idx[nnz + i], w[i]);
		// Rows of the output layer, read as they are first wanted.
		this.w2Rows = new Map();
		llm.print(`  MHR: ${JOINTS} joints, ${this.vertexCount} vertices, read in ${llm.since(t0)}`);
	}

	close() {
		this.m.close();
		if (this.gpu) for (const t of [this.gpu.w1, this.gpu.w, this.gpu.feat, this.gpu.x, this.gpu.y]) t.dispose();
		this.gpu = null;
	}

	// The rest shape of `vertices` on the GPU, for a vertex list that is
	// skinned again and again (each decoder layer's keypoints): the corrective
	// MLP, its sparse first layer made dense, and each vertex coordinate's
	// output row with its identity blend shapes and base shape as further
	// columns (the input ending in the identity and a 1). One pass there in
	// place of 9000 multiply-adds a vertex here. `forward` uses it when it is
	// given this same list.
	offload(vertices) {
		const H = this.hidden, F = (JOINTS - 2) * 6, n = vertices.length, V = this.vertexCount, cols = H + IDENTITY + 1;
		const w = new Float32Array(3 * n * cols);
		vertices.forEach((v, i) => {
			const rows = this.correctiveRows(v);
			for (let k = 0; k < 3; k++) {
				const r = (3 * i + k) * cols;
				w.set(rows.subarray(k * H, (k + 1) * H), r);
				for (let c = 0; c < IDENTITY; c++) w[r + H + c] = this.shapeVectors[(c * V + v) * 3 + k];
				w[r + H + IDENTITY] = this.baseShape[v * 3 + k];
			}
		});
		const w1 = new Float32Array(H * F);
		this.w1.forEach((row, u) => { for (let i = 0; i < row.length; i += 2) w1[u * F + row[i]] += row[i + 1]; });
		const up = (data, shape) => { const t = f32(data.length, shape); t.buffer.write(data); return t; };
		this.gpu = { list: vertices, w1: up(w1, [F, H]), w: up(w, [cols, 3 * n]), feat: f32(F), x: f32(cols), y: f32(3 * n) };
	}

	// The offloaded list's rest positions [3 * count] from the corrective
	// features and the identity.
	restOnGpu(feat, identity) {
		const g = this.gpu, H = this.hidden, cols = H + IDENTITY + 1, n = g.list.length;
		const tail = new Float32Array(IDENTITY + 1);
		for (let c = 0; c < IDENTITY; c++) tail[c] = identity[c];
		tail[IDENTITY] = 1;
		g.feat.buffer.write(Float32Array.from(feat));
		g.x.buffer.write(tail, H * 4);
		const hidden = g.x.view(0, H * 4);
		op.matmul(g.w1, g.feat, hidden, H, feat.length, 1);
		op.relu(hidden, H);
		op.matmul(g.w, g.x, g.y, 3 * n, cols, 1);
		submit();
		return new Float32Array(g.y.buffer.readBytes(0, n * 12).buffer);
	}

	// The rest positions [3 * count] of `list` before correctives: the base
	// shape and the identity blend.
	restShape(list, identity) {
		const V = this.vertexCount, out = new Float64Array(list.length * 3);
		list.forEach((v, n) => {
			for (let k = 0; k < 3; k++) {
				let s = this.baseShape[v * 3 + k];
				for (let i = 0; i < IDENTITY; i++) s += identity[i] * this.shapeVectors[(i * V + v) * 3 + k];
				out[n * 3 + k] = s;
			}
		});
		return out;
	}

	// Each vertex's corrective offset [3 * count] from the hidden layer. A long
	// list (the mesh) goes to the GPU a chunk of output rows at a time, read
	// and not kept: all of them are 660 MB.
	correctives(hidden, list) {
		const H = this.hidden, out = new Float64Array(list.length * 3);
		if (list.length >= 2048) {
			const CHUNK = 4096;
			const x = f32(H), w = f32(3 * CHUNK * H, [H, 3 * CHUNK]), y = f32(3 * CHUNK);
			try {
				x.buffer.write(Float32Array.from(hidden));
				const rows = new Float32Array(3 * CHUNK * H);
				for (let s = 0; s < list.length; s += CHUNK) {
					const n = Math.min(CHUNK, list.length - s);
					// A run of consecutive vertices is one read.
					for (let i = 0; i < n;) {
						let j = i + 1;
						while (j < n && list[s + j] === list[s + j - 1] + 1) j++;
						rows.set(this.m.floatRange(CORRECTIVE_W2, 3 * list[s + i] * H, 3 * (j - i) * H), 3 * i * H);
						i = j;
					}
					w.buffer.write(rows.subarray(0, 3 * n * H));
					op.matmul(w, x, y, 3 * n, H, 1);
					submit();
					out.set(new Float32Array(y.buffer.readBytes(0, n * 12).buffer), 3 * s);
				}
			} finally {
				x.dispose();
				w.dispose();
				y.dispose();
			}
			return out;
		}
		list.forEach((v, n) => {
			const rows = this.correctiveRows(v);
			for (let k = 0; k < 3; k++) {
				let s = 0;
				for (let u = 0; u < H; u++) s += rows[k * H + u] * hidden[u];
				out[n * 3 + k] = s;
			}
		});
		return out;
	}

	// The mesh's triangles, [36874 * 3] vertex indices.
	faces() { return this.m.ints('character_torch.mesh.faces'); }

	// The three output rows of vertex v, [3 * hidden].
	correctiveRows(v) {
		let r = this.w2Rows.get(v);
		if (!r) {
			r = this.m.floatRange(CORRECTIVE_W2, 3 * v * this.hidden, 3 * this.hidden);
			this.w2Rows.set(v, r);
		}
		return r;
	}

	// Joint parameters [127 * 7] from model parameters [204] (the identity
	// coefficients' 45 columns of the transform are multiplied by zero, as
	// MHRDemo does).
	jointParameters(params) {
		const jp = new Float64Array(JOINTS * 7);
		for (let r = 0; r < jp.length; r++) {
			const row = this.transform[r];
			let s = 0;
			for (let i = 0; i < row.length; i += 2) if (row[i] < params.length) s += row[i + 1] * params[row[i]];
			jp[r] = s;
		}
		return jp;
	}

	// Every joint's world (t, q, s), centimetres.
	skeleton(jp) {
		const out = [];
		for (let j = 0; j < JOINTS; j++) {
			const p = j * 7;
			const t = [jp[p] + this.offsets[j * 3], jp[p + 1] + this.offsets[j * 3 + 1], jp[p + 2] + this.offsets[j * 3 + 2]];
			const pre = Array.from(this.prerotations.subarray(j * 4, j * 4 + 4));
			const q = qmul(pre, eulerXyzQuat(jp[p + 3], jp[p + 4], jp[p + 5]), [0, 0, 0, 0]);
			const s = Math.exp(jp[p + 6] * 0.6931471824645996);
			const par = this.parents[j];
			out.push(par < 0 ? { t, q, s } : (() => { const g = out[par]; return compose(g.t, g.q, g.s, t, q, s); })());
		}
		return out;
	}

	// The corrective MLP's input [750] from the joint parameters: joints 2..
	// as 6D rotations less the identity's.
	correctiveFeatures(jp) {
		const feat = new Float64Array((JOINTS - 2) * 6);
		for (let j = 2; j < JOINTS; j++) {
			const d = sixDFromXyz(jp[j * 7 + 3], jp[j * 7 + 4], jp[j * 7 + 5]);
			d[0] -= 1;
			d[4] -= 1;
			feat.set(d, (j - 2) * 6);
		}
		return feat;
	}

	// The corrective MLP's hidden layer, [3000]: the sparse layer, ReLU.
	correctiveHidden(jp) {
		const feat = this.correctiveFeatures(jp);
		const h = new Float64Array(this.hidden);
		for (let u = 0; u < this.hidden; u++) {
			const row = this.w1[u];
			let s = 0;
			for (let i = 0; i < row.length; i += 2) s += row[i + 1] * feat[row[i]];
			h[u] = s > 0 ? s : 0;
		}
		return h;
	}

	// `params` [204] model parameters, `identity` [45]. Returns the joints'
	// world states and positions [127 * 3] and the skinned positions
	// [count * 3] of `vertices` (indices; all of them when null), in metres.
	forward(params, identity, { vertices = null, correctives = true } = {}) {
		const jp = this.jointParameters(params);
		const skel = this.skeleton(jp);
		const joints = new Float64Array(JOINTS * 3);
		skel.forEach((g, j) => { for (let k = 0; k < 3; k++) joints[j * 3 + k] = g.t[k] / 100; });
		// Each joint's state times its inverse bind pose, what skinning applies,
		// as its scaled rotation matrix and translation [12].
		const bound = new Float64Array(JOINTS * 12);
		skel.forEach((g, j) => {
			const b = this.inverseBind.subarray(j * 8, j * 8 + 8);
			const c = compose(g.t, g.q, g.s, [b[0], b[1], b[2]], [b[3], b[4], b[5], b[6]], b[7]);
			const R = quatToMatrix(qnormalize(c.q));
			for (let i = 0; i < 9; i++) bound[j * 12 + i] = c.s * R[i];
			bound.set(c.t, j * 12 + 9);
		});
		const list = vertices ?? Array.from({ length: this.vertexCount }, (_, i) => i);
		let rest;
		if (correctives && this.gpu?.list === list) rest = this.restOnGpu(this.correctiveFeatures(jp), identity);
		else {
			rest = this.restShape(list, identity);
			if (correctives) {
				const offsets = this.correctives(this.correctiveHidden(jp), list);
				for (let i = 0; i < rest.length; i++) rest[i] += offsets[i];
			}
		}
		const out = new Float64Array(list.length * 3);
		for (let n = 0; n < list.length; n++) {
			const rx = rest[n * 3], ry = rest[n * 3 + 1], rz = rest[n * 3 + 2];
			const sk = this.skin[list[n]];
			let x = 0, y = 0, z = 0;
			for (let i = 0; i < sk.length; i += 2) {
				const b = sk[i] * 12, w = sk[i + 1];
				x += w * (bound[b] * rx + bound[b + 1] * ry + bound[b + 2] * rz + bound[b + 9]);
				y += w * (bound[b + 3] * rx + bound[b + 4] * ry + bound[b + 5] * rz + bound[b + 10]);
				z += w * (bound[b + 6] * rx + bound[b + 7] * ry + bound[b + 8] * rz + bound[b + 11]);
			}
			out[n * 3] = x / 100;
			out[n * 3 + 1] = y / 100;
			out[n * 3 + 2] = z / 100;
		}
		return { skeleton: skel, joints, vertices: out };
	}
}
