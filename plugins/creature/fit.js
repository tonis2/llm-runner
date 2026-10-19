// A rig posed to match 2D tracks: every frame's joint turns, the root's place
// and one orthographic camera fitted together, on the host, with gradients
// written out by hand.
//
//   const fit = fitRig(rig, tracks, { markers });   // rig.js, track.js
//   fit.theta, fit.root, fit.camera, fit.report
//
// The rig is posed by forward kinematics from its rest pose (rest turns all
// identity, as Rigsmith stores rigs), so its bone lengths hold exactly and a
// joint nobody labelled follows its chain. Each tracked point is a joint, or a
// marker: a point on a joint's segment at an offset the fit learns (a snout,
// an eye, the skin over a thigh). The losses, after the fit this replaces
// (fit.py of the T-rex test):
//
//   reprojection  each point to its nearest hypothesis, robustly (Cauchy,
//                 15 px), softly nearest at first and sharply by the end; a
//                 hypothesis a point is far from in every frame costs little
//   smoothness    the joints' and markers' accelerations, and the turns'
//   twist         a one-child joint's turn about its own bone, which no point
//                 sees, held near none
//   rest          the turns near rest: weakly where points see them, firmly
//                 where none do (a hand nobody labelled)
//   chain         neighbouring one-child joints bend alike (spines, tails)
//   floor         each frame's lowest foot on y = 0, and none below it
//   in place      the root stays put (the camera follows the animal); its
//                 travel is put back afterwards from the planted feet
//   camera        yaw and pitch once, pitch near `pitch` degrees; its scale and
//                 aim on screen smooth from frame to frame
//
// Gradients: a turn δ of joint a's rotation vector θ turns everything below a
// about it by ω = G_parent J_l(θ) δ in the world, moving a point x by
// ω × (x - p_a), so dL/dθ_a = J_l(θ)ᵀ G_parentᵀ Σ (x - p_a) × dL/dx over the
// points below a: one pass up the tree with each subtree's Σ g and Σ x × g.

import { llm } from '../lib/llm.js';
import { bad } from './request.js';

const SIGMA2 = 15 * 15;
const DEG = Math.PI / 180;
const W_PLANE = 10, W_MIDLINE = 0.5;

// Row-major R (9 at `o`) from the rotation vector (x, y, z).
function rodrigues(x, y, z, R, o) {
	const t2 = x * x + y * y + z * z, t = Math.sqrt(t2);
	let a, b;
	if (t < 1e-6) { a = 1 - t2 / 6; b = 0.5 - t2 / 24; } else { a = Math.sin(t) / t; b = (1 - Math.cos(t)) / t2; }
	R[o] = 1 - b * (y * y + z * z); R[o + 1] = b * x * y - a * z; R[o + 2] = b * x * z + a * y;
	R[o + 3] = b * x * y + a * z; R[o + 4] = 1 - b * (x * x + z * z); R[o + 5] = b * y * z - a * x;
	R[o + 6] = b * x * z - a * y; R[o + 7] = b * y * z + a * x; R[o + 8] = 1 - b * (x * x + y * y);
}

// J_l(θ)ᵀ g into out[0..2]: g - a θ × g + b θ × (θ × g).
function leftJacobianT(x, y, z, g0, g1, g2, out) {
	const t2 = x * x + y * y + z * z, t = Math.sqrt(t2);
	let a, b;
	if (t < 1e-6) { a = 0.5 - t2 / 24; b = 1 / 6 - t2 / 120; } else { a = (1 - Math.cos(t)) / t2; b = (t - Math.sin(t)) / (t2 * t); }
	const c0 = y * g2 - z * g1, c1 = z * g0 - x * g2, c2 = x * g1 - y * g0;
	const d0 = y * c2 - z * c1, d1 = z * c0 - x * c2, d2 = x * c1 - y * c0;
	out[0] = g0 - a * c0 + b * d0;
	out[1] = g1 - a * c1 + b * d1;
	out[2] = g2 - a * c2 + b * d2;
}

// Adam over named Float64Arrays, each with its own learning rate.
class Adam {
	constructor(groups) {
		this.groups = groups.map((g) => ({ ...g, m: new Float64Array(g.value.length), v: new Float64Array(g.value.length) }));
		this.t = 0;
	}

	step(scale) {
		this.t++;
		const c1 = 1 - 0.9 ** this.t, c2 = 1 - 0.999 ** this.t;
		for (const g of this.groups) {
			const { value, grad, m, v } = g, lr = g.lr * scale;
			for (let i = 0; i < value.length; i++) {
				const d = grad[i];
				m[i] = 0.9 * m[i] + 0.1 * d;
				v[i] = 0.999 * v[i] + 0.001 * d * d;
				value[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
			}
		}
	}
}

// The problem: the rig, the points and their hypotheses over T frames.
export class RigFit {
	// `obs` { T, points: [{ name, joint, marker }], markers: count, x, y, w }
	// with x, y, w [T, P, H] (y up, w 0 for none).
	constructor(rig, obs, { pitch = 8, k0 = 1000 } = {}) {
		const J = rig.names.length;
		this.rig = rig;
		this.J = J;
		this.T = obs.T;
		this.P = obs.points.length;
		this.H = obs.H;
		this.M = obs.markers;
		this.obs = obs;
		this.pitchPrior = pitch * DEG;
		this.k0 = k0;
		const T = this.T;
		this.parents = Int32Array.from(rig.parents);
		this.offsets = new Float64Array(J * 3);
		for (let j = 0; j < J; j++) {
			const p = rig.parents[j];
			for (let d = 0; d < 3; d++) this.offsets[j * 3 + d] = p < 0 ? 0 : rig.positions[j][d] - rig.positions[p][d];
		}
		this.rootRest = rig.positions[0].slice();
		this.prepareStructure();

		// Parameters and their gradients.
		this.theta = new Float64Array(T * J * 3);
		this.root = new Float64Array(T * 3);
		for (let t = 0; t < T; t++) this.root.set(this.rootRest, t * 3);
		this.marks = new Float64Array(this.M * 3);
		this.logS = new Float64Array(T).fill(Math.log(k0));
		this.shift = new Float64Array(T * 2);
		this.cam = new Float64Array([0, this.pitchPrior]);   // yaw, pitch
		this.g = {
			theta: new Float64Array(T * J * 3), root: new Float64Array(T * 3), marks: new Float64Array(this.M * 3),
			logS: new Float64Array(T), shift: new Float64Array(T * 2), cam: new Float64Array(2),
		};
		// Forward kinematics' results.
		this.R = new Float64Array(T * J * 9);
		this.G = new Float64Array(T * J * 9);
		this.pos = new Float64Array(T * J * 3);
		this.mpos = new Float64Array(T * this.M * 3);
		this.gPos = new Float64Array(T * J * 3);
		this.gM = new Float64Array(T * this.M * 3);
		this.rootTau = new Float64Array(T * 3);   // the root's turn's own share (its frame in the lateral terms)
		this.weights = { reproj: 1, acc: 2, thetaAcc: 2, twist: 1, rest: 0.0005, oneMarker: 0.05, unseen: 1, chain: 0.1, floor: 2, inPlace: 1, marks: 0.001, follow: 1, level: 1e4, lateral: 1, heading: 1 };
	}

	// Which joints the points see, each joint's reach (the farthest rest point
	// below it), the one-child joints' bone directions and the feet.
	prepareStructure() {
		const { rig, J } = this;
		const children = Array.from({ length: J }, () => []);
		for (let j = 1; j < J; j++) children[this.parents[j]].push(j);
		this.children = children;
		// How well the points see a joint's turn: 2 with a point below it (a
		// joint or a marker further down) or two markers of its own, 1 with one
		// marker of its own alone (its fitted offset trades with the turn: a jaw
		// with one point on it), 0 with none (a hand nobody labelled).
		const below = new Int32Array(J), own = new Int32Array(J);
		for (const p of this.obs.points) {
			if (p.marker >= 0) own[p.joint]++;
			else if (p.joint > 0) below[this.parents[p.joint]]++;
		}
		for (let j = J - 1; j > 0; j--) below[this.parents[j]] += below[j] + own[j];
		this.seenBelow = new Uint8Array(J);
		for (let j = 0; j < J; j++) this.seenBelow[j] = below[j] > 0 || own[j] > 1 ? 2 : own[j] === 1 ? 1 : 0;
		this.reach = new Float64Array(J);
		for (let j = J - 1; j >= 0; j--) {
			let r = 0;
			for (const c of children[j]) {
				const o = this.offsets.subarray(c * 3, c * 3 + 3);
				r = Math.max(r, Math.hypot(o[0], o[1], o[2]) + this.reach[c]);
			}
			this.reach[j] = r;
		}
		this.boneDir = new Float64Array(J * 3);
		this.oneChild = new Uint8Array(J);
		for (let j = 0; j < J; j++) {
			if (children[j].length !== 1) continue;
			const c = children[j][0], o = this.offsets.subarray(c * 3, c * 3 + 3), n = Math.hypot(o[0], o[1], o[2]);
			if (n < 1e-9) continue;
			this.oneChild[j] = 1;
			for (let d = 0; d < 3; d++) this.boneDir[j * 3 + d] = o[d] / n;
		}
		// Chains that bend evenly: a one-child joint and its one-child child
		// whose bones run within 35 degrees of each other at rest (a spine, a
		// tail), not a limb's zigzag of knee and ankle.
		this.chainChild = new Int32Array(J).fill(-1);
		for (let j = 0; j < J; j++) {
			if (!this.oneChild[j]) continue;
			const c = children[j][0];
			if (!this.oneChild[c]) continue;
			const b = this.boneDir;
			if (b[j * 3] * b[c * 3] + b[j * 3 + 1] * b[c * 3 + 1] + b[j * 3 + 2] * b[c * 3 + 2] > Math.cos(35 * DEG)) this.chainChild[j] = c;
		}
		// Feet: the leaves well below the root, and their parents.
		const rootY = rig.positions[0][1], lowest = Math.min(...rig.positions.map((p) => p[1]));
		const feet = new Set();
		for (let j = 0; j < J; j++) {
			if (children[j].length === 0 && rig.positions[j][1] < lowest + 0.3 * (rootY - lowest)) {
				feet.add(j);
				if (this.parents[j] > 0) feet.add(this.parents[j]);
			}
		}
		this.feet = Int32Array.from([...feet]);
		this.footLeaves = [...feet].filter((j) => children[j].length === 0);

		// Sideways offsets in the root's frame (x is across the body: rest turns
		// are identity and the rig faces +Z), which one view does not see: each
		// leg's joints held at their rest offsets from its hip (a leg swings in
		// its own plane; the chain from a foot up to where it branches off), and
		// the middle joints (no side in their names) on the root's midline -
		// firmly near the root, letting go toward the tail's tip and the head,
		// whatever curve the model was made in.
		const lateral = [];
		for (const leaf of this.footLeaves) {
			const chain = [leaf];
			let j = leaf;
			while (this.parents[j] > 0 && children[this.parents[j]].length === 1) { j = this.parents[j]; chain.push(j); }
			const hip = j === leaf ? this.parents[leaf] : j;
			for (const c of chain) if (c !== hip) lateral.push([c, hip, rig.positions[c][0] - rig.positions[hip][0], W_PLANE]);
		}
		const side = (n) => /(_L|_R|\.L|\.R|_l|_r|Left|Right|left|right)$/.test(n) || /^(l_|r_|L_|R_|left|right)/.test(n);
		const along = new Float64Array(J);
		for (let j = 1; j < J; j++) along[j] = along[this.parents[j]] + Math.hypot(...this.offsets.subarray(j * 3, j * 3 + 3));
		const middle = [];
		for (let j = 1; j < J; j++) if (!side(rig.names[j])) middle.push(j);
		const far = Math.max(1e-9, ...middle.map((j) => along[j]));
		for (const j of middle) lateral.push([j, 0, 0, W_MIDLINE * (1 - along[j] / far) ** 2]);
		this.lateral = lateral;
	}

	forward() {
		const { T, J, theta, R, G, pos, parents, offsets, root } = this;
		for (let t = 0; t < T; t++) {
			for (let j = 0; j < J; j++) {
				const tj = t * J + j, o9 = tj * 9, o3 = tj * 3;
				rodrigues(theta[o3], theta[o3 + 1], theta[o3 + 2], R, o9);
				const p = parents[j];
				if (p < 0) {
					for (let k = 0; k < 9; k++) G[o9 + k] = R[o9 + k];
					pos[o3] = root[t * 3]; pos[o3 + 1] = root[t * 3 + 1]; pos[o3 + 2] = root[t * 3 + 2];
					continue;
				}
				const q9 = (t * J + p) * 9, q3 = (t * J + p) * 3;
				for (let r = 0; r < 3; r++) {
					const g0 = G[q9 + r * 3], g1 = G[q9 + r * 3 + 1], g2 = G[q9 + r * 3 + 2];
					G[o9 + r * 3] = g0 * R[o9] + g1 * R[o9 + 3] + g2 * R[o9 + 6];
					G[o9 + r * 3 + 1] = g0 * R[o9 + 1] + g1 * R[o9 + 4] + g2 * R[o9 + 7];
					G[o9 + r * 3 + 2] = g0 * R[o9 + 2] + g1 * R[o9 + 5] + g2 * R[o9 + 8];
					pos[o3 + r] = pos[q3 + r] + g0 * offsets[j * 3] + g1 * offsets[j * 3 + 1] + g2 * offsets[j * 3 + 2];
				}
			}
			for (const p of this.obs.points) {
				if (p.marker < 0) continue;
				const g9 = (t * J + p.joint) * 9, j3 = (t * J + p.joint) * 3, m3 = p.marker * 3, x3 = (t * this.M + p.marker) * 3;
				for (let r = 0; r < 3; r++) {
					this.mpos[x3 + r] = pos[j3 + r] + G[g9 + r * 3] * this.marks[m3] + G[g9 + r * 3 + 1] * this.marks[m3 + 1] + G[g9 + r * 3 + 2] * this.marks[m3 + 2];
				}
			}
		}
	}

	// A point's world position array and offset.
	where(t, p) {
		return p.marker < 0 ? [this.pos, (t * this.J + p.joint) * 3, this.gPos] : [this.mpos, (t * this.M + p.marker) * 3, this.gM];
	}

	// The loss, and its gradient into this.g. `tau` the soft-min's temperature.
	evaluate(tau) {
		const { T, J, P, H, obs, g, k0 } = this, W = this.weights;
		this.forward();
		for (const a of Object.values(g)) a.fill(0);
		this.gPos.fill(0);
		this.gM.fill(0);
		this.rootTau.fill(0);
		const terms = {};
		const yaw = this.cam[0], pitch = this.cam[1];
		const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
		const r0 = [cy, 0, sy], r1 = [sp * sy, cp, -sp * cy];
		const dy0 = [-sy, 0, cy], dy1 = [sp * cy, 0, sp * sy], dp1 = [cp * sy, -sp, -cp * cy];

		// Reprojection.
		let reproj = 0, count = 0;
		const norm = 1 / (T * P);
		const a = new Float64Array(H), e2 = new Float64Array(H);
		for (let t = 0; t < T; t++) {
			const s = Math.exp(this.logS[t]), sx = this.shift[t * 2], syy = this.shift[t * 2 + 1];
			for (let pi = 0; pi < P; pi++) {
				const p = obs.points[pi];
				const [X, o, gX] = this.where(t, p);
				const x0 = X[o], x1 = X[o + 1], x2 = X[o + 2];
				const q0 = r0[0] * x0 + r0[2] * x2, q1 = r1[0] * x0 + r1[1] * x1 + r1[2] * x2;
				const u0 = s * q0 + sx, u1 = s * q1 + syy;
				let best = -Infinity, any = false;
				const base = (t * P + pi) * H;
				for (let h = 0; h < H; h++) {
					const w = obs.w[base + h];
					if (!(w > 0)) { a[h] = -Infinity; continue; }
					any = true;
					const d0 = u0 - obs.x[base + h], d1 = u1 - obs.y[base + h];
					e2[h] = d0 * d0 + d1 * d1;
					a[h] = (-SIGMA2 * Math.log1p(e2[h] / SIGMA2)) / tau + Math.log(w);
					if (a[h] > best) best = a[h];
				}
				if (!any) continue;
				let z = 0;
				for (let h = 0; h < H; h++) if (a[h] > -Infinity) z += Math.exp(a[h] - best);
				reproj += -tau * (best + Math.log(z)) * norm;
				count++;
				let du0 = 0, du1 = 0;
				for (let h = 0; h < H; h++) {
					if (a[h] === -Infinity) continue;
					const wh = Math.exp(a[h] - best) / z;
					const k = (wh * 2) / (1 + e2[h] / SIGMA2) * norm * W.reproj;
					du0 += k * (u0 - obs.x[base + h]);
					du1 += k * (u1 - obs.y[base + h]);
				}
				g.shift[t * 2] += du0;
				g.shift[t * 2 + 1] += du1;
				g.logS[t] += du0 * s * q0 + du1 * s * q1;
				g.cam[0] += s * (du0 * (dy0[0] * x0 + dy0[2] * x2) + du1 * (dy1[0] * x0 + dy1[2] * x2));
				g.cam[1] += s * du1 * (dp1[0] * x0 + dp1[1] * x1 + dp1[2] * x2);
				gX[o] += s * (du0 * r0[0] + du1 * r1[0]);
				gX[o + 1] += s * du1 * r1[1];
				gX[o + 2] += s * (du0 * r0[2] + du1 * r1[2]);
			}
		}
		terms.reproj = reproj * W.reproj;

		// Accelerations of the joints and markers, in pixels.
		if (T > 2) {
			const pts = J + this.M;
			const c = (W.acc * k0 * k0) / ((T - 2) * pts);
			let acc = 0;
			for (const [X, gX, n] of [[this.pos, this.gPos, J], [this.mpos, this.gM, this.M]]) {
				for (let t = 1; t < T - 1; t++) {
					for (let i = 0; i < n * 3; i++) {
						const A = X[(t + 1) * n * 3 + i] - 2 * X[t * n * 3 + i] + X[(t - 1) * n * 3 + i];
						acc += c * A * A;
						gX[(t + 1) * n * 3 + i] += 2 * c * A;
						gX[t * n * 3 + i] -= 4 * c * A;
						gX[(t - 1) * n * 3 + i] += 2 * c * A;
					}
				}
			}
			terms.acc = acc;
		}

		// The floor: each frame's lowest foot point at 0 (softly lowest), none below.
		if (this.feet.length) {
			const F = this.feet.length, tf = 3;
			let floor = 0;
			const ys = new Float64Array(F);
			for (let t = 0; t < T; t++) {
				let m = Infinity;
				for (let f = 0; f < F; f++) { ys[f] = this.pos[(t * J + this.feet[f]) * 3 + 1] * k0; if (ys[f] < m) m = ys[f]; }
				let z = 0;
				for (let f = 0; f < F; f++) z += Math.exp(-(ys[f] - m) / tf);
				const low = m - tf * Math.log(z);
				floor += (W.floor * low * low) / T;
				for (let f = 0; f < F; f++) {
					const wf = Math.exp(-(ys[f] - m) / tf) / z;
					let d = (W.floor * 2 * low * wf) / T;
					if (ys[f] < 0) {
						floor += (W.floor * ys[f] * ys[f]) / (T * F);
						d += (W.floor * 2 * ys[f]) / (T * F);
					}
					this.gPos[(t * J + this.feet[f]) * 3 + 1] += d * k0;
				}
			}
			terms.floor = floor;
		}

		// In place: the root's spread about its mean (and that mean at rest).
		{
			const mean = [0, 0, 0];
			for (let t = 0; t < T; t++) for (let d = 0; d < 3; d++) mean[d] += this.root[t * 3 + d] / T;
			let ip = 0;
			const wd = [1, 0.1, 1];
			for (let t = 0; t < T; t++) {
				for (let d = 0; d < 3; d++) {
					const v = (this.root[t * 3 + d] - mean[d]) * k0;
					ip += (W.inPlace * wd[d] * v * v) / T;
					g.root[t * 3 + d] += ((W.inPlace * wd[d] * 2 * v) / T) * k0;   // the mean's own share sums to zero
				}
			}
			for (const d of [0, 2]) {
				const v = (mean[d] - this.rootRest[d]) * k0;
				ip += W.inPlace * v * v;
				for (let t = 0; t < T; t++) g.root[t * 3 + d] += ((W.inPlace * 2 * v) / T) * k0;
			}
			terms.inPlace = ip;
		}

		// Markers stay near their joints (weakly: their offsets are what is fitted).
		{
			let mk = 0;
			for (let i = 0; i < this.marks.length; i++) {
				const v = this.marks[i] * k0;
				mk += (W.marks * v * v) / Math.max(1, this.M);
				g.marks[i] += ((W.marks * 2 * v) / Math.max(1, this.M)) * k0;
			}
			terms.marks = mk;
		}

		// Sideways offsets in the root's frame: lat = (p_j - p_anchor) . a, a the
		// root's x axis; a turn ω of the root moves a by ω × a.
		if (this.lateral.length) {
			let lat = 0;
			const n = this.lateral.length, scale = (W.lateral * k0 * k0) / (T * n);
			for (let t = 0; t < T; t++) {
				const g9 = t * J * 9, a0 = this.G[g9], a1 = this.G[g9 + 3], a2 = this.G[g9 + 6];
				for (const [j, anchor, rest, w] of this.lateral) {
					const o = (t * J + j) * 3, q = (t * J + anchor) * 3;
					const d0 = this.pos[o] - this.pos[q], d1 = this.pos[o + 1] - this.pos[q + 1], d2 = this.pos[o + 2] - this.pos[q + 2];
					const e = d0 * a0 + d1 * a1 + d2 * a2 - rest;
					lat += scale * w * e * e;
					const c = 2 * scale * w * e;
					this.gPos[o] += c * a0; this.gPos[o + 1] += c * a1; this.gPos[o + 2] += c * a2;
					this.gPos[q] -= c * a0; this.gPos[q + 1] -= c * a1; this.gPos[q + 2] -= c * a2;
					this.rootTau[t * 3] += c * (a1 * d2 - a2 * d1);
					this.rootTau[t * 3 + 1] += c * (a2 * d0 - a0 * d2);
					this.rootTau[t * 3 + 2] += c * (a0 * d1 - a1 * d0);
				}
			}
			terms.lateral = lat;
		}

		this.backward();
		this.turnPriors(terms);
		this.cameraPriors(terms);
		let total = 0;
		for (const v of Object.values(terms)) total += v;
		this.terms = terms;
		this.observed = count;
		return total;
	}

	// The joints' and markers' position gradients up the tree.
	backward() {
		const { T, J, M, G, pos, mpos, gPos, gM, parents, theta, g } = this;
		const S = new Float64Array(J * 3), X = new Float64Array(J * 3), out = new Float64Array(3);
		for (let t = 0; t < T; t++) {
			for (let j = 0; j < J; j++) {
				const o = (t * J + j) * 3;
				S[j * 3] = gPos[o]; S[j * 3 + 1] = gPos[o + 1]; S[j * 3 + 2] = gPos[o + 2];
				X[j * 3] = pos[o + 1] * gPos[o + 2] - pos[o + 2] * gPos[o + 1];
				X[j * 3 + 1] = pos[o + 2] * gPos[o] - pos[o] * gPos[o + 2];
				X[j * 3 + 2] = pos[o] * gPos[o + 1] - pos[o + 1] * gPos[o];
			}
			for (const p of this.obs.points) {
				if (p.marker < 0) continue;
				const m = (t * M + p.marker) * 3, j = p.joint, g9 = (t * J + j) * 9;
				const gx = gM[m], gy = gM[m + 1], gz = gM[m + 2], x = mpos[m], y = mpos[m + 1], z = mpos[m + 2];
				S[j * 3] += gx; S[j * 3 + 1] += gy; S[j * 3 + 2] += gz;
				X[j * 3] += y * gz - z * gy; X[j * 3 + 1] += z * gx - x * gz; X[j * 3 + 2] += x * gy - y * gx;
				// The offset, in the joint's frame: Gᵀ g.
				for (let d = 0; d < 3; d++) g.marks[p.marker * 3 + d] += G[g9 + d] * gx + G[g9 + 3 + d] * gy + G[g9 + 6 + d] * gz;
			}
			for (let j = J - 1; j > 0; j--) {
				const p = parents[j];
				for (let d = 0; d < 3; d++) { S[p * 3 + d] += S[j * 3 + d]; X[p * 3 + d] += X[j * 3 + d]; }
			}
			for (let j = 0; j < J; j++) {
				const o = (t * J + j) * 3, s0 = S[j * 3], s1 = S[j * 3 + 1], s2 = S[j * 3 + 2];
				const p0 = pos[o], p1 = pos[o + 1], p2 = pos[o + 2];
				let t0 = X[j * 3] - (p1 * s2 - p2 * s1), t1 = X[j * 3 + 1] - (p2 * s0 - p0 * s2), t2 = X[j * 3 + 2] - (p0 * s1 - p1 * s0);
				if (j === 0) { t0 += this.rootTau[t * 3]; t1 += this.rootTau[t * 3 + 1]; t2 += this.rootTau[t * 3 + 2]; }
				const par = parents[j];
				let l0 = t0, l1 = t1, l2 = t2;
				if (par >= 0) {
					const q9 = (t * J + par) * 9;
					l0 = G[q9] * t0 + G[q9 + 3] * t1 + G[q9 + 6] * t2;
					l1 = G[q9 + 1] * t0 + G[q9 + 4] * t1 + G[q9 + 7] * t2;
					l2 = G[q9 + 2] * t0 + G[q9 + 5] * t1 + G[q9 + 8] * t2;
				} else {
					g.root[t * 3] += s0; g.root[t * 3 + 1] += s1; g.root[t * 3 + 2] += s2;
				}
				leftJacobianT(theta[o], theta[o + 1], theta[o + 2], l0, l1, l2, out);
				g.theta[o] += out[0]; g.theta[o + 1] += out[1]; g.theta[o + 2] += out[2];
			}
		}
	}

	// Twist, rest, chain and the turns' own smoothness, each joint's in pixels
	// (its turn times its reach).
	turnPriors(terms) {
		const { T, J, theta, g, k0 } = this, W = this.weights;
		let twist = 0, rest = 0, chain = 0, tacc = 0;
		const norm = 1 / (T * J);
		for (let j = 0; j < J; j++) {
			const L = Math.max(this.reach[j], 0.05 * this.rig.height) * k0, L2 = L * L;
			const wr = [W.unseen, W.oneMarker, W.rest][this.seenBelow[j]] * L2 * norm;
			const wt = W.twist * L2 * norm;
			const b = this.boneDir.subarray(j * 3, j * 3 + 3);
			const c = this.chainChild[j];
			for (let t = 0; t < T; t++) {
				const o = (t * J + j) * 3;
				for (let d = 0; d < 3; d++) {
					rest += wr * theta[o + d] * theta[o + d];
					g.theta[o + d] += 2 * wr * theta[o + d];
				}
				if (this.oneChild[j]) {
					const tw = theta[o] * b[0] + theta[o + 1] * b[1] + theta[o + 2] * b[2];
					twist += wt * tw * tw;
					for (let d = 0; d < 3; d++) g.theta[o + d] += 2 * wt * tw * b[d];
				}
				if (c >= 0) {
					const wc = W.chain * L2 * norm, q = (t * J + c) * 3;
					for (let d = 0; d < 3; d++) {
						const v = theta[q + d] - theta[o + d];
						chain += wc * v * v;
						g.theta[q + d] += 2 * wc * v;
						g.theta[o + d] -= 2 * wc * v;
					}
				}
			}
			if (T > 2) {
				const wa = (W.thetaAcc * L2) / ((T - 2) * J);
				for (let t = 1; t < T - 1; t++) {
					for (let d = 0; d < 3; d++) {
						const i = (t * J + j) * 3 + d, n = J * 3;
						const A = theta[i + n] - 2 * theta[i] + theta[i - n];
						tacc += wa * A * A;
						g.theta[i + n] += 2 * wa * A;
						g.theta[i] -= 4 * wa * A;
						g.theta[i - n] += 2 * wa * A;
					}
				}
			}
		}
		// Heading: the root's mean turn about the vertical held at none, so the
		// clip faces the rig's own way and the camera takes the view's yaw (the
		// two trade otherwise); the animal may still turn within the clip.
		{
			const L = Math.max(this.reach[0], 0.05 * this.rig.height) * k0;
			let mean = 0;
			for (let t = 0; t < T; t++) mean += theta[t * J * 3 + 1] / T;
			const w = W.heading * L * L;
			Object.assign(terms, { heading: w * mean * mean });
			for (let t = 0; t < T; t++) g.theta[t * J * 3 + 1] += (2 * w * mean) / T;
		}
		Object.assign(terms, { twist, rest, chain, thetaAcc: tacc });
	}

	cameraPriors(terms) {
		const { T, g } = this, W = this.weights;
		const dp = this.cam[1] - this.pitchPrior;
		terms.level = W.level * dp * dp;
		g.cam[1] += 2 * W.level * dp;
		let follow = 0;
		if (T > 2) {
			const cs = (W.follow * 1e6) / (T - 2), ch = W.follow / (T - 2);
			for (let t = 1; t < T - 1; t++) {
				const A = this.logS[t + 1] - 2 * this.logS[t] + this.logS[t - 1];
				follow += cs * A * A;
				g.logS[t + 1] += 2 * cs * A; g.logS[t] -= 4 * cs * A; g.logS[t - 1] += 2 * cs * A;
				for (let d = 0; d < 2; d++) {
					const B = this.shift[(t + 1) * 2 + d] - 2 * this.shift[t * 2 + d] + this.shift[(t - 1) * 2 + d];
					follow += ch * B * B;
					g.shift[(t + 1) * 2 + d] += 2 * ch * B; g.shift[t * 2 + d] -= 4 * ch * B; g.shift[(t - 1) * 2 + d] += 2 * ch * B;
				}
			}
		}
		terms.follow = follow;
	}

	// Projections of every point [T][P][x, y] (y up).
	projections() {
		this.forward();
		const yaw = this.cam[0], pitch = this.cam[1];
		const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
		const out = [];
		for (let t = 0; t < this.T; t++) {
			const s = Math.exp(this.logS[t]), row = [];
			for (const p of this.obs.points) {
				const [X, o] = this.where(t, p);
				row.push([s * (cy * X[o] + sy * X[o + 2]) + this.shift[t * 2], s * (sp * sy * X[o] + cp * X[o + 1] - sp * cy * X[o + 2]) + this.shift[t * 2 + 1]]);
			}
			out.push(row);
		}
		return out;
	}

	// Every joint's projection [T][J][x, y] (y up).
	jointProjections() {
		this.forward();
		const yaw = this.cam[0], pitch = this.cam[1];
		const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
		const out = [];
		for (let t = 0; t < this.T; t++) {
			const s = Math.exp(this.logS[t]), row = [];
			for (let j = 0; j < this.J; j++) {
				const o = (t * this.J + j) * 3, X = this.pos;
				row.push([s * (cy * X[o] + sy * X[o + 2]) + this.shift[t * 2], s * (sp * sy * X[o] + cp * X[o + 1] - sp * cy * X[o + 2]) + this.shift[t * 2 + 1]]);
			}
			out.push(row);
		}
		return out;
	}

	run(iterations, { lr = {}, tau0 = 500, tau1 = 20, onStep = null } = {}) {
		const k0 = this.k0;
		const adam = new Adam([
			{ value: this.theta, grad: this.g.theta, lr: lr.theta ?? 0.01 },
			{ value: this.root, grad: this.g.root, lr: lr.root ?? 1 / k0 },
			{ value: this.marks, grad: this.g.marks, lr: lr.marks ?? 1 / k0 },
			{ value: this.logS, grad: this.g.logS, lr: lr.logS ?? 0.002 },
			{ value: this.shift, grad: this.g.shift, lr: lr.shift ?? 1 },
			{ value: this.cam, grad: this.g.cam, lr: lr.cam ?? 0.003 },
		]);
		let loss = 0;
		for (let it = 0; it < iterations; it++) {
			const tau = tau0 * (tau1 / tau0) ** Math.min(1, it / (0.6 * iterations));
			loss = this.evaluate(tau);
			if (it === iterations - 1) break;
			adam.step(0.5 * (1 + Math.cos((Math.PI * it) / iterations)));
			onStep?.(it + 1, iterations, loss);
		}
		return loss;
	}
}

// The tracks (track.js) as the fit's observations, and the points as joints
// and markers of `rig`.
export function observations(rig, tracks, markers = {}) {
	const joints = new Map(rig.names.map((n, j) => [n, j]));
	const T = tracks.frames, H = tracks.hypotheses.length, P = tracks.points.length;
	const marked = [];
	const points = tracks.points.map((name) => {
		if (joints.has(name)) return { name, joint: joints.get(name), marker: -1 };
		if (!(name in markers)) throw bad(`"${name}" is neither one of the rig's joints nor a marker`);
		if (!joints.has(markers[name])) throw bad(`marker ${name} is on "${markers[name]}", not one of the rig's joints`);
		marked.push(name);
		return { name, joint: joints.get(markers[name]), marker: marked.length - 1 };
	});
	const x = new Float64Array(T * P * H), y = new Float64Array(T * P * H), w = new Float64Array(T * P * H);
	tracks.hypotheses.forEach((hyp, h) => {
		hyp.frames.forEach((f, i) => {
			for (let p = 0; p < P; p++) {
				const q = hyp.tracks[i][p];
				if (!q) continue;
				const k = (f * P + p) * H + h;
				x[k] = q[0];
				y[k] = -q[1];
				w[k] = Math.min(1, Math.max(0.05, ((hyp.scores[i][p] ?? 1) - 0.2) / 0.4));
			}
		});
	});
	return { T, H, points, markers: marked.length, x, y, w };
}

// One frame's observations: the labels of keyframe `k` alone.
function keyframeObservations(obs, tracks, k) {
	const P = obs.points.length;
	const x = new Float64Array(P), y = new Float64Array(P), w = new Float64Array(P);
	const h = tracks.hypotheses.findIndex((hyp) => hyp.from === k);
	const row = tracks.hypotheses[h].frames.indexOf(k);
	for (let p = 0; p < P; p++) {
		const q = tracks.hypotheses[h].tracks[row][p];
		if (!q) continue;
		x[p] = q[0]; y[p] = -q[1]; w[p] = 1;
	}
	return { T: 1, H: 1, points: obs.points, markers: obs.markers, x, y, w };
}

// The centroid of frame t's observations, weighted.
function centroid(obs, t) {
	const P = obs.points.length, H = obs.H;
	let sx = 0, sy = 0, sw = 0;
	for (let i = t * P * H; i < (t + 1) * P * H; i++) {
		if (!(obs.w[i] > 0)) continue;
		sx += obs.w[i] * obs.x[i]; sy += obs.w[i] * obs.y[i]; sw += obs.w[i];
	}
	return sw > 0 ? [sx / sw, sy / sw] : null;
}

/**
 * Fits `rig` (rig.js) to `tracks` (track.js). Returns { problem (the RigFit,
 * solved), camera { yaw, pitch } in degrees, report }. `iterations` of the
 * whole clip after a search over the first keyframe's yaw; `onStep(done,
 * total)` as it goes.
 */
export function fitRig(rig, tracks, { markers = {}, pitch = 8, iterations = 1500, onStep = null } = {}) {
	const t0 = llm.now();
	const obs = observations(rig, tracks, markers);
	const key = Math.min(...tracks.hypotheses.map((h) => h.from));
	const keyObs = keyframeObservations(obs, tracks, key);

	// Pixels per rig unit, roughly: the labels' spread over the rig's.
	const lab = [];
	for (let p = 0; p < keyObs.points.length; p++) if (keyObs.w[p] > 0) lab.push([keyObs.x[p], keyObs.y[p]]);
	const spread = (pts) => Math.hypot(Math.max(...pts.map((q) => q[0])) - Math.min(...pts.map((q) => q[0])), Math.max(...pts.map((q) => q[1])) - Math.min(...pts.map((q) => q[1])));
	const restPts = keyObs.points.filter((p, i) => keyObs.w[i] > 0).map((p) => rig.positions[p.joint]);
	const rest3 = Math.hypot(...[0, 1, 2].map((d) => Math.max(...restPts.map((q) => q[d])) - Math.min(...restPts.map((q) => q[d]))));
	const k0 = spread(lab) / Math.max(1e-6, rest3) * 1.1;

	// The first keyframe alone, from rest, over a ring of yaws: which way the
	// animal faces, and a pose to start every frame from.
	const keyCentre = centroid(keyObs, 0);
	let best = null;
	const initSteps = 400, yaws = 12;
	for (let i = 0; i < yaws; i++) {
		const f = new RigFit(rig, keyObs, { pitch, k0 });
		f.cam[0] = (i * 2 * Math.PI) / yaws;
		f.forward();
		// Aim: the rig's projected centroid on the labels'.
		const proj = f.projections()[0].filter((_, p) => keyObs.w[p] > 0);
		const mx = proj.reduce((s, q) => s + q[0], 0) / proj.length, my = proj.reduce((s, q) => s + q[1], 0) / proj.length;
		f.shift[0] = keyCentre[0] - mx;
		f.shift[1] = keyCentre[1] - my;
		const loss = f.run(initSteps, { tau0: 20, tau1: 20 });
		const err = f.terms.reproj;
		if (!best || err < best.err) best = { f, err, loss, yaw: f.cam[0] };
		onStep?.(i + 1, yaws + iterations / 10);
	}
	llm.print(`  start: frame ${key} alone, yaw ${(best.yaw / DEG).toFixed(0)}° fits best (reprojection ${best.err.toFixed(1)})`);

	// Every frame from that pose, aimed where its points are.
	const fit = new RigFit(rig, obs, { pitch, k0 });
	const T = obs.T, J = rig.names.length;
	const b = best.f;
	for (let t = 0; t < T; t++) {
		fit.theta.set(b.theta.subarray(0, J * 3), t * J * 3);
		fit.root.set(b.root.subarray(0, 3), t * 3);
		fit.logS[t] = b.logS[0];
		const c = centroid(obs, t);
		fit.shift[t * 2] = b.shift[0] + (c ? c[0] - keyCentre[0] : 0);
		fit.shift[t * 2 + 1] = b.shift[1] + (c ? c[1] - keyCentre[1] : 0);
	}
	fit.marks.set(b.marks);
	fit.cam.set(b.cam);
	let lastPrint = llm.now();
	fit.run(iterations, {
		onStep: (done, total, loss) => {
			onStep?.(yaws + done / 10, yaws + total / 10);
			if (llm.now() - lastPrint > 5000) {
				llm.print(`  fit ${done}/${total}: loss ${loss.toFixed(2)}, reprojection ${fit.terms.reproj.toFixed(2)}`);
				lastPrint = llm.now();
			}
		},
	});

	// How well it fits: each observed point's distance to its nearest hypothesis.
	const proj = fit.projections();
	const errs = [];
	for (let t = 0; t < T; t++) {
		for (let p = 0; p < obs.points.length; p++) {
			let e = Infinity;
			for (let h = 0; h < obs.H; h++) {
				const k = (t * obs.points.length + p) * obs.H + h;
				if (obs.w[k] > 0) e = Math.min(e, Math.hypot(proj[t][p][0] - obs.x[k], proj[t][p][1] - obs.y[k]));
			}
			if (e < Infinity) errs.push(e);
		}
	}
	errs.sort((x, y) => x - y);
	const q = (f) => errs[Math.min(errs.length - 1, Math.floor(f * errs.length))];
	const report = {
		reprojection_px: { median: round(q(0.5), 2), p90: round(q(0.9), 2), over_30: round(errs.filter((e) => e > 30).length / errs.length, 3) },
		camera: { yaw: round(fit.cam[0] / DEG, 2), pitch: round(fit.cam[1] / DEG, 2) },
		terms: Object.fromEntries(Object.entries(fit.terms).map(([k, v]) => [k, round(v, 3)])),
		seconds: round((llm.now() - t0) / 1000, 1),
	};
	llm.print(`  fit: reprojection median ${report.reprojection_px.median}px, 90% ${report.reprojection_px.p90}px, ${(report.reprojection_px.over_30 * 100).toFixed(1)}% over 30px; yaw ${report.camera.yaw}°, pitch ${report.camera.pitch}°, in ${report.seconds}s`);
	return { problem: fit, projections: proj, report };
}

const round = (v, d) => Number(v.toFixed(d));
