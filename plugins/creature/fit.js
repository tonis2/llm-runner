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
// an eye, the skin over a thigh), or at one it is given (a point of the mesh a
// finder was trained to find: the nose, a heel), which shows the joint's turn
// about its own bone. The losses, after the fit this replaces
// (fit.py of the T-rex test):
//
//   reprojection  each point to its nearest hypothesis, robustly (Cauchy,
//                 15 px), softly nearest at first and sharply by the end; a
//                 hypothesis a point is far from in every frame costs little;
//                 a found point (a finder's) pulls as hard as it is sure
//   smoothness    the joints' and markers' accelerations, and the turns'
//   twist         a one-child joint's turn about its own bone, which no point
//                 sees, held near none
//   rest          the turns near rest: weakly where points see them, firmly
//                 where none do (a hand nobody labelled)
//   chain         neighbouring joints along a line bend alike (spines, tails,
//                 a chest and the neck that carries on from it)
//   floor         each frame's lowest foot on y = 0, and none below it
//   in place      the root stays put (the camera follows the animal); its
//                 travel is put back afterwards from the planted feet
//   limits        the rig's rotation limits (rig.js), firmly by the end:
//                 Rigsmith's player holds every frame inside them
//   camera        yaw and pitch once, pitch near `pitch` degrees; its scale and
//                 aim on screen smooth from frame to frame
//
// Views: clips of the same moments from other cameras (a front and a side,
// frame N of each the same instant) each bring their own camera and points,
// and all of them project the one motion. What one view cannot see along its
// line of sight (an arm swinging toward it or hanging back) the other sees
// across.
//
// Gradients: a turn δ of joint a's rotation vector θ turns everything below a
// about it by ω = G_parent J_l(θ) δ in the world, moving a point x by
// ω × (x - p_a), so dL/dθ_a = J_l(θ)ᵀ G_parentᵀ Σ (x - p_a) × dL/dx over the
// points below a: one pass up the tree with each subtree's Σ g and Σ x × g.

import { llm } from '../lib/llm.js';
import { bad, markerJoint } from './request.js';

const DEG = Math.PI / 180;
const W_PLANE = 10, W_FOOT = 0.05, W_MIDLINE = 0.5;
const WIDE = 100;

// Every term's weight; a request's `weights` overrides any of them.
export const DEFAULT_WEIGHTS = Object.freeze({
	reproj: 1, acc: 6, thetaAcc: 6, twist: 1, twistMarked: 0, rest: 0.0005, oneMarker: 0.05, unseen: 1, chain: 0.1, chainBranch: 0.1,
	floor: 2, inPlace: 1, marks: 0.001, follow: 1, level: 1e4, lateral: 1, heading: 1, limit: 10,
});

// The rotation vector s of the way from a to b (each 3 at `ao`, `bo`), the short
// way round, into out at `oo`.
function slerpTurn(a, ao, b, bo, s, out, oo) {
	const quat = (v, o) => {
		const t = Math.hypot(v[o], v[o + 1], v[o + 2]), k = t < 1e-12 ? 0.5 : Math.sin(t / 2) / t;
		return [v[o] * k, v[o + 1] * k, v[o + 2] * k, Math.cos(t / 2)];
	};
	const p = quat(a, ao), q = quat(b, bo);
	let dot = p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3];
	if (dot < 0) { for (let i = 0; i < 4; i++) q[i] = -q[i]; dot = -dot; }
	let wa = 1 - s, wb = s;
	if (dot < 0.9995) {
		const om = Math.acos(Math.min(1, dot)), so = Math.sin(om);
		wa = Math.sin((1 - s) * om) / so; wb = Math.sin(s * om) / so;
	}
	const r = [0, 1, 2, 3].map((i) => wa * p[i] + wb * q[i]);
	const n = Math.hypot(...r);
	let [x, y, z, w] = r.map((v) => v / n);
	if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
	const sn = Math.hypot(x, y, z), angle = 2 * Math.atan2(sn, w), k = sn < 1e-12 ? 2 : angle / sn;
	out[oo] = x * k; out[oo + 1] = y * k; out[oo + 2] = z * k;
}

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

// How far the turn θ is past a joint's limits, as Rigsmith reads them
// (joint_limits.c3): the turn in the limit's frame F, M = F⁻¹ R F, split into
// a twist about F's Y (the bone) and a swing across it, each axis's angle
// past its range the short way round the circle. Σ of the excesses².
function limitExcess(lim, x, y, z) {
	const t = Math.hypot(x, y, z), k = t < 1e-12 ? 0.5 : Math.sin(t / 2) / t;
	const rx = x * k, ry = y * k, rz = z * k, rw = Math.cos(t / 2);
	const [fx, fy, fz, fw] = lim.frame;
	// a = R F, then m = F⁻¹ a.
	const ax = rw * fx + rx * fw + ry * fz - rz * fy, ay = rw * fy - rx * fz + ry * fw + rz * fx;
	const az = rw * fz + rx * fy - ry * fx + rz * fw, aw = rw * fw - rx * fx - ry * fy - rz * fz;
	let mx = fw * ax - fx * aw - fy * az + fz * ay, my = fw * ay + fx * az - fy * aw - fz * ax;
	let mz = fw * az - fx * ay + fy * ax - fz * aw, mw = fw * aw + fx * ax + fy * ay + fz * az;
	if (mw < 0) { mx = -mx; my = -my; mz = -mz; mw = -mw; }
	// Twist (0, ty, 0, tw) about Y; swing m twist⁻¹, which has no Y part.
	const reach = Math.hypot(mw, my), ty = reach > 1e-6 ? my / reach : 0, tw = reach > 1e-6 ? mw / reach : 1;
	const sx = mx * tw + mz * ty, sz = mz * tw - mx * ty, sw = mw * tw + my * ty;
	const across = Math.hypot(sx, sz);
	const scale = across < 1e-8 ? 2 / Math.max(sw, 1e-8) : (2 * Math.atan2(across, sw)) / across;
	let twist = 2 * Math.atan2(ty, tw);
	if (twist <= -Math.PI) twist += 2 * Math.PI;
	let e = 0;
	for (let i = 0; i < 3; i++) {
		const r = lim.axes[i];
		if (!r) continue;
		const angle = i === 0 ? sx * scale : i === 1 ? twist : sz * scale;
		if (angle >= r[0] && angle <= r[1]) continue;
		let along = (angle - r[0]) % (2 * Math.PI);
		if (along < 0) along += 2 * Math.PI;
		const span = r[1] - r[0];
		if (along <= span) continue;
		const d = Math.min(along - span, 2 * Math.PI - along);
		e += d * d;
	}
	return e;
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

// The problem: the rig, the points and their hypotheses over T frames, seen
// by one camera or several.
export class RigFit {
	// `views` one obs or a list, each { T, points: [{ name, joint, marker }],
	// markers: count, x, y, w } with x, y, w [T, P, H] (y up, w 0 for none),
	// the markers numbered across all of them. `pitch` degrees, one for every
	// view or each its own. `foot` the feet's share of the leg plane: firm while
	// the view's yaw is found, loose after. `sigma` the reprojection's
	// robust scale in pixels: tracks stray, exact labels may be trusted wider.
	// `fixed`: per marker, its offset when it is known (a point of the mesh) and
	// held, or null to fit it.
	constructor(rig, views, { pitch = 8, k0 = 1000, foot = W_PLANE, sigma = 15, weights = null, fixed = null } = {}) {
		views = Array.isArray(views) ? views : [views];
		const J = rig.names.length, V = views.length;
		this.rig = rig;
		this.J = J;
		this.T = views[0].T;
		this.V = V;
		this.views = views;
		this.M = Math.max(...views.map((o) => o.markers));
		// Every point once (a view's points are each one name; two views may share
		// one), and the joint each marker sits on.
		this.points = [...new Map(views.flatMap((o) => o.points).map((p) => [p.name, p])).values()];
		this.markerJoint = new Int32Array(this.M);
		for (const p of this.points) if (p.marker >= 0) this.markerJoint[p.marker] = p.joint;
		this.fixed = new Uint8Array(this.M);
		for (let m = 0; m < this.M; m++) if (fixed?.[m]) this.fixed[m] = 1;
		this.pitchPriors = views.map((_, v) => (Array.isArray(pitch) ? pitch[v] : pitch) * DEG);
		this.sigma2 = sigma * sigma;
		this.k0 = k0;
		const T = this.T;
		this.parents = Int32Array.from(rig.parents);
		this.offsets = new Float64Array(J * 3);
		for (let j = 0; j < J; j++) {
			const p = rig.parents[j];
			for (let d = 0; d < 3; d++) this.offsets[j * 3 + d] = p < 0 ? 0 : rig.positions[j][d] - rig.positions[p][d];
		}
		this.rootRest = rig.positions[0].slice();
		this.prepareStructure(foot);

		// Parameters and their gradients.
		this.theta = new Float64Array(T * J * 3);
		this.root = new Float64Array(T * 3);
		for (let t = 0; t < T; t++) this.root.set(this.rootRest, t * 3);
		this.marks = new Float64Array(this.M * 3);
		for (let m = 0; m < this.M; m++) if (this.fixed[m]) this.marks.set(fixed[m], m * 3);
		// Each view's camera: its scale and aim per frame, its yaw and pitch (view
		// v's at v T, v T 2 and 2 v).
		this.logS = new Float64Array(V * T).fill(Math.log(k0));
		this.shift = new Float64Array(V * T * 2);
		this.cam = new Float64Array(2 * V);
		for (let v = 0; v < V; v++) this.cam[2 * v + 1] = this.pitchPriors[v];
		this.g = {
			theta: new Float64Array(T * J * 3), root: new Float64Array(T * 3), marks: new Float64Array(this.M * 3),
			logS: new Float64Array(V * T), shift: new Float64Array(V * T * 2), cam: new Float64Array(2 * V),
		};
		// Forward kinematics' results.
		this.R = new Float64Array(T * J * 9);
		this.G = new Float64Array(T * J * 9);
		this.pos = new Float64Array(T * J * 3);
		this.mpos = new Float64Array(T * this.M * 3);
		this.gPos = new Float64Array(T * J * 3);
		this.gM = new Float64Array(T * this.M * 3);
		this.rootTau = new Float64Array(T * 3);   // the root's turn's own share (its frame in the lateral terms)
		this.weights = { ...DEFAULT_WEIGHTS, ...weights };
		// The limited joints, their frames normalised; an axis whose range is a
		// whole turn is free.
		this.limits = (rig.limits ?? []).filter((l) => l.joint > 0).map((l) => {
			const n = Math.hypot(...l.frame), frame = l.frame.map((v) => v / n);
			return { joint: l.joint, frame, axes: l.axes.map((r) => (r && r[1] - r[0] < 2 * Math.PI - 1e-6 ? r : null)) };
		}).filter((l) => l.axes.some(Boolean));
		this.limitScale = 1;
	}

	// Which joints the points see, each joint's reach (the farthest rest point
	// below it), the one-child joints' bone directions and the feet.
	prepareStructure(foot) {
		const { rig, J } = this;
		const children = Array.from({ length: J }, () => []);
		for (let j = 1; j < J; j++) children[this.parents[j]].push(j);
		this.children = children;
		// How well the points see a joint's turn: 2 with a point below it (a
		// joint or a marker further down) or two markers of its own, 1 with one
		// marker of its own alone (its fitted offset trades with the turn: a jaw
		// with one point on it), 0 with none (a hand nobody labelled). A marker
		// whose offset is known counts as two: it does not trade. A joint with
		// one sees its own twist (the point is off its bone), so its twist prior
		// weighs `twistMarked` instead of `twist`.
		const below = new Int32Array(J), own = new Int32Array(J);
		this.twistSeen = new Uint8Array(J);
		for (const p of this.points) {
			if (p.marker >= 0) {
				own[p.joint] += this.fixed[p.marker] ? 2 : 1;
				if (this.fixed[p.marker]) this.twistSeen[p.joint] = 1;
			} else if (p.joint > 0) below[this.parents[p.joint]]++;
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
		// Chains that bend evenly: a joint and the child that carries its line on,
		// whose bones run within 35 degrees of each other at rest (a spine, a
		// tail), not a limb's zigzag of knee and ankle. The child that carries a
		// line on: a one-child joint's child, or of several the one whose bone
		// runs within 35 degrees of the joint's own while the others turn off
		// past 60 (a chest's neck, not its collarbones; not a hand's five
		// fingers). A pair through a branch weighs `chainBranch`, and reaches
		// only along its line (a chest's arms do not make its bend dearer).
		const unit = (o) => { const n = Math.hypot(o[0], o[1], o[2]); return n < 1e-9 ? null : [o[0] / n, o[1] / n, o[2] / n]; };
		const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
		const carries = new Int32Array(J).fill(-1);
		for (let j = 0; j < J; j++) {
			if (children[j].length === 1) { if (unit(this.offsets.subarray(children[j][0] * 3, children[j][0] * 3 + 3))) carries[j] = children[j][0]; continue; }
			// A root's own turn is its heading as well: no line through a branch there.
			const own = j > 0 && unit(this.offsets.subarray(j * 3, j * 3 + 3));
			if (!own || children[j].length < 2) continue;
			let along = -1, near = 0, far = 0;
			for (const c of children[j]) {
				const d = unit(this.offsets.subarray(c * 3, c * 3 + 3));
				if (!d) continue;
				const cos = dot(own, d);
				if (cos > Math.cos(35 * DEG)) { along = c; near++; } else if (cos < Math.cos(60 * DEG)) far++;
			}
			if (near === 1 && near + far === children[j].length) carries[j] = along;
		}
		this.chainChild = new Int32Array(J).fill(-1);
		this.chainBranch = new Uint8Array(J);
		this.chainReach = new Float64Array(J);
		for (let j = 0; j < J; j++) {
			const c = carries[j];
			if (c < 0 || carries[c] < 0) continue;
			const a = unit(this.offsets.subarray(c * 3, c * 3 + 3)), b = unit(this.offsets.subarray(carries[c] * 3, carries[c] * 3 + 3));
			if (dot(a, b) <= Math.cos(35 * DEG)) continue;
			this.chainChild[j] = c;
			this.chainBranch[j] = this.oneChild[j] && this.oneChild[c] ? 0 : 1;
			this.chainReach[j] = Math.hypot(...this.offsets.subarray(c * 3, c * 3 + 3)) + this.reach[c];
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
		// its own plane; the chain from a foot up to where it branches off; the
		// foot itself loosely once the view is known, as it may turn in from a
		// splayed rest stance), and the middle joints (no side in their names)
		// on the root's midline - firmly near the root, letting go toward the
		// tail's tip and the head, whatever curve the model was made in.
		const lateral = [];
		for (const leaf of this.footLeaves) {
			const chain = [leaf];
			let j = leaf;
			while (this.parents[j] > 0 && children[this.parents[j]].length === 1) { j = this.parents[j]; chain.push(j); }
			const hip = j === leaf ? this.parents[leaf] : j;
			for (const c of chain) if (c !== hip) lateral.push([c, hip, rig.positions[c][0] - rig.positions[hip][0], feet.has(c) ? foot : W_PLANE]);
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
			for (let m = 0; m < this.M; m++) {
				const j = this.markerJoint[m], g9 = (t * J + j) * 9, j3 = (t * J + j) * 3, m3 = m * 3, x3 = (t * this.M + m) * 3;
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
		const { T, J, g, k0, sigma2 } = this, W = this.weights;
		this.forward();
		for (const a of Object.values(g)) a.fill(0);
		this.gPos.fill(0);
		this.gM.fill(0);
		this.rootTau.fill(0);
		const terms = {};

		// Reprojection, each view through its own camera.
		let reproj = 0, count = 0;
		for (let v = 0; v < this.V; v++) {
			const obs = this.views[v], P = obs.points.length, H = obs.H;
			const yaw = this.cam[2 * v], pitch = this.cam[2 * v + 1];
			const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
			const r0 = [cy, 0, sy], r1 = [sp * sy, cp, -sp * cy];
			const dy0 = [-sy, 0, cy], dy1 = [sp * cy, 0, sp * sy], dp1 = [cp * sy, -sp, -cp * cy];
			const norm = 1 / (T * P);
			const a = new Float64Array(H), e2 = new Float64Array(H);
			for (let t = 0; t < T; t++) {
				const vt = v * T + t;
				const s = Math.exp(this.logS[vt]), sx = this.shift[vt * 2], syy = this.shift[vt * 2 + 1];
				for (let pi = 0; pi < P; pi++) {
					const p = obs.points[pi];
					const [X, o, gX] = this.where(t, p);
					const x0 = X[o], x1 = X[o + 1], x2 = X[o + 2];
					const q0 = r0[0] * x0 + r0[2] * x2, q1 = r1[0] * x0 + r1[1] * x1 + r1[2] * x2;
					const u0 = s * q0 + sx, u1 = s * q1 + syy;
					let best = -Infinity, any = false, pull = 0;
					const base = (t * P + pi) * H;
					for (let h = 0; h < H; h++) {
						const w = obs.w[base + h];
						if (!(w > 0)) { a[h] = -Infinity; continue; }
						any = true;
						if (w > pull) pull = w;
						const d0 = u0 - obs.x[base + h], d1 = u1 - obs.y[base + h];
						e2[h] = d0 * d0 + d1 * d1;
						a[h] = (-sigma2 * Math.log1p(e2[h] / sigma2)) / tau + Math.log(w);
						if (a[h] > best) best = a[h];
					}
					if (!any) continue;
					// How hard the point pulls: as sure as its surest hypothesis, to the
					// view's `gamma` (0 for labels and tracks: each pulls fully).
					const c = obs.gamma > 0 ? pull ** obs.gamma : 1;
					let z = 0;
					for (let h = 0; h < H; h++) if (a[h] > -Infinity) z += Math.exp(a[h] - best);
					reproj += -tau * (best + Math.log(z)) * norm * c;
					count++;
					let du0 = 0, du1 = 0;
					for (let h = 0; h < H; h++) {
						if (a[h] === -Infinity) continue;
						const wh = Math.exp(a[h] - best) / z;
						const k = (wh * 2) / (1 + e2[h] / sigma2) * norm * W.reproj * c;
						du0 += k * (u0 - obs.x[base + h]);
						du1 += k * (u1 - obs.y[base + h]);
					}
					g.shift[vt * 2] += du0;
					g.shift[vt * 2 + 1] += du1;
					g.logS[vt] += du0 * s * q0 + du1 * s * q1;
					g.cam[2 * v] += s * (du0 * (dy0[0] * x0 + dy0[2] * x2) + du1 * (dy1[0] * x0 + dy1[2] * x2));
					g.cam[2 * v + 1] += s * du1 * (dp1[0] * x0 + dp1[1] * x1 + dp1[2] * x2);
					gX[o] += s * (du0 * r0[0] + du1 * r1[0]);
					gX[o + 1] += s * du1 * r1[1];
					gX[o + 2] += s * (du0 * r0[2] + du1 * r1[2]);
				}
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
				if (this.fixed[Math.floor(i / 3)]) continue;
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
		// A known offset is held.
		for (let m = 0; m < this.M; m++) if (this.fixed[m]) g.marks.fill(0, m * 3, m * 3 + 3);
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
			for (let mk = 0; mk < M; mk++) {
				const m = (t * M + mk) * 3, j = this.markerJoint[mk], g9 = (t * J + j) * 9;
				const gx = gM[m], gy = gM[m + 1], gz = gM[m + 2], x = mpos[m], y = mpos[m + 1], z = mpos[m + 2];
				S[j * 3] += gx; S[j * 3 + 1] += gy; S[j * 3 + 2] += gz;
				X[j * 3] += y * gz - z * gy; X[j * 3 + 1] += z * gx - x * gz; X[j * 3 + 2] += x * gy - y * gx;
				// The offset, in the joint's frame: Gᵀ g.
				for (let d = 0; d < 3; d++) g.marks[mk * 3 + d] += G[g9 + d] * gx + G[g9 + 3 + d] * gy + G[g9 + 6 + d] * gz;
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
			const wt = (this.twistSeen[j] ? W.twistMarked : W.twist) * L2 * norm;
			const b = this.boneDir.subarray(j * 3, j * 3 + 3);
			const c = this.chainChild[j];
			for (let t = 0; t < T; t++) {
				const o = (t * J + j) * 3;
				for (let d = 0; d < 3; d++) {
					rest += wr * theta[o + d] * theta[o + d];
					g.theta[o + d] += 2 * wr * theta[o + d];
				}
				if (this.oneChild[j] && wt > 0) {
					const tw = theta[o] * b[0] + theta[o + 1] * b[1] + theta[o + 2] * b[2];
					twist += wt * tw * tw;
					for (let d = 0; d < 3; d++) g.theta[o + d] += 2 * wt * tw * b[d];
				}
				if (c >= 0) {
					const Lc = Math.max(this.chainReach[j], 0.05 * this.rig.height) * k0;
					const wc = (this.chainBranch[j] ? W.chainBranch : W.chain) * Lc * Lc * norm, q = (t * J + c) * 3;
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
		// Limits, in pixels as the rest are; the excess is cheap, so its
		// gradient is taken by central differences.
		let limit = 0;
		for (const lim of this.limits) {
			const j = lim.joint, L = Math.max(this.reach[j], 0.05 * this.rig.height) * k0, w = W.limit * this.limitScale * L * L * norm, h = 1e-6;
			for (let t = 0; t < T; t++) {
				const o = (t * J + j) * 3, x = theta[o], y = theta[o + 1], z = theta[o + 2];
				const e = limitExcess(lim, x, y, z);
				if (e === 0) continue;
				limit += w * e;
				g.theta[o] += (w * (limitExcess(lim, x + h, y, z) - limitExcess(lim, x - h, y, z))) / (2 * h);
				g.theta[o + 1] += (w * (limitExcess(lim, x, y + h, z) - limitExcess(lim, x, y - h, z))) / (2 * h);
				g.theta[o + 2] += (w * (limitExcess(lim, x, y, z + h) - limitExcess(lim, x, y, z - h))) / (2 * h);
			}
		}
		Object.assign(terms, { twist, rest, chain, thetaAcc: tacc, limit });
	}

	// Each view's pitch near its prior, its scale and aim smooth.
	cameraPriors(terms) {
		const { T, g } = this, W = this.weights;
		let level = 0, follow = 0;
		for (let v = 0; v < this.V; v++) {
			const dp = this.cam[2 * v + 1] - this.pitchPriors[v];
			level += W.level * dp * dp;
			g.cam[2 * v + 1] += 2 * W.level * dp;
			if (T <= 2) continue;
			const cs = (W.follow * 1e6) / (T - 2), ch = W.follow / (T - 2), o = v * T;
			for (let t = o + 1; t < o + T - 1; t++) {
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
		terms.level = level;
		terms.follow = follow;
	}

	// World points through view v's camera on frame t: [x, y] (y up) of
	// `at(i)` = [array, offset] for i < n.
	project(v, t, n, at) {
		const yaw = this.cam[2 * v], pitch = this.cam[2 * v + 1];
		const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
		const vt = v * this.T + t, s = Math.exp(this.logS[vt]), row = [];
		for (let i = 0; i < n; i++) {
			const [X, o] = at(i);
			row.push([s * (cy * X[o] + sy * X[o + 2]) + this.shift[vt * 2], s * (sp * sy * X[o] + cp * X[o + 1] - sp * cy * X[o + 2]) + this.shift[vt * 2 + 1]]);
		}
		return row;
	}

	// Projections of view v's points [T][P][x, y] (y up).
	projections(v = 0) {
		this.forward();
		const points = this.views[v].points, out = [];
		for (let t = 0; t < this.T; t++) out.push(this.project(v, t, points.length, (i) => this.where(t, points[i])));
		return out;
	}

	// Every joint's projection through view v's camera [T][J][x, y] (y up).
	jointProjections(v = 0) {
		this.forward();
		const out = [];
		for (let t = 0; t < this.T; t++) out.push(this.project(v, t, this.J, (j) => [this.pos, (t * this.J + j) * 3]));
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
			// The limits come in over the first half: the fit finds the pose the
			// points ask for, then the nearest one the rig allows.
			this.limitScale = Math.min(1, it / (0.5 * iterations));
			loss = this.evaluate(tau);
			if (it === iterations - 1) break;
			adam.step(0.5 * (1 + Math.cos((Math.PI * it) / iterations)));
			onStep?.(it + 1, iterations, loss);
		}
		return loss;
	}
}

// The tracks (track.js) as the fit's observations, and the points as joints
// and markers of `rig`: the first `frames` of them, the markers numbered in
// `marked` (shared by the views of one fit, so a marker two views see is one).
//
// `gamma`: how a point's score weighs its pull, score-weight^gamma; by default 2
// for a finder's tracks (they carry `boxes`), and 0 (every point pulls fully)
// for labels and the tracker's tracks.
export function observations(rig, tracks, markers = {}, { frames = tracks.frames, marked = [], gamma = null } = {}) {
	const joints = new Map(rig.names.map((n, j) => [n, j]));
	const T = frames, H = tracks.hypotheses.length, P = tracks.points.length;
	const points = tracks.points.map((name) => {
		if (joints.has(name)) return { name, joint: joints.get(name), marker: -1 };
		if (!(name in markers)) throw bad(`"${name}" is neither one of the rig's joints nor a marker`);
		const on = markerJoint(markers[name]);
		if (!joints.has(on)) throw bad(`marker ${name} is on "${on}", not one of the rig's joints`);
		if (!marked.includes(name)) marked.push(name);
		return { name, joint: joints.get(on), marker: marked.indexOf(name) };
	});
	const x = new Float64Array(T * P * H), y = new Float64Array(T * P * H), w = new Float64Array(T * P * H);
	tracks.hypotheses.forEach((hyp, h) => {
		hyp.frames.forEach((f, i) => {
			if (f >= T) return;
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
	return { T, H, points, markers: marked.length, x, y, w, gamma: gamma ?? (tracks.boxes ? 2 : 0) };
}

// One frame's observations: the labels of keyframe `k` alone, weighted by
// their scores as the whole clip's are (a label's is 1, a found joint's how
// sure the finder was).
function keyframeObservations(obs, tracks, k) {
	const P = obs.points.length;
	const x = new Float64Array(P), y = new Float64Array(P), w = new Float64Array(P);
	const h = tracks.hypotheses.findIndex((hyp) => hyp.from === k);
	const row = tracks.hypotheses[h].frames.indexOf(k);
	for (let p = 0; p < P; p++) {
		const q = tracks.hypotheses[h].tracks[row][p];
		if (!q) continue;
		x[p] = q[0]; y[p] = -q[1]; w[p] = Math.min(1, Math.max(0.05, ((tracks.hypotheses[h].scores[row][p] ?? 1) - 0.2) / 0.4));
	}
	return { T: 1, H: 1, points: obs.points, markers: obs.markers, x, y, w, gamma: obs.gamma };
}

// One frame's observations: every hypothesis on frame t.
function frameObservations(obs, t) {
	const n = obs.points.length * obs.H, at = (a) => a.slice(t * n, (t + 1) * n);
	return { T: 1, H: obs.H, points: obs.points, markers: obs.markers, x: at(obs.x), y: at(obs.y), w: at(obs.w), gamma: obs.gamma };
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

// Pixels per rig unit, roughly: one frame's labels' spread over the rig's.
function pixelsPerUnit(rig, o) {
	const lab = [];
	for (let p = 0; p < o.points.length; p++) if (o.w[p] > 0) lab.push([o.x[p], o.y[p]]);
	const spread = (pts) => Math.hypot(Math.max(...pts.map((q) => q[0])) - Math.min(...pts.map((q) => q[0])), Math.max(...pts.map((q) => q[1])) - Math.min(...pts.map((q) => q[1])));
	const restPts = o.points.filter((p, i) => o.w[i] > 0).map((p) => rig.positions[p.joint]);
	const rest3 = Math.hypot(...[0, 1, 2].map((d) => Math.max(...restPts.map((q) => q[d])) - Math.min(...restPts.map((q) => q[d]))));
	return spread(lab) / Math.max(1e-6, rest3) * 1.1;
}

// Aim view v of a one-frame fit: the rig's projected centroid on its points'.
function aim(f, v = 0) {
	const o = f.views[v], H = o.H, c = centroid(o, 0);
	if (!c) return;
	const seen = (p) => { for (let h = 0; h < H; h++) if (o.w[p * H + h] > 0) return true; return false; };
	const proj = f.projections(v)[0].filter((_, p) => seen(p));
	f.shift[v * 2] = c[0] - proj.reduce((s, q) => s + q[0], 0) / proj.length;
	f.shift[v * 2 + 1] = c[1] - proj.reduce((s, q) => s + q[1], 0) / proj.length;
}

// The solved keyframes either side of frame t and how far between: [a, b, s].
function bracket(solved, t) {
	let n = 0;
	while (n + 1 < solved.length && solved[n + 1].key <= t) n++;
	const a = solved[n], b = solved[Math.min(n + 1, solved.length - 1)];
	return [a, b, b.key > a.key ? Math.min(1, Math.max(0, (t - a.key) / (b.key - a.key))) : 0];
}

const lerp = (x, y, s) => x + (y - x) * s;

// The pose of frame t blended from the solved keyframes, into a fit's frame 0.
function blendPose(f, solved, t) {
	const [a, b, s] = bracket(solved, t);
	for (let j = 0; j < f.J; j++) slerpTurn(a.f.theta, j * 3, b.f.theta, j * 3, s, f.theta, j * 3);
	for (let d = 0; d < 3; d++) f.root[d] = lerp(a.f.root[d], b.f.root[d], s);
	return [a, b, s];
}

// Frame 0 of fit `f` started from `from` (a solved one-frame fit): its pose, its
// root and each view's scale.
function startFrom(f, from) {
	f.theta.set(from.theta.subarray(0, f.J * 3));
	f.root.set(from.root.subarray(0, 3));
	for (let v = 0; v < Math.min(f.V, from.V); v++) f.logS[v] = from.logS[v];
}

/**
 * Fits `rig` (rig.js) to `views`: [{ tracks (track.js), pitch? }], frame N of
 * each the same moment (or one view's tracks alone). Returns { problem (the
 * RigFit, solved), projections (the first view's), report }. `iterations` of
 * the whole clip after a search over the first keyframe's yaw (and each other
 * view's); `onStep(done, total)` as it goes.
 */
export function fitRig(rig, views, { markers = {}, pitch = 8, iterations = 1500, onStep = null, weights = null, sigma = 15, confidence = null } = {}) {
	const t0 = llm.now();
	if (!Array.isArray(views)) views = [{ tracks: views }];
	const V = views.length, T = views[0].tracks.frames, J = rig.names.length;
	const marked = [];
	const all = views.map((v) => observations(rig, v.tracks, markers, { frames: T, marked, gamma: v.confidence ?? confidence }));
	for (const o of all) o.markers = marked.length;
	// Markers whose offsets are known (points of the mesh), held at them.
	const fixed = marked.map((name) => (markers[name] && typeof markers[name] === 'object' ? markers[name].offset.map(Number) : null));
	const pitches = views.map((v) => Number(v.pitch ?? pitch));
	const keysOf = views.map((v) => [...new Set(v.tracks.hypotheses.map((h) => h.from))].filter((k) => k < T).sort((x, y) => x - y));
	for (const [v, ks] of keysOf.entries()) if (!ks.length) throw bad(`view ${v}: no keyframe within the first view's ${T} frames`);
	const tracks = views[0].tracks, obs = all[0];
	const keys = keysOf[0], key = keys[0];
	const keyObs = keyframeObservations(obs, tracks, key);
	const k0 = pixelsPerUnit(rig, keyObs);

	// The first keyframe alone, from rest, over a ring of yaws: which way the
	// animal faces, and where the markers sit.
	const allKeys = [...new Set(keysOf.flat())].sort((x, y) => x - y);
	const initSteps = 400, yaws = 12;
	const total = V * yaws + keys.length - 1 + (V > 1 ? allKeys.length : 0) + iterations / 10;
	let done = 0;
	let best = null;
	for (let i = 0; i < yaws; i++) {
		const f = new RigFit(rig, keyObs, { pitch: pitches[0], k0, sigma, weights, fixed });
		f.cam[0] = (i * 2 * Math.PI) / yaws;
		aim(f);
		const loss = f.run(initSteps, { tau0: 20, tau1: 20 });
		const err = f.terms.reproj;
		if (!best || err < best.err) best = { f, err, loss, yaw: f.cam[0] };
		onStep?.(++done, total);
	}
	llm.print(`  start: frame ${key} alone, yaw ${(best.yaw / DEG).toFixed(0)}° fits best (reprojection ${best.err.toFixed(1)})`);

	// Every other keyframe alone, in order, with that camera and those markers,
	// from the first's pose and from the keyframe's before it, the better kept:
	// a clip that leaves its first pose far behind (arms out, then down) starts
	// near each of its poses, and a pose that is a fold of the one before it (a
	// chest bent one way and the neck the other, which a short bone's error
	// along itself asks for) does not win against carrying on.
	let solved = [{ key, f: best.f, centres: [centroid(keyObs, 0)] }];
	for (const k of keys.slice(1)) {
		const o = keyframeObservations(obs, tracks, k);
		let kept = null;
		for (const from of [best.f, solved[solved.length - 1].f]) {
			if (kept && from === best.f) continue;
			const f = new RigFit(rig, o, { pitch: pitches[0], k0, sigma, weights, fixed });
			startFrom(f, from);
			f.marks.set(best.f.marks);
			f.cam.set(best.f.cam);
			aim(f);
			const loss = f.run(initSteps, { tau0: 20, tau1: 20, lr: { cam: 0, marks: 0 } });
			if (!kept || loss < kept.loss) kept = { f, loss };
		}
		solved.push({ key: k, f: kept.f, centres: [centroid(o, 0)] });
		onStep?.(++done, total);
	}
	if (solved.length > 1) llm.print(`  keyframes ${keys.join(', ')} alone: reprojection ${solved.map((s) => s.f.terms.reproj.toFixed(1)).join(', ')}`);

	let marks = Float64Array.from(best.f.marks);
	let cams = [{ cam: Array.from(best.f.cam), logS: null }];
	// View v's observations on keyframe k: its labels there, or every hypothesis.
	const at = (v, k) => (keysOf[v].includes(k) ? keyframeObservations(all[v], views[v].tracks, k) : frameObservations(all[v], k));
	let lastSearch;
	if (V > 1) {
		// Each other view's camera: its first keyframe over a ring of yaws, solved
		// with the views placed before it and the pose free, since one view
		// alone may have a limb the wrong way along its line of sight (a stride
		// backwards): only the right yaw lets every view fit at once. The best
		// ring's cameras and markers are kept. Labels are trusted wide here
		// (WIDE px): a limb the first view put the wrong way along its line of
		// sight is that far off in another, and must still be pulled round.
		for (let v = 1; v < V; v++) {
			const k = keysOf[v][0], os = Array.from({ length: v + 1 }, (_, u) => at(u, k));
			let bestV = null;
			for (let i = 0; i < yaws; i++) {
				const f = new RigFit(rig, os, { pitch: pitches.slice(0, v + 1), k0, sigma: WIDE, weights, fixed });
				const [a, b, s] = blendPose(f, solved, k);
				f.marks.set(marks);
				for (let u = 0; u < v; u++) {
					f.cam[2 * u] = cams[u].cam[0];
					f.cam[2 * u + 1] = cams[u].cam[1];
					f.logS[u] = u === 0 ? lerp(a.f.logS[0], b.f.logS[0], s) : cams[u].logS;
					aim(f, u);
				}
				f.logS[v] = Math.log(pixelsPerUnit(rig, os[v]));
				f.cam[2 * v] = (i * 2 * Math.PI) / yaws;
				aim(f, v);
				f.run(initSteps, { tau0: 20, tau1: 20 });
				if (!bestV || f.terms.reproj < bestV.err) bestV = { f, err: f.terms.reproj };
				onStep?.(++done, total);
			}
			cams = Array.from({ length: v + 1 }, (_, u) => ({ cam: [bestV.f.cam[2 * u], bestV.f.cam[2 * u + 1]], logS: bestV.f.logS[u] }));
			marks = Float64Array.from(bestV.f.marks);
			if (v === V - 1 && k === allKeys[0]) lastSearch = bestV.f;
			llm.print(`  view ${v}: frame ${k} with the views before it, yaw ${(bestV.f.cam[2 * v] / DEG).toFixed(0)}° fits best (reprojection ${bestV.err.toFixed(1)}; yaws now ${cams.map((c) => (c.cam[0] / DEG).toFixed(0)).join('°, ')}°)`);
		}

		// Every keyframe of every view again, all views together, the cameras
		// held, the labels trusted wide: what one view leaves along its line of
		// sight, another sees.
		// Each from the one-view keyframes blended, and from the keyframe before
		// it in all the views (the first from the last camera search's pose),
		// the better kept.
		const both = [];
		let before = lastSearch;
		for (const k of allKeys) {
			const os = all.map((_, v) => at(v, k));
			let kept = null;
			for (const from of [null, before]) {
				if (from === undefined) continue;
				const f = new RigFit(rig, os, { pitch: pitches, k0, sigma: WIDE, weights, fixed });
				const [a, b, s] = blendPose(f, solved, k);
				f.marks.set(marks);
				for (let v = 0; v < V; v++) {
					f.cam[2 * v] = cams[v].cam[0];
					f.cam[2 * v + 1] = cams[v].cam[1];
					f.logS[v] = v === 0 ? lerp(a.f.logS[0], b.f.logS[0], s) : cams[v].logS;
				}
				if (from) startFrom(f, from);
				for (let v = 0; v < V; v++) aim(f, v);
				const loss = f.run(initSteps, { tau0: 20, tau1: 20, lr: { cam: 0, marks: 0 } });
				if (!kept || loss < kept.loss) kept = { f, loss };
			}
			before = kept.f;
			both.push({ key: k, f: kept.f, centres: os.map((o) => centroid(o, 0)) });
			onStep?.(++done, total);
		}
		llm.print(`  keyframes ${allKeys.join(', ')} in all ${V} views: reprojection ${both.map((s) => s.f.terms.reproj.toFixed(1)).join(', ')}`);
		solved = both;
	}

	// Every frame from the keyframes either side of it, blended by time, aimed
	// where each view's points are.
	const fit = new RigFit(rig, all, { pitch: V > 1 ? pitches : pitches[0], k0, foot: W_FOOT, sigma, weights, fixed });
	for (let t = 0; t < T; t++) {
		const [a, b, s] = bracket(solved, t);
		for (let j = 0; j < J; j++) slerpTurn(a.f.theta, j * 3, b.f.theta, j * 3, s, fit.theta, (t * J + j) * 3);
		for (let d = 0; d < 3; d++) fit.root[t * 3 + d] = lerp(a.f.root[d], b.f.root[d], s);
		for (let v = 0; v < V; v++) {
			const vt = v * T + t;
			fit.logS[vt] = lerp(a.f.logS[v], b.f.logS[v], s);
			const c = centroid(all[v], t);
			for (let d = 0; d < 2; d++) {
				const ofA = a.centres[v] && a.f.shift[v * 2 + d] - a.centres[v][d], ofB = b.centres[v] && b.f.shift[v * 2 + d] - b.centres[v][d];
				const off = a.centres[v] && b.centres[v] ? lerp(ofA, ofB, s) : a.centres[v] ? ofA : ofB;
				fit.shift[vt * 2 + d] = c && off !== null ? c[d] + off : lerp(a.f.shift[v * 2 + d], b.f.shift[v * 2 + d], s);
			}
		}
	}
	fit.marks.set(marks);
	if (V > 1) for (let v = 0; v < V; v++) { fit.cam[2 * v] = cams[v].cam[0]; fit.cam[2 * v + 1] = cams[v].cam[1]; }
	else fit.cam.set(best.f.cam);
	const before = done;
	let lastPrint = llm.now();
	fit.run(iterations, {
		onStep: (n, of, loss) => {
			onStep?.(before + n / 10, total);
			if (llm.now() - lastPrint > 5000) {
				llm.print(`  fit ${n}/${of}: loss ${loss.toFixed(2)}, reprojection ${fit.terms.reproj.toFixed(2)}`);
				lastPrint = llm.now();
			}
		},
	});

	// How well it fits: each observed point's distance to its nearest
	// hypothesis, per view and over all of them.
	const viewErrors = all.map((o, v) => {
		const proj = fit.projections(v), errs = [];
		for (let t = 0; t < T; t++) {
			for (let p = 0; p < o.points.length; p++) {
				let e = Infinity;
				for (let h = 0; h < o.H; h++) {
					const k = (t * o.points.length + p) * o.H + h;
					if (o.w[k] > 0) e = Math.min(e, Math.hypot(proj[t][p][0] - o.x[k], proj[t][p][1] - o.y[k]));
				}
				if (e < Infinity) errs.push(e);
			}
		}
		return errs;
	});
	const stats = (errs) => {
		errs = errs.slice().sort((x, y) => x - y);
		const q = (f) => errs[Math.min(errs.length - 1, Math.floor(f * errs.length))];
		return { median: round(q(0.5), 2), p90: round(q(0.9), 2), over_30: round(errs.filter((e) => e > 30).length / errs.length, 3) };
	};
	const camera = (v) => ({ yaw: round(fit.cam[2 * v] / DEG, 2), pitch: round(fit.cam[2 * v + 1] / DEG, 2) });
	const report = {
		reprojection_px: stats(viewErrors.flat()),
		camera: camera(0),
		terms: Object.fromEntries(Object.entries(fit.terms).map(([k, v]) => [k, round(v, 3)])),
		seconds: round((llm.now() - t0) / 1000, 1),
	};
	if (V > 1) report.views = viewErrors.map((errs, v) => ({ reprojection_px: stats(errs), camera: camera(v) }));
	if (fit.limits.length) {
		// How far past the rig's limits the clip goes at worst; Rigsmith plays it held at them.
		let worst = 0;
		for (const lim of fit.limits) {
			for (let t = 0; t < T; t++) {
				const o = (t * J + lim.joint) * 3;
				worst = Math.max(worst, Math.sqrt(limitExcess(lim, fit.theta[o], fit.theta[o + 1], fit.theta[o + 2])));
			}
		}
		report.limits = { joints: fit.limits.length, worst_deg: round(worst / DEG, 2) };
	}
	const perView = report.views ? `; ${report.views.map((r, v) => `view ${v} ${r.reprojection_px.median}px at yaw ${r.camera.yaw}°`).join(', ')}` : '';
	llm.print(`  fit: reprojection median ${report.reprojection_px.median}px, 90% ${report.reprojection_px.p90}px, ${(report.reprojection_px.over_30 * 100).toFixed(1)}% over 30px; yaw ${report.camera.yaw}°, pitch ${report.camera.pitch}°${perView}${report.limits ? `, at worst ${report.limits.worst_deg}° past a limit` : ''}, in ${report.seconds}s`);
	return { problem: fit, projections: fit.projections(0), report };
}

const round = (v, d) => Number(v.toFixed(d));
