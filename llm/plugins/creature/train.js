// Training a joint finder (detect.js) for one rig, on pictures of that rig in
// poses it can take: Rigsmith's render_poses folders, each picture an RGBA PNG
// with every joint's pixel beside it.
//
//   const report = await trainFinder(encoder, { renders, out }, onStep);
//
// 1. Pictures. Each render is laid over a background of its own (flat,
//    two-tone with a horizon, tiles, soft colour fields, or a crop of an image
//    from `backgrounds`: frames of the video itself work well), its colours
//    jittered and noise added, then cropped a little larger or smaller and
//    off centre. DINOv3 encodes it once.
// 2. PCA. The tokens of the first pictures give the directions most of their
//    variance lies along (subspace iteration on the GPU); every picture's
//    tokens are kept as their `rank` whitened components, on the GPU, so the
//    head trains from memory and not from the backbone.
// 3. The head (a hidden layer and a score per joint, per patch) is fitted for
//    `epochs` passes: each joint's scores over a picture softmaxed against a
//    Gaussian of `sigma` patches about its pixel, gradients written out by
//    hand as matmuls on the GPU, Adam on the host. A twentieth of the pictures
//    are held out and scored as the finder would: the peak against the label.
//
// Joints: those given, or every joint whose bone from its parent is at least
// 4% of the rig's height (and the root): fingers and zero-length twist bones
// are too small for 16-pixel patches to tell apart. Then the set's landmarks,
// points of the mesh (render_poses' `landmarks`), unless `landmarks: false`: a
// joint sits inside the body, where nothing marks it, but a nose or a heel
// shows, and shows which way the head or the foot is turned. A joint is learnt
// everywhere, as the body round it says where it is; a landmark where the
// picture shows it, and where it does not (a nose seen from behind) its heatmap
// is taught to be flat, at `hidden_weight` (0.3): it lights nothing, so its
// score is low, and the fit hardly listens to it.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit, copy, zero, breathe, breathing } from '/core/gpu.js';
import { PATCH } from '/core/models/dinov3.js';
import { uploadImage, warpCropGpu, boxAffine } from '/core/image/warp.js';
import { Head, saveFinder, peaks, partner, FINDER_VERSION } from './detect.js';
import { bad } from './request.js';

const IMAGE = /\.(png|jpe?g)$/i;
// A target's width (patches) that makes its heatmap flat: a hidden landmark's.
const FLAT = 1000;

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

const readFloats = (t, n) => new Float32Array(t.buffer.readBytes(0, n * 4).buffer);

// A seeded generator (mulberry32), for everything the trainer draws.
export function generator(seed) {
	let s = seed >>> 0;
	const next = () => {
		s = (s + 0x6d2b79f5) >>> 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
	next.between = (a, b) => a + (b - a) * next();
	next.int = (n) => Math.floor(next() * n);
	return next;
}

// render_poses folders: the pictures and their joints, checked to be one rig.
export function readRenders(folders) {
	folders = Array.isArray(folders) ? folders : [folders];
	let head = null;
	const images = [];
	for (const folder of folders) {
		const dir = String(folder).replace(/\/+$/, '');
		const sets = llm.listDir(dir).filter((e) => !e.dir && /^poses_\d+\.json$/.test(e.name));
		if (!sets.length) throw bad(`renders: ${dir} has no poses_*.json (draw it with Rigsmith's render_poses)`);
		for (const e of sets) {
			const set = JSON.parse(llm.readText(`${dir}/${e.name}`));
			if (set.motion) throw bad(`renders: ${dir}/${e.name} is an exam (motion); train on random poses`);
			const landmarks = set.landmarks ?? [];
			if (!head) head = { joints: set.joints, parents: set.parents, rest: set.rest, size: set.size, height: set.height, fill: set.fill ?? [0.45, 0.75], landmarks };
			if (set.joints.join() !== head.joints.join()) throw bad(`renders: ${dir}/${e.name} is another rig's (its joints differ from the first set's)`);
			if (set.size !== head.size) throw bad(`renders: ${dir}/${e.name} is ${set.size} pixels; the first set is ${head.size}`);
			if (landmarks.map((m) => m.name).join() !== head.landmarks.map((m) => m.name).join()) throw bad(`renders: ${dir}/${e.name} follows other landmarks than the first set (draw every set with the same "landmarks")`);
			for (const im of set.images) images.push({ path: `${dir}/${im.file}`, points: im.points, marks: im.marks ?? [], camera: im.camera });
		}
	}
	if (!images.length) throw bad('renders: no pictures');
	return { ...head, images };
}

// The joints a finder is trained on: `names`, or the ones big enough to see.
export function chooseJoints(set, names = null) {
	if (names) {
		const out = names.map((n) => set.joints.indexOf(n));
		const missing = names.filter((_, i) => out[i] < 0);
		if (missing.length) throw bad(`joints: ${missing.join(', ')} not in the rig (${set.joints.join(', ')})`);
		return out;
	}
	const ys = set.rest.map((p) => p[1]);
	const height = Math.max(...ys) - Math.min(...ys) || 1;
	const out = [];
	set.joints.forEach((_, j) => {
		const p = set.parents[j];
		if (p < 0) { out.push(j); return; }
		const a = set.rest[j], b = set.rest[p];
		if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) >= 0.04 * height) out.push(j);
	});
	return out;
}

// Backgrounds to lay pictures over: made up, and crops of `folder`'s images.
export class Backdrops {
	constructor(size, folder, random) {
		this.size = size;
		this.random = random;
		this.photos = [];
		if (folder) {
			const dir = String(folder).replace(/\/+$/, '');
			const names = llm.listDir(dir).filter((e) => !e.dir && IMAGE.test(e.name)).map((e) => `${dir}/${e.name}`);
			if (!names.length) throw bad(`backgrounds: ${dir} holds no .png or .jpg images`);
			// At most 64 of them, spread over the folder (a video's frames are alike).
			const step = Math.max(1, names.length / 64);
			for (let i = 0; i < names.length && this.photos.length < 64; i += step) this.photos.push(llm.image.load(names[Math.floor(i)]));
		}
		this.pool = Array.from({ length: 48 }, () => this.made());
	}

	colour() {
		const r = this.random;
		return [r.int(256), r.int(256), r.int(256)];
	}

	// One made-up background, size x size x 3. Its draws here, its loops in
	// native code when the host has it (lib/host/finder_train.c3, the same
	// arithmetic): in QuickJS one takes about 75 ms.
	made(native = llm.trainNative) {
		const S = this.size, r = this.random, px = new Uint8Array(S * S * 3);
		const kind = r.int(5);
		const fill = (params) => {
			llm.imageBackdrop(kind, Float64Array.from(params), S, px);
			return px;
		};
		if (kind === 0) {
			// Flat, with a slight gradient.
			const a = this.colour(), b = a.map((v) => Math.max(0, Math.min(255, v + r.between(-40, 40))));
			if (native) return fill([...a, ...b]);
			for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) for (let c = 0; c < 3; c++) px[(y * S + x) * 3 + c] = a[c] + ((b[c] - a[c]) * y) / S;
		} else if (kind === 1) {
			// Sky and ground.
			const sky = this.colour(), ground = this.colour(), horizon = r.between(0.2, 0.8) * S;
			if (native) return fill([...sky, ...ground, horizon]);
			for (let y = 0; y < S; y++) {
				const col = y < horizon ? sky : ground, shade = y < horizon ? 1 - (0.3 * y) / S : 0.7 + (0.3 * y) / S;
				for (let x = 0; x < S; x++) for (let c = 0; c < 3; c++) px[(y * S + x) * 3 + c] = Math.min(255, col[c] * shade);
			}
		} else if (kind === 2) {
			// Tiles.
			const a = this.colour(), b = this.colour(), t = 8 + r.int(56);
			if (native) return fill([...a, ...b, t]);
			for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
				const col = (Math.floor(x / t) + Math.floor(y / t)) % 2 ? a : b;
				for (let c = 0; c < 3; c++) px[(y * S + x) * 3 + c] = col[c];
			}
		} else if (kind === 4) {
			// A studio: a pale wall going a little darker toward a pale floor, the
			// way a turntable or a generated clip is shot.
			const tone = r.between(170, 250), tint = [0, 1, 2].map(() => r.between(-12, 12)), horizon = r.between(0.45, 0.9) * S;
			const floor = r.between(0.75, 0.95);
			if (native) return fill([tone, ...tint, horizon, floor]);
			for (let y = 0; y < S; y++) {
				const shade = y < horizon ? 1 - (0.08 * y) / horizon : floor * (0.9 + (0.1 * (y - horizon)) / Math.max(1, S - horizon));
				for (let x = 0; x < S; x++) for (let c = 0; c < 3; c++) px[(y * S + x) * 3 + c] = Math.max(0, Math.min(255, (tone + tint[c]) * shade));
			}
		} else {
			// A soft colour field: a coarse random grid, smoothly blown up.
			const g = 3 + r.int(6), grid = Array.from({ length: (g + 1) * (g + 1) }, () => this.colour());
			if (native) return fill([g, ...grid.flat()]);
			for (let y = 0; y < S; y++) {
				const fy = (y / S) * g, y0 = Math.floor(fy), ty = fy - y0;
				for (let x = 0; x < S; x++) {
					const fx = (x / S) * g, x0 = Math.floor(fx), tx = fx - x0;
					const q = (i, j) => grid[(y0 + j) * (g + 1) + x0 + i];
					for (let c = 0; c < 3; c++) {
						px[(y * S + x) * 3 + c] = (q(0, 0)[c] * (1 - tx) + q(1, 0)[c] * tx) * (1 - ty) + (q(0, 1)[c] * (1 - tx) + q(1, 1)[c] * tx) * ty;
					}
				}
			}
		}
		return px;
	}

	// A crop of a photo, nearest-pixel, size x size x 3.
	photo(native = llm.trainNative) {
		const S = this.size, r = this.random, img = this.photos[r.int(this.photos.length)], px = new Uint8Array(S * S * 3);
		const side = Math.min(img.width, img.height) * r.between(0.3, 1);
		const x0 = r.between(0, img.width - side), y0 = r.between(0, img.height - side), flip = r() < 0.5, ch = img.channels;
		if (native) {
			llm.imageCropNearest(img.pixels, img.width, img.height, ch, x0, y0, side, flip ? 1 : 0, S, px);
			return px;
		}
		for (let y = 0; y < S; y++) {
			const sy = Math.min(img.height - 1, Math.floor(y0 + (y * side) / S));
			for (let x = 0; x < S; x++) {
				let sx = Math.min(img.width - 1, Math.floor(x0 + (x * side) / S));
				if (flip) sx = img.width - 1 - sx;
				const s = (sy * img.width + sx) * ch, d = (y * S + x) * 3;
				px[d] = img.pixels[s]; px[d + 1] = img.pixels[s + 1]; px[d + 2] = img.pixels[s + 2];
			}
		}
		return px;
	}

	next() {
		const r = this.random;
		if (this.photos.length && r() < 0.6) return this.photo();
		// Now and then a fresh one, so the pool is not all there is.
		if (r() < 0.1) this.pool[r.int(this.pool.length)] = this.made();
		return this.pool[r.int(this.pool.length)];
	}
}

// A render laid over a background, its colours jittered and noise added. The
// jitter is a table per channel; most pixels are background or solid, which
// need no blend.
export function composite(render, backdrop, random, noise, native = llm.trainNative) {
	const S = render.width, ch = render.channels, src = render.pixels, out = new Uint8ClampedArray(S * S * 3);
	const bright = random.between(0.7, 1.3), lift = random.between(-20, 20), grain = random() < 0.5 ? random.between(0, 8) : 0;
	if (native) {
		// The same draws in the same order as below, the loops in lib/host/finder_train.c3.
		const gains = Float64Array.from([0, 1, 2], () => bright * random.between(0.8, 1.2));
		const at = grain > 0 ? random.int(noise.length - out.length) : 0;
		llm.imageComposite(src, ch, backdrop, out, gains, lift, grain, noise, at);
		return { width: S, height: S, channels: 3, pixels: out };
	}
	const [l0, l1, l2] = [0, 1, 2].map(() => {
		const gain = bright * random.between(0.8, 1.2);
		return Float32Array.from({ length: 256 }, (_, v) => v * gain + lift);
	});
	out.set(backdrop);
	for (let i = 0, s = 0, d = 0; i < S * S; i++, s += ch, d += 3) {
		const a = ch === 4 ? src[s + 3] : 255;
		if (a === 0) continue;
		if (a === 255) {
			out[d] = l0[src[s]]; out[d + 1] = l1[src[s + 1]]; out[d + 2] = l2[src[s + 2]];
			continue;
		}
		const t = a / 255, u = 1 - t;
		out[d] = l0[src[s]] * t + backdrop[d] * u;
		out[d + 1] = l1[src[s + 1]] * t + backdrop[d + 1] * u;
		out[d + 2] = l2[src[s + 2]] * t + backdrop[d + 2] * u;
	}
	if (grain > 0) {
		let n = random.int(noise.length - out.length);
		for (let d = 0; d < out.length; d++) out[d] += grain * noise[n++];
	}
	return { width: S, height: S, channels: 3, pixels: out };
}

// The top `rank` directions of the tokens' variance, whitened: the projection
// [rank, D] and its bias [rank] (P x + c, the mean taken off). `raw` is the
// tokens [N, D] of some pictures, on the GPU.
export async function principal(raw, N, D, rank, random, iterations = 10) {
	const ones = upload(new Float32Array(N).fill(1));
	const gram = upload(new Float32Array(D * D)), sums = upload(new Float32Array(D));
	const xt = f32(N * D), part = f32(D * D), partSum = f32(D);
	try {
		for (const x of raw) {
			op.transposeChannelSpatial(x, xt, D, N, 1);
			op.matmul(xt, xt, part, D, N, D);
			op.add(gram, part, D * D);
			op.matmul(xt, ones, partSum, D, N, 1);
			op.add(sums, partSum, D);
			submit();
			if (breathing()) await breathe();
		}
		const count = raw.length * N;
		const g = readFloats(gram, D * D), s = readFloats(sums, D);
		const mean = Float64Array.from(s, (v) => v / count);
		const cov = new Float32Array(D * D);
		for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) cov[i * D + j] = g[i * D + j] / count - mean[i] * mean[j];
		const { q, z } = await subspace(cov, D, rank, random, iterations);
		// Each direction's variance, for whitening: (q Cov q^T)_kk, z = q Cov.
		const variance = new Float64Array(rank);
		for (let k = 0; k < rank; k++) for (let i = 0; i < D; i++) variance[k] += q[k * D + i] * z[k * D + i];
		const P = new Float32Array(rank * D), c = new Float32Array(rank);
		for (let k = 0; k < rank; k++) {
			const w = 1 / Math.sqrt(Math.max(variance[k], 1e-12));
			let b = 0;
			for (let i = 0; i < D; i++) {
				P[k * D + i] = q[k * D + i] * w;
				b -= P[k * D + i] * mean[i];
			}
			c[k] = b;
		}
		const total = Array.from({ length: D }, (_, i) => cov[i * D + i]).reduce((a, b) => a + b, 0);
		const kept = variance.reduce((a, b) => a + b, 0);
		return { pca: P, pca_bias: c, kept: kept / total };
	} finally {
		for (const t of [ones, gram, sums, xt, part, partSum]) t.dispose();
	}
}

// `rank` orthonormal rows spanning the top of symmetric `cov` [D, D]: subspace
// iteration, Q <- orth(Q Cov), the product on the GPU and the orthonormalising
// on the host in float64 (orthonormalRows): its Gram matrix squares the spread
// of the variances, which a middle block's features (or noise) take past
// float32's precision.
async function subspace(cov, D, rank, random, iterations) {
	const C = upload(cov);
	const start = new Float64Array(rank * D);
	for (let i = 0; i < start.length; i++) start[i] = random() - 0.5;
	const q = f32(rank * D), z = f32(rank * D);
	try {
		q.buffer.write(Float32Array.from(orthonormalRows(start, rank, D)));
		for (let it = 0; it < iterations; it++) {
			op.matmul(C, q, z, D, D, rank);
			submit();
			q.buffer.write(Float32Array.from(orthonormalRows(Float64Array.from(readFloats(z, rank * D)), rank, D)));
			if (breathing()) await breathe();
		}
		op.matmul(C, q, z, D, D, rank);
		submit();
		return { q: readFloats(q, rank * D), z: readFloats(z, rank * D) };
	} finally {
		for (const t of [C, q, z]) t.dispose();
	}
}

// The rows of Z [n, D] made orthonormal, spanning the same space: Cholesky QR
// twice over (Q = L^-1 Z for Z Z^T = L L^T, then again on Q), in float64, each
// row brought to unit length first (rows of very different lengths square to
// a Gram matrix past float64's precision), a little added to the diagonal for
// rows with next to nothing left of their own. Native code when the host has
// it (lib/host/finder_train.c3, the same steps): in QuickJS a 384 x 2560 call
// takes about fifteen seconds, and the PCA makes twenty-two.
export function orthonormalRows(Z, n, D, native = llm.trainNative) {
	if (native) return llm.orthonormalRows(Z, n, D);
	for (let pass = 0; pass < 2; pass++) {
		for (let i = 0; i < n; i++) {
			const zi = Z.subarray(i * D, (i + 1) * D);
			let s = 0;
			for (let d = 0; d < D; d++) s += zi[d] * zi[d];
			const inv = s > 0 ? 1 / Math.sqrt(s) : 0;
			for (let d = 0; d < D; d++) zi[d] *= inv;
		}
		const L = new Float64Array(n * n);
		let trace = 0;
		for (let i = 0; i < n; i++) {
			const zi = Z.subarray(i * D, (i + 1) * D);
			for (let j = 0; j <= i; j++) {
				const zj = Z.subarray(j * D, (j + 1) * D);
				let s = 0;
				for (let d = 0; d < D; d++) s += zi[d] * zj[d];
				L[i * n + j] = s;
			}
			trace += L[i * n + i];
		}
		const eps = (1e-12 * trace) / n + 1e-300;
		for (let i = 0; i < n; i++) {
			for (let j = 0; j <= i; j++) {
				let s = L[i * n + j] + (i === j ? eps : 0);
				for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
				L[i * n + j] = i === j ? Math.sqrt(Math.max(s, eps)) : s / L[j * n + j];
			}
		}
		// Forward substitution, row by row: Q_i = (Z_i - sum_k<i L_ik Q_k) / L_ii.
		for (let i = 0; i < n; i++) {
			const zi = Z.subarray(i * D, (i + 1) * D);
			for (let k = 0; k < i; k++) {
				const l = L[i * n + k], zk = Z.subarray(k * D, (k + 1) * D);
				for (let d = 0; d < D; d++) zi[d] -= l * zk[d];
			}
			const inv = 1 / L[i * n + i];
			for (let d = 0; d < D; d++) zi[d] *= inv;
		}
	}
	return Z;
}

// Adam on the GPU: each parameter is the head's own tensor, its gradient the
// step's buffer of the same layout, its moments kept beside it. Nothing comes
// back to the host until the head is saved (in QuickJS the step's loops took
// longer than the step's matmuls).
class Adam {
	constructor(params, { lr, decay }) {
		this.params = params.map((p) => {
			const m = f32(p.n), v = f32(p.n);
			zero(m);
			zero(v);
			return { ...p, m, v };
		});
		this.lr = lr;
		this.decay = decay;
		this.t = 0;
	}

	step(scale) {
		this.t++;
		const c1 = 1 - 0.9 ** this.t, c2 = 1 - 0.999 ** this.t, lr = this.lr * scale;
		for (const p of this.params) op.adamStep(p.value, p.m, p.v, p.grad, p.n, lr, c1, c2, p.decay ? this.decay : 0);
	}

	dispose() {
		for (const p of this.params) { p.m.dispose(); p.v.dispose(); }
	}
}

// The rig's hands: joints with two children or more whose whole reach is
// small (a wrist and its fingers, not a hip or a chest). The first hand's
// points (it, its children, the finger tips) are the hand head's; a hand that
// is its mirror image (wrist_L of wrist_R) is looked at mirrored. `anchors`:
// the joints of each hand the body's look finds, round which it is cropped.
// `scale`: the crop's side, a share of the joints' box. Null without hands.
export function handsOf(set, chosen) {
	const J = set.joints.length, children = Array.from({ length: J }, () => []);
	set.parents.forEach((p, j) => { if (p >= 0) children[p].push(j); });
	const ys = set.rest.map((p) => p[1]);
	const height = Math.max(...ys) - Math.min(...ys) || 1;
	const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
	const below = (j) => children[j].flatMap((c) => [c, ...below(c)]);
	const found = [];
	for (let j = 0; j < J; j++) {
		if (children[j].length < 2 || !chosen.includes(j)) continue;
		const under = below(j), reach = Math.max(...under.map((c) => dist(set.rest[c], set.rest[j])));
		if (reach >= 0.15 * height || reach < 0.02 * height) continue;
		const tips = under.filter((c) => !children[c].length && !children[j].includes(c));
		found.push({ j, reach, points: [j, ...children[j], ...tips], anchors: [j, ...children[j].filter((c) => chosen.includes(c))] });
	}
	if (!found.length) return null;
	const name = (k) => set.joints[k], first = found[0];
	const points = first.points.map(name), anchors = first.anchors.map(name);
	const hands = [{ points, anchors, mirror: false }];
	for (const h of found.slice(1)) {
		const names = h.points.map(name), mirrored = points.map(partner);
		if (name(h.j) === partner(name(first.j)) && mirrored.every((n) => names.includes(n))) hands.push({ points: mirrored, anchors: anchors.map(partner), mirror: true });
	}
	return { scale: (2.6 * first.reach) / height, points, hands };
}

// The longer side of the box round `chosen` joints of a picture's points.
function jointBox(points, chosen) {
	const xs = chosen.map((j) => points[j][0]), ys = chosen.map((j) => points[j][1]);
	return Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
}

// The body of a finder already at `path` trained as this one would be and
// without hands: its weights and metadata, so only the hands need training.
function bodyFrom(path, { names, S, D, layers, rank, hidden }) {
	if (!path || !llm.exists(path)) return null;
	const m = llm.open(path);
	try {
		const text = m.meta('finder', null);
		if (!text) return null;
		const meta = JSON.parse(text);
		if (meta.version !== FINDER_VERSION || meta.hands || meta.joints.join() !== names.join() || meta.size !== S || meta.dim !== D
			|| (meta.layers ?? []).join() !== layers.join() || meta.rank !== rank || meta.hidden !== hidden) return null;
		const weights = {};
		for (const n of ['pca', 'pca_bias', 'w1', 'b1', 'w2', 'b2']) weights[n] = Float32Array.from(m.floats(n));
		return { weights, meta };
	} finally {
		m.close();
	}
}

/**
 * Pictures through the backbone, kept as a head's features: the first `sample`
 * give the PCA, then every picture's tokens are kept as their whitened
 * components, on the GPU, so the head trains from memory. `picture(i)` makes
 * picture i's input (S x S, on the GPU) and writes its targets. Returns the
 * head (the PCA and fresh weights, `host` their copies), `features(i)` and
 * dispose().
 */
async function encodePictures(encoder, { M, S, layers, rank, hidden, J, seed, random, sample, picture, step, spent, what = 'encoding pictures' }) {
	const N = (S / PATCH) ** 2, D = encoder.dim * (layers.length + 1), perChunk = 256;
	const raw = [], chunks = [];
	let head = null, kept = 0;
	const timed = (key, t) => { spent[key] += llm.now() - t; };
	const features = (i) => [chunks[Math.floor(i / perChunk)], (i % perChunk) * N * rank * 4];
	const keep = (tokens, i) => {
		const [chunk, at] = features(i);
		const f = f32(N * rank);
		head.project(tokens, f, N);
		copy(f, chunk, N * rank * 4, 0, at);
		submit();
		f.dispose();
	};
	const dispose = () => {
		for (const t of raw) t.dispose();
		for (const t of chunks) t.dispose();
		raw.length = chunks.length = 0;
		head?.dispose();
	};
	try {
		for (let i = 0; i < M; i++) {
			let t = llm.now();
			const input = picture(i);
			timed('pictures', t);
			t = llm.now();
			const tokens = await encoder.encode(input, S, S, { tokens: true, quiet: true, layers });
			input.dispose();
			timed('backbone', t);
			t = llm.now();
			if (!head) {
				raw.push(tokens);
				if (raw.length === sample) {
					step?.(i + 1, M, 'principal components');
					const tp = llm.now();
					const p = await principal(raw, N, D, rank, random);
					spent.pca += llm.now() - tp;
					t += llm.now() - tp;
					kept = p.kept;
					llm.print(`  PCA: ${rank} of ${D} components keep ${(kept * 100).toFixed(1)}% of the variance (${sample} pictures)`);
					const w1 = llm.randomNormal(hidden * rank, seed + 1).map((v) => v * Math.sqrt(2 / rank));
					const w2 = llm.randomNormal(J * hidden, seed + 2).map((v) => v * 0.01);
					head = new Head({ pca: p.pca, pca_bias: p.pca_bias, w1, b1: new Float32Array(hidden), w2, b2: new Float32Array(J) }, { dim: D, rank, hidden, joints: J });
					head.host = { w1: Float32Array.from(w1), b1: new Float32Array(hidden), w2: Float32Array.from(w2), b2: new Float32Array(J), pca: p.pca, pca_bias: p.pca_bias };
					for (let c = 0; c * perChunk < M; c++) chunks.push(f32(Math.min(perChunk, M - c * perChunk) * N * rank));
					raw.forEach((x, k) => keep(x, k));
					for (const x of raw) x.dispose();
					raw.length = 0;
				}
			} else {
				keep(tokens, i);
				tokens.dispose();
			}
			timed('keep', t);
			step?.(i + 1, M, what);
			if (breathing()) await breathe();
		}
	} catch (e) {
		dispose();
		throw e;
	}
	return { head, features, kept, dispose };
}

/**
 * The head fitted to the kept features of pictures [0, train), scored on
 * [train, M) every five epochs. Its weights are read back into head.host.
 * Returns the last held-out scoring and the history.
 */
async function fitHead(head, features, targets, { M, train, B, S, J, rank, hidden, sigma, epochs, lr, random, step, landmarksFrom = J, what = 'training', label = '' }) {
	const G = S / PATCH, N = G * G;
	const g = stepBuffers({ B, N, J, rank, hidden });
	const adam = new Adam([
		{ value: head.g.w1, grad: g.dw1, n: hidden * rank, decay: true }, { value: head.g.b1, grad: g.db1, n: hidden },
		{ value: head.g.w2, grad: g.dw2, n: J * hidden, decay: true }, { value: head.g.b2, grad: g.db2, n: J },
	], { lr, decay: 1e-4 });
	try {
		// w2 [J, hidden] transposed for the step's dA = dS W2.
		const transposeW2 = () => op.transposeChannelSpatial(head.g.w2, g.w2t, hidden, J, 1);
		const gather = (batch) => {
			batch.forEach((i, b) => {
				const [chunk, at] = features(i);
				copy(chunk, g.x, N * rank * 4, at, b * N * rank * 4);
			});
			const t = new Float32Array(B * J * 4);
			batch.forEach((i, b) => t.set(targets.subarray(i * J * 4, (i + 1) * J * 4), b * J * 4));
			g.target.buffer.write(t);
		};
		const order = Array.from({ length: train }, (_, i) => i);
		const steps = epochs * Math.floor(train / B), warm = Math.min(200, steps / 10);
		const opts = { B, N, J, G, rank, hidden, S, landmarksFrom };
		let done = 0, lossSum = 0, lossCount = 0;
		const history = [];
		transposeW2();
		for (let epoch = 0; epoch < epochs; epoch++) {
			for (let i = order.length - 1; i > 0; i--) { const k = random.int(i + 1); [order[i], order[k]] = [order[k], order[i]]; }
			for (let b0 = 0; b0 + B <= train; b0 += B) {
				gather(order.slice(b0, b0 + B));
				const { loss } = headGradients(head, g, { B, N, J, G, rank, hidden, sigma }, false);
				for (const v of loss) lossSum += v;
				lossCount += B;
				const scale = Math.min(1, (done + 1) / warm) * 0.5 * (1 + Math.cos((Math.PI * done) / steps));
				adam.step(scale);
				transposeW2();
				done++;
				step?.(done, steps, what);
				if (done % 50 === 0 && breathing()) await breathe();
			}
			const mean = lossSum / Math.max(1, lossCount);
			lossSum = 0;
			lossCount = 0;
			if (epoch % 5 === 4 || epoch === epochs - 1) {
				const v = evaluate(head, g, features, targets, train, M, opts);
				history.push({ epoch: epoch + 1, loss: round(mean, 3), held_out_px: v.median });
				llm.print(`  ${label}epoch ${epoch + 1}/${epochs}: loss ${mean.toFixed(3)}, held out: median ${v.median.toFixed(1)}px, ${(v.within * 100).toFixed(0)}% within ${v.near}px`);
			}
		}
		const v = evaluate(head, g, features, targets, train, M, opts);
		for (const name of ['w1', 'b1', 'w2', 'b2']) head.host[name] = readFloats(head.g[name], head.host[name].length);
		return { v, history };
	} finally {
		for (const t of Object.values(g)) t.dispose();
		adam.dispose();
	}
}

/**
 * The hand head: every picture's hands cropped square, the hand filling the
 * crop as it fills the detector's (the joints' box times `hands.scale`), the
 * crop's centre and side jittered as much as the body's look strays, the left
 * hand mirrored. `images` in the body's order, the last `held` held out.
 */
async function trainHands(encoder, set, images, held, chosen, hands, request, step, spent) {
	const S = Number(request.hand_size ?? 256), seed = Number(request.seed ?? 1) + 7;
	if (S % PATCH) throw bad(`hand_size: ${S} pixels is not a multiple of ${PATCH}`);
	const layers = (request.layers ?? []).map(Number), D = encoder.dim * (layers.length + 1);
	const rank = Number(request.hand_rank ?? 256), hidden = Number(request.hand_hidden ?? 256);
	const epochs = Number(request.hand_epochs ?? request.epochs ?? 40), B = Number(request.batch ?? 32), sigma = Number(request.sigma ?? 1);
	if (!(rank >= 16 && rank <= D)) throw bad(`hand_rank: 16 to ${D}`);
	const random = generator(seed), P = hands.points.length;
	const at = (n) => set.joints.indexOf(n);
	const kinds = hands.hands.map((h) => ({ ...h, idx: h.points.map(at), anchorIdx: h.anchors.map(at) }));
	// Every picture's hands whose wrist is in it; the held-out pictures' last.
	const crops = [];
	images.forEach((im, i) => kinds.forEach((h, k) => {
		const w = im.points[h.idx[0]];
		if (w[0] >= 0 && w[1] >= 0 && w[0] < set.size && w[1] < set.size) crops.push({ i, k });
	}));
	const M = crops.length, train = crops.filter((c) => c.i < images.length - held).length;
	if (train < B) throw bad(`renders: ${train} hands to train on, too few for batches of ${B}`);
	llm.print(`  hands: ${M} crops (${M - train} held out) of ${S}x${S}, ${kinds.length} hand${kinds.length > 1 ? 's' : ''}, points ${hands.points.join(', ')}`);

	const backdrops = new Backdrops(set.size, request.backgrounds ?? null, random);
	const noise = llm.randomNormal(1 << 21, seed);
	const targets = new Float32Array(M * P * 4);
	let cached = -1, picture = null;
	const make = (c) => {
		const { i, k } = crops[c], im = images[i], h = kinds[k];
		if (cached !== i) {
			const render = llm.image.load(im.path);
			picture = composite(render, backdrops.next(), random, noise);
			cached = i;
		}
		const anchors = h.anchorIdx.map((j) => im.points[j]);
		const ax = anchors.reduce((s, p) => s + p[0], 0) / anchors.length, ay = anchors.reduce((s, p) => s + p[1], 0) / anchors.length;
		const side = Math.max(16, hands.scale * jointBox(im.points, chosen) * random.between(0.85, 1.2));
		const cx = ax + random.between(-0.18, 0.18) * side, cy = ay + random.between(-0.18, 0.18) * side;
		const affine = boxAffine([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], S, S), [a, , tx, , d, ty] = affine;
		for (let n = 0; n < P; n++) {
			const p = im.points[h.idx[n]], o = (c * P + n) * 4;
			// Rigsmith's pixels are counted from the left edge, the warp's from the first centre.
			let x = a * (p[0] - 0.5) + tx;
			const y = d * (p[1] - 0.5) + ty;
			if (h.mirror) x = S - 1 - x;
			targets[o] = (x - (PATCH - 1) / 2) / PATCH;
			targets[o + 1] = (y - (PATCH - 1) / 2) / PATCH;
			targets[o + 2] = x >= 0 && y >= 0 && x < S && y < S ? 1 : 0;
		}
		const image = uploadImage(picture);
		const m = h.mirror ? [a, 0, S - 1 - tx - a * (picture.width - 1), 0, d, ty] : affine;
		const input = warpCropGpu(image, picture, m, { width: S, height: S, mirror: h.mirror });
		image.dispose();
		return input;
	};
	const t0 = llm.now();
	const enc = await encodePictures(encoder, {
		M, S, layers, rank, hidden, J: P, seed, random, sample: Math.min(train, Number(request.pca_pictures ?? 48)),
		picture: make, step, spent, what: 'encoding hands',
	});
	try {
		const encoded = (llm.now() - t0) / 1000;
		llm.print(`  encoded ${M} hands in ${encoded.toFixed(0)}s`);
		const { v, history } = await fitHead(enc.head, enc.features, targets, {
			M, train, B, S, J: P, rank, hidden, sigma, epochs, lr: Number(request.lr ?? 1e-3), random, step, what: 'training hands', label: 'hands ',
		});
		const host = enc.head.host;
		return {
			weights: { hand_pca: host.pca, hand_pca_bias: host.pca_bias, hand_w1: host.w1, hand_b1: host.b1, hand_w2: host.w2, hand_b2: host.b2 },
			meta: {
				size: S, rank, hidden, sigma, scale: round(hands.scale, 4), points: hands.points, hands: hands.hands,
				report: {
					crops: M, held_out: M - train, pca_variance_kept: round(enc.kept, 3),
					held_out_px: { median: round(v.median, 2), mean: round(v.mean, 2), p90: round(v.p90, 2), within: round(v.within, 3), near: v.near },
					worst_points: v.perJoint.map((e, n) => ({ point: hands.points[n], px: round(e, 1) })).sort((x, y) => y.px - x.px).slice(0, 6),
					history, seconds: { encode: round(encoded, 1), total: round((llm.now() - t0) / 1000, 1) },
				},
			},
		};
	} finally {
		enc.dispose();
	}
}

/**
 * Trains a finder on `request.renders` and writes it to `request.out`.
 * Options: joints, landmarks (true), hidden_weight (0.3), layers ([]), backgrounds, rank (384), hidden (256), epochs (40),
 * batch (32), lr (0.001), sigma (1, patches), holdout (0.05), seed (1),
 * limit (pictures at most), hands (true), hand_size (256), hand_rank (256),
 * hand_hidden (256), hand_epochs (epochs). A finder already at `out` for the
 * same body and without hands keeps its body: only the hands are trained.
 * Returns the report.
 */
export async function trainFinder(encoder, request, step = null) {
	const t0 = llm.now();
	if (!request.renders) throw bad('renders: render_poses folders (one, or a list)');
	if (!request.out) throw bad('out: where to write the finder (.safetensors)');
	const set = readRenders(request.renders);
	const chosen = chooseJoints(set, request.joints ?? null);
	const landmarks = request.landmarks === false ? [] : set.landmarks;
	const names = [...chosen.map((j) => set.joints[j]), ...landmarks.map((m) => m.name)];
	// Blocks laid beside the last one's patches (dinov3.js `layers`): none by default.
	const layers = (request.layers ?? []).map(Number);
	if (layers.some((l) => !Number.isInteger(l) || l < 0 || l >= encoder.depth - 1)) throw bad(`layers: block indices from 0 to ${encoder.depth - 2}`);
	const S = set.size, G = S / PATCH, N = G * G, D = encoder.dim * (layers.length + 1), J = names.length;
	const rank = Number(request.rank ?? 384), hidden = Number(request.hidden ?? 256);
	const epochs = Number(request.epochs ?? 40), B = Number(request.batch ?? 32), sigma = Number(request.sigma ?? 1);
	if (S % PATCH) throw bad(`renders: ${S} pixels is not a multiple of ${PATCH}`);
	if (!(rank >= 16 && rank <= D)) throw bad(`rank: 16 to ${D}`);
	const seed = Number(request.seed ?? 1);
	const random = generator(seed);
	const hiddenWeight = Number(request.hidden_weight ?? 0.3);

	// The pictures, shuffled, a twentieth held out.
	const images = set.images.slice();
	for (let i = images.length - 1; i > 0; i--) { const k = random.int(i + 1); [images[i], images[k]] = [images[k], images[i]]; }
	if (request.limit) images.length = Math.min(images.length, Number(request.limit));
	const held = Math.max(1, Math.round(images.length * Number(request.holdout ?? 0.05)));
	const M = images.length, train = M - held;
	if (train < B) throw bad(`renders: ${M} pictures, too few for batches of ${B}`);
	const hands = request.hands === false ? null : handsOf(set, chosen);
	// Where the time goes, in ms: making the pictures, the backbone, the PCA,
	// projecting and keeping the features, then fitting the heads.
	const spent = { pictures: 0, backbone: 0, pca: 0, keep: 0, head: 0 };
	const kept = hands ? bodyFrom(request.out, { names, S, D, layers, rank, hidden }) : null;
	let body, bodyReport;
	if (kept) {
		llm.print(`  finder: ${request.out} already finds this body; training its hands`);
		body = kept.weights;
		bodyReport = kept.meta.report;
	} else {
		llm.print(`  finder: ${M} pictures (${held} held out) of ${S}x${S}, ${chosen.length} of ${set.joints.length} joints and ${landmarks.length} landmarks: ${names.join(', ')}`);
		// 1. Every picture through the backbone; each one's labels in its crop
		// ([image, joint, (grid x, grid y, weight, -)]).
		const backdrops = new Backdrops(S, request.backgrounds ?? null, random);
		const noise = llm.randomNormal(1 << 21, seed);
		const targets = new Float32Array(M * J * 4);
		const picture = (i) => {
			const im = images[i];
			const render = llm.image.load(im.path);
			if (render.width !== S || render.height !== S) throw bad(`renders: ${im.path} is ${render.width}x${render.height}, not ${S}`);
			const pic = composite(render, backdrops.next(), random, noise);
			// Cropped a little larger or smaller and off centre.
			const zoom = random.between(0.85, 1.2), side = S / zoom;
			const cx = S / 2 + random.between(-0.06, 0.06) * S, cy = S / 2 + random.between(-0.06, 0.06) * S;
			const affine = boxAffine([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], S, S);
			// Each point: a joint's, or a landmark's (learnt only where it shows).
			const point = (n) => (n < chosen.length ? im.points[chosen[n]] : im.marks[n - chosen.length]);
			for (let n = 0; n < J; n++) {
				const p = point(n);
				const o = (i * J + n) * 4;
				if (!p) { targets[o + 2] = 0; continue; }
				// Rigsmith's pixels are counted from the left edge, the warp's from the first centre.
				const x = affine[0] * (p[0] - 0.5) + affine[2], y = affine[4] * (p[1] - 0.5) + affine[5];
				targets[o] = (x - (PATCH - 1) / 2) / PATCH;
				targets[o + 1] = (y - (PATCH - 1) / 2) / PATCH;
				const inside = x >= 0 && y >= 0 && x < S && y < S;
				if (n < chosen.length) targets[o + 2] = inside && p[2] > 0 ? 1 : 0;
				else if (inside && p[3] === 1) targets[o + 2] = 1;
				else { targets[o] = targets[o + 1] = G / 2; targets[o + 2] = hiddenWeight; targets[o + 3] = FLAT; }
			}
			const image = uploadImage(pic);
			const input = warpCropGpu(image, pic, affine, { width: S, height: S });
			image.dispose();
			return input;
		};
		const enc = await encodePictures(encoder, {
			M, S, layers, rank, hidden, J, seed, random, sample: Math.min(train, Number(request.pca_pictures ?? 48)), picture, step, spent,
		});
		try {
			const encoded = (llm.now() - t0) / 1000;
			llm.print(`  encoded ${M} pictures in ${encoded.toFixed(0)}s`);
			// 3. The head.
			const headStart = llm.now();
			const { v, history } = await fitHead(enc.head, enc.features, targets, {
				M, train, B, S, J, rank, hidden, sigma, epochs, lr: Number(request.lr ?? 1e-3), random, step, landmarksFrom: chosen.length,
			});
			spent.head += llm.now() - headStart;
			const host = enc.head.host;
			body = { pca: host.pca, pca_bias: host.pca_bias, w1: host.w1, b1: host.b1, w2: host.w2, b2: host.b2 };
			bodyReport = {
				pictures: M, held_out: held, joints: names, size: S,
				pca_variance_kept: round(enc.kept, 3),
				held_out_px: { median: round(v.median, 2), mean: round(v.mean, 2), p90: round(v.p90, 2), within: round(v.within, 3), near: v.near },
				worst_joints: v.perJoint.map((e, n) => ({ joint: names[n], px: round(e, 1) })).sort((a, b) => b.px - a.px).slice(0, 8),
				// How sure the finder is of a landmark the picture shows, and of one it
				// hides (median scores): far apart, the fit can tell them apart.
				...(landmarks.length ? { landmark_score: { seen: round(v.seenScore, 3), hidden: round(v.hiddenScore, 3) } } : {}),
				history,
				seconds: {
					encode: round(encoded, 1), total: round((llm.now() - t0) / 1000, 1),
					...Object.fromEntries(Object.entries(spent).map(([k, ms]) => [k, round(ms / 1000, 1)])),
				},
			};
		} finally {
			enc.dispose();
		}
	}
	const hand = hands ? await trainHands(encoder, set, images, held, chosen, hands, request, step, spent) : null;
	const report = { ...bodyReport, ...(hand ? { hands: hand.meta.report } : {}) };
	saveFinder(request.out, { ...body, ...(hand?.weights ?? {}) }, {
		joints: names, size: S, dim: D, layers, rank, hidden, sigma, fill: set.fill, report,
		landmarks: landmarks.map((m) => ({ name: m.name, joint: m.joint, offset: m.offset })),
		...(hand ? { hands: hand.meta } : {}),
	});
	llm.print(`  finder written to ${request.out}: held out median ${report.held_out_px.median}px${hand ? `, hands ${hand.meta.report.held_out_px.median}px of ${hand.meta.size}` : ''}, in ${round((llm.now() - t0) / 1000, 1)}s`);
	return report;
}

const round = (v, d) => Number(v.toFixed(d));

// The GPU buffers of a training step over B pictures of N patches.
export function stepBuffers({ B, N, J, rank, hidden }) {
	const BN = B * N;
	return {
		x: f32(BN * rank), a: f32(BN * hidden), s: f32(BN * J), ds: f32(BN * J), dst: f32(BN * J),
		at: f32(BN * hidden), da: f32(BN * hidden), dat: f32(BN * hidden), xt: f32(BN * rank),
		dw1: f32(hidden * rank), db1: f32(hidden), dw2: f32(J * hidden), db2: f32(J),
		ones: upload(new Float32Array(BN).fill(1)), w2t: f32(hidden * J), target: f32(B * J * 4), loss: f32(B * J),
	};
}

/**
 * One step's loss and gradients for the features in g.x [B N, rank] and the
 * targets in g.target, the head's own weights (g.w2t its w2 transposed). The
 * gradients stay in g (dw1, db1, dw2, db2) and come back with `read`:
 * forward through the head, the heatmaps' loss, and back, every product a
 * matmul (W [out, in] times rows): dW2 = dS^T A, dA = dS W2, through the relu,
 * dW1 = dA^T X; the biases' gradients are the columns' sums (a matmul by ones).
 */
export function headGradients(head, g, { B, N, J, G, rank, hidden, sigma }, read = true) {
	const BN = B * N;
	head.hide(g.x, g.a, BN);
	head.score(g.a, g.s, BN);
	op.spatialSoftmaxXent(g.s, g.ds, g.target, g.loss, B, N, J, G, sigma, 1 / B);
	op.transposeChannelSpatial(g.ds, g.dst, J, BN, 1);
	op.transposeChannelSpatial(g.a, g.at, hidden, BN, 1);
	op.matmul(g.at, g.dst, g.dw2, hidden, BN, J);
	op.matmul(g.dst, g.ones, g.db2, J, BN, 1);
	op.matmul(g.w2t, g.ds, g.da, hidden, J, BN);
	op.reluGrad(g.da, g.a, BN * hidden);
	op.transposeChannelSpatial(g.da, g.dat, hidden, BN, 1);
	op.transposeChannelSpatial(g.x, g.xt, rank, BN, 1);
	op.matmul(g.xt, g.dat, g.dw1, rank, BN, hidden);
	op.matmul(g.dat, g.ones, g.db1, hidden, BN, 1);
	submit();
	return {
		grads: read ? { w1: readFloats(g.dw1, hidden * rank), b1: readFloats(g.db1, hidden), w2: readFloats(g.dw2, J * hidden), b2: readFloats(g.db2, J) } : null,
		loss: readFloats(g.loss, B * J),
	};
}

// The held-out pictures [from, to) scored as the finder would: each joint's
// peak against its label, in pixels.
function evaluate(head, g, features, targets, from, to, { B, N, J, G, rank, hidden, S, landmarksFrom = J }) {
	const errs = [], perJoint = new Float64Array(J), perCount = new Float64Array(J), seen = [], hid = [];
	for (let b0 = from; b0 < to; b0 += B) {
		const n = Math.min(B, to - b0);
		for (let b = 0; b < n; b++) {
			const [chunk, at] = features(b0 + b);
			copy(chunk, g.x, N * rank * 4, at, b * N * rank * 4);
		}
		head.hide(g.x, g.a, n * N);
		head.score(g.a, g.s, n * N);
		submit();
		const s = readFloats(g.s, n * N * J);
		for (let b = 0; b < n; b++) {
			const one = s.subarray(b * N * J, (b + 1) * N * J);
			for (let j = 0; j < J; j++) {
				const o = ((b0 + b) * J + j) * 4;
				if (!(targets[o + 2] > 0)) continue;
				const [p] = peaks(one, j, J, G, G, { second: 2 });
				if (j >= landmarksFrom) (targets[o + 3] > 0 ? hid : seen).push(p.mass);
				// A hidden landmark's flat target has no place to be found at.
				if (targets[o + 3] > 0) continue;
				const e = Math.hypot(p.at[0] - targets[o], p.at[1] - targets[o + 1]) * PATCH;
				errs.push(e);
				perJoint[j] += e;
				perCount[j]++;
			}
		}
	}
	errs.sort((a, b) => a - b);
	const q = (f) => errs[Math.min(errs.length - 1, Math.floor(f * errs.length))] ?? 0;
	const mid = (a) => (a.length ? a.slice().sort((x, y) => x - y)[a.length >> 1] : 0);
	const near = Math.round(S * 0.025);
	return {
		median: q(0.5), p90: q(0.9), mean: errs.reduce((a, b) => a + b, 0) / Math.max(1, errs.length),
		within: errs.filter((e) => e <= near).length / Math.max(1, errs.length), near,
		perJoint: Array.from(perJoint, (v, j) => v / Math.max(1, perCount[j])),
		seenScore: mid(seen), hiddenScore: mid(hid),
	};
}
