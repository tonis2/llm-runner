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
// are too small for 16-pixel patches to tell apart.

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, copy, breathe, breathing } from '../lib/gpu.js';
import { PATCH } from '../lib/dinov3.js';
import { uploadImage, warpCropGpu, boxAffine } from '../lib/warp.js';
import { Head, saveFinder, peaks } from './detect.js';
import { bad } from './request.js';

const IMAGE = /\.(png|jpe?g)$/i;

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
			if (!head) head = { joints: set.joints, parents: set.parents, rest: set.rest, size: set.size, height: set.height, fill: set.fill ?? [0.45, 0.75] };
			if (set.joints.join() !== head.joints.join()) throw bad(`renders: ${dir}/${e.name} is another rig's (its joints differ from the first set's)`);
			if (set.size !== head.size) throw bad(`renders: ${dir}/${e.name} is ${set.size} pixels; the first set is ${head.size}`);
			for (const im of set.images) images.push({ path: `${dir}/${im.file}`, points: im.points, camera: im.camera });
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

	// One made-up background, size x size x 3.
	made() {
		const S = this.size, r = this.random, px = new Uint8Array(S * S * 3);
		const kind = r.int(4);
		if (kind === 0) {
			// Flat, with a slight gradient.
			const a = this.colour(), b = a.map((v) => Math.max(0, Math.min(255, v + r.between(-40, 40))));
			for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) for (let c = 0; c < 3; c++) px[(y * S + x) * 3 + c] = a[c] + ((b[c] - a[c]) * y) / S;
		} else if (kind === 1) {
			// Sky and ground.
			const sky = this.colour(), ground = this.colour(), horizon = r.between(0.2, 0.8) * S;
			for (let y = 0; y < S; y++) {
				const col = y < horizon ? sky : ground, shade = y < horizon ? 1 - (0.3 * y) / S : 0.7 + (0.3 * y) / S;
				for (let x = 0; x < S; x++) for (let c = 0; c < 3; c++) px[(y * S + x) * 3 + c] = Math.min(255, col[c] * shade);
			}
		} else if (kind === 2) {
			// Tiles.
			const a = this.colour(), b = this.colour(), t = 8 + r.int(56);
			for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
				const col = (Math.floor(x / t) + Math.floor(y / t)) % 2 ? a : b;
				for (let c = 0; c < 3; c++) px[(y * S + x) * 3 + c] = col[c];
			}
		} else {
			// A soft colour field: a coarse random grid, smoothly blown up.
			const g = 3 + r.int(6), grid = Array.from({ length: (g + 1) * (g + 1) }, () => this.colour());
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
	photo() {
		const S = this.size, r = this.random, img = this.photos[r.int(this.photos.length)], px = new Uint8Array(S * S * 3);
		const side = Math.min(img.width, img.height) * r.between(0.3, 1);
		const x0 = r.between(0, img.width - side), y0 = r.between(0, img.height - side), flip = r() < 0.5, ch = img.channels;
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
export function composite(render, backdrop, random, noise) {
	const S = render.width, ch = render.channels, src = render.pixels, out = new Uint8ClampedArray(S * S * 3);
	const bright = random.between(0.7, 1.3), lift = random.between(-20, 20), grain = random() < 0.5 ? random.between(0, 8) : 0;
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
async function principal(raw, N, D, rank, random, iterations = 10) {
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
// iteration, Q <- orth(Q Cov), orthonormalised by Cholesky (Q = L^-1 Z for
// Z Z^T = L L^T), the products on the GPU.
async function subspace(cov, D, rank, random, iterations) {
	const C = upload(cov);
	const start = new Float32Array(rank * D);
	for (let i = 0; i < start.length; i++) start[i] = random() - 0.5;
	const q = upload(start), z = f32(rank * D), zt = f32(rank * D), gram = f32(rank * rank), inv = f32(rank * rank);
	try {
		for (let it = 0; it <= iterations; it++) {
			// The first pass orthonormalises the random start alone.
			if (it > 0) op.matmul(C, q, z, D, D, rank);
			else copy(q, z, rank * D * 4);
			op.matmul(z, z, gram, rank, D, rank);
			submit();
			inv.buffer.write(choleskyInverse(readFloats(gram, rank * rank), rank));
			op.transposeChannelSpatial(z, zt, D, rank, 1);
			op.matmul(zt, inv, q, D, rank, rank);
			submit();
			if (breathing()) await breathe();
		}
		op.matmul(C, q, z, D, D, rank);
		submit();
		return { q: readFloats(q, rank * D), z: readFloats(z, rank * D) };
	} finally {
		for (const t of [C, q, z, zt, gram, inv]) t.dispose();
	}
}

// L^-1 for the Cholesky factor of symmetric positive `g` [n, n] (a little
// added to the diagonal, for directions with next to no variance).
function choleskyInverse(g, n) {
	let trace = 0;
	for (let i = 0; i < n; i++) trace += g[i * n + i];
	const eps = (1e-9 * trace) / n + 1e-30;
	const L = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j <= i; j++) {
			let s = g[i * n + j] + (i === j ? eps : 0);
			for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
			L[i * n + j] = i === j ? Math.sqrt(Math.max(s, eps)) : s / L[j * n + j];
		}
	}
	const inv = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		inv[i * n + i] = 1 / L[i * n + i];
		for (let j = 0; j < i; j++) {
			let s = 0;
			for (let k = j; k < i; k++) s += L[i * n + k] * inv[k * n + j];
			inv[i * n + j] = -s / L[i * n + i];
		}
	}
	return Float32Array.from(inv);
}

// Adam over Float32Arrays (the head is small; the host keeps the master copy).
class Adam {
	constructor(params, { lr, decay }) {
		this.params = params.map((p) => ({ ...p, m: new Float32Array(p.value.length), v: new Float32Array(p.value.length) }));
		this.lr = lr;
		this.decay = decay;
		this.t = 0;
	}

	step(grads, scale) {
		this.t++;
		const c1 = 1 - 0.9 ** this.t, c2 = 1 - 0.999 ** this.t, lr = this.lr * scale;
		for (const p of this.params) {
			const g = grads[p.name], { value, m, v } = p, decay = p.decay ? this.decay : 0;
			for (let i = 0; i < value.length; i++) {
				m[i] = 0.9 * m[i] + 0.1 * g[i];
				v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
				value[i] -= lr * ((m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8) + decay * value[i]);
			}
		}
	}
}

/**
 * Trains a finder on `request.renders` and writes it to `request.out`.
 * Options: joints, backgrounds, rank (384), hidden (256), epochs (40),
 * batch (32), lr (0.001), sigma (1, patches), holdout (0.05), seed (1),
 * limit (pictures at most). Returns the report.
 */
export async function trainFinder(encoder, request, step = null) {
	const t0 = llm.now();
	if (!request.renders) throw bad('renders: render_poses folders (one, or a list)');
	if (!request.out) throw bad('out: where to write the finder (.safetensors)');
	const set = readRenders(request.renders);
	const chosen = chooseJoints(set, request.joints ?? null);
	const names = chosen.map((j) => set.joints[j]);
	const S = set.size, G = S / PATCH, N = G * G, D = encoder.dim, J = chosen.length;
	const rank = Number(request.rank ?? 384), hidden = Number(request.hidden ?? 256);
	const epochs = Number(request.epochs ?? 40), B = Number(request.batch ?? 32), sigma = Number(request.sigma ?? 1);
	if (S % PATCH) throw bad(`renders: ${S} pixels is not a multiple of ${PATCH}`);
	if (!(rank >= 16 && rank <= D)) throw bad(`rank: 16 to ${D}`);
	const random = generator(Number(request.seed ?? 1));

	// The pictures, shuffled, a twentieth held out.
	const images = set.images.slice();
	for (let i = images.length - 1; i > 0; i--) { const k = random.int(i + 1); [images[i], images[k]] = [images[k], images[i]]; }
	if (request.limit) images.length = Math.min(images.length, Number(request.limit));
	const held = Math.max(1, Math.round(images.length * Number(request.holdout ?? 0.05)));
	const M = images.length, train = M - held;
	if (train < B) throw bad(`renders: ${M} pictures, too few for batches of ${B}`);
	llm.print(`  finder: ${M} pictures (${held} held out) of ${S}x${S}, ${J} of ${set.joints.length} joints: ${names.join(', ')}`);

	// 1. Every picture through the backbone; each one's labels in its crop
	// ([image, joint, (grid x, grid y, weight, -)]).
	const backdrops = new Backdrops(S, request.backgrounds ?? null, random);
	const noise = llm.randomNormal(1 << 21, Number(request.seed ?? 1));
	const targets = new Float32Array(M * J * 4);
	const sample = Math.min(train, Number(request.pca_pictures ?? 48));
	const raw = [];
	let head = null, chunks = [], perChunk = 256, kept = 0;
	const features = (i) => [chunks[Math.floor(i / perChunk)], (i % perChunk) * N * rank * 4];
	const keep = (tokens, i) => {
		const [chunk, at] = features(i);
		const f = f32(N * rank);
		head.project(tokens, f, N);
		copy(f, chunk, N * rank * 4, 0, at);
		submit();
		f.dispose();
	};
	try {
		for (let i = 0; i < M; i++) {
			const im = images[i];
			const render = llm.image.load(im.path);
			if (render.width !== S || render.height !== S) throw bad(`renders: ${im.path} is ${render.width}x${render.height}, not ${S}`);
			const picture = composite(render, backdrops.next(), random, noise);
			// Cropped a little larger or smaller and off centre.
			const zoom = random.between(0.85, 1.2), side = S / zoom;
			const cx = S / 2 + random.between(-0.06, 0.06) * S, cy = S / 2 + random.between(-0.06, 0.06) * S;
			const box = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2];
			const affine = boxAffine(box, S, S);
			chosen.forEach((j, n) => {
				// Rigsmith's pixels are counted from the left edge, the warp's from the first centre.
				const x = affine[0] * (im.points[j][0] - 0.5) + affine[2], y = affine[4] * (im.points[j][1] - 0.5) + affine[5];
				const o = (i * J + n) * 4;
				targets[o] = (x - (PATCH - 1) / 2) / PATCH;
				targets[o + 1] = (y - (PATCH - 1) / 2) / PATCH;
				targets[o + 2] = x >= 0 && y >= 0 && x < S && y < S && im.points[j][2] > 0 ? 1 : 0;
			});
			const image = uploadImage(picture);
			const input = warpCropGpu(image, picture, affine, { width: S, height: S });
			image.dispose();
			const tokens = await encoder.encode(input, S, S, { tokens: true, quiet: true });
			input.dispose();
			if (!head) {
				raw.push(tokens);
				if (raw.length === sample) {
					// 2. The PCA, from the pictures so far; then they are kept as features.
					step?.(i + 1, M, 'principal components');
					const p = await principal(raw, N, D, rank, random);
					kept = p.kept;
					llm.print(`  PCA: ${rank} of ${D} components keep ${(kept * 100).toFixed(1)}% of the variance (${sample} pictures)`);
					const w1 = llm.randomNormal(hidden * rank, Number(request.seed ?? 1) + 1).map((v) => v * Math.sqrt(2 / rank));
					const w2 = llm.randomNormal(J * hidden, Number(request.seed ?? 1) + 2).map((v) => v * 0.01);
					head = new Head({ pca: p.pca, pca_bias: p.pca_bias, w1, b1: new Float32Array(hidden), w2, b2: new Float32Array(J) }, { dim: D, rank, hidden, joints: J });
					head.host = { w1: Float32Array.from(w1), b1: new Float32Array(hidden), w2: Float32Array.from(w2), b2: new Float32Array(J), pca: p.pca, pca_bias: p.pca_bias };
					for (let c = 0; c * perChunk < M; c++) chunks.push(f32(Math.min(perChunk, M - c * perChunk) * N * rank));
					raw.forEach((t, k) => keep(t, k));
					for (const t of raw) t.dispose();
					raw.length = 0;
				}
			} else {
				keep(tokens, i);
				tokens.dispose();
			}
			step?.(i + 1, M, 'encoding pictures');
			if (breathing()) await breathe();
		}
		const encoded = (llm.now() - t0) / 1000;
		llm.print(`  encoded ${M} pictures in ${encoded.toFixed(0)}s`);

		// 3. The head.
		const BN = B * N, g = stepBuffers({ B, N, J, rank, hidden });
		const host = head.host;
		const adam = new Adam([
			{ name: 'w1', value: host.w1, decay: true }, { name: 'b1', value: host.b1 },
			{ name: 'w2', value: host.w2, decay: true }, { name: 'b2', value: host.b2 },
		], { lr: Number(request.lr ?? 1e-3), decay: 1e-4 });
		const writeWeights = () => {
			head.set('w1', host.w1); head.set('b1', host.b1); head.set('w2', host.w2); head.set('b2', host.b2);
			const t = new Float32Array(hidden * J);
			for (let j = 0; j < J; j++) for (let h = 0; h < hidden; h++) t[h * J + j] = host.w2[j * hidden + h];
			g.w2t.buffer.write(t);
		};
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
		let done = 0, lossSum = 0, lossCount = 0;
		const history = [];
		writeWeights();
		for (let epoch = 0; epoch < epochs; epoch++) {
			for (let i = order.length - 1; i > 0; i--) { const k = random.int(i + 1); [order[i], order[k]] = [order[k], order[i]]; }
			for (let b0 = 0; b0 + B <= train; b0 += B) {
				gather(order.slice(b0, b0 + B));
				const { grads, loss } = headGradients(head, g, { B, N, J, G, rank, hidden, sigma });
				for (const v of loss) lossSum += v;
				lossCount += B;
				const scale = Math.min(1, (done + 1) / warm) * 0.5 * (1 + Math.cos((Math.PI * done) / steps));
				adam.step(grads, scale);
				writeWeights();
				done++;
				step?.(done, steps, 'training');
				if (done % 50 === 0 && breathing()) await breathe();
			}
			const mean = lossSum / Math.max(1, lossCount);
			lossSum = 0;
			lossCount = 0;
			if (epoch % 5 === 4 || epoch === epochs - 1) {
				const v = evaluate(head, g, features, targets, train, M, { B, N, J, G, rank, hidden, S });
				history.push({ epoch: epoch + 1, loss: round(mean, 3), held_out_px: v.median });
				llm.print(`  epoch ${epoch + 1}/${epochs}: loss ${mean.toFixed(3)}, held out: median ${v.median.toFixed(1)}px, ${(v.within * 100).toFixed(0)}% within ${v.near}px`);
			}
		}
		const v = evaluate(head, g, features, targets, train, M, { B, N, J, G, rank, hidden, S }, true);
		for (const t of Object.values(g)) t.dispose();

		const report = {
			pictures: M, held_out: held, joints: names, size: S,
			pca_variance_kept: round(kept, 3),
			held_out_px: { median: round(v.median, 2), mean: round(v.mean, 2), p90: round(v.p90, 2), within: round(v.within, 3), near: v.near },
			worst_joints: v.perJoint.map((e, n) => ({ joint: names[n], px: round(e, 1) })).sort((a, b) => b.px - a.px).slice(0, 8),
			history,
			seconds: { encode: round(encoded, 1), total: round((llm.now() - t0) / 1000, 1) },
		};
		saveFinder(request.out, { pca: host.pca, pca_bias: host.pca_bias, w1: host.w1, b1: host.b1, w2: host.w2, b2: host.b2 }, {
			joints: names, size: S, dim: D, rank, hidden, sigma, fill: set.fill, report,
		});
		llm.print(`  finder written to ${request.out}: held out median ${report.held_out_px.median}px, in ${report.seconds.total}s`);
		return report;
	} finally {
		for (const t of raw) t.dispose();
		for (const t of chunks) t.dispose();
		head?.dispose();
	}
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
 * targets in g.target, the head's own weights (g.w2t its w2 transposed):
 * forward through the head, the heatmaps' loss, and back, every product a
 * matmul (W [out, in] times rows): dW2 = dS^T A, dA = dS W2, through the relu,
 * dW1 = dA^T X; the biases' gradients are the columns' sums (a matmul by ones).
 */
export function headGradients(head, g, { B, N, J, G, rank, hidden, sigma }) {
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
		grads: { w1: readFloats(g.dw1, hidden * rank), b1: readFloats(g.db1, hidden), w2: readFloats(g.dw2, J * hidden), b2: readFloats(g.db2, J) },
		loss: readFloats(g.loss, B * J),
	};
}

// The held-out pictures [from, to) scored as the finder would: each joint's
// peak against its label, in pixels.
function evaluate(head, g, features, targets, from, to, { B, N, J, G, rank, hidden, S }) {
	const errs = [], perJoint = new Float64Array(J), perCount = new Float64Array(J);
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
				const e = Math.hypot(p.at[0] - targets[o], p.at[1] - targets[o + 1]) * PATCH;
				errs.push(e);
				perJoint[j] += e;
				perCount[j]++;
			}
		}
	}
	errs.sort((a, b) => a - b);
	const q = (f) => errs[Math.min(errs.length - 1, Math.floor(f * errs.length))] ?? 0;
	const near = Math.round(S * 0.025);
	return {
		median: q(0.5), p90: q(0.9), mean: errs.reduce((a, b) => a + b, 0) / Math.max(1, errs.length),
		within: errs.filter((e) => e <= near).length / Math.max(1, errs.length), near,
		perJoint: Array.from(perJoint, (v, j) => v / Math.max(1, perCount[j])),
	};
}
