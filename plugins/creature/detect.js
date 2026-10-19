// A joint finder made for one rig: DINOv3's patch features under a small head
// trained on pictures of that rig in poses it can take (train.js, from
// Rigsmith's render_poses). It finds every joint it was trained on in a frame,
// with no labels, as the tracker's tracks: what /fit takes.
//
//   const finder = Finder.load('zidane.finder.safetensors');
//   const tracks = await finder.detect(encoder, frames, { region });
//
// The head, per patch: the token [D] through the PCA the trainer found
// (whitened, [K]), a hidden layer [H] with relu, a score per joint [J]. A
// joint's scores over the picture's patches, softmaxed, are where it is: the
// peak, refined over the 3x3 about it, and the mass near the peak as how sure
// (the trainer fitted the softmax to a Gaussian of `sigma` patches there). The
// second peak is kept too, as a second hypothesis, when it holds mass of its
// own: a left leg's heatmap often lights the right leg as well, and the fit
// takes whichever its motion agrees with.
//
// Frames are looked at twice: once whole (or `region`), to find the figure,
// and again cropped square round the joints found then, as the trainer framed
// its pictures (the joints' box `fill` of the side), the crop's centre and side
// smoothed over the clip.
//
// The finder file is a .safetensors: pca [K, D], pca_bias [K], w1 [H, K],
// b1 [H], w2 [J, H], b2 [J] in float32, and `finder` in its metadata, JSON:
// { version, joints, size, dim, rank, hidden, sigma, fill, report }.

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, breathe, breathing } from '../lib/gpu.js';
import { PATCH } from '../lib/dinov3.js';
import { uploadImage, warpCropGpu, boxAffine } from '../lib/warp.js';
import { bad } from './request.js';

export const FINDER_VERSION = 1;
const NAMES = ['pca', 'pca_bias', 'w1', 'b1', 'w2', 'b2'];

function upload(data, shape = [data.length]) {
	const t = f32(data.length, shape);
	t.buffer.write(data);
	return t;
}

// The head's weights on the GPU and the matmuls through them. Training keeps
// its own copies of the trainable four and writes them back here (set()).
export class Head {
	// `w` { pca, pca_bias, w1, b1, w2, b2 } as Float32Arrays.
	constructor(w, { dim, rank, hidden, joints }) {
		Object.assign(this, { dim, rank, hidden, joints });
		this.g = {
			pca: upload(w.pca, [dim, rank]), pcaBias: upload(w.pca_bias),
			w1: upload(w.w1, [rank, hidden]), b1: upload(w.b1),
			w2: upload(w.w2, [hidden, joints]), b2: upload(w.b2),
		};
	}

	set(name, data) { this.g[name].buffer.write(data); }

	dispose() {
		for (const t of Object.values(this.g)) t.dispose();
		this.g = null;
	}

	// tokens [n, D] -> features [n, K] (whitened PCA).
	project(tokens, out, n) {
		op.matmul(this.g.pca, tokens, out, this.rank, this.dim, n);
		op.biasAdd(out, this.g.pcaBias, n * this.rank, this.rank);
	}

	// features [n, K] -> hidden [n, H], relu'd.
	hide(features, out, n) {
		op.matmul(this.g.w1, features, out, this.hidden, this.rank, n);
		op.biasAdd(out, this.g.b1, n * this.hidden, this.hidden);
		op.relu(out, n * this.hidden);
	}

	// hidden [n, H] -> scores [n, J].
	score(hidden, out, n) {
		op.matmul(this.g.w2, hidden, out, this.joints, this.hidden, n);
		op.biasAdd(out, this.g.b2, n * this.joints, this.joints);
	}

	// tokens [n, D] -> scores [n, J], read back.
	scores(tokens, n) {
		const f = f32(n * this.rank), h = f32(n * this.hidden), s = f32(n * this.joints);
		try {
			this.project(tokens, f, n);
			this.hide(f, h, n);
			this.score(h, s, n);
			submit();
			return new Float32Array(s.buffer.readBytes(0, n * this.joints * 4).buffer);
		} finally {
			f.dispose();
			h.dispose();
			s.dispose();
		}
	}
}

// A finder's file: the weights and what they were trained for.
export function saveFinder(path, weights, meta) {
	const header = {}, parts = [];
	let at = 0;
	const shapes = {
		pca: [meta.rank, meta.dim], pca_bias: [meta.rank], w1: [meta.hidden, meta.rank], b1: [meta.hidden],
		w2: [meta.joints.length, meta.hidden], b2: [meta.joints.length],
	};
	for (const name of NAMES) {
		const data = weights[name];
		if (data.length !== shapes[name].reduce((a, b) => a * b, 1)) throw new Error(`finder: ${name} has ${data.length} values for shape ${shapes[name]}`);
		header[name] = { dtype: 'F32', shape: shapes[name], data_offsets: [at, at + data.length * 4] };
		parts.push(data);
		at += data.length * 4;
	}
	header.__metadata__ = { finder: JSON.stringify({ version: FINDER_VERSION, ...meta }) };
	let json = JSON.stringify(header);
	while ((8 + json.length) % 8) json += ' ';
	const bytes = new Uint8Array(8 + json.length + at);
	new DataView(bytes.buffer).setUint32(0, json.length, true);
	for (let i = 0; i < json.length; i++) bytes[8 + i] = json.charCodeAt(i);
	let p = 8 + json.length;
	for (const data of parts) {
		bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), p);
		p += data.byteLength;
	}
	llm.writeBytes(path, bytes);
}

// A joint's best peaks in its scores (column j of [N, J]) over a gw x gh grid:
// [{ at: [gx, gy], mass }], the best first, a second one at least 3 patches
// away when it holds `second` of the mass or more.
export function peaks(scores, j, J, gw, gh, { second = 0.15 } = {}) {
	const N = gw * gh;
	let m = -Infinity;
	for (let n = 0; n < N; n++) if (scores[n * J + j] > m) m = scores[n * J + j];
	const p = new Float64Array(N);
	let z = 0;
	for (let n = 0; n < N; n++) { p[n] = Math.exp(scores[n * J + j] - m); z += p[n]; }
	for (let n = 0; n < N; n++) p[n] /= z;
	const around = (b, r) => {
		const bx = b % gw, by = Math.floor(b / gw);
		let sw = 0, sx = 0, sy = 0, mass = 0;
		for (let y = Math.max(0, by - 2); y <= Math.min(gh - 1, by + 2); y++) {
			for (let x = Math.max(0, bx - 2); x <= Math.min(gw - 1, bx + 2); x++) {
				const w = p[y * gw + x];
				if (Math.abs(x - bx) <= 1 && Math.abs(y - by) <= 1) { sw += w; sx += w * x; sy += w * y; }
				if ((x - bx) ** 2 + (y - by) ** 2 <= r * r) mass += w;
			}
		}
		return { at: [sx / sw, sy / sw], mass };
	};
	let best = 0;
	for (let n = 1; n < N; n++) if (p[n] > p[best]) best = n;
	const out = [around(best, 2)];
	const bx = best % gw, by = Math.floor(best / gw);
	let next = -1;
	for (let n = 0; n < N; n++) {
		if ((n % gw - bx) ** 2 + (Math.floor(n / gw) - by) ** 2 < 9) continue;
		if (next < 0 || p[n] > p[next]) next = n;
	}
	if (next >= 0) {
		const q = around(next, 2);
		if (q.mass >= second) out.push(q);
	}
	return out;
}

// A square box of side `side` about (cx, cy).
const square = (cx, cy, side) => [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2];

// Values over frames, missing ones (null) filled from their neighbours and the
// rest averaged over a window of `half` frames either side.
function smoothed(values, half) {
	const n = values.length, known = values.map((v, i) => [i, v]).filter(([, v]) => v !== null);
	if (!known.length) return values.map(() => null);
	const filled = values.map((v, i) => {
		if (v !== null) return v;
		let a = null, b = null;
		for (const k of known) { if (k[0] < i) a = k; else { b = k; break; } }
		if (!a) return b[1];
		if (!b) return a[1];
		return a[1] + ((b[1] - a[1]) * (i - a[0])) / (b[0] - a[0]);
	});
	return filled.map((_, i) => {
		let s = 0, c = 0;
		for (let k = Math.max(0, i - half); k <= Math.min(n - 1, i + half); k++) { s += filled[k]; c++; }
		return s / c;
	});
}

export class Finder {
	constructor(head, meta) {
		this.head = head;
		this.meta = meta;
		this.joints = meta.joints;
		this.size = meta.size;
		this.fill = meta.fill ?? [0.45, 0.75];
	}

	static load(path) {
		if (!llm.exists(path)) throw bad(`finder: ${path} does not exist (train one with POST /train)`);
		const m = llm.open(path);
		try {
			const text = m.meta('finder', null);
			if (!text) throw bad(`finder: ${path} is not a joint finder (no "finder" in its metadata)`);
			const meta = JSON.parse(text);
			if (meta.version !== FINDER_VERSION) throw bad(`finder: ${path} is version ${meta.version}; this plugin reads ${FINDER_VERSION}`);
			const w = {};
			for (const name of NAMES) w[name] = m.floats(name);
			const head = new Head(w, { dim: meta.dim, rank: meta.rank, hidden: meta.hidden, joints: meta.joints.length });
			const finder = new Finder(head, meta);
			finder.path = path;
			return finder;
		} finally {
			m.close();
		}
	}

	dispose() {
		this.head?.dispose();
		this.head = null;
	}

	// Every joint in `box` (frame pixels, square) of `img`: per joint its peaks
	// in frame pixels, [{ at: [x, y], score }].
	async findIn(encoder, img, box) {
		const S = this.size, G = S / PATCH, N = G * G, J = this.joints.length;
		if (encoder.dim !== this.meta.dim) throw bad(`finder: trained on a ${this.meta.dim}-wide backbone, the loaded one is ${encoder.dim}`);
		const affine = boxAffine(box, S, S);
		const image = uploadImage(img);
		const input = warpCropGpu(image, img, affine, { width: S, height: S });
		image.dispose();
		const tokens = await encoder.encode(input, S, S, { tokens: true, quiet: true });
		input.dispose();
		let scores;
		try {
			scores = this.head.scores(tokens, N);
		} finally {
			tokens.dispose();
		}
		const [a, , tx, , d, ty] = affine;
		const toFrame = ([gx, gy]) => [(gx * PATCH + (PATCH - 1) / 2 - tx) / a, (gy * PATCH + (PATCH - 1) / 2 - ty) / d];
		const out = [];
		for (let j = 0; j < J; j++) out.push(peaks(scores, j, J, G, G).map((q) => ({ at: toFrame(q.at), score: q.mass })));
		return out;
	}

	// The joints the first peaks of `found` put with score `least` or more:
	// their box's centre and longer side, or null.
	static boxOf(found, least = 0.3) {
		const pts = found.map((q) => q[0]).filter((q) => q.score >= least).map((q) => q.at);
		if (pts.length < 3) return null;
		const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
		const x1 = Math.min(...xs), x2 = Math.max(...xs), y1 = Math.min(...ys), y2 = Math.max(...ys);
		return { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, side: Math.max(x2 - x1, y2 - y1) };
	}

	/**
	 * Every frame's joints, as track.js's tracks. `frames` { count, image(k) };
	 * `region` [x1, y1, x2, y2] where the figure is (the whole frame when null);
	 * a keyframe every `keyEvery` frames (the fit starts from those). `every`:
	 * the first, whole-frame look on every so many frames only.
	 */
	async detect(encoder, frames, { region = null, keyEvery = 6, every = 3, onFrame = null } = {}) {
		const count = frames.count, t0 = llm.now();
		const first = frames.image(0);
		const whole = region ?? [0, 0, first.width, first.height];
		const wide = square((whole[0] + whole[2]) / 2, (whole[1] + whole[3]) / 2, Math.max(whole[2] - whole[0], whole[3] - whole[1]));
		const looks = Math.ceil(count / every) + count;
		let done = 0;

		// The figure, frame by frame: where its joints are when the frame is
		// looked at whole.
		const cx = new Array(count).fill(null), cy = new Array(count).fill(null), side = new Array(count).fill(null);
		for (let k = 0; k < count; k += every) {
			const box = Finder.boxOf(await this.findIn(encoder, frames.image(k), wide));
			if (box) { cx[k] = box.cx; cy[k] = box.cy; side[k] = box.side; }
			onFrame?.(++done, looks);
			if (breathing()) await breathe();
		}
		if (!side.some((v) => v !== null)) throw bad('finder: found no figure in the frames (is it the rig the finder was trained on? give region)');
		const fillMid = (this.fill[0] + this.fill[1]) / 2;
		const scx = smoothed(cx, 2), scy = smoothed(cy, 2), sside = smoothed(side, 4);

		// Then each frame cropped round it.
		const J = this.joints.length, found = [], boxes = [];
		for (let k = 0; k < count; k++) {
			const box = square(scx[k], scy[k], Math.max(32, sside[k] / fillMid));
			boxes.push(box.map((v) => Math.round(v * 10) / 10));
			found.push(await this.findIn(encoder, frames.image(k), box));
			onFrame?.(++done, looks);
			if (breathing()) await breathe();
		}

		// As the tracker's hypotheses: one per stretch from a keyframe, the
		// first peaks; and one over every frame with the second peaks.
		const round = (p) => [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100];
		const score = (s) => Math.round(s * 1000) / 1000;
		const hypotheses = [];
		for (let from = 0; from < count; from += keyEvery) {
			const to = Math.min(count, from + keyEvery), ks = [];
			for (let k = from; k < to; k++) ks.push(k);
			hypotheses.push({
				from, direction: 1, frames: ks,
				tracks: ks.map((k) => found[k].map((q) => round(q[0].at))),
				scores: ks.map((k) => found[k].map((q) => score(q[0].score))),
			});
		}
		const all = Array.from({ length: count }, (_, k) => k);
		if (found.some((f) => f.some((q) => q.length > 1))) {
			hypotheses.push({
				from: 0, direction: 1, frames: all,
				tracks: all.map((k) => found[k].map((q) => (q[1] ? round(q[1].at) : null))),
				scores: all.map((k) => found[k].map((q) => (q[1] ? score(q[1].score) : null))),
			});
		}
		const seconds = (llm.now() - t0) / 1000;
		const sure = found.flat().map((q) => q[0].score).sort((a, b) => a - b);
		llm.print(`  found ${J} joints in ${count} frames in ${seconds.toFixed(1)}s (${looks} looks), median score ${sure[sure.length >> 1].toFixed(2)}`);
		return {
			points: this.joints.slice(), frames: count, size: [first.width, first.height], region: whole, width: this.size,
			boxes, hypotheses, seconds: Math.round(seconds * 10) / 10,
		};
	}
}
