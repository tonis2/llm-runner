// Named points followed through video frames with DINOv3's patch features
// alone, on the GPU.
//
//   const tracker = new Tracker(encoder);           // a loaded lib/dinov3.js DinoV3
//   const tracks = await tracker.track(frames, { labels, region, width });
//
// Each labelled keyframe starts a track forward to the last frame and one
// backward to the first. A point's descriptor is the feature under its label
// on the keyframe plus the one under where it was the frame before, matched
// within `radius` patches of there, and refined to a fraction of a patch by a
// soft-argmax over the 3x3 around the best. The features stay on the GPU: the
// descriptors are sampled with a matmul by bilinear weights, every point is
// scored against every patch with another, and only the scores come back. All
// the tracks going one way share one sweep through the frames, so a frame is
// encoded at most twice.
//
// `frames` is { count, image(k) } (request.js); `labels` { frame: { point:
// [x, y] } } in frame pixels; `region` [x1, y1, x2, y2] the part of every frame
// to look at (the whole frame when null), resized to `width` (a multiple of
// 16) wide. The result has the tracks in frame pixels: { points, frames,
// size, region, hypotheses: [{ from, direction, frames, tracks, scores }] },
// `tracks[i][p]` point p's [x, y] on frame `frames[i]` (null where the
// hypothesis does not have it) and `scores[i][p]` how much it still looks like
// its label (1 on the keyframe).

import { llm, f32 } from '../lib/llm.js';
import * as op from '../lib/ops.js';
import { submit, copy, breathe } from '../lib/gpu.js';
import { PATCH } from '../lib/dinov3.js';
import { uploadImage, warpCropGpu, boxAffine } from '../lib/warp.js';
import { bad } from './request.js';

function upload(data) {
	const t = f32(data.length);
	t.buffer.write(data);
	return t;
}

export class Tracker {
	constructor(encoder, { radius = 5, temperature = 0.02 } = {}) {
		this.enc = encoder;
		this.radius = radius;
		this.temperature = temperature;
		const D = encoder.dim;
		// rmsNorm by this weight is an L2 normalisation: x / |x|.
		this.unit = upload(new Float32Array(D).fill(1 / Math.sqrt(D)));
	}

	dispose() {
		this.unit?.dispose();
		this.unit = null;
	}

	async track(frames, { labels, region = null, width = 1280, onFrame = null } = {}) {
		const count = frames.count;
		const keys = Object.keys(labels).map(Number).sort((a, b) => a - b);
		if (!keys.length) throw bad('labels: no keyframe');
		for (const k of keys) if (!(Number.isInteger(k) && k >= 0 && k < count)) throw bad(`labels: frame ${k} is not one of the ${count} frames`);
		const points = [...new Set(keys.flatMap((k) => Object.keys(labels[k])))];
		const first = frames.image(keys[0]);
		const box = region ?? [0, 0, first.width, first.height];
		const W = Math.max(PATCH, Math.round(width / PATCH) * PATCH);
		const H = Math.max(PATCH, Math.round((W * (box[3] - box[1])) / (box[2] - box[0]) / PATCH) * PATCH);
		const affine = boxAffine(box, W, H);
		this.geo = { W, H, GW: W / PATCH, GH: H / PATCH, affine, imageSize: [first.width, first.height] };

		// One hypothesis per keyframe and direction that has frames to go.
		const hyps = [];
		for (const k of keys) {
			const named = points.filter((p) => labels[k][p]);
			if (k < count - 1) hyps.push({ from: k, direction: 1, points: named });
			if (k > 0) hyps.push({ from: k, direction: -1, points: named });
		}
		const sweeps = [1, -1].map((dir) => hyps.filter((h) => h.direction === dir)).filter((s) => s.length);
		const total = sweeps.reduce((n, s) => n + (s[0].direction > 0 ? count - Math.min(...s.map((h) => h.from)) : Math.max(...s.map((h) => h.from)) + 1), 0);
		let done = 0;
		const t0 = llm.now();
		for (const sweep of sweeps) {
			await this.sweep(frames, sweep, labels, () => {
				done++;
				onFrame?.(done, total);
			});
		}
		const seconds = (llm.now() - t0) / 1000;
		llm.print(`  tracked ${points.length} points, ${hyps.length} tracks, ${done} frames encoded in ${seconds.toFixed(1)}s (${(seconds / Math.max(1, done)).toFixed(2)}s a frame), ${this.geo.GW}x${this.geo.GH} patches`);
		return {
			points, frames: count, size: this.geo.imageSize, region: box, width: W,
			hypotheses: hyps.map((h) => ({
				from: h.from, direction: h.direction, frames: h.frames,
				tracks: h.tracks.map((row) => points.map((p) => (h.points.includes(p) ? row[h.points.indexOf(p)] : null))),
				scores: h.scores.map((row) => points.map((p) => (h.points.includes(p) ? row[h.points.indexOf(p)] : null))),
			})),
			seconds: Math.round(seconds * 10) / 10,
		};
	}

	// Frame pixels to grid cells and back.
	toGrid([x, y]) {
		const [a, , tx, , d, ty] = this.geo.affine;
		return [(a * x + tx - (PATCH - 1) / 2) / PATCH, (d * y + ty - (PATCH - 1) / 2) / PATCH];
	}

	toImage([gx, gy]) {
		const [a, , tx, , d, ty] = this.geo.affine;
		return [(gx * PATCH + (PATCH - 1) / 2 - tx) / a, (gy * PATCH + (PATCH - 1) / 2 - ty) / d];
	}

	// A frame's features: the normalised tokens [N, D] and the same channel
	// first [D, N], for sampling.
	async features(frames, k) {
		const { W, H, affine } = this.geo, D = this.enc.dim, N = this.geo.GW * this.geo.GH;
		const img = frames.image(k);
		const image = uploadImage(img);
		const input = warpCropGpu(image, img, affine, { width: W, height: H });
		image.dispose();
		const raw = await this.enc.encode(input, H, W, { tokens: true, quiet: true });
		input.dispose();
		const rows = f32(N * D, [D, N]), cols = f32(N * D, [N, D]);
		op.rmsNorm(raw, this.unit, rows, D, N, 1e-12 / D);
		op.transposeChannelSpatial(rows, cols, D, N, 1);
		submit();
		raw.dispose();
		return { rows, cols };
	}

	// Bilinear weights [K, N] putting row r on its point's four patches.
	weights(at, K) {
		const { GW, GH } = this.geo, N = GW * GH;
		const w = new Float32Array(K * N);
		for (const [r, g] of at) {
			const x0 = Math.max(0, Math.min(GW - 2, Math.floor(g[0]))), y0 = Math.max(0, Math.min(GH - 2, Math.floor(g[1])));
			const fx = Math.min(1, Math.max(0, g[0] - x0)), fy = Math.min(1, Math.max(0, g[1] - y0));
			w[r * N + y0 * GW + x0] += (1 - fx) * (1 - fy);
			w[r * N + y0 * GW + x0 + 1] += fx * (1 - fy);
			w[r * N + (y0 + 1) * GW + x0] += (1 - fx) * fy;
			w[r * N + (y0 + 1) * GW + x0 + 1] += fx * fy;
		}
		return w;
	}

	// The tracks one way. Every hypothesis has its rows in the stacked
	// matrices from the start; rows not started yet have no weights.
	async sweep(frames, hyps, labels, tick) {
		const D = this.enc.dim, dir = hyps[0].direction, count = frames.count;
		const N = this.geo.GW * this.geo.GH;
		let K = 0;
		for (const h of hyps) {
			h.row = K;
			K += h.points.length;
			h.frames = [];
			h.tracks = [];
			h.scores = [];
		}
		const start = dir > 0 ? Math.min(...hyps.map((h) => h.from)) : Math.max(...hyps.map((h) => h.from));
		const end = dir > 0 ? count - 1 : 0;
		const g = {
			wts: f32(K * N), sampled: f32(K * D), ref: f32(K * D), query: f32(2 * K * D, [D, 2 * K]), scores: f32(N * 2 * K),
		};
		g.ref.buffer.write(new Float32Array(K * D));
		const at = new Array(K).fill(null);                 // each row's grid position the frame before
		let prev = null;
		try {
			for (let k = start; dir > 0 ? k <= end : k >= end; k += dir) {
				const cur = await this.features(frames, k);
				const live = hyps.filter((h) => (dir > 0 ? h.from < k : h.from > k));
				if (live.length && prev) {
					// Descriptors: the label's plus where each point was, then every
					// patch scored against those and against the label's alone.
					const rows = live.flatMap((h) => h.points.map((_, j) => [h.row + j, at[h.row + j]]));
					g.wts.buffer.write(this.weights(rows, K));
					op.matmul(prev, g.wts, g.sampled, D, N, K);
					op.rmsNorm(g.sampled, this.unit, g.query.view(0, K * D * 4), D, K, 1e-12 / D);
					op.add(g.query.view(0, K * D * 4), g.ref, K * D);
					copy(g.ref, g.query, K * D * 4, 0, K * D * 4);
					op.matmul(g.query, cur.rows, g.scores, 2 * K, D, N);
					submit();
					const sc = new Float32Array(g.scores.buffer.readBytes(0, N * 2 * K * 4).buffer);
					for (const h of live) {
						const row = [], score = [];
						for (let j = 0; j < h.points.length; j++) {
							const r = h.row + j;
							const m = this.match(sc, 2 * K, r, at[r]);
							at[r] = m.at;
							row.push(this.toImage(m.at).map((v) => Math.round(v * 100) / 100));
							const p = Math.round(m.at[1]) * this.geo.GW + Math.round(m.at[0]);
							score.push(Math.round(sc[p * 2 * K + K + r] * 1000) / 1000);
						}
						h.frames.push(k);
						h.tracks.push(row);
						h.scores.push(score);
					}
				}
				// Hypotheses starting here: their labels, and the features under them.
				const starting = hyps.filter((h) => h.from === k);
				if (starting.length) {
					const rows = starting.flatMap((h) => h.points.map((p, j) => [h.row + j, this.toGrid(labels[k][p])]));
					g.wts.buffer.write(this.weights(rows, K));
					op.matmul(cur.cols, g.wts, g.sampled, D, N, K);
					const fresh = f32(K * D);
					op.rmsNorm(g.sampled, this.unit, fresh, D, K, 1e-12 / D);
					for (const h of starting) copy(fresh, g.ref, h.points.length * D * 4, h.row * D * 4, h.row * D * 4);
					submit();
					fresh.dispose();
					for (const h of starting) {
						h.points.forEach((p, j) => { at[h.row + j] = this.toGrid(labels[k][p]); });
						h.frames.push(k);
						h.tracks.push(h.points.map((p) => labels[k][p].slice()));
						h.scores.push(h.points.map(() => 1));
					}
				}
				// Only the channel-first copy is needed from here, for sampling.
				prev?.dispose();
				cur.rows.dispose();
				prev = cur.cols;
				tick();
				await breathe();
			}
		} finally {
			submit();
			prev?.dispose();
			for (const t of Object.values(g)) t.dispose();
		}
		// Backward sweeps ran from the keyframe down: put every track in frame order.
		if (dir < 0) {
			for (const h of hyps) {
				h.frames.reverse();
				h.tracks.reverse();
				h.scores.reverse();
			}
		}
	}

	// Row r's best patch within `radius` of `around` in the scores [N, stride],
	// refined by a soft-argmax over the 3x3 about it.
	match(sc, stride, r, [cx, cy]) {
		const { GW, GH } = this.geo, R = this.radius;
		let best = -Infinity, bx = 0, by = 0;
		for (let y = Math.max(0, Math.round(cy) - R); y <= Math.min(GH - 1, Math.round(cy) + R); y++) {
			for (let x = Math.max(0, Math.round(cx) - R); x <= Math.min(GW - 1, Math.round(cx) + R); x++) {
				const s = sc[(y * GW + x) * stride + r];
				if (s > best) { best = s; bx = x; by = y; }
			}
		}
		let sw = 0, sx = 0, sy = 0;
		for (let y = Math.max(0, by - 1); y <= Math.min(GH - 1, by + 1); y++) {
			for (let x = Math.max(0, bx - 1); x <= Math.min(GW - 1, bx + 1); x++) {
				const w = Math.exp((sc[(y * GW + x) * stride + r] - best) / this.temperature);
				sw += w;
				sx += w * x;
				sy += w * y;
			}
		}
		return { at: [sx / sw, sy / sw], score: best };
	}
}
