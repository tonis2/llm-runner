// Are a studio clip's two halves in step? A video model may play one view
// ahead of the other. Both cameras are level and share one scale and floor,
// so a joint is at the same height in both, frame for frame: how far apart
// the two views put the joints' heights says how out of step they are, as
// anymove's sync check does.
//
//   const s = syncViews(frontTracks, sideTracks, [frontCamera, sideCamera]);
//   s.report    { height_gap_px, worst_gap_px, best_lag, gap_at_lag_px, feet, retimed, ... }
//   s.front     the front view's tracks, re-timed onto the side's when that helps
//
// The heights are taken without trusting left and right: a middle joint's as
// it is, a left/right pair's as its mean and its spread, so a view that swaps
// a pair still agrees. When matching the views frame to frame by dynamic time
// warping (DTW) brings the heights at least 15% closer, the front view's
// tracks are resampled onto the side view's timeline.

import { partner } from './detect.js';

// Each frame's first hypothesis of every point: [T][P] of [x, y] or null, and
// its score. The keyframe stretches come first in a finder's (and the
// tracker's) hypotheses, the second peaks after.
function firstPoints(tracks) {
	const T = tracks.frames, P = tracks.points.length;
	const at = Array.from({ length: T }, () => null), score = Array.from({ length: T }, () => null);
	for (const h of tracks.hypotheses) {
		h.frames.forEach((t, i) => {
			if (t < T && at[t] === null) { at[t] = h.tracks[i]; score[t] = h.scores[i]; }
		});
	}
	for (let t = 0; t < T; t++) if (at[t] === null) { at[t] = Array(P).fill(null); score[t] = Array(P).fill(null); }
	return { at, score };
}

// The label-free heights of one view, in rig units: [T][K] and their weights.
function features(tracks, camera, pairs) {
	const { at, score } = firstPoints(tracks), s = Math.exp(camera.logS);
	const height = (q) => (q ? (-q[1] - camera.shift[1]) / s : NaN);
	const weight = (v) => (v === null || v === undefined ? 0 : Math.min(1, Math.max(0, (v - 0.2) / 0.4)));
	return at.map((row, t) => {
		const f = [], w = [];
		for (const [a, b] of pairs) {
			if (b < 0) { f.push(height(row[a])); w.push(weight(score[t][a])); continue; }
			const ha = height(row[a]), hb = height(row[b]);
			const wab = Math.min(weight(score[t][a]), weight(score[t][b]));
			f.push((ha + hb) / 2, Math.abs(ha - hb));
			w.push(wab, wab);
		}
		return { f, w };
	});
}

// The weighted mean height gap between frames i and j of the two views.
function gap(F, S, i, j) {
	let sum = 0, sw = 0;
	for (let k = 0; k < F[i].f.length; k++) {
		const w = Math.min(F[i].w[k], S[j].w[k]);
		if (!(w > 0) || !Number.isFinite(F[i].f[k]) || !Number.isFinite(S[j].f[k])) continue;
		sum += w * Math.abs(F[i].f[k] - S[j].f[k]);
		sw += w;
	}
	return sw > 0 ? sum / sw : NaN;
}

const mean = (xs) => { const v = xs.filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; };

// The front frame matched to each side frame, by DTW in a band of `band`
// frames, a step that holds one view still costing `hold` rig units; smoothed
// so it has no stair steps.
function warp(F, S, band, hold) {
	const T = F.length, C = (i, j) => { const g = gap(F, S, i, j); return Number.isFinite(g) ? g : hold; };
	const D = Array.from({ length: T }, () => new Float64Array(T).fill(Infinity));
	for (let i = 0; i < T; i++) {
		for (let j = Math.max(0, i - band); j <= Math.min(T - 1, i + band); j++) {
			if (i === 0 && j === 0) { D[0][0] = C(0, 0); continue; }
			const best = Math.min(i && j ? D[i - 1][j - 1] : Infinity, i ? D[i - 1][j] + hold : Infinity, j ? D[i][j - 1] + hold : Infinity);
			D[i][j] = C(i, j) + best;
		}
	}
	let i = T - 1, j = T - 1;
	const path = [[i, j]];
	while (i || j) {
		const steps = [i && j ? [D[i - 1][j - 1], i - 1, j - 1] : null, i ? [D[i - 1][j], i - 1, j] : null, j ? [D[i][j - 1], i, j - 1] : null].filter(Boolean);
		[, i, j] = steps.reduce((a, b) => (b[0] < a[0] ? b : a));
		path.push([i, j]);
	}
	const match = Array.from({ length: T }, (_, jj) => mean(path.filter((p) => p[1] === jj).map((p) => p[0])));
	return match.map((_, t) => mean(match.slice(Math.max(0, t - 3), t + 4)));
}

// `tracks` resampled at fractional frames `when` (one per frame): each point
// between its two frames, or the nearer when one is missing.
function resample(tracks, when) {
	const T = tracks.frames, P = tracks.points.length;
	const lerp = (a, b, s) => (a && b ? [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s] : s < 0.5 ? a ?? b : b ?? a);
	const num = (a, b, s) => (a !== null && b !== null ? a + (b - a) * s : s < 0.5 ? a ?? b : b ?? a);
	return {
		...tracks,
		hypotheses: tracks.hypotheses.map((h) => {
			const row = new Map(h.frames.map((t, i) => [t, i]));
			return {
				...h,
				tracks: h.frames.map((t) => {
					const u = Math.min(T - 1, Math.max(0, when[t])), a = Math.floor(u), b = Math.min(T - 1, a + 1), s = u - a;
					const ra = h.tracks[row.get(a)] ?? null, rb = h.tracks[row.get(b)] ?? null;
					return Array.from({ length: P }, (_, p) => lerp(ra?.[p] ?? null, rb?.[p] ?? null, s));
				}),
				scores: h.frames.map((t) => {
					const u = Math.min(T - 1, Math.max(0, when[t])), a = Math.floor(u), b = Math.min(T - 1, a + 1), s = u - a;
					const ra = h.scores[row.get(a)] ?? null, rb = h.scores[row.get(b)] ?? null;
					return Array.from({ length: P }, (_, p) => num(ra?.[p] ?? null, rb?.[p] ?? null, s));
				}),
			};
		}),
	};
}

/**
 * How far out of step the studio's two views are, and the front view re-timed
 * onto the side's when matching them frame by frame brings their heights at
 * least `gain` closer (`retime: false` to only report). Heights in the report
 * are pixels of the first view.
 */
export function syncViews(front, side, cameras, { gain = 0.15, retime = true, band = 48, lags = 12 } = {}) {
	const names = front.points;
	if (side.points.length !== names.length || side.points.some((n, p) => n !== names[p])) return { report: null, front };
	const T = Math.min(front.frames, side.frames);
	// Middle points alone ([p, -1]), left/right pairs together ([left, right]).
	const pairs = [];
	names.forEach((n, p) => {
		const q = names.indexOf(partner(n));
		if (q < 0 || q === p) pairs.push([p, -1]);
		else if (p < q) pairs.push([p, q]);
	});
	const F = features(front, cameras[0], pairs).slice(0, T), S = features(side, cameras[1], pairs).slice(0, T);
	const px = Math.exp(cameras[0].logS);
	const perFrame = F.map((_, t) => gap(F, S, t, t));
	const at = (lag) => mean(Array.from({ length: T }, (_, t) => (t + lag >= 0 && t + lag < T ? gap(F, S, t + lag, t) : NaN)));
	let bestLag = 0;
	for (let lag = -lags; lag <= lags; lag++) if (at(lag) < at(bestLag)) bestLag = lag;
	// The feet: does each one rise and fall together in the two views?
	const feet = {};
	names.forEach((n, p) => {
		if (!/toe|ankle|heel|foot|paw/i.test(n)) return;
		const a = [], b = [];
		const fp = firstPoints(front).at, sp = firstPoints(side).at;
		for (let t = 0; t < T; t++) if (fp[t][p] && sp[t][p]) { a.push(-fp[t][p][1]); b.push(-sp[t][p][1]); }
		if (a.length < 8) return;
		const ma = mean(a), mb = mean(b);
		let sab = 0, saa = 0, sbb = 0;
		for (let k = 0; k < a.length; k++) { sab += (a[k] - ma) * (b[k] - mb); saa += (a[k] - ma) ** 2; sbb += (b[k] - mb) ** 2; }
		const spread = Math.sqrt(saa / a.length), spreadB = Math.sqrt(sbb / b.length);
		if (spread > 0.5 && spreadB > 0.5) feet[n] = Number((sab / Math.sqrt(saa * sbb)).toFixed(2));
	});
	const round = (v) => (Number.isFinite(v) ? Number(v.toFixed(2)) : null);
	const report = {
		height_gap_px: round(mean(perFrame) * px), worst_gap_px: round(Math.max(...perFrame.filter(Number.isFinite)) * px),
		best_lag: bestLag, gap_at_lag_px: round(at(bestLag) * px), feet, retimed: false,
	};
	if (!retime || T < 8) return { report, front };
	const match = warp(F, S, Math.min(band, T - 1), mean(perFrame) || 1e-3);
	const trial = features(resample(front, match), cameras[0], pairs).slice(0, T);
	const after = mean(trial.map((_, t) => gap(trial, S, t, t)));
	report.retime_gap_px = round(after * px);
	if (!(after < (1 - gain) * mean(perFrame))) return { report, front };
	report.retimed = true;
	report.shift_frames = round(Math.max(...match.map((m, t) => Math.abs(m - t))));
	return { report, front: resample(front, match) };
}
