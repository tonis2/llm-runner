// Are a studio clip's two halves in step? A video model may play one view
// ahead of the other. Both cameras are level and share one scale and floor,
// so a joint is at the same height in both, frame for frame: how far apart
// the two views put the joints' heights says how out of step they are, as
// anymove's sync check does.
//
//   const s = syncViews(frontTracks, sideTracks, [frontCamera, sideCamera]);
//   s.report    { height_gap_px, worst_gap_px, best_lag, gap_at_lag_px, repeat_frames, feet, retimed, slip, ... }
//   s.front     the front view's tracks, re-timed onto the side's when that helps
//   s.side      the side view's tracks, shorter when a view slipped
//
// The heights are taken without trusting left and right: a middle joint's as
// it is, a left/right pair's as its mean and its spread, so a view that swaps
// a pair still agrees. When matching the views frame to frame by dynamic time
// warping (DTW) brings the heights at least 15% closer, the front view's
// tracks are resampled onto the side view's timeline.
// Heights that repeat, as a run's do every step, can't say which step they are
// on, so no shift goes as far as half a repeat.
// A view that jumps back partway through is another matter: see slipOf.

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

// How many frames the side view's heights take to come round again, or null when
// they do not: a run's repeat every step, left and right being left out. The
// first dip of the heights against themselves, well under the gap before it.
const REPEAT_DIP = 0.5;
function repeatOf(S) {
	const most = Math.floor(S.length / 2), gaps = [];
	for (let lag = 1; lag <= most; lag++) gaps[lag] = mean(Array.from({ length: S.length - lag }, (_, t) => gap(S, S, t + lag, t)));
	let peak = 0;
	for (let lag = 2; lag < most; lag++) {
		peak = Math.max(peak, gaps[lag - 1]);
		if (gaps[lag] <= gaps[lag + 1] && gaps[lag] < REPEAT_DIP * peak) return lag;
	}
	return null;
}

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

// A view that jumped back and stayed behind, so the legs only meet crossed: the
// cut that keeps the legs' gap across the body at the first third's. Any cut
// holding the jump joins well, its frames being repeats; the latest is nearest it.
const SLIP_MOST = 20, SLIP_GAIN = 0.5, SLIP_STRAY = 0.05, SLIP_CAP = 0.6, SLIP_FLAT = 1.1, SLIP_JOIN = 1.5, SLIP_EDGE = 8, SLIP_TAIL = 16;
function slipOf(front, side, cameras, height) {
	const T = Math.min(front.frames, side.frames), names = front.points;
	const [a, b] = cameras, det = Math.cos(a.yaw) * Math.sin(b.yaw) - Math.sin(a.yaw) * Math.cos(b.yaw);
	const legs = names.flatMap((n, p) => {
		const q = names.indexOf(partner(n));
		return /knee|ankle|foot|paw/i.test(n) && q > p ? [[p, q]] : [];
	});
	if (!legs.length || !(height > 0) || !(Math.abs(det) > 0.1) || T < 2 * SLIP_TAIL) return null;
	const F = firstPoints(front).at, S = firstPoints(side).at;
	const sa = Math.exp(a.logS), sb = Math.exp(b.logS);
	// Across the body, in rig heights: the front's frame f and the side's frame s.
	const across = (f, s, [p, q]) => {
		const u = F[f][p] && F[f][q] ? (F[f][p][0] - F[f][q][0]) / sa : NaN, v = S[s][p] && S[s][q] ? (S[s][p][0] - S[s][q][0]) / sb : NaN;
		return (u * Math.sin(b.yaw) - v * Math.sin(a.yaw)) / det / height;
	};
	const normal = legs.map((leg) => {
		const gaps = Array.from({ length: Math.floor(T / 3) }, (_, t) => across(t, t, leg)).filter(Number.isFinite).sort((x, y) => x - y);
		return gaps[gaps.length >> 1];
	});
	const stray = (f, s) => mean(legs.map((leg, k) => Math.min(Math.abs(across(f, s, leg) - normal[k]), SLIP_CAP)));
	// Running sums of a frame's stray, skipping frames with no reads.
	const sums = (at) => {
		const total = [0], count = [0];
		for (let t = 0; t < T; t++) {
			const v = at(t);
			total.push(total[t] + (Number.isFinite(v) ? v : 0));
			count.push(count[t] + (Number.isFinite(v) ? 1 : 0));
		}
		return { total, count };
	};
	const still = sums((t) => stray(t, t));
	const now = still.total[T] / still.count[T];
	if (!(now >= SLIP_STRAY)) return null;
	const tried = [];
	for (const late of ['front', 'side']) {
		for (let k = 1; k <= SLIP_MOST; k++) {
			const moved = sums((t) => (t + k >= T ? NaN : late === 'front' ? stray(t + k, t) : stray(t, t + k)));
			for (let c = SLIP_EDGE; c + k + SLIP_TAIL <= T; c++) {
				const n = still.count[c] + moved.count[T] - moved.count[c];
				if (n) tried.push({ late, frames: k, frame: c, left: (still.total[c] + moved.total[T] - moved.total[c]) / n });
			}
		}
	}
	const best = tried.reduce((x, y) => (y.left < x.left ? y : x), { left: Infinity });
	if (!(best.left <= (1 - SLIP_GAIN) * now)) return null;
	const near = tried.filter((c) => c.late === best.late && c.frames === best.frames && c.left <= SLIP_FLAT * best.left).map((c) => c.frame);
	const reads = best.late === 'front' ? F : S;
	const join = (c) => mean(names.map((_, p) => {
		const u = reads[c - 1][p], v = reads[c + best.frames][p];
		return u && v ? Math.hypot(u[0] - v[0], u[1] - v[1]) : NaN;
	}));
	const smoothest = Math.min(...near.map(join).filter(Number.isFinite));
	const frame = Math.max(...near.filter((c) => !(join(c) > SLIP_JOIN * smoothest)));
	return { late: best.late, frames: best.frames, frame, stray: Number(now.toFixed(3)), left: Number(best.left.toFixed(3)) };
}

// `tracks` with `count` frames taken out from `from` on, the rest moved up.
// Reads come in short stretches, so they move whole, not through resample.
function withoutFrames(tracks, from, count) {
	const moved = (t) => (t < from ? t : Math.max(from, t - count));
	const kept = (t) => t < from || t >= from + count;
	return {
		...tracks,
		frames: tracks.frames - count,
		boxes: tracks.boxes?.filter((_, t) => kept(t)),
		hypotheses: tracks.hypotheses.filter((h) => h.frames.some(kept)).map((h) => {
			const rows = h.frames.flatMap((t, i) => (kept(t) ? [i] : []));
			return { ...h, from: moved(h.from), frames: rows.map((i) => moved(h.frames[i])), tracks: rows.map((i) => h.tracks[i]), scores: rows.map((i) => h.scores[i]) };
		}),
	};
}

/**
 * How far out of step the studio's two views are, and the front view re-timed
 * onto the side's when matching them frame by frame brings their heights at
 * least `gain` closer (`retime: false` to only report). Heights in the report
 * are pixels of the first view. With the rig's `height`, a view that slipped
 * is put back first: its repeated frames cut, both views ending that much sooner.
 */
export function syncViews(front, side, cameras, { gain = 0.15, retime = true, band = 48, lags = 12, height = 0 } = {}) {
	const names = front.points;
	if (side.points.length !== names.length || side.points.some((n, p) => n !== names[p])) return { report: null, front, side };
	const slip = slipOf(front, side, cameras, height);
	if (slip && retime) {
		const end = Math.min(front.frames, side.frames) - slip.frames;
		front = slip.late === 'front' ? withoutFrames(front, slip.frame, slip.frames) : { ...front, frames: end };
		side = slip.late === 'side' ? withoutFrames(side, slip.frame, slip.frames) : { ...side, frames: end };
	}
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
	// Half a repeat or more may only be matching the next step, the other foot
	// forward, so neither the lag nor the warp goes that far.
	const repeat = repeatOf(S), reach = repeat ? Math.ceil(repeat / 2) - 1 : Infinity;
	lags = Math.min(lags, reach);
	band = Math.min(band, reach);
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
		best_lag: bestLag, gap_at_lag_px: round(at(bestLag) * px), repeat_frames: repeat, feet, retimed: false,
	};
	if (slip) report.slip = { ...slip, put_back: retime };
	if (!retime || T < 8) return { report, front, side };
	const match = warp(F, S, Math.min(band, T - 1), mean(perFrame) || 1e-3);
	const trial = features(resample(front, match), cameras[0], pairs).slice(0, T);
	const after = mean(trial.map((_, t) => gap(trial, S, t, t)));
	report.retime_gap_px = round(after * px);
	if (!(after < (1 - gain) * mean(perFrame))) return { report, front, side };
	report.retimed = true;
	report.shift_frames = round(Math.max(...match.map((m, t) => Math.abs(m - t))));
	return { report, front: resample(front, match), side };
}
