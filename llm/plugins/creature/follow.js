// Which limb is which, held against joints followed from a frame where every
// one of them is known.
//
// A finder names each joint on its own, frame by frame. On a body with many
// limbs of one shape (a spider's legs) it puts two names on one leg and none
// on the leg beside it, most of all once a leg leaves the pose it rests in.
// But a studio clip starts on its reference picture, where every joint's place
// is known, and track.js can follow those places through the frames. What it
// follows keeps its name, though it is less exact and can be lost.
//
//   const tracks = followLimbs(rig, found, followed, camera, { markers });
//
// `found` are the finder's tracks of one view (detect.js), `followed` the
// tracker's of the same view from frame 0's joints (track.js). Each limb
// joint's read, each frame, from the body out:
//
// 1. A peak of the finder's near where the joint was followed to is the joint:
//    the finder's place, which is the finer, at no less than `AGREED`.
// 2. Else the finder's first peak, when it is sure enough and not *taken*: no
//    other joint at that depth of an alike limb was followed to there. A taken
//    peak is the finder naming the leg beside it.
// 3. Else where the joint was followed to, weighing little, unless the tracker
//    is unsure of it or has it on top of another such joint it did not start on.
// 4. Else it has no read on that frame.
//
// A read must also be within the bone's reach of the joint above it, so a foot
// that was followed to where another leg stands is not kept once its own knee
// has moved off.
//
// The middle of the body and its landmarks stay as found. A landmark on a limb
// joint is kept where the finder's first peak for that joint was (1 or 2).
// The result is tracks as the finder's, one read a limb joint.

import { partner } from './detect.js';
import { markerJoint } from './request.js';

// Distances as shares of the rig's height in pixels: how near a peak is where
// the joint was followed to (a finder's patch or two), how near two followed
// joints are on top of each other, and the slack on a bone's reach.
const NEAR = 0.15, PILED = 0.05, REACH = 0.05;
// How far past a bone's length two joints may sit, each found a little off.
const REACH_SLACK = 1.15;
// The least score of a peak the tracker agrees with, of a first peak taken
// without it, and of the tracker's own (a joint that turns looks less like
// its label, about 0.6); and what a read from the tracker alone scores.
const AGREED = 0.45, UNFOLLOWED = 0.25, FOLLOWED_LEAST = 0.45, FOLLOWED = 0.35;

/**
 * The rig's limbs: per joint its limb (-1 for the middle; a limb is a joint
 * with a mirror partner and all below it), its depth in the limb, and which
 * limbs are alike (as many joints, as long within 30%), as lists of limbs.
 */
export function limbsOf(rig) {
	const J = rig.names.length, limb = new Int32Array(J).fill(-1), depth = new Int32Array(J), roots = [];
	for (let j = 0; j < J; j++) {
		const up = rig.parents[j];
		if (up >= 0 && limb[up] >= 0) { limb[j] = limb[up]; depth[j] = depth[up] + 1; continue; }
		const mate = rig.names.indexOf(partner(rig.names[j]));
		if (mate < 0 || mate === j) continue;
		limb[j] = roots.length;
		roots.push(j);
	}
	const length = (j) => (rig.parents[j] < 0 ? 0 : Math.hypot(...rig.positions[j].map((v, d) => v - rig.positions[rig.parents[j]][d])));
	const size = roots.map((_, l) => { let n = 0, sum = 0; for (let j = 0; j < J; j++) if (limb[j] === l) { n++; sum += length(j); } return { n, sum }; });
	const alike = [];
	roots.forEach((_, l) => {
		const g = alike.find((list) => size[list[0]].n === size[l].n && Math.abs(size[list[0]].sum - size[l].sum) <= 0.3 * Math.max(size[list[0]].sum, size[l].sum));
		if (g) g.push(l); else alike.push([l]);
	});
	return { limb, depth, alike, length };
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/**
 * One view's tracks with each limb joint's read held against where it was
 * followed to from frame 0 (see the header). `camera` is the view's (its scale
 * gives the bones' lengths in pixels), `markers` the fit's ({ name: joint or
 * { joint } }), for the landmarks among the points. `report`, when given,
 * gets how many reads went each way: { reads, agreed, unfollowed, followed,
 * none, taken }.
 */
export function followLimbs(rig, found, followed, camera, { markers = {}, report = null } = {}) {
	const T = found.frames, P = found.points.length, J = rig.names.length, scale = Math.exp(camera.logS);
	const { limb, depth, alike, length } = limbsOf(rig);
	const height = rig.height ?? Math.max(...rig.positions.map((p) => p[1])) - Math.min(...rig.positions.map((p) => p[1]));
	const px = height * scale, near = NEAR * px, piled = PILED * px;
	// The points: a joint's own, or a landmark on a joint.
	const jointOf = found.points.map((n) => rig.names.indexOf(n));
	const rides = found.points.map((n, p) => (jointOf[p] >= 0 || !(n in markers) ? -1 : rig.names.indexOf(markerJoint(markers[n]))));
	const pointOf = new Int32Array(J).fill(-1);
	jointOf.forEach((j, p) => { if (j >= 0) pointOf[j] = p; });
	// Each frame's first and second peaks, and where the tracker has each joint.
	const peaks = Array.from({ length: T }, () => Array.from({ length: P }, () => []));
	for (const h of found.hypotheses) {
		h.frames.forEach((t, i) => {
			if (t >= T) return;
			h.tracks[i].forEach((q, p) => { if (q && peaks[t][p].length < 2) peaks[t][p].push({ at: q, score: h.scores[i][p] ?? 0 }); });
		});
	}
	const trail = Array.from({ length: T }, () => new Array(J).fill(null));
	const named = followed.points.map((n) => rig.names.indexOf(n));
	for (const h of followed.hypotheses) {
		h.frames.forEach((t, i) => {
			if (t >= T) return;
			h.tracks[i].forEach((q, p) => { if (q && named[p] >= 0 && !trail[t][named[p]]) trail[t][named[p]] = { at: q, score: h.scores[i][p] ?? 0 }; });
		});
	}
	// The limb joints from the body out, and each one's like: the joints at its
	// depth of the limbs alike to its own.
	const order = [];
	for (let j = 0; j < J; j++) if (limb[j] >= 0 && pointOf[j] >= 0) order.push(j);
	order.sort((a, b) => depth[a] - depth[b]);
	const list = new Int32Array(J).fill(-1);
	alike.forEach((limbs, g) => { for (let j = 0; j < J; j++) if (limbs.includes(limb[j])) list[j] = g; });
	const like = order.map((j) => order.filter((k) => k !== j && list[k] === list[j] && depth[k] === depth[j]));
	const started = (j, k) => trail[0][j] && trail[0][k] && dist(trail[0][j].at, trail[0][k].at) <= piled;

	const counts = { reads: 0, agreed: 0, unfollowed: 0, followed: 0, none: 0, taken: 0 };
	const read = Array.from({ length: T }, () => new Array(J).fill(null));
	for (let t = 0; t < T; t++) {
		// Where a joint is on this frame, as far as is known: a limb joint's read
		// (or where it was followed to), the middle as found (or followed).
		const place = (j) => {
			if (j < 0) return null;
			if (limb[j] >= 0) return read[t][j]?.at ?? trail[t][j]?.at ?? null;
			const own = pointOf[j] >= 0 ? peaks[t][pointOf[j]][0] : null;
			return own?.at ?? trail[t][j]?.at ?? null;
		};
		order.forEach((j, n) => {
			counts.reads++;
			const up = place(rig.parents[j]), reach = REACH_SLACK * length(j) * scale + REACH * px;
			const within = (q) => !up || dist(q, up) <= reach;
			const mine = peaks[t][pointOf[j]], f = trail[t][j];
			const agreed = f ? mine.filter((q) => dist(q.at, f.at) <= near && within(q.at)).sort((a, b) => b.score - a.score)[0] : null;
			const first = mine[0];
			const taken = !!first && first !== agreed && like[n].some((k) => trail[t][k] && dist(trail[t][k].at, first.at) <= near);
			if (taken) counts.taken++;
			if (agreed) {
				read[t][j] = { at: agreed.at, score: Math.max(agreed.score, AGREED), first: agreed === first };
				counts.agreed++;
				return;
			}
			if (first && first.score >= UNFOLLOWED && within(first.at) && !taken) {
				read[t][j] = { at: first.at, score: first.score, first: true };
				counts.unfollowed++;
				return;
			}
			const lost = f && like[n].some((k) => trail[t][k] && dist(trail[t][k].at, f.at) <= piled && !started(j, k));
			if (f && f.score >= FOLLOWED_LEAST && within(f.at) && !lost) {
				read[t][j] = { at: f.at, score: FOLLOWED, first: false };
				counts.followed++;
				return;
			}
			counts.none++;
		});
	}
	if (report) Object.assign(report, counts);

	// As the finder's tracks: a hypothesis a stretch with the reads, and the
	// second peaks of the points left as found.
	const round = (q) => [Math.round(q[0] * 100) / 100, Math.round(q[1] * 100) / 100];
	const kept = (t, p) => !(rides[p] >= 0 && limb[rides[p]] >= 0 && !read[t][rides[p]]?.first);
	const first = (t, p) => (jointOf[p] >= 0 && limb[jointOf[p]] >= 0 ? read[t][jointOf[p]] : kept(t, p) ? peaks[t][p][0] ?? null : null);
	const second = (t, p) => (jointOf[p] >= 0 && limb[jointOf[p]] >= 0 ? null : kept(t, p) ? peaks[t][p][1] ?? null : null);
	const rows = (frames, pick) => ({
		tracks: frames.map((t) => found.points.map((_, p) => { const q = pick(t, p); return q ? round(q.at) : null; })),
		scores: frames.map((t) => found.points.map((_, p) => { const q = pick(t, p); return q ? Math.round(q.score * 1000) / 1000 : null; })),
	});
	const stretches = [...new Set(found.hypotheses.map((h) => h.from))].filter((k) => k < T).sort((a, b) => a - b);
	const hypotheses = stretches.map((from, i) => {
		const frames = [];
		for (let t = from; t < (stretches[i + 1] ?? T); t++) frames.push(t);
		return { from, direction: 1, frames, ...rows(frames, first) };
	});
	const all = Array.from({ length: T }, (_, t) => t);
	if (all.some((t) => found.points.some((_, p) => second(t, p)))) hypotheses.push({ from: 0, direction: 1, frames: all, ...rows(all, second) });
	return { ...found, hypotheses };
}
