// The studio exam: a known clip drawn through a studio's two cameras, found
// and fitted as /animate does a video model's clip, then held against the
// joints that drew it.
//
//   llm-runner tests/creature_studio_exam.js model=sam-3d-body-dinov3 rig=rig.glb \
//       finder=rig.finder.safetensors exam=exam_sprint/ [tracks_out=t.json | tracks_in=t.json]
//       [swap=1:30-40,60-70] [swap_score=0.9] [request='{"weights":{"swap":2}}']
//       [median_px=12] [p90_px=40] [crossed=2] [draw=9] [reads_out=reads.png]
//
// `exam` holds frames/NNNN.png and truth/NNNN.json: Rigsmith's render_studio of
// the clip at each frame with `layout` kept (every frame the same cameras), its
// studio.json each frame's truth: every joint's pixel in both views.
//
// Graded, on the rig's joints in both views (two level cameras pin a point in
// 3D, so both views' pixels are its place):
// - every joint's distance from the truth: median, 90%, share over 30 px, and
//   per group (body, feet, hands, tail);
// - crossed: frames where the fit has a pair of limbs the wrong way round
//   (each pair laid on the truth's swapped is nearer than as it is).
// Reported, to say where it goes wrong: the same distances per limb (a joint
// with a mirror partner and all below it, named by its first joint) and per
// side, for joints a view shows against ones the body hides there (the
// truth's `shown`), and the finder's reads that sit on another limb: over 20
// px from their own joint and under half as far from a joint of the mirror
// limb, or of some other one.
// `draw` frames of the reads go to `reads_out` (exam/reads.png).
//
// `swap=<view>:<frames>` relabels the legs left for right in one view's tracks
// on those frames, scored `swap_score`: a finder's confident mistake, to try
// how the fit gets out of it. `tracks_out` keeps what the finder found, and
// `tracks_in` fits that again without the GPU.
import { llm } from '/core/llm.js';
import { utf8 } from '/core/graph/plugins.js';
import { animate } from '/plugins/creature/main.js';
import { partner } from '/plugins/creature/detect.js';
import { readRig } from '/plugins/creature/rig.js';
import { limbPairs } from '/plugins/creature/fit.js';

const cfg = globalThis.__llm_config;
for (const k of ['rig', 'finder', 'exam']) if (!cfg[k]) throw new Error(`${k}= is needed (see the header)`);
if (!cfg.tracks_in && !cfg.model) throw new Error('model= is needed to find the joints (or tracks_in=)');
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

const dir = String(cfg.exam).replace(/\/+$/, '');
const names = llm.listDir(`${dir}/truth`).map((e) => e.name).filter((n) => n.endsWith('.json')).sort();
const truth = names.map((n) => JSON.parse(llm.readText(`${dir}/truth/${n}`)));
const T = truth.length, layout = truth[0];
llm.print(`  exam: ${T} frames of ${layout.clip}, ${layout.width} x ${layout.height}`);

// The tracks: found, or kept from before, with any swaps laid in.
const request = {
	rig: cfg.rig, finder: cfg.finder, studio: `${dir}/truth/${names[0]}`, fps: 24, name: 'exam', outputs: ['tracks'],
	...(typeof cfg.request === 'string' ? JSON.parse(cfg.request) : cfg.request ?? {}),
};
let tracks = null;
if (cfg.tracks_in) tracks = JSON.parse(llm.readText(cfg.tracks_in));
else {
	const found = await animate({ ...request, frames: `${dir}/frames` }, { model: cfg.model });
	tracks = found.reply.tracks;
	if (cfg.tracks_out) llm.writeBytes(cfg.tracks_out, utf8(JSON.stringify(tracks)));
}
const LEGS = /^(thigh|knee|ankle|toe|toe_tip)_[LR]$|^ankle_[LR]\.heel$/;
// As the finder read them, before any swap is laid in.
const asRead = JSON.parse(JSON.stringify(tracks));
if (cfg.swap) {
	const score = Number(cfg.swap_score ?? 0.9);
	for (const part of String(cfg.swap).split(';')) {
		const [view, spans] = part.split(':');
		const t = tracks[Number(view)], cols = t.points.map((n, p) => (LEGS.test(n) ? [p, t.points.indexOf(partner(n))] : null)).filter((c) => c && c[1] > c[0]);
		const frames = new Set(spans.split(',').flatMap((s) => { const [a, b] = s.split('-').map(Number); return Array.from({ length: (b ?? a) - a + 1 }, (_, i) => a + i); }));
		for (const h of t.hypotheses) {
			h.frames.forEach((k, i) => {
				if (!frames.has(k)) return;
				const row = h.tracks[i], sc = h.scores[i];
				for (const [a, b] of cols) {
					[row[a], row[b]] = [row[b], row[a]];
					if (sc) { [sc[a], sc[b]] = [sc[b], sc[a]]; if (sc[a] != null) sc[a] = Math.max(sc[a], score); if (sc[b] != null) sc[b] = Math.max(sc[b], score); }
				}
			});
		}
		llm.print(`  swap: legs relabelled in view ${view} on ${frames.size} frames, scored at least ${score}`);
	}
}

// The fit, through the same cameras.
const drawn = cfg.draw ? { frames: `${dir}/frames`, draw: Number(cfg.draw) } : { size: [layout.width, layout.height] };
const { reply, fit, files } = await animate({ ...request, views: tracks.map((t) => ({ tracks: t })), ...drawn });
if (files['reads.png']) llm.writeBytes(cfg.reads_out ?? `${dir}/reads.png`, files['reads.png']);
const report = reply.report, boxes = report.studio.views.map((v) => v.box);
const rigNames = reply.joints;

// Each joint's fitted pixel against the truth's, both views, whole-picture pixels.
const graded = rigNames.map((n, j) => [j, layout.joints.indexOf(n)]).filter(([, k]) => k >= 0);
const fitted = [0, 1].map((v) => fit.problem.jointProjections(v).map((row) => row.map(([x, y]) => [x + boxes[v][0], boxes[v][1] - y])));
const at = (v, t, j) => fitted[v][t][j];
const want = (v, t, k) => truth[t].pixels[v][k];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const group = (n) => (/^tail/.test(n) ? 'tail' : /thumb|index|middle|ring|pinky/.test(n) ? 'hands' : /ankle|toe|heel/.test(n) ? 'feet' : 'body');
const errs = [], byGroup = {};
for (let t = 0; t < T; t++) {
	for (const [j, k] of graded) {
		for (let v = 0; v < 2; v++) {
			const e = dist(at(v, t, j), want(v, t, k));
			errs.push(e);
			(byGroup[group(rigNames[j])] ??= []).push(e);
		}
	}
}
const stats = (a) => {
	a = a.slice().sort((x, y) => x - y);
	const q = (f) => a[Math.min(a.length - 1, Math.floor(f * a.length))];
	return { median: Number(q(0.5).toFixed(2)), p90: Number(q(0.9).toFixed(2)), over_30: Number((a.filter((e) => e > 30).length / a.length).toFixed(3)) };
};
const all = stats(errs);
llm.print(`  joints against the truth: median ${all.median}px, 90% ${all.p90}px, ${(all.over_30 * 100).toFixed(1)}% over 30px`);
for (const [g, e] of Object.entries(byGroup)) { const s = stats(e); llm.print(`    ${g}: median ${s.median}px, 90% ${s.p90}px, ${(s.over_30 * 100).toFixed(1)}% over 30px`); }

// The rig's limbs: per joint its mirror pair (-1 for the middle), its partner,
// and the limb's first joint, which names it.
const rig = readRig(cfg.rig);
const limbs = limbPairs(rig, rig.names.map((name, joint) => ({ name, joint })));
const limbOf = rigNames.map((n) => {
	let j = rig.names.indexOf(n);
	if (j < 0 || limbs.group[j] < 0) return null;
	while (rig.parents[j] >= 0 && limbs.group[rig.parents[j]] === limbs.group[j]) j = rig.parents[j];
	return rig.names[j];
});
const leads = (n) => rigNames.indexOf(n) < rigNames.indexOf(partner(n));
const sideOf = (n) => (n === null ? 'middle' : /(^|[_.])l$|left$/i.test(n) ? 'left' : /(^|[_.])r$|right$/i.test(n) ? 'right' : leads(n) ? 'first side' : 'second side');
const line = (label, e) => { const s = stats(e); llm.print(`    ${label}: median ${s.median}px, 90% ${s.p90}px, ${(s.over_30 * 100).toFixed(1)}% over 30px (${e.length})`); };
const byLimb = {}, bySide = {}, bySight = { shown: [], hidden: [] };
for (let t = 0; t < T; t++) {
	for (const [j, k] of graded) {
		for (let v = 0; v < 2; v++) {
			const e = dist(at(v, t, j), want(v, t, k));
			(byLimb[limbOf[j] ?? 'middle'] ??= []).push(e);
			(bySide[sideOf(limbOf[j])] ??= []).push(e);
			if (truth[t].shown) bySight[truth[t].shown[v][k] ? 'shown' : 'hidden'].push(e);
		}
	}
}
if (Object.keys(byLimb).length > 1) {
	llm.print('  the fit, by limb:');
	for (const [g, e] of Object.entries(byLimb)) line(g, e);
	for (const [g, e] of Object.entries(bySide)) line(g, e);
	for (const [g, e] of Object.entries(bySight)) if (e.length) line(`${g} in its view`, e);
}

// The finder's reads against the truth, per view: how far, shown and hidden,
// and how many sit on another limb's joint instead of their own.
const whole = (v, q) => [q[0] + boxes[v][0], q[1] + boxes[v][1]];
asRead.forEach((tr, v) => {
	const errs = { shown: [], hidden: [] }, strays = { mirror: 0, other: 0 }, taken = new Set();
	let reads = 0;
	// A point's first read on a frame is its first peak; any later is a second.
	for (const h of tr.hypotheses) {
		h.frames.forEach((t, i) => {
			if (t >= T) return;
			tr.points.forEach((n, p) => {
				const q = h.tracks[i][p], k = layout.joints.indexOf(n), j = rigNames.indexOf(n);
				if (!q || k < 0 || j < 0 || taken.has(t * tr.points.length + p)) return;
				taken.add(t * tr.points.length + p);
				const own = dist(whole(v, q), want(v, t, k));
				errs[!truth[t].shown || truth[t].shown[v][k] ? 'shown' : 'hidden'].push(own);
				if (limbOf[j] === null) return;
				reads++;
				if (own <= 20) return;
				let near = Infinity, mirror = false;
				for (const [j2, k2] of graded) {
					if (limbOf[j2] === null || limbOf[j2] === limbOf[j]) continue;
					const d = dist(whole(v, q), want(v, t, k2));
					if (d < near) { near = d; mirror = limbOf[j2] === partner(limbOf[j]); }
				}
				if (near < own / 2) strays[mirror ? 'mirror' : 'other']++;
			});
		});
	}
	llm.print(`  view ${v} (${report.studio.views[v].name}), the finder's reads against the truth:`);
	for (const [g, e] of Object.entries(errs)) if (e.length) line(g, e);
	llm.print(`    on another limb: ${strays.mirror} on the mirror limb, ${strays.other} on some other, of ${reads} limb reads (${(((strays.mirror + strays.other) / Math.max(1, reads)) * 100).toFixed(1)}%)`);
});

// Crossed: a limb pair nearer the truth's swapped than as it is.
const firstOfPair = [...new Set(limbOf.filter((n) => n !== null && leads(n)))];
const pairs = firstOfPair
	.map((root) => rigNames.filter((n, j) => limbOf[j] === root).map((n) => [rigNames.indexOf(n), rigNames.indexOf(partner(n)), layout.joints.indexOf(n), layout.joints.indexOf(partner(n))]).filter((p) => p.every((i) => i >= 0)))
	.filter((pts) => pts.length > 0);
const crossedFrames = [];
for (let t = 0; t < T; t++) {
	for (const pts of pairs) {
		let straight = 0, swapped = 0;
		for (const [jl, jr, kl, kr] of pts) {
			for (let v = 0; v < 2; v++) {
				straight += dist(at(v, t, jl), want(v, t, kl)) + dist(at(v, t, jr), want(v, t, kr));
				swapped += dist(at(v, t, jl), want(v, t, kr)) + dist(at(v, t, jr), want(v, t, kl));
			}
		}
		if (swapped < 0.7 * straight) { crossedFrames.push(t); break; }
	}
}
llm.print(`  crossed against the truth: ${crossedFrames.length} frames${crossedFrames.length ? ` (${crossedFrames.join(', ')})` : ''}; the fit relabelled ${JSON.stringify(report.crossed ?? [])}`);
llm.print(`  fit's own: reprojection median ${report.reprojection_px.median}px, 90% ${report.reprojection_px.p90}px${report.recheck ? `; recheck ${JSON.stringify(report.recheck)}` : ''}`);

check(all.median <= Number(cfg.median_px ?? 12), `median ${all.median}px (at most ${cfg.median_px ?? 12})`);
check(all.p90 <= Number(cfg.p90_px ?? 40), `90% ${all.p90}px (at most ${cfg.p90_px ?? 40})`);
check(crossedFrames.length <= Number(cfg.crossed ?? 2), `${crossedFrames.length} frames crossed (at most ${cfg.crossed ?? 2})`);
llm.print(failures ? `${failures} failed` : 'all passed');
