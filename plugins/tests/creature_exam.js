// The joint finder's exam: a movement whose every turn is known, found and
// fitted as a video would be, then held against the turns that made it.
//
//   llm-runner tests/creature_exam.js model=sam-3d-body-dinov3 rig=rig.glb \
//       finder=rig.finder.safetensors exam=exam/ [backgrounds=frames/] \
//       [iterations=1500] [place_pct=6] [found_px=12] [tracks_out=t.json | tracks_in=t.json]
//       [weights='{"lateral":0}'] [noise=side:15] [chain=hips,spine,...,head]
//       [chain_deg=10] [head_deg=12] [wobble=2] [flip=false] [landmarks=all|none|a,b,...]
//
// `exam` is Rigsmith's render_poses with motion: one smooth movement inside
// the rig's limits, from two cameras held still (or more). Each camera's
// pictures are laid over a background as a video's frames, the finder finds
// the joints in each (its own cameras, its own crops), and the rig is fitted
// to all the views at once, as POST /animate does with a finder.
//
// Graded:
// 1. Finding: every found joint against where the exam drew it, in pixels.
// 2. The fit: every joint the finder sees, its place from the root against
//    the truth's, as a share of the rig's height, in the world (the exam's
//    camera yaws are the rig's, so the fit's world is the truth's) and in the
//    root's frame. Passes when the world's median is under `place_pct` and
//    the finding's median under `found_px`.
//
// 3. The turns a viewer sees, along `chain` (the root to the head by default):
//    - each joint's turn against its parent, against the truth's (median over
//      the frames, the worst joint of the chain graded under `chain_deg`);
//    - the head's turn in the root's frame (under `head_deg`): the fit's world
//      may turn about the vertical as a whole, which nobody sees;
//    - wobble: how far each joint strays from its own mean turn, against how far
//      the truth's does, (fit + 2°) / (truth + 2°), under `wobble`;
//    - jitter: the change of each joint's frame-to-frame turn, against the
//      truth's, under max(1.5 × the truth's, 0.5°).
//    A fold (the chest bent one way, the neck and head the other) passes the
//    joint places but not these.
//
//    Printed beside it: each bone's direction against the truth's in degrees,
//    in the root's frame and in the world, and how far the root's own up and
//    forward are from the truth's. Short bones (a finger's root, a toe's tip)
//    turn tens of degrees on a few pixels, and the root's turn is barely
//    pinned by its joints, so these say where the fit goes wrong rather than
//    how wrong the clip looks.
//
// A finder's landmarks (points of the mesh) are graded as found where the exam
// drew them in sight (render the exam with the same "landmarks"), and fitted as
// markers with known offsets, as /animate does.
//
// `tracks_out` keeps what the finder found, and `tracks_in` fits that again
// without the GPU, for trying the fit. `noise=<camera>:<px>` adds seeded noise
// of that many pixels to one camera's tracks before the fit (after the finding
// is graded): a real video's side view is found less well than renders are.
import { llm } from '../lib/llm.js';
import { utf8 } from '../lib/graph/plugins.js';
import { loadBackbone } from '../creature/backbone.js';
import { Finder, fittedLandmarks } from '../creature/detect.js';
import { generator, Backdrops, composite } from '../creature/train.js';
import { readRig, qmul, qrotate, qconj } from '../creature/rig.js';
import { fitRig } from '../creature/fit.js';

const cfg = globalThis.__llm_config;
for (const k of ['model', 'rig', 'finder', 'exam']) if (!cfg[k]) throw new Error(`${k}= is needed (see the header)`);
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
const DEG = 180 / Math.PI;

// The exam: its sets in order of frame, each picture with its camera.
const dir = String(cfg.exam).replace(/\/+$/, '');
const sets = llm.listDir(dir).filter((e) => /^poses_\d+\.json$/.test(e.name)).map((e) => JSON.parse(llm.readText(`${dir}/${e.name}`)));
if (!sets.length || !sets[0].motion) throw new Error(`${dir}: no exam (render_poses with motion: true)`);
sets.sort((a, b) => a.first - b.first);
const exam = sets[0], cameras = exam.cameras.length;
const poses = sets.flatMap((s) => s.poses), images = sets.flatMap((s) => s.images);
const T = poses.length;
const views = Array.from({ length: cameras }, (_, c) => images.filter((im) => im.camera === c).sort((a, b) => a.pose - b.pose));
llm.print(`  exam: ${T} frames from ${cameras} cameras (${exam.cameras.map((c) => c.name).join(', ')}), ${exam.joints.length} joints`);

const rig = readRig(cfg.rig);
const finder = Finder.load(cfg.finder);
const backbone = cfg.tracks_in ? null : await loadBackbone(cfg.model);

// 1. Every view found as a video's frames would be.
const random = generator(17);
const backdrops = new Backdrops(exam.size, cfg.backgrounds ?? null, random);
const noise = llm.randomNormal(1 << 21, 17);
const fitted = [];
const foundErr = [];
const saved = cfg.tracks_in ? JSON.parse(llm.readText(cfg.tracks_in)) : null;
for (let c = 0; c < cameras; c++) {
	let tracks;
	if (saved) tracks = saved[c];
	else {
		const backdrop = backdrops.next();
		const pictures = views[c].map((im) => composite(llm.image.load(`${dir}/${im.file}`), backdrop, random, noise));
		const frames = { count: T, image: (k) => pictures[k] };
		tracks = await finder.detect(backbone.encoder, frames, { flip: cfg.flip !== false && cfg.flip !== 'false' });
	}
	fitted.push({ tracks, pitch: views[c][0].pitch });
	for (const h of tracks.hypotheses.filter((h) => h.frames.length < T || tracks.hypotheses.length === 1)) {
		h.frames.forEach((k, i) => {
			tracks.points.forEach((name, p) => {
				const q = h.tracks[i][p], im = views[c][k], j = exam.joints.indexOf(name);
				// A joint where it was drawn; a landmark where it was drawn in sight.
				const m = j < 0 ? (exam.landmarks ?? []).findIndex((l) => l.name === name) : -1;
				const truth = j >= 0 ? im.points[j] : m >= 0 && im.marks?.[m]?.[3] === 1 ? im.marks[m] : null;
				if (q && truth) foundErr.push(Math.hypot(q[0] - (truth[0] - 0.5), q[1] - (truth[1] - 0.5)));
			});
		});
	}
}
if (cfg.tracks_out) llm.writeBytes(cfg.tracks_out, utf8(JSON.stringify(fitted.map((f) => f.tracks))));
foundErr.sort((a, b) => a - b);
const pct = (a, f) => a[Math.min(a.length - 1, Math.floor(f * a.length))];
const foundPx = pct(foundErr, 0.5);
llm.print(`  found: median ${foundPx.toFixed(1)}px, 90% ${pct(foundErr, 0.9).toFixed(1)}px of ${exam.size}px pictures`);

// 2. The fit, and every bone against the truth in the root's frame.
if (cfg.noise) {
	const [camera, px] = String(cfg.noise).split(':');
	const c = exam.cameras.findIndex((k) => k.name === camera);
	if (c < 0 || !(Number(px) >= 0)) throw new Error(`noise=${cfg.noise}: wants <camera>:<pixels>, a camera of ${exam.cameras.map((k) => k.name).join(', ')}`);
	const gauss = generator(29);
	const normal = () => Math.sqrt(-2 * Math.log(1 - gauss())) * Math.cos(2 * Math.PI * gauss());
	for (const h of fitted[c].tracks.hypotheses) for (const row of h.tracks) for (const q of row) if (q) { q[0] += normal() * Number(px); q[1] += normal() * Number(px); }
	llm.print(`  noise: ${px}px on the ${camera} camera's tracks`);
}
// The finder's landmarks the fit takes: as /animate takes them (all but a mirror
// pair on one joint), `landmarks=all`, `landmarks=none`, or those listed.
const named = String(cfg.landmarks ?? '');
const used = cfg.landmarks === undefined ? fittedLandmarks(finder.landmarks) : named === 'all' ? finder.landmarks : named === 'none' ? [] : finder.landmarks.filter((m) => named.split(',').includes(m.name));
const markers = Object.fromEntries(used.map((m) => [m.name, { joint: m.joint, offset: m.offset }]));
if (finder.landmarks.length) llm.print(`  landmarks fitted: ${used.map((m) => m.name).join(', ') || 'none'}`);
const dropped = new Set(finder.landmarks.filter((m) => !used.includes(m)).map((m) => m.name));
if (dropped.size) {
	for (const f of fitted) {
		const t = f.tracks, keep = t.points.map((n, p) => (dropped.has(n) ? -1 : p)).filter((p) => p >= 0);
		const pick = (row) => (row ? keep.map((p) => row[p]) : row);
		f.tracks = { ...t, points: keep.map((p) => t.points[p]), hypotheses: t.hypotheses.map((h) => ({ ...h, tracks: h.tracks.map(pick), scores: h.scores.map(pick) })) };
	}
}
const fit = fitRig(rig, fitted, { markers, iterations: Number(cfg.iterations ?? 1500), weights: typeof cfg.weights === 'string' ? JSON.parse(cfg.weights) : cfg.weights ?? null });
const p = fit.problem;
p.forward();
const J = rig.names.length;
const truthIndex = rig.names.map((n) => exam.joints.indexOf(n));
if (truthIndex.some((i) => i < 0)) throw new Error('the rig and the exam name their joints differently');
// The truth's world turns and places, Rigsmith's way: W_j = W_parent R_j, the
// joint at its parent plus the rest offset turned by the parent.
function truthFrame(t) {
	const R = poses[t].rotations, E = exam.joints.length, W = new Array(E), P = new Array(E);
	for (let j = 0; j < E; j++) {
		const q = exam.parents[j];
		if (q < 0) { W[j] = R[j]; P[j] = exam.rest[j]; continue; }
		W[j] = qmul(W[q], R[j]);
		const o = qrotate(W[q], [0, 1, 2].map((d) => exam.rest[j][d] - exam.rest[q][d]));
		P[j] = [0, 1, 2].map((d) => P[q][d] + o[d]);
	}
	return { W, P };
}
const unit = (v) => { const n = Math.hypot(...v); return n < 1e-9 ? null : v.map((x) => x / n); };
const seen = new Set(finder.joints);
const perBone = new Float64Array(J), counted = new Float64Array(J), worldErr = [], rootUp = [], rootAhead = [], placeErr = [], placeRootErr = [];
const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * DEG;
for (let t = 0; t < T; t++) {
	const { W, P } = truthFrame(t);
	const rootT = qconj(W[truthIndex[0]]);
	const G = p.G.subarray(t * J * 9, t * J * 9 + 9);
	// The root's own axes: its up (y) and forward (z), fitted and true.
	rootUp.push(angle([G[1], G[4], G[7]], qrotate(W[truthIndex[0]], [0, 1, 0])));
	rootAhead.push(angle([G[2], G[5], G[8]], qrotate(W[truthIndex[0]], [0, 0, 1])));
	// Every seen joint's place against the root's, as a share of the rig's
	// height: in the world, and in the root's frame.
	const r = t * J * 3, tr = P[truthIndex[0]];
	for (let j = 1; j < J; j++) {
		if (!seen.has(rig.names[j])) continue;
		const a = (t * J + j) * 3, f = [0, 1, 2].map((d) => p.pos[a + d] - p.pos[r + d]), g = [0, 1, 2].map((d) => P[truthIndex[j]][d] - tr[d]);
		placeErr.push(Math.hypot(f[0] - g[0], f[1] - g[1], f[2] - g[2]) / rig.height * 100);
		const fr = [0, 1, 2].map((k) => G[k] * f[0] + G[3 + k] * f[1] + G[6 + k] * f[2]), gr = qrotate(rootT, g);
		placeRootErr.push(Math.hypot(fr[0] - gr[0], fr[1] - gr[1], fr[2] - gr[2]) / rig.height * 100);
	}
	for (let j = 1; j < J; j++) {
		const q = rig.parents[j];
		const a = (t * J + j) * 3, b = (t * J + q) * 3;
		const fitDir = unit([0, 1, 2].map((d) => p.pos[a + d] - p.pos[b + d]));
		const truthDir = unit([0, 1, 2].map((d) => P[truthIndex[j]][d] - P[truthIndex[q]][d]));
		if (!fitDir || !truthDir) continue;
		// Into the root's frame: Gᵀ v for the fit, W⁻¹ v for the truth.
		const f = [0, 1, 2].map((r) => G[r] * fitDir[0] + G[3 + r] * fitDir[1] + G[6 + r] * fitDir[2]);
		const g = qrotate(rootT, truthDir);
		perBone[j] += angle(f, g);
		counted[j]++;
		if (seen.has(rig.names[j])) worldErr.push(angle(fitDir, truthDir));
	}
}
const bones = [];
for (let j = 1; j < J; j++) if (counted[j]) bones.push({ joint: rig.names[j], deg: perBone[j] / counted[j], seen: seen.has(rig.names[j]) });
const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const seenDeg = median(bones.filter((b) => b.seen).map((b) => b.deg)), allDeg = median(bones.map((b) => b.deg));
llm.print(`  bones in the root's frame: median ${seenDeg.toFixed(1)} deg over the ${bones.filter((b) => b.seen).length} the finder sees, ${allDeg.toFixed(1)} deg over all ${bones.length}`);
llm.print(`  joints from the root: median ${median(placeErr).toFixed(1)}% of the rig's height in the world, ${median(placeRootErr).toFixed(1)}% in the root's frame`);
llm.print(`  in the world: median ${median(worldErr).toFixed(1)} deg a seen bone; the root's up ${median(rootUp).toFixed(1)} deg off, its forward ${median(rootAhead).toFixed(1)} deg`);
llm.print(`  worst: ${bones.filter((b) => b.seen).sort((a, b) => b.deg - a.deg).slice(0, 8).map((b) => `${b.joint} ${b.deg.toFixed(0)}`).join(', ')}`);

// 3. The turns along the chain.
const matOf = (q) => {
	const [x, y, z, w] = q;
	return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
};
// Aᵀ B, A B (row-major 3×3).
const tmul = (A, B) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[r] * B[c] + A[3 + r] * B[3 + c] + A[6 + r] * B[6 + c]));
const mul = (A, B) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]));
const turnDeg = (A, B) => { const M = tmul(A, B); return Math.acos(Math.max(-1, Math.min(1, (M[0] + M[4] + M[8] - 1) / 2))) * DEG; };
// The rotation vector of R (axis times angle), in degrees.
function logDeg(R) {
	const a = Math.acos(Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2)));
	const v = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
	const s = Math.sin(a);
	if (s < 1e-6) return a < 1 ? v.map((x) => (x / 2) * DEG) : [0, 0, 0].map((_, k) => Math.sqrt(Math.max(0, (R[k * 4] + 1) / 2)) * a * DEG);
	return v.map((x) => (x * a) / (2 * s) * DEG);
}
// Turns over the clip: their spread about their own mean, and the median change
// of the frame-to-frame turn.
function spreadDeg(Rs) {
	let mean = Rs[0];
	for (let it = 0; it < 3; it++) {
		const m = [0, 0, 0];
		for (const R of Rs) logDeg(tmul(mean, R)).forEach((x, k) => (m[k] += x / Rs.length));
		const a = Math.hypot(...m) / DEG;
		const n = a > 1e-9 ? m.map((x) => x / DEG / a) : [1, 0, 0];
		const q = [n[0] * Math.sin(a / 2), n[1] * Math.sin(a / 2), n[2] * Math.sin(a / 2), Math.cos(a / 2)];
		mean = mul(mean, matOf(q));
	}
	return Math.sqrt(Rs.reduce((s, R) => s + turnDeg(mean, R) ** 2, 0) / Rs.length);
}
function jitterDeg(Rs) {
	const w = [];
	for (let t = 1; t < Rs.length; t++) w.push(logDeg(tmul(Rs[t - 1], Rs[t])));
	const d = [];
	for (let t = 1; t < w.length; t++) d.push(Math.hypot(w[t][0] - w[t - 1][0], w[t][1] - w[t - 1][1], w[t][2] - w[t - 1][2]));
	return median(d);
}
const chain = cfg.chain ? String(cfg.chain).split(',').map((n) => rig.names.indexOf(n.trim())) : (() => {
	let j = rig.names.indexOf('head');
	if (j < 0) j = [...seen].map((n) => rig.names.indexOf(n)).filter((i) => i >= 0).reduce((a, b) => (rig.positions[b][1] > rig.positions[a][1] ? b : a));
	const path = [];
	for (; j >= 0; j = rig.parents[j]) path.unshift(j);
	return path;
})();
if (chain.some((j) => j < 0)) throw new Error(`chain=${cfg.chain}: names a joint the rig does not have`);
const local = (t, j) => p.R.subarray((t * J + j) * 9, (t * J + j) * 9 + 9);
const world = (t, j) => p.G.subarray((t * J + j) * 9, (t * J + j) * 9 + 9);
const truthLocal = (t, j) => matOf(poses[t].rotations[truthIndex[j]]);
const rows = [];
for (const j of chain.slice(1)) {
	const err = [], fitTurns = [], truthTurns = [];
	for (let t = 0; t < T; t++) {
		const F = Array.from(local(t, j)), R = truthLocal(t, j);
		err.push(turnDeg(F, R));
		fitTurns.push(F);
		truthTurns.push(R);
	}
	rows.push({ joint: rig.names[j], deg: median(err), wobble: (spreadDeg(fitTurns) + 2) / (spreadDeg(truthTurns) + 2), spread: spreadDeg(fitTurns), truthSpread: spreadDeg(truthTurns), jitter: jitterDeg(fitTurns), truthJitter: jitterDeg(truthTurns) });
}
const headJ = chain[chain.length - 1], rootJ = chain[0];
const headErr = [];
for (let t = 0; t < T; t++) {
	const { W } = truthFrame(t);
	const truthRel = matOf(qmul(qconj(W[truthIndex[rootJ]]), W[truthIndex[headJ]]));
	headErr.push(turnDeg(tmul(Array.from(world(t, rootJ)), Array.from(world(t, headJ))), truthRel));
}
llm.print(`  the chain ${chain.map((j) => rig.names[j]).join(' > ')}:`);
for (const r of rows) llm.print(`    ${r.joint.padEnd(14)} turn off ${r.deg.toFixed(1).padStart(5)} deg  spread ${r.spread.toFixed(1)} vs ${r.truthSpread.toFixed(1)} (wobble ${r.wobble.toFixed(2)})  jitter ${r.jitter.toFixed(2)} vs ${r.truthJitter.toFixed(2)}`);
const chainWorst = Math.max(...rows.map((r) => r.deg)), wobbleWorst = Math.max(...rows.map((r) => r.wobble));
const jitterBad = rows.filter((r) => r.jitter > Math.max(1.5 * r.truthJitter, 0.5));
llm.print(`  ${rig.names[headJ]} in the ${rig.names[rootJ]}'s frame: median ${median(headErr).toFixed(1)} deg off, 90% ${pct(headErr.slice().sort((a, b) => a - b), 0.9).toFixed(1)}`);

check(foundPx < Number(cfg.found_px ?? 12), `the finder finds the exam's joints (median ${foundPx.toFixed(1)}px)`);
check(median(placeErr) < Number(cfg.place_pct ?? 6), `the fit puts the exam's joints back (median ${median(placeErr).toFixed(1)}% of the rig's height from the truth)`);
check(chainWorst < Number(cfg.chain_deg ?? 10), `the chain's turns follow the truth's (worst joint ${chainWorst.toFixed(1)} deg off)`);
check(median(headErr) < Number(cfg.head_deg ?? 12), `the ${rig.names[headJ]} turns as the truth's on the ${rig.names[rootJ]} (median ${median(headErr).toFixed(1)} deg off)`);
check(wobbleWorst < Number(cfg.wobble ?? 2), `the chain wobbles no more than the truth (worst ${wobbleWorst.toFixed(2)}×)`);
check(!jitterBad.length, `the chain is no shakier than the truth${jitterBad.length ? ` (${jitterBad.map((r) => r.joint).join(', ')} shake)` : ''}`);

finder.dispose();
backbone?.dispose();
llm.print(failures ? `${failures} failed` : 'all passed');
