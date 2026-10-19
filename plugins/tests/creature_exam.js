// The joint finder's exam: a movement whose every turn is known, found and
// fitted as a video would be, then held against the turns that made it.
//
//   llm-runner tests/creature_exam.js model=sam-3d-body-dinov3 rig=rig.glb \
//       finder=rig.finder.safetensors exam=exam/ [backgrounds=frames/] \
//       [iterations=1500] [bone_deg=20] [found_px=12]
//
// `exam` is Rigsmith's render_poses with motion: one smooth movement inside
// the rig's limits, from two cameras held still (or more). Each camera's
// pictures are laid over a background as a video's frames, the finder finds
// the joints in each (its own cameras, its own crops), and the rig is fitted
// to all the views at once, as POST /animate does with a finder.
//
// Graded:
// 1. Finding: every found joint against where the exam drew it, in pixels.
// 2. The fit: each bone's direction against the truth's, both in their
//    root's frame (the fit leaves the heading to the camera), in degrees per
//    bone over the clip; the bones the finder sees and all of them. Passes
//    when the seen bones' median is under `bone_deg` and the finding's median
//    under `found_px`.
import { llm } from '../lib/llm.js';
import { loadBackbone } from '../creature/backbone.js';
import { Finder } from '../creature/detect.js';
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
const backbone = await loadBackbone(cfg.model);

// 1. Every view found as a video's frames would be.
const random = generator(17);
const backdrops = new Backdrops(exam.size, cfg.backgrounds ?? null, random);
const noise = llm.randomNormal(1 << 21, 17);
const fitted = [];
const foundErr = [];
for (let c = 0; c < cameras; c++) {
	const backdrop = backdrops.next();
	const pictures = views[c].map((im) => composite(llm.image.load(`${dir}/${im.file}`), backdrop, random, noise));
	const frames = { count: T, image: (k) => pictures[k] };
	const tracks = await finder.detect(backbone.encoder, frames, {});
	fitted.push({ tracks, pitch: views[c][0].pitch });
	for (const h of tracks.hypotheses.filter((h) => h.frames.length < T || tracks.hypotheses.length === 1)) {
		h.frames.forEach((k, i) => {
			tracks.points.forEach((name, p) => {
				const q = h.tracks[i][p], truth = views[c][k].points[exam.joints.indexOf(name)];
				if (q) foundErr.push(Math.hypot(q[0] - (truth[0] - 0.5), q[1] - (truth[1] - 0.5)));
			});
		});
	}
}
foundErr.sort((a, b) => a - b);
const pct = (a, f) => a[Math.min(a.length - 1, Math.floor(f * a.length))];
const foundPx = pct(foundErr, 0.5);
llm.print(`  found: median ${foundPx.toFixed(1)}px, 90% ${pct(foundErr, 0.9).toFixed(1)}px of ${exam.size}px pictures`);

// 2. The fit, and every bone against the truth in the root's frame.
const fit = fitRig(rig, fitted, { iterations: Number(cfg.iterations ?? 1500) });
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
const perBone = new Float64Array(J), counted = new Float64Array(J);
for (let t = 0; t < T; t++) {
	const { W, P } = truthFrame(t);
	const rootT = qconj(W[truthIndex[0]]);
	const G = p.G.subarray(t * J * 9, t * J * 9 + 9);
	for (let j = 1; j < J; j++) {
		const q = rig.parents[j];
		const a = (t * J + j) * 3, b = (t * J + q) * 3;
		const fitDir = unit([0, 1, 2].map((d) => p.pos[a + d] - p.pos[b + d]));
		const truthDir = unit([0, 1, 2].map((d) => P[truthIndex[j]][d] - P[truthIndex[q]][d]));
		if (!fitDir || !truthDir) continue;
		// Into the root's frame: Gᵀ v for the fit, W⁻¹ v for the truth.
		const f = [0, 1, 2].map((r) => G[r] * fitDir[0] + G[3 + r] * fitDir[1] + G[6 + r] * fitDir[2]);
		const g = qrotate(rootT, truthDir);
		perBone[j] += Math.acos(Math.max(-1, Math.min(1, f[0] * g[0] + f[1] * g[1] + f[2] * g[2]))) * DEG;
		counted[j]++;
	}
}
const bones = [];
for (let j = 1; j < J; j++) if (counted[j]) bones.push({ joint: rig.names[j], deg: perBone[j] / counted[j], seen: seen.has(rig.names[j]) });
const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const seenDeg = median(bones.filter((b) => b.seen).map((b) => b.deg)), allDeg = median(bones.map((b) => b.deg));
llm.print(`  bones in the root's frame: median ${seenDeg.toFixed(1)} deg over the ${bones.filter((b) => b.seen).length} the finder sees, ${allDeg.toFixed(1)} deg over all ${bones.length}`);
llm.print(`  worst: ${bones.filter((b) => b.seen).sort((a, b) => b.deg - a.deg).slice(0, 8).map((b) => `${b.joint} ${b.deg.toFixed(0)}`).join(', ')}`);

check(foundPx < Number(cfg.found_px ?? 12), `the finder finds the exam's joints (median ${foundPx.toFixed(1)}px)`);
check(seenDeg < Number(cfg.bone_deg ?? 20), `the fit gives the exam's movement back (median ${seenDeg.toFixed(1)} deg a bone)`);

finder.dispose();
backbone.dispose();
llm.print(failures ? `${failures} failed` : 'all passed');
