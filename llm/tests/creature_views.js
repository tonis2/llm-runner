// Two views of a known motion: does the second recover what the first cannot
// see along its line of sight? No GPU.
//
//   llm-runner tests/creature_views.js rig=rig.glb tracks=tracks.json request=request.json \
//       [angle=90] [iterations=600] [pitch=8]
//
// The truth is the fit of real tracks (tests/creature_fit.js's inputs). Its
// points are drawn through two cameras, A at the fitted yaw plus `angle` and
// B at the fitted yaw, with a pixel of noise and the real tracks' keyframes;
// the rig is fitted to A alone and to A and B together, and each fit's
// tracked points are held against the truth's.
import { llm } from '/core/llm.js';
import { readRig } from '/plugins/creature/rig.js';
import { fitRig } from '/plugins/creature/fit.js';

const cfg = globalThis.__llm_config;
const rig = readRig(cfg.rig);
const tracks = JSON.parse(llm.readText(cfg.tracks));
const request = cfg.request ? JSON.parse(llm.readText(cfg.request)) : {};
const markers = request.markers ?? {};
const pitch = Number(cfg.pitch ?? request.pitch ?? 8);
const iterations = Number(cfg.iterations ?? 600);
const angle = (Number(cfg.angle ?? 90) * Math.PI) / 180;
let failures = 0;

const truth = fitRig(rig, tracks, { markers, pitch, iterations }).problem;
truth.forward();
const T = truth.T, J = truth.J;

// The truth's points through a camera at `yaw` (its pitch, scale and aim),
// laid out as the real tracks are, each a pixel or so off.
let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
function draw(yaw) {
	const cam = Float64Array.from(truth.cam);
	truth.cam[0] = yaw;
	const points = truth.views[0].points;
	const frames = Array.from({ length: T }, (_, t) => truth.project(0, t, points.length, (i) => truth.where(t, points[i])));
	truth.cam.set(cam);
	const at = (t, p) => [frames[t][p][0] + rnd(), -frames[t][p][1] + rnd()];
	return {
		...tracks,
		hypotheses: tracks.hypotheses.map((h) => ({
			from: h.from, direction: h.direction, frames: h.frames.slice(),
			tracks: h.frames.map((t) => points.map((_, p) => at(t, p))),
			scores: h.frames.map(() => points.map(() => 1)),
		})),
	};
}
const A = draw(truth.cam[0] + angle), B = draw(truth.cam[0]);

// How far each tracked point (a joint or a marker) sits from the points' own
// centre against the truth's (where the root sits among them is the priors'
// to say when no point is on it, and each view's aim hides it), the whole
// clip turned about the vertical to the truth's heading first (likewise): the
// mean, and the worst point's mean, in % of the rig's height.
function error(fit) {
	const p = fit.problem, points = truth.views[0].points, P = points.length;
	p.forward();
	const centred = (q, t) => {
		const xs = points.map((pt) => { const [X, o] = q.where(t, pt); return [X[o], X[o + 1], X[o + 2]]; });
		const c = [0, 1, 2].map((d) => xs.reduce((s, x) => s + x[d], 0) / P);
		return xs.map((x) => x.map((v, d) => v - c[d]));
	};
	const pairs = [];
	for (let t = 0; t < T; t++) {
		const a = centred(p, t), b = centred(truth, t);
		for (let i = 0; i < P; i++) pairs.push([a[i], b[i], i]);
	}
	let c = 0, s = 0;
	for (const [a, b] of pairs) { c += a[0] * b[0] + a[2] * b[2]; s += a[2] * b[0] - a[0] * b[2]; }
	const turn = Math.atan2(s, c), ct = Math.cos(turn), st = Math.sin(turn);
	const each = new Float64Array(P);
	for (const [a, b, i] of pairs) each[i] += Math.hypot(ct * a[0] + st * a[2] - b[0], a[1] - b[1], -st * a[0] + ct * a[2] - b[2]) / T;
	const worst = each.indexOf(Math.max(...each));
	return { mean: (100 * each.reduce((x, e) => x + e, 0)) / (P * rig.height), worst: (100 * each[worst]) / rig.height, at: points[worst].name, turn: (turn * 180) / Math.PI };
}

const one = fitRig(rig, A, { markers, pitch, iterations });
const two = fitRig(rig, [{ tracks: A }, { tracks: B }], { markers, pitch, iterations });
const e1 = error(one), e2 = error(two);
llm.print(`  truth's yaw ${(truth.cam[0] * 180 / Math.PI).toFixed(1)}°; view A at +${(angle * 180 / Math.PI).toFixed(0)}°`);
llm.print(`  A alone: points ${e1.mean.toFixed(2)}% of height off, worst ${e1.at} ${e1.worst.toFixed(2)}% (turned ${e1.turn.toFixed(1)}°); yaw ${one.report.camera.yaw}°`);
llm.print(`  A and B: points ${e2.mean.toFixed(2)}% of height off, worst ${e2.at} ${e2.worst.toFixed(2)}% (turned ${e2.turn.toFixed(1)}°); yaws ${two.report.views.map((r) => r.camera.yaw).join('°, ')}°`);
if (!(e2.mean <= 0.5 * e1.mean && e2.mean <= Number(cfg.max_error ?? 2))) {
	llm.print('FAIL: two views should at least halve the error and come under 2% of the height');
	failures++;
}
llm.print(failures ? `${failures} failed` : 'ok');
