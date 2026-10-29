// A studio clip's known cameras (studio.js): do they put the rig where
// Rigsmith's render_studio drew it, and does a fit through them get the
// motion back, the root's own path included, and are its halves in step?
// No GPU.
//
//   llm-runner tests/creature_studio.js rig=rig.glb studio=studio_rest/studio.json [iterations=600]
//
// `studio` is render_studio's layout of the rig in its rest pose (no clip), so
// its `pixels` are the rest joints as the picture shows them. The motion is
// made up: the root walking a little to the side and back, the spine leaning,
// one arm raised and lowered; each view's points are drawn through its
// camera with a pixel of noise, at 1.5 times the picture's size (as MiniMax
// gave), and fitted.
import { llm } from '../lib/llm.js';
import { readRig } from '../creature/rig.js';
import { RigFit, observations, fitRig } from '../creature/fit.js';
import { readStudio, studioViews, calibrateViews } from '../creature/studio.js';
import { syncViews } from '../creature/sync.js';

const cfg = globalThis.__llm_config;
const rig = readRig(cfg.rig);
const layout = readStudio(cfg.studio);
const iterations = Number(cfg.iterations ?? 600);
let failures = 0;
const check = (ok, what) => {
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
	if (!ok) failures++;
};

// Point X (model space) through a studio camera, in its box's pixels (y down).
const pixel = (c, X) => {
	const s = Math.exp(c.logS);
	return [s * (Math.cos(c.yaw) * X[0] + Math.sin(c.yaw) * X[2]) + c.shift[0], -(s * X[1] + c.shift[1])];
};

// 1. The rest joints land where render_studio drew them, at its own size and
//    at 1.5 times it.
for (const scale of [1, 1.5]) {
	const views = studioViews(layout, layout.width * scale, layout.height * scale);
	let worst = 0, n = 0;
	views.forEach((v, i) => {
		layout.joints.forEach((name, k) => {
			const j = rig.names.indexOf(name);
			if (j < 0) return;
			const [x, y] = pixel(v.camera, rig.positions[j]);
			const want = layout.pixels[i][k];
			worst = Math.max(worst, Math.hypot(x + v.box[0] - want[0] * scale, y + v.box[1] - want[1] * scale));
			n++;
		});
	});
	check(n > 0 && worst < 0.5 * scale + 0.05, `at ${scale}x, ${n} rest joints land on render_studio's pixels (worst ${worst.toFixed(3)} px)`);
}
const views = studioViews(layout, layout.width * 1.5, layout.height * 1.5);
const front = views[0].camera, side = views[1].camera;
check(Math.abs(front.yaw) < 0.05 && Math.abs(Math.abs(side.yaw) - Math.PI / 2) < 0.05, `front looks at the face, side across (yaws ${(front.yaw * 180 / Math.PI).toFixed(1)}°, ${(side.yaw * 180 / Math.PI).toFixed(1)}°)`);
const left = rig.positions[0].slice();
left[0] += 0.2;
check(pixel(front, left)[0] > pixel(front, rig.positions[0])[0], 'the rig\'s left (+X) is on the front view\'s right, as it faces the viewer');

// 2. A made-up motion through the cameras, fitted with them held.
const T = 48, J = rig.names.length;
const blank = { points: rig.names, frames: T, hypotheses: [{ from: 0, direction: 1, frames: [...Array(T).keys()], tracks: Array.from({ length: T }, () => rig.names.map(() => null)), scores: Array.from({ length: T }, () => rig.names.map(() => null)) }] };
const truth = new RigFit(rig, observations(rig, blank), {});
const joint = (name) => rig.names.indexOf(name);
const arm = joint('shoulder_L') >= 0 ? joint('shoulder_L') : 1, spine = joint('spine') >= 0 ? joint('spine') : 1;
for (let t = 0; t < T; t++) {
	const a = Math.sin((Math.PI * t) / (T - 1));
	truth.theta[(t * J + arm) * 3 + 2] = 0.9 * a;
	truth.theta[(t * J + spine) * 3] = 0.15 * a;
	truth.root[t * 3] = rig.positions[0][0] + 0.12 * a;
	truth.root[t * 3 + 1] = rig.positions[0][1] - 0.02 * a;
	truth.root[t * 3 + 2] = rig.positions[0][2] + 0.05 * a;
}
truth.forward();
let seed = 11;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
const drawn = views.map((v) => {
	const rows = Array.from({ length: T }, (_, t) => rig.names.map((_, j) => {
		const [x, y] = pixel(v.camera, Array.from(truth.pos.subarray((t * J + j) * 3, (t * J + j) * 3 + 3)));
		return [x + rnd(), y + rnd()];
	}));
	const hypotheses = [];
	for (let from = 0; from < T; from += 6) {
		const frames = [...Array(Math.min(6, T - from)).keys()].map((k) => from + k);
		hypotheses.push({ from, direction: 1, frames, tracks: frames.map((t) => rows[t]), scores: frames.map(() => rig.names.map(() => 1)) });
	}
	return { tracks: { points: rig.names, frames: T, hypotheses }, camera: v.camera };
});
const fit = fitRig(rig, drawn, { iterations });
const p = fit.problem;
p.forward();
let rootWorst = 0, jointMean = 0;
for (let t = 0; t < T; t++) {
	rootWorst = Math.max(rootWorst, Math.hypot(...[0, 1, 2].map((d) => p.root[t * 3 + d] - truth.root[t * 3 + d])));
	for (let j = 0; j < J; j++) jointMean += Math.hypot(...[0, 1, 2].map((d) => p.pos[(t * J + j) * 3 + d] - truth.pos[(t * J + j) * 3 + d])) / (T * J);
}
const camsHeld = views.every((v, i) => Math.abs(p.cam[2 * i] - v.camera.yaw) < 1e-12 && p.logS.subarray(i * T, (i + 1) * T).every((s) => s === v.camera.logS));
check(camsHeld, 'the cameras are held exactly as given');
check(rootWorst < 0.01 * Math.max(1, rig.height), `the root's path comes back (worst ${(rootWorst * 100).toFixed(2)} cm off, it moves 13 cm)`);
check(jointMean < 0.02 * rig.height, `joints within 2% of the rig's height on average (${(jointMean / rig.height * 100).toFixed(2)}%)`);
check(fit.report.reprojection_px.median < 3, `reprojection median ${fit.report.reprojection_px.median} px (a pixel of noise drawn)`);

// 3. The halves in step (sync.js): the drawn pair is left alone; a front view
//    played 1 to 5 frames behind is found out and re-timed.
const inStep = syncViews(drawn[0].tracks, drawn[1].tracks, views.map((v) => v.camera));
check(!inStep.report.retimed && inStep.report.best_lag === 0, `views in step are left alone (heights ${inStep.report.height_gap_px}px apart, best lag ${inStep.report.best_lag})`);
const late = (t) => Math.max(0, t - 3 - 2 * Math.sin((2 * Math.PI * t) / T));
const behind = {
	...drawn[0].tracks,
	hypotheses: drawn[0].tracks.hypotheses.map((h) => ({
		...h,
		tracks: h.frames.map((t) => rig.names.map((_, j) => {
			const u = late(t), a = Math.floor(u), b = Math.min(T - 1, a + 1), s = u - a;
			const X = [0, 1, 2].map((d) => truth.pos[(a * J + j) * 3 + d] * (1 - s) + truth.pos[(b * J + j) * 3 + d] * s);
			return pixel(views[0].camera, X);
		})),
	})),
};
const caught = syncViews(behind, drawn[1].tracks, views.map((v) => v.camera));
check(caught.report.retimed && caught.report.retime_gap_px < 0.5 * caught.report.height_gap_px,
	`a front view 1 to 5 frames late is re-timed (heights ${caught.report.height_gap_px}px apart, ${caught.report.retime_gap_px}px after, best constant lag ${caught.report.best_lag})`);

// 4. Calibration: a first frame drawn 1.5 times the picture's size, each
//    half 1% bigger about its middle and the floor and moved out from the
//    divider and down (as MiniMax drew one), is found out, and the corrected
//    cameras put the rest joints where that frame has them.
const reference = llm.image.load(cfg.studio.replace(/[^/]*$/, 'reference.png'));
const W = layout.width * 1.5, H = layout.height * 1.5, floorRow = layout.ground_row * 1.5;
const plain = studioViews(layout, W, H);
const moved = [{ k: 1.01, d: [-8, 4] }, { k: 1.01, d: [8, 4] }];
const middle = (v) => (plain[v].box[0] + plain[v].box[2]) / 2;
const framePixels = new Uint8Array(W * H * 3);
for (let y = 0; y < H; y++) {
	for (let x = 0; x < W; x++) {
		const v = x < plain[1].box[0] - 2 ? 0 : 1, { k, d } = moved[v];
		const fx = middle(v) + (x - middle(v) - d[0]) / k, fy = floorRow + (y - floorRow - d[1]) / k;
		const rx = Math.min(reference.width - 1, Math.max(0, Math.round(fx / 1.5))), ry = Math.min(reference.height - 1, Math.max(0, Math.round(fy / 1.5)));
		for (let c = 0; c < 3; c++) framePixels[(y * W + x) * 3 + c] = reference.pixels[(ry * reference.width + rx) * reference.channels + c];
	}
}
const first = { width: W, height: H, channels: 3, pixels: framePixels };
const calibrating = llm.now();
const calibrated = calibrateViews(layout, plain, reference, first);
const calibrationSeconds = (llm.now() - calibrating) / 1000;
check(calibrationSeconds < 5, `calibration takes ${calibrationSeconds.toFixed(1)} s for the two views`);
calibrated.report.forEach((c, v) => {
	check(Math.abs(c.scale - 1.01) < 0.004 && Math.abs(c.shift[0] - moved[v].d[0]) <= 2 && Math.abs(c.shift[1] - moved[v].d[1]) <= 2,
		`${c.name}: found scaled ${c.scale}, moved ${c.shift.join(', ')} (drawn 1.01, ${moved[v].d.join(', ')}); overlap ${c.overlap.join(' -> ')}`);
	let worst = 0;
	layout.joints.forEach((name, n) => {
		const j = rig.names.indexOf(name);
		if (j < 0) return;
		const [x, y] = pixel(calibrated.views[v].camera, rig.positions[j]), box = calibrated.views[v].box;
		const want = [middle(v) + moved[v].k * (layout.pixels[v][n][0] * 1.5 - middle(v)) + moved[v].d[0], floorRow + moved[v].k * (layout.pixels[v][n][1] * 1.5 - floorRow) + moved[v].d[1]];
		worst = Math.max(worst, Math.hypot(x + box[0] - want[0], y + box[1] - want[1]));
	});
	check(worst < 4, `${c.name}: the corrected camera puts the rest joints where the frame has them (worst ${worst.toFixed(2)} px)`);
});

llm.print(failures ? `${failures} failed` : 'all passed');
if (failures) throw new Error(`${failures} checks failed`);
