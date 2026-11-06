// Re-timing a studio clip's halves on a cycle (sync.js): heights leave left and
// right out, so a run's read the same a step later, and a shift must stop short
// of that. A view that jumps back partway through is cut back into step. No GPU,
// no files.
//
//   llm-runner tests/creature_sync.js
import { llm } from '/core/llm.js';
import { syncViews } from '/plugins/creature/sync.js';

let failures = 0;
const check = (ok, what) => {
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
	if (!ok) failures++;
};

const T = 120, STEP = 10, points = ['hips', 'foot_L', 'foot_R'];
const cameras = [{ logS: 0, shift: [0, 0] }, { logS: 0, shift: [0, 0] }];
// Heights over the frames as tracks: y down, every point sure.
const tracks = (heights) => ({
	points, frames: T,
	hypotheses: [{
		from: 0,
		frames: Array.from({ length: T }, (_, t) => t),
		tracks: Array.from({ length: T }, (_, t) => heights(t).map((h) => [0, -h])),
		scores: Array.from({ length: T }, () => points.map(() => 1)),
	}],
});
// The hips bob once a step; the feet swap once every two.
const run = (t) => {
	const bob = Math.sin((2 * Math.PI * t) / STEP), swing = 30 * Math.sin((Math.PI * t) / STEP);
	return [100 + 10 * bob, 40 + swing, 40 - swing];
};
const side = tracks(run);

const inStep = syncViews(side, side, cameras);
check(inStep.report.repeat_frames === STEP, `a run's heights repeat every step (${inStep.report.repeat_frames} frames)`);
check(!inStep.report.retimed && inStep.report.best_lag === 0, 'views in step are left alone');

// Seven frames late is as near the next step as this one: nothing tells them
// apart, so no shift reaches half a step.
const late = syncViews(tracks((t) => run(t - 7)), side, cameras);
check(Math.abs(late.report.best_lag) < STEP / 2, `the best lag stops short of half a step (${late.report.best_lag})`);
check((late.report.shift_frames ?? 0) < STEP / 2, `so does a re-time (${late.report.shift_frames ?? 'none'})`);

// Motion that does not repeat keeps the whole range.
const once = (t) => [100 + t, 40 + 0.5 * t, 40 + 0.2 * t];
const free = syncViews(tracks((t) => once(Math.max(0, t - 8))), tracks(once), cameras);
check(free.report.repeat_frames === null, 'a one-way motion has no repeat');
check(free.report.best_lag === -8 || free.report.best_lag === 8, `its lag is found in full (${free.report.best_lag})`);
check(free.report.retimed && free.report.shift_frames > 5, `and it is re-timed (${free.report.shift_frames} frames)`);

// A walk seen by the studio's two cameras: the legs a steady gap apart across the
// body, swinging forward and back. `at(t)` is the moment each frame shows.
const WALK = 38, HEIGHT = 100, legs = ['hips', 'knee_L', 'knee_R', 'ankle_L', 'ankle_R'];
const seen = [{ yaw: -Math.PI / 4, pitch: 0, logS: 0, shift: [0, 0] }, { yaw: Math.PI / 2, pitch: 0, logS: 0, shift: [0, 0] }];
const pose = (t) => {
	const swing = Math.sin((2 * Math.PI * t) / WALK), lift = (s) => Math.max(0, s);
	return [[0, 50, 0], [9, 28, 15 * swing], [-9, 28, -15 * swing], [9, 8 + 6 * lift(swing), 33 * swing], [-9, 8 + 6 * lift(-swing), -33 * swing]];
};
// Reads in stretches of six frames, as a finder's come, and one chain over them all.
const walk = (camera, at, points = legs, length = T) => {
	const row = (t) => pose(at(t)).map(([x, y, z]) => [Math.cos(camera.yaw) * x + Math.sin(camera.yaw) * z, -y]);
	const stretch = (from, frames) => ({ from, direction: 1, frames, tracks: frames.map(row), scores: frames.map(() => points.map(() => 1)) });
	const all = Array.from({ length }, (_, t) => t), stretches = [];
	for (let t = 0; t < length; t += 6) stretches.push(stretch(t, all.slice(t, t + 6)));
	return { points, frames: length, boxes: all.map((t) => [t, 0, 1, 1]), hypotheses: [...stretches, stretch(0, all)] };
};
const JUMP = 53, BACK = 14, slipped = (t) => (t < JUMP ? t : t - BACK), steady = (t) => t;
const options = { height: HEIGHT, band: 4, lags: 4 };

const clean = syncViews(walk(seen[0], steady), walk(seen[1], steady), seen, options);
check(!clean.report.slip && clean.front.frames === T && clean.side.frames === T, 'views in step are not cut');

const put = syncViews(walk(seen[0], slipped), walk(seen[1], steady), seen, options);
const slip = put.report.slip;
check(slip?.late === 'front' && slip.frames === BACK, `a view stepping back is found (${JSON.stringify(slip)})`);
check(slip?.frame <= JUMP && slip.frame + BACK > JUMP, `the cut holds the jump (from ${slip?.frame})`);
check(put.front.frames === T - BACK && put.side.frames === T - BACK, 'both views end that much sooner');
const framesOf = (tracks) => new Set(tracks.hypotheses.flatMap((h) => h.frames));
check(put.front.hypotheses.filter((h) => h.frames.length < T - BACK).length >= 15, 'the short stretches of reads survive the cut');
check(Array.from({ length: T - BACK }, (_, t) => t).every((t) => framesOf(put.front).has(t)), 'every frame left still has its reads');
const firstAt = (tracks, t) => {
	for (const h of tracks.hypotheses) { const i = h.frames.indexOf(t); if (i >= 0) return h.tracks[i]; }
	return null;
};
const sideways = (tracks, t) => { const r = firstAt(tracks, t); return r[3][0] - r[4][0]; };
const after = walk(seen[0], steady);
check(Array.from({ length: T - BACK }, (_, t) => t).every((t) => Math.abs(sideways(put.front, t) - sideways(after, t)) < 1e-9), 'and shows the same moment as the side view');

const behind = syncViews(walk(seen[0], steady), walk(seen[1], slipped), seen, options);
check(behind.report.slip?.late === 'side' && behind.report.slip.frames === BACK, `the side view slipping is found too (${JSON.stringify(behind.report.slip)})`);
check(behind.front.frames === T - BACK && behind.side.frames === T - BACK, 'and cut the same way');

const asked = syncViews(walk(seen[0], slipped), walk(seen[1], steady), seen, { ...options, retime: false });
check(asked.report.slip?.frames === BACK && asked.report.slip.put_back === false, 'told only to report, it says so');
check(asked.front.frames === T && asked.side.frames === T, 'and leaves the tracks alone');

const armless = ['hips', 'wrist_L', 'wrist_R', 'elbow_L', 'elbow_R'];
const noLegs = syncViews(walk(seen[0], slipped, armless), walk(seen[1], steady, armless), seen, options);
check(!noLegs.report.slip, 'without legs to read there is no check');

// No reads where the cut would join: it still lands on a frame.
const gapped = walk(seen[0], slipped);
for (const h of gapped.hypotheses) h.tracks = h.tracks.map((row, i) => (h.frames[i] > 30 && h.frames[i] < 75 && h.frames[i] % 2 ? row.map(() => null) : row));
const holes = syncViews(gapped, walk(seen[1], steady), seen, options);
check(Number.isFinite(holes.report.slip?.frame), `reads missing around the join leave a real frame (${holes.report.slip?.frame})`);

const noHeight = syncViews(walk(seen[0], slipped), walk(seen[1], steady), seen, { band: 4, lags: 4 });
check(!noHeight.report.slip, 'nor without the rig\'s height');

llm.print(failures ? `${failures} failed` : 'all passed');
if (failures) throw new Error(`${failures} checks failed`);
