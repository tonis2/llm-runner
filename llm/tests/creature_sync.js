// Re-timing a studio clip's halves on a cycle (sync.js): heights leave left and
// right out, so a run's read the same a step later, and a shift must stop short
// of that. No GPU, no files.
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

llm.print(failures ? `${failures} failed` : 'all passed');
if (failures) throw new Error(`${failures} checks failed`);
