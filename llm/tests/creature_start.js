// /animate's `start: "reference"`: a studio clip whose first frame is the
// studio's picture takes frame 0's joints from studio.json's pixels, whatever
// was read there. No GPU, no files.
//
//   llm-runner tests/creature_start.js
import { llm } from '/core/llm.js';
import { animate, startOnReference } from '/plugins/creature/main.js';
import { studioViews, referenceJoints } from '/plugins/creature/studio.js';

let failures = 0;
const check = (ok, what) => {
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
	if (!ok) failures++;
};

const rig = {
	names: ['hips', 'leg_L', 'foot_L', 'leg_R', 'foot_R', 'head'],
	parents: [-1, 0, 1, 0, 3, 0],
	positions: [[0, 1, 0], [0.2, 1, 0], [0.2, 0, 0], [-0.2, 1, 0], [-0.2, 0, 0], [0, 1.6, 0.1]],
};
// The picture's pose: the left leg swung forward 45 degrees.
const posed = rig.positions.map((p) => p.slice());
posed[2] = [0.2, 1 - Math.SQRT1_2, Math.SQRT1_2];

// render_studio's layout of it: a three-quarter view and the right side.
const W = 640, H = 360, PPU = 100, FLOOR = 320, R = Math.SQRT1_2;
const layout = {
	version: 1, width: W, height: H, ground_row: FLOOR, pixels_per_unit: PPU, divider_width: 4,
	views: [
		{ name: 'three_quarter', x: 0, width: 318, yaw: 45, origin: [0.01, 0, 0.02], right: [R, 0, -R], up: [0, 1, 0] },
		{ name: 'side', x: 322, width: 318, yaw: -90, origin: [0.01, 0, 0.02], right: [0, 0, 1], up: [0, 1, 0] },
	],
	joints: rig.names,
};
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => a.map((v, d) => v - b[d]);
layout.pixels = layout.views.map((v) => posed.map((p) => [v.x + v.width / 2 + dot(sub(p, v.origin), v.right) * PPU, FLOOR - dot(sub(p, v.origin), v.up) * PPU]));

const known = referenceJoints(layout);
const off = Math.max(...known.positions.map((p, j) => Math.hypot(...sub(p, posed[j]))));
check(known.names === rig.names && off < 1e-9, `the picture's pixels give its joints back in the model (${off.toExponential(1)} off at worst)`);
check(referenceJoints({ ...layout, pixels: undefined }) === null, 'a layout without pixels has none to give');
check(referenceJoints({ ...layout, views: [layout.views[1], layout.views[1]] }) === null, 'nor two views that look the same way');

// The clip holds the picture's pose, at 1.5 times its size; on frame 0 the
// side view reads the feet the wrong way round, sure of itself.
const T = 6, views = studioViews(layout, W * 1.5, H * 1.5);
const pixel = (c, X) => { const s = Math.exp(c.logS); return [s * (Math.cos(c.yaw) * X[0] + Math.sin(c.yaw) * X[2]) + c.shift[0], -(s * X[1] + c.shift[1])]; };
const points = [...rig.names, 'head.top'], markers = { 'head.top': { joint: 'head', offset: [0, 0.1, 0] } };
const place = (name) => (name === 'head.top' ? [0, 1.7, 0.1] : posed[rig.names.indexOf(name)]);
const frames = [...Array(T).keys()];
const tracks = views.map((v, i) => {
	const row = (t) => points.map((n) => pixel(v.camera, place(i === 1 && t === 0 && n === 'foot_L' ? 'foot_R' : i === 1 && t === 0 && n === 'foot_R' ? 'foot_L' : n)));
	return {
		points, frames: T, size: [v.box[2] - v.box[0], v.box[3] - v.box[1]], boxes: [],
		hypotheses: [
			{ from: 0, direction: 1, frames, tracks: frames.map(row), scores: frames.map(() => points.map(() => 0.9)) },
			// Second peaks: each point's a little to one side.
			{ from: 0, direction: 1, frames, tracks: frames.map((t) => row(t).map(([x, y]) => [x + 30, y])), scores: frames.map(() => points.map(() => 0.3)) },
		],
	};
});
const request = { rig, studio: layout, views: tracks.map((t) => ({ tracks: t })), size: [W * 1.5, H * 1.5], markers, iterations: 300, outputs: ['tracks'] };

const plain = await animate(request);
const wrongFoot = pixel(views[1].camera, posed[4]), rightFoot = pixel(views[1].camera, posed[2]);
const near = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.02;
check(near(plain.reply.tracks[1].hypotheses[0].tracks[0][2], wrongFoot) && plain.reply.report.start === undefined, 'without start, frame 0 is as it was read');

const { reply, fit } = await animate({ ...request, start: 'reference' });
check(reply.report.start?.joints === rig.names.length && reply.report.start.overlap === null, `the report says how many joints the picture gave (${JSON.stringify(reply.report.start)})`);
let worst = 0;
reply.tracks.forEach((t, i) => rig.names.forEach((n, p) => { const q = t.hypotheses[0].tracks[0][p], w = pixel(views[i].camera, posed[p]); worst = Math.max(worst, Math.hypot(q[0] - w[0], q[1] - w[1])); }));
check(worst < 0.02, `frame 0's joints are the picture's in both views (${worst.toFixed(3)} px off at worst)`);
check(near(reply.tracks[1].hypotheses[0].tracks[0][2], rightFoot), 'the left foot the side view read on the right one is put back');
check(reply.tracks.every((t) => rig.names.every((_, p) => t.hypotheses[0].scores[0][p] === 1 && t.hypotheses[1].tracks[0][p] === null && t.hypotheses[1].scores[0][p] === null)), 'at full score, with no second peak beside them');
const top = points.indexOf('head.top');
check(reply.tracks.every((t, i) => near(t.hypotheses[0].tracks[0][top], pixel(views[i].camera, place('head.top'))) && t.hypotheses[0].scores[0][top] === 0.9 && t.hypotheses[1].tracks[0][top] !== null), 'a landmark, which the picture does not place, keeps its reads');
check(reply.tracks.every((t, i) => near(t.hypotheses[0].tracks[1][2], pixel(views[i].camera, posed[2])) && t.hypotheses[0].scores[1][2] === 0.9 && t.hypotheses[1].tracks[1][2] !== null), 'and every later frame is left as read');
check(tracks[1].hypotheses[0].scores[0][2] === 0.9 && near(tracks[1].hypotheses[0].tracks[0][2], wrongFoot), 'the tracks handed in are not written over');
const fitted = fit.problem.jointProjections(1)[0][reply.joints.indexOf('foot_L')];
check(Math.hypot(fitted[0] - rightFoot[0], -fitted[1] - rightFoot[1]) < 3, `the fit's first frame has the foot where the picture has it (${fitted.map((v) => v.toFixed(1))} for ${rightFoot.map((v) => v.toFixed(1))})`);

// A first frame measured against the picture and found to be something else.
const strange = views.map((v) => ({ ...v, calibration: [{ name: 'three_quarter', overlap: [0.31, 0.42] }, { name: 'side', overlap: [0.9, 0.95] }] }));
const held = tracks.map((t) => ({ tracks: t }));
const left = startOnReference(layout, strange, held, rig);
check(left.joints === 0 && left.overlap === 0.42 && held.every((v, i) => v.tracks === tracks[i]), `a first frame that is not the picture is left as read (${JSON.stringify(left)})`);
const same = views.map((v) => ({ ...v, calibration: [{ name: 'three_quarter', overlap: [0.84, 0.93] }, { name: 'side', overlap: [0.9, 0.88] }] }));
check(startOnReference(layout, same, held, rig).overlap === 0.9, 'one that is, by its better overlap in each view, is taken');
let refused = '';
try { await animate({ ...request, start: 'rest' }); } catch (e) { refused = e.message; }
check(refused.startsWith('start:'), `a start that is not "reference" is refused (${refused})`);
refused = '';
try { await animate({ rig, markers, views: request.views.map((v, i) => ({ ...v, camera: views[i].camera })), start: 'reference', outputs: ['tracks'] }); } catch (e) { refused = e.message; }
check(refused.includes('needs studio'), `and so is one with no studio (${refused})`);

llm.print(failures ? `${failures} failed` : 'all passed');
if (failures) throw new Error(`${failures} failed`);
