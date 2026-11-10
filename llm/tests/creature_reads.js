// /animate's `draw` and `report.reads`: a clip fitted to made-up tracks in two
// known views, one read put on the wrong leg. No GPU, no files.
//
//   llm-runner tests/creature_reads.js
import { llm } from '/core/llm.js';
import { animate } from '/plugins/creature/main.js';

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
const W = 320, H = 240, T = 4, S = 100;
// A front and a side camera, level, the floor 20 px above the picture's foot.
const cameras = [0, Math.PI / 2].map((yaw) => ({ yaw, pitch: 0, logS: Math.log(S), shift: [W / 2, -(H - 20)] }));
const seen = (c, p) => [S * (Math.cos(c.yaw) * p[0] + Math.sin(c.yaw) * p[2]) + c.shift[0], -(S * p[1] + c.shift[1])];
const WRONG = { view: 0, frame: 2, point: 4 };
const tracks = cameras.map((c, v) => ({
	points: rig.names, frames: T, size: [W, H], boxes: [],
	hypotheses: [{
		from: 0, direction: 1, frames: [0, 1, 2, 3],
		tracks: [0, 1, 2, 3].map((t) => rig.positions.map((p, j) => (v === WRONG.view && t === WRONG.frame && j === WRONG.point ? seen(c, rig.positions[2]) : seen(c, p)))),
		scores: [0, 1, 2, 3].map(() => rig.names.map(() => 0.9)),
	}],
}));
// A grey picture with a slope across it, big enough to be told from a path.
const grey = new Uint8Array(W * H * 3);
for (let i = 0; i < W * H; i++) grey.fill(90 + ((i % W) * 40) / W + ((i * 7919) % 13), i * 3, i * 3 + 3);
const picture = llm.base64Encode(llm.image.encodePng({ width: W, height: H, channels: 3, pixels: grey }));
const frames = Array.from({ length: T }, () => picture);

const request = { rig, views: tracks.map((t, v) => ({ tracks: t, frames, camera: cameras[v] })), iterations: 300, outputs: ['tracks'] };
const { reply, files } = await animate({ ...request, draw: [0, WRONG.frame] });

const sheet = files['reads.png'] ? llm.image.decode(files['reads.png']) : null;
check(sheet !== null && sheet.width === 2 * 2 * W && sheet.height === H, `the sheet is two frames of two views (${sheet?.width} x ${sheet?.height})`);
check(typeof reply.reads === 'string' && reply.reads.length > 100, 'and comes back in the reply too');
check(JSON.stringify(reply.report.drawn.frames) === `[0,${WRONG.frame}]`, `the report names the frames drawn (${JSON.stringify(reply.report.drawn.frames)})`);
const marks = reply.report.drawn.marks;
check(marks.leg_L?.endsWith('discs') && marks.leg_R?.endsWith('squares') && marks.leg_L.slice(0, 7) === marks.leg_R.slice(0, 7), `a limb pair shares a colour, in discs and squares (${JSON.stringify(marks)})`);

const pixel = (x, y) => { const o = (Math.round(y) * sheet.width + Math.round(x)) * sheet.channels; return [sheet.pixels[o], sheet.pixels[o + 1], sheet.pixels[o + 2]]; };
const colour = marks.leg_L.slice(1, 7).match(/../g).map((h) => parseInt(h, 16));
const foot = seen(cameras[0], rig.positions[2]);
check(pixel(foot[0], foot[1]).join() === colour.join(), `the left foot's read is drawn in its limb's colour (${pixel(foot[0], foot[1])} for ${colour})`);
const head = seen(cameras[1], rig.positions[5]);
check(pixel(W + head[0], head[1]).join() === '255,255,255', `the head's, in the side view, in white (${pixel(W + head[0], head[1])})`);
// On the wrong frame the right foot's read sits on the left foot, a white line back to where the fit has it.
const right = seen(cameras[0], rig.positions[4]), x0 = 2 * W;
let white = 0;
for (let x = Math.round(right[0]) + 12; x < Math.round(foot[0]) - 12; x++) {
	for (let dy = -5; dy <= 5; dy++) if (pixel(x0 + x, foot[1] + dy).join() === '255,255,255') { white++; break; }
}
check(white > 10, `the read on the wrong leg is tied to its own joint (${white} white pixels between the feet)`);

const reads = reply.report.reads;
check(reads.length === 2 && reads.every((v) => v.length === rig.names.length), 'every point of both views is in the report');
const row = reads[WRONG.view][WRONG.point], other = reads[1][WRONG.point];
check(row[0] === 'foot_R' && row[1] === 1 && row[3] === 0.25 && other[3] === 0, `one frame in four of the right foot is over 30 px off in the first view only (${JSON.stringify(row)}, ${JSON.stringify(other)})`);

const plain = await animate(request);
check(plain.files['reads.png'] === undefined && plain.reply.report.drawn === undefined, 'without draw nothing is drawn');
let refused = '';
try { await animate({ ...request, draw: 'all' }); } catch (e) { refused = e.message; }
check(refused.startsWith('draw:'), `a draw that is no count is refused (${refused})`);
check((await animate({ ...request, draw: 3 })).reply.report.drawn.frames.join() === '0,2,3', 'a count is spread over the clip');

llm.print(failures ? `${failures} failed` : 'all passed');
if (failures) throw new Error(`${failures} failed`);
