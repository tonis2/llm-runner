// follow.js: a finder's limb reads held against joints followed from frame 0,
// on a body with two legs a side of one shape. No GPU, no files.
//
//   llm-runner tests/creature_follow.js
import { llm } from '/core/llm.js';
import { followLimbs, limbsOf } from '/plugins/creature/follow.js';

let failures = 0;
const check = (ok, what) => {
	llm.print(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
	if (!ok) failures++;
};

// A body, a front and a mid leg a side (a hip and a foot each), and two short horns.
const rig = {
	names: ['body', 'front_1_l', 'front_2_l', 'mid_1_l', 'mid_2_l', 'front_1_r', 'front_2_r', 'mid_1_r', 'mid_2_r', 'horn_l', 'horn_r'],
	parents: [-1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0],
	positions: [[0, 1, 0], [0.3, 1, 0], [0.3, 0, 0], [0.6, 1, 0], [0.6, 0, 0], [-0.3, 1, 0], [-0.3, 0, 0], [-0.6, 1, 0], [-0.6, 0, 0], [0.1, 1.2, 0], [-0.1, 1.2, 0]],
	height: 1,
};
const limbs = limbsOf(rig);
check(limbs.limb[0] === -1 && limbs.limb[2] === limbs.limb[1] && limbs.depth[2] === 1 && limbs.depth[1] === 0, 'a limb is a joint with a mirror partner and all below it; the body is the middle');
check(limbs.alike.length === 2 && limbs.alike[0].length === 4 && limbs.alike[1].length === 2, `the four legs are alike, the horns are their own kind (${JSON.stringify(limbs.alike)})`);

// One view at 100 pixels a unit: the body at (200, 100), the floor at row 200.
const camera = { logS: Math.log(100) }, T = 4, frames = [0, 1, 2, 3];
const rest = { body: [200, 100], front_1_l: [230, 100], front_2_l: [230, 200], mid_1_l: [260, 100], mid_2_l: [260, 200], front_1_r: [170, 100], front_2_r: [170, 200], mid_1_r: [140, 100], mid_2_r: [140, 200], horn_l: [210, 80], horn_r: [190, 80] };
const points = [...rig.names, 'front_2_l.heel'], markers = { 'front_2_l.heel': { joint: 'front_2_l', offset: [0, 0.05, 0] } };
// The left front foot rises; everything else holds still.
const raised = [[230, 200], [250, 150], [290, 60], [300, 20]];
const truth = (t, n) => (n === 'front_2_l' ? raised[t] : n === 'front_2_l.heel' ? [raised[t][0], raised[t][1] - 5] : rest[n]);

// The finder: right, but for four things.
const first = frames.map((t) => points.map((n) => ({ at: truth(t, n), score: 0.7 })));
const second = frames.map(() => points.map(() => null));
// - From frame 2 it names the raised foot's leg on the mid leg's foot, the true place a weak second peak.
for (const t of [2, 3]) {
	first[t][points.indexOf('front_2_l')] = { at: [262, 199], score: 0.6 };
	second[t][points.indexOf('front_2_l')] = { at: [raised[t][0] + 4, raised[t][1] + 3], score: 0.15 };
}
// - On frame 2 it names the right mid foot on the right front one, with no
//   second peak, and on frame 3 it has none for it.
first[2][points.indexOf('mid_2_r')] = { at: [172, 201], score: 0.6 };
first[3][points.indexOf('mid_2_r')] = null;
// - On frame 3 it has the right front hip twice a bone's length off the body.
first[3][points.indexOf('front_1_r')] = { at: [170, 160], score: 0.7 };
second[0][0] = { at: [205, 104], score: 0.2 };
const tracksOf = (rows) => ({ tracks: rows.map((r) => r.map((q) => (q ? q.at : null))), scores: rows.map((r) => r.map((q) => (q ? q.score : null))) });
const found = { points, frames: T, boxes: [], hypotheses: [{ from: 0, direction: 1, frames: [0, 1], ...tracksOf(first.slice(0, 2)) }, { from: 2, direction: 1, frames: [2, 3], ...tracksOf(first.slice(2)) }, { from: 0, direction: 1, frames, ...tracksOf(second) }] };

// The tracker: each joint from its place on frame 0, a few pixels off, but:
const trail = frames.map((t) => rig.names.map((n) => ({ at: [truth(t, n)[0] - 6, truth(t, n)[1] + 5], score: 0.8 })));
const J = (n) => rig.names.indexOf(n);
rig.names.forEach((n) => { trail[0][J(n)] = { at: truth(0, n), score: 1 }; });
// - it loses the left mid foot for a frame, to nowhere near a leg,
trail[1][J('mid_2_l')] = { at: [330, 230], score: 0.7 };
// - on frame 3 it has the right mid foot on top of the right front one,
trail[3][J('mid_2_r')] = { at: [165, 204], score: 0.8 };
// - and it goes with the finder to the hip that is out of reach.
trail[3][J('front_1_r')] = { at: [172, 158], score: 0.8 };
const followed = { points: rig.names, frames: T, hypotheses: [{ from: 0, direction: 1, frames, ...tracksOf(trail) }] };

const report = {};
const out = followLimbs(rig, found, followed, camera, { markers, report });
const at = (t, n) => { const h = out.hypotheses.find((x) => x.frames.includes(t)), i = h.frames.indexOf(t), p = points.indexOf(n); return h.tracks[i][p] && { at: h.tracks[i][p], score: h.scores[i][p] }; };
const is = (q, want, score) => q !== null && Math.hypot(q.at[0] - want[0], q.at[1] - want[1]) < 0.01 && Math.abs(q.score - score) < 1e-9;

check(out.hypotheses.length === 3 && out.hypotheses[0].from === 0 && out.hypotheses[1].from === 2 && out.hypotheses[1].frames.join() === '2,3', 'the tracks keep the finder\'s stretches');
check(is(at(1, 'front_2_l'), raised[1], 0.7), 'a read the tracker agrees with is the finder\'s, as it scored it');
check(is(at(2, 'front_2_l'), [raised[2][0] + 4, raised[2][1] + 3], 0.45) && is(at(3, 'front_2_l'), [raised[3][0] + 4, raised[3][1] + 3], 0.45), `the raised foot is the finder's weak peak where it was followed to, not its sure one on the leg beside it (${JSON.stringify(at(2, 'front_2_l'))})`);
check(is(at(2, 'mid_2_l'), rest.mid_2_l, 0.7), 'and that leg keeps its own foot');
check(is(at(1, 'mid_2_l'), rest.mid_2_l, 0.7), 'where the tracker is lost and no other joint was followed there, the finder\'s first peak stands');
check(is(at(2, 'mid_2_r'), [rest.mid_2_r[0] - 6, rest.mid_2_r[1] + 5], 0.35), `a joint the finder put on the leg beside it is where it was followed to, weighing little (${JSON.stringify(at(2, 'mid_2_r'))})`);
check(at(3, 'mid_2_r') === null, 'and has no read once the tracker has it on top of that leg\'s own');
check(at(3, 'front_1_r') === null, 'a read out of the bone\'s reach of the joint above has none either, though finder and tracker agree');
check(is(at(3, 'front_2_r'), rest.front_2_r, 0.7), 'the joint below it is still read');
check(is(at(0, 'body'), rest.body, 0.7) && out.hypotheses[2].tracks[0][0].join() === '205,104', 'the middle of the body is left as found, its second peak too');
check(out.hypotheses[2].tracks.every((row) => row.slice(1, rig.names.length).every((q) => q === null)), 'a limb joint has one read, no second peak');
check(at(1, 'front_2_l.heel') !== null && at(2, 'front_2_l.heel') === null, 'a landmark goes with its joint: dropped where the finder\'s first peak for the joint was not it');
check(report.reads === 40 && report.taken === 3 && report.followed === 1 && report.none === 2 && report.unfollowed === 1 && report.agreed === 36, `the report counts them (${JSON.stringify(report)})`);
check(found.hypotheses[1].tracks[0][points.indexOf('front_2_l')].join() === '262,199', 'the tracks handed in are not written over');

// From the side a left and a right leg are in line: followed on top of each
// other from the start, which is not being lost.
const side = { ...rest, front_1_r: rest.front_1_l, front_2_r: rest.front_2_l, mid_1_r: rest.mid_1_l, mid_2_r: rest.mid_2_l };
const sFirst = frames.map(() => points.map((n) => ({ at: side[n] ?? side.front_2_l, score: 0.7 })));
sFirst[1][points.indexOf('front_2_r')] = null;
const sTrail = frames.map(() => rig.names.map((n) => ({ at: side[n], score: 0.9 })));
const lined = followLimbs(rig, { points, frames: T, boxes: [], hypotheses: [{ from: 0, direction: 1, frames, ...tracksOf(sFirst) }] }, { points: rig.names, frames: T, hypotheses: [{ from: 0, direction: 1, frames, ...tracksOf(sTrail) }] }, camera, { markers });
const lAt = (t, n) => { const p = points.indexOf(n); return lined.hypotheses[0].tracks[t][p] && { at: lined.hypotheses[0].tracks[t][p], score: lined.hypotheses[0].scores[t][p] }; };
check(is(lAt(2, 'front_2_r'), side.front_2_l, 0.7) && is(lAt(2, 'front_2_l'), side.front_2_l, 0.7), 'two legs in line both keep the one place the finder reads them at');
check(is(lAt(1, 'front_2_r'), side.front_2_l, 0.35), 'and one the finder missed is where it was followed to, on top of the other as it began');

llm.print(failures ? `${failures} failed` : 'all passed');
if (failures) throw new Error(`${failures} failed`);
