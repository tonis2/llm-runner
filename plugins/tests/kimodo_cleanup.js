// The Kimodo plugin's host-side motion work, checked on synthetic motions of the
// SOMA skeleton: turns read off guide-pose positions (poses.js) and upstream's
// clean-up (cleanup.js) planting feet and meeting whole-body and hand keys.
// `llm-runner tests/kimodo_cleanup.js`.
import { llm } from '../lib/llm.js';
import { SKELETONS } from '../kimodo/skeletons.js';
import { restPositions, placePoses, globalTurns, encodePoses } from '../kimodo/poses.js';
import { cleanUpMotion } from '../kimodo/cleanup.js';
import { forward, axisAngle, mul, rotate, sub, length, IDENTITY } from '../kimodo/rotation.js';

const sk = SKELETONS.soma30;
const J = sk.parents.length;
const idx = new Map(sk.names.map((n, j) => [n, j]));
let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };
const qdist = (a, b) => 1 - Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);

// 1. Turns of the rest pose are identity; of a turned rest pose, the turn.
{
	const rest = restPositions(sk);
	const turns = globalTurns(rest, sk);
	check(turns.every((q) => qdist(q, IDENTITY) < 1e-9), 'rest turns are identity');
	const R = axisAngle([0.3, 0.9, 0.1].map((v) => v / Math.hypot(0.3, 0.9, 0.1)), 0.8);
	const turned = rest.map((p) => rotate(R, p));
	const t2 = globalTurns(turned, sk);
	check(t2.every((q) => qdist(q, R) < 1e-9), 'a turned pose reads as that turn for every joint');
}

// A pose: rest with the right arm raised and the left knee bent, standing on the floor.
const raise = axisAngle([0, 0, 1], 1.2); // right arm up (about +Z, towards +Y for -X arm)
const bend = axisAngle([1, 0, 0], -0.9);
function localOf(j) {
	if (sk.names[j] === 'RightArm') return raise;
	if (sk.names[j] === 'RightForeArm') return axisAngle([0, 1, 0], 0.7);
	if (sk.names[j] === 'LeftShin') return axisAngle([1, 0, 0], 0.9);
	if (sk.names[j] === 'LeftLeg') return bend;
	return IDENTITY;
}
const floor = Math.min(...restPositions(sk).map((p) => p[1]));
const hipsAt = [0.1, -floor, 0.3];
const key = forward(sk, localOf, hipsAt);

// 2. Turns from positions reproduce the posed joints (FK from derived locals).
{
	const turns = globalTurns(key.positions, sk);
	const locals = turns.map((q, j) => (sk.parents[j] < 0 ? q : mul([-turns[sk.parents[j]][0], -turns[sk.parents[j]][1], -turns[sk.parents[j]][2], turns[sk.parents[j]][3]], q)));
	const again = forward(sk, (j) => locals[j], hipsAt);
	const err = Math.max(...again.positions.map((p, j) => length(sub(p, key.positions[j]))));
	check(err < 1e-6, `turns from positions rebuild the pose (max ${err.toExponential(2)} m)`);
}

// A drifting "generated" motion: rest pose, hips sliding along +Z at 1 m/s, feet flagged down.
const N = 150;
function generated() {
	const rotations = new Float32Array(N * J * 4);
	const root = new Float32Array(N * 3);
	const contacts = new Float32Array(N * 4);
	for (let f = 0; f < N; f++) {
		// Knees bent: thighs forward, shins back, hips lowered to keep the feet down.
		for (let j = 0; j < J; j++) {
			const n = sk.names[j];
			const q = n.endsWith('Leg') ? axisAngle([1, 0, 0], -0.4) : n.endsWith('Shin') ? axisAngle([1, 0, 0], 0.8) : n.endsWith('Foot') ? axisAngle([1, 0, 0], -0.4) : IDENTITY;
			rotations.set(q, (f * J + j) * 4);
		}
		root[f * 3] = 0;
		root[f * 3 + 1] = -floor - 0.07 + 0.01 * Math.sin(f / 7);
		root[f * 3 + 2] = f / 30 * 0.1;
		for (let k = 0; k < 4; k++) contacts[f * 4 + k] = f < 60 ? 1 : 0;
	}
	return { rotations, root, contacts };
}
const footAt = (m, f, name) => forward(sk, (j) => Array.from(m.rotations.subarray((f * J + j) * 4, (f * J + j) * 4 + 4)), Array.from(m.root.subarray(f * 3, f * 3 + 3))).positions[idx.get(name)];

// 3. Planted feet stop sliding.
{
	const m = generated();
	const before = length(sub(footAt(m, 50, 'LeftFoot'), footAt(m, 10, 'LeftFoot')));
	const t0 = llm.now();
	const r = cleanUpMotion({ skeleton: sk, key: 'soma30', frames: N, ...m, keys: [] });
	const after = length(sub(footAt(m, 50, 'LeftFoot'), footAt(m, 10, 'LeftFoot')));
	check(after < 0.01 && before > 0.1, `planted left foot slides ${before.toFixed(3)} m before, ${after.toFixed(4)} m after (${r.plantedFrames} frames, ${(llm.now() - t0).toFixed(0)} ms)`);
	const finite = m.rotations.every(Number.isFinite) && m.root.every(Number.isFinite);
	check(finite, 'all values finite');
}

// 4. A whole-body key is met exactly.
{
	const m = generated();
	const rig = { rest: {}, keys: [] };
	// Hand the key over the way crig does: named joint positions (here every joint, in metres).
	const rest = restPositions(sk);
	sk.names.forEach((n, j) => { rig.rest[n] = [rest[j][0], rest[j][1] - floor, rest[j][2]]; });
	const joints = {};
	sk.names.forEach((n, j) => { joints[n] = key.positions[j]; });
	rig.keys.push({ frame: 90, joints });
	const placed = placePoses(rig, sk, N);
	const errPlaced = Math.max(...placed[0].positions.map((p, j) => length(sub(p, key.positions[j]))));
	check(errPlaced < 1e-6, `placing a pose of the skeleton itself keeps it (max ${errPlaced.toExponential(2)} m)`);
	cleanUpMotion({ skeleton: sk, key: 'soma30', frames: N, ...m, keys: placed });
	const hit = Math.max(...sk.names.map((n, j) => length(sub(footAt(m, 90, n), key.positions[j]))));
	check(hit < 1e-3, `whole-body key met: max joint error ${hit.toExponential(2)} m`);
	const near = length(sub(footAt(m, 89, 'RightHand'), footAt(m, 90, 'RightHand')));
	check(near < 0.1, `frame before the key eases in (right hand moves ${near.toFixed(3)} m in one frame)`);
}

// 5. A right-hand-only key: the hand reaches it, the rest is the motion's.
{
	const m = generated();
	const rig = { rest: {}, keys: [] };
	const rest = restPositions(sk);
	sk.names.forEach((n, j) => { rig.rest[n] = [rest[j][0], rest[j][1] - floor, rest[j][2]]; });
	const joints = {};
	sk.names.forEach((n, j) => { joints[n] = key.positions[j]; });
	rig.keys.push({ frame: 100, joints, effectors: ['RightHand'] });
	const placed = placePoses(rig, sk, N);
	const enc = encodePoses(placed, sk, { globalMean: new Float32Array(5), globalStd: new Float32Array(5).fill(1), bodyMean: new Float32Array(9 + 12 * J - 5), bodyStd: new Float32Array(9 + 12 * J - 5).fill(1) }, N);
	const D = 9 + 12 * J;
	const masked = enc.mask.subarray(100 * D, 101 * D).reduce((a, b) => a + b, 0);
	check(masked === 5 + 2 * 3 + 6, `a hand key masks root 5 + 2 positions + one 6D turn (${masked})`);
	cleanUpMotion({ skeleton: sk, key: 'soma30', frames: N, ...m, keys: placed });
	const hand = length(sub(footAt(m, 100, 'RightHand'), key.positions[idx.get('RightHand')]));
	check(hand < 0.005, `right hand key met: ${hand.toExponential(2)} m`);
	const hips = footAt(m, 100, 'Hips');
	const off = Math.hypot(hips[0] - key.positions[0][0], hips[2] - key.positions[0][2]);
	check(off <= 0.0401, `hips kept within 4 cm of the key's over the floor (${off.toFixed(4)} m)`);
}

if (failures) throw new Error(`${failures} check(s) failed`);
