// The walk's travel (output.js fittedMotion): a toy two-legged walk on the
// spot, as the fit leaves one, gets its travel back at an even speed, with no
// jolt where one foot takes over from the other. No model, no GPU.
//
//   llm-runner tests/creature_travel.js
//
// Each foot slides back along the floor through 60% of the stride and swings
// forward, lifted, through the rest, the two half a stride apart: so for a
// tenth of the stride both are down. Through the first half of that the foot
// coming down has not settled yet and slides at half speed, a millimetre lower
// than the one leaving: what a fit's feet do, and what a travel that follows
// the lower foot alone turns into a hitch.
import { llm } from '../lib/llm.js';
import { fittedMotion } from '../creature/output.js';

let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

const names = ['hips', 'ankle_L', 'toe_L', 'ankle_R', 'toe_R'];
const parents = [-1, 0, 1, 0, 3];
const positions = [[0, 0.5, 0], [0.1, 0.05, 0], [0.1, 0, 0.1], [-0.1, 0.05, 0], [-0.1, 0, 0.1]];
const rig = { names, parents, positions, height: 1 };

const T = 120, J = names.length, period = 32, stance = 0.6, reach = 0.15, settle = 0.05 / stance;
const speed = (2 * reach) / (stance * period);
const pos = new Float64Array(T * J * 3);
const G = new Float64Array(T * J * 9);
for (let t = 0; t < T; t++) {
	for (let j = 0; j < J; j++) [0, 4, 8].forEach((k) => (G[(t * J + j) * 9 + k] = 1));
	pos.set([0, 0.5, 0], t * J * 3);
	for (const [side, ankle, offset] of [[0.1, 1, 0], [-0.1, 3, 0.5]]) {
		const phase = (t / period + offset) % 1;
		let z, y;
		if (phase < stance) {
			// Down: sliding back, settling at half speed and a millimetre low at first.
			const s = phase / stance;
			if (s < settle) z = reach - 2 * reach * 0.5 * s;
			else z = reach - 2 * reach * (settle / 2 + (s - settle) * ((1 - settle / 2) / (1 - settle)));
			y = s < settle ? -0.001 : 0;
		} else {
			const s = (phase - stance) / (1 - stance);
			z = -reach + 2 * reach * s;
			y = 0.04 * Math.sin(Math.PI * s);
		}
		pos.set([side, y + 0.05, z], (t * J + ankle) * 3);
		pos.set([side, y, z + 0.1], (t * J + ankle + 1) * 3);
	}
}
const problem = {
	T, J, G, pos, parents: Int32Array.from(parents), theta: new Float64Array(T * J * 3), root: new Float64Array(T * 3),
	feet: Int32Array.from([1, 2, 3, 4]), footLeaves: [2, 4], forward() {},
};

// The root's speed frame to frame, away from the clip's ends.
function speeds(motion) {
	const out = [];
	for (let t = 9; t < T - 9; t++) out.push(motion.root[t * 3 + 2] - motion.root[(t - 1) * 3 + 2]);
	return out;
}
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];

const motion = fittedMotion({ problem }, rig, { fps: 24 });
const v = speeds(motion), mid = median(v);
const worst = Math.max(...v.map((x) => Math.abs(x - mid) / mid));
check(Math.abs(mid - speed) / speed < 0.08, `the walk travels at its feet's speed (${(mid * 1000).toFixed(2)} vs ${(speed * 1000).toFixed(2)} mm a frame)`);
check(worst < 0.1, `its speed holds within 10% from frame to frame (worst ${(worst * 100).toFixed(1)}%)`);
const sideways = Math.max(...Array.from({ length: T }, (_, t) => Math.abs(motion.root[t * 3])));
check(sideways < 1e-9, `a walk straight ahead travels straight ahead (${sideways.toExponential(1)} sideways)`);

const raw = fittedMotion({ problem }, rig, { fps: 24, smooth: 1 });
const rawWorst = Math.max(...speeds(raw).map((x) => Math.abs(x - mid) / mid));
llm.print(`  unsmoothed, the worst frame is ${(rawWorst * 100).toFixed(1)}% off`);
check(rawWorst < 0.35, `planted by degrees, the handover is no worse than a third without smoothing (${(rawWorst * 100).toFixed(1)}%)`);

llm.print(failures ? `${failures} failed` : 'all passed');
