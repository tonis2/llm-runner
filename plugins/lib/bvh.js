// A motion as BVH text: the HIERARCHY from the skeleton's rest offsets (rest
// rotations identity), the MOTION as the root's position and every joint's
// turn as Z, X, Y Euler angles in degrees, the order most importers expect.
//
// `skeleton` { names, parents, offsets } with parents before children,
// `rotations` [frames, J, 4] parent-local (x, y, z, w), `root` [frames, 3].
// The root's position channels are where it stands, not an offset from
// `rest` (its OFFSET, where it stands in the rest pose). `scale` multiplies
// lengths: 100 writes centimetres from metres.

const DEG = 180 / Math.PI;

// R = Rz(z) Rx(x) Ry(y) from a unit quaternion, in degrees: [z, x, y].
function zxyFromQuat(x, y, z, w) {
	const m01 = 2 * (x * y - z * w), m11 = 1 - 2 * (x * x + z * z);
	const m20 = 2 * (x * z - y * w), m21 = 2 * (y * z + x * w), m22 = 1 - 2 * (x * x + y * y);
	const rx = Math.asin(Math.max(-1, Math.min(1, m21)));
	return [Math.atan2(-m01, m11) * DEG, rx * DEG, Math.atan2(-m20, m22) * DEG];
}

// The angle nearest `previous` among `angle` + k 360.
const unwrap = (angle, previous) => angle + 360 * Math.round((previous - angle) / 360);

export function motionBvh({ skeleton, fps, frames, rotations, root, rest = null, scale = 1 }) {
	const { names, parents, offsets } = skeleton;
	const J = parents.length;
	const children = Array.from({ length: J }, () => []);
	for (let j = 1; j < J; j++) children[parents[j]].push(j);
	const num = (v) => (Math.abs(v) < 5e-7 ? '0' : v.toFixed(6).replace(/\.?0+$/, ''));
	const lines = ['HIERARCHY'];
	const order = [];
	const write = (j, depth) => {
		const pad = '\t'.repeat(depth);
		const o = j === 0 ? (rest ?? [0, 0, 0]) : offsets[j];
		lines.push(`${pad}${j === 0 ? 'ROOT' : 'JOINT'} ${names[j]}`, `${pad}{`);
		lines.push(`${pad}\tOFFSET ${o.map((v) => num(v * scale)).join(' ')}`);
		lines.push(`${pad}\tCHANNELS ${j === 0 ? '6 Xposition Yposition Zposition ' : '3 '}Zrotation Xrotation Yrotation`);
		order.push(j);
		if (children[j].length === 0) {
			lines.push(`${pad}\tEnd Site`, `${pad}\t{`, `${pad}\t\tOFFSET 0 0 0`, `${pad}\t}`);
		}
		for (const c of children[j]) write(c, depth + 1);
		lines.push(`${pad}}`);
	};
	write(0, 0);
	lines.push('MOTION', `Frames: ${frames}`, `Frame Time: ${num(1 / fps)}`);
	const previous = new Float64Array(J * 3);
	for (let f = 0; f < frames; f++) {
		const row = [];
		for (let d = 0; d < 3; d++) row.push(num(root[f * 3 + d] * scale));
		for (const j of order) {
			const q = (f * J + j) * 4;
			const e = zxyFromQuat(rotations[q], rotations[q + 1], rotations[q + 2], rotations[q + 3]);
			for (let k = 0; k < 3; k++) {
				const a = f === 0 ? e[k] : unwrap(e[k], previous[j * 3 + k]);
				previous[j * 3 + k] = a;
				row.push(num(a));
			}
		}
		lines.push(row.join(' '));
	}
	return lines.join('\n') + '\n';
}
