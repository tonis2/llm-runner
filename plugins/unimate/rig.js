// A rig's skeleton at rest out of a .glb, for running UniMate on a file: the
// first skin's joints, each one's parent (the nearest joint above it) and its
// model-space position. A caller that has the rig already (crig) passes
// { names, parents, positions } instead.

import { llm } from '../lib/llm.js';

// UTF-8 bytes as a string (QuickJS has no TextDecoder).
export function utf8(bytes) {
	let s = '';
	for (let i = 0; i < bytes.length;) {
		const b = bytes[i++];
		let c = b;
		if (b >= 0xf0) c = ((b & 7) << 18) | ((bytes[i++] & 63) << 12) | ((bytes[i++] & 63) << 6) | (bytes[i++] & 63);
		else if (b >= 0xe0) c = ((b & 15) << 12) | ((bytes[i++] & 63) << 6) | (bytes[i++] & 63);
		else if (b >= 0xc0) c = ((b & 31) << 6) | (bytes[i++] & 63);
		s += String.fromCodePoint(c);
	}
	return s;
}

// Column-major 4x4 of a node.
function nodeMatrix(n) {
	if (n.matrix) return n.matrix.slice();
	const [x, y, z, w] = n.rotation ?? [0, 0, 0, 1];
	const [sx, sy, sz] = n.scale ?? [1, 1, 1];
	const [tx, ty, tz] = n.translation ?? [0, 0, 0];
	return [
		(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
		2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
		2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
		tx, ty, tz, 1,
	];
}

function mul(a, b) {
	const o = new Array(16).fill(0);
	for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
	return o;
}

export function readRigGlb(path) {
	const bytes = llm.readBytes(path);
	const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	if (dv.getUint32(0, true) !== 0x46546c67) throw new Error(`${path} is not a .glb`);
	const len = dv.getUint32(12, true);
	const gltf = JSON.parse(utf8(bytes.subarray(20, 20 + len)));
	const skin = gltf.skins?.[0];
	if (!skin) throw new Error(`${path} has no skin: no skeleton to animate`);
	const parentOf = new Map();
	gltf.nodes.forEach((n, i) => (n.children ?? []).forEach((c) => parentOf.set(c, i)));
	const world = new Map();
	const worldOf = (i) => {
		if (!world.has(i)) world.set(i, parentOf.has(i) ? mul(worldOf(parentOf.get(i)), nodeMatrix(gltf.nodes[i])) : nodeMatrix(gltf.nodes[i]));
		return world.get(i);
	};
	const joints = skin.joints, index = new Map(joints.map((j, k) => [j, k]));
	const parents = joints.map((j) => {
		let p = parentOf.get(j);
		while (p !== undefined && !index.has(p)) p = parentOf.get(p);
		return p === undefined ? -1 : index.get(p);
	});
	return {
		names: joints.map((j, k) => gltf.nodes[j].name ?? `joint${k}`),
		parents,
		positions: joints.map((j) => worldOf(j).slice(12, 15)),
	};
}
