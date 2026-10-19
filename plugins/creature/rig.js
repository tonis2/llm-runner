// The rig a clip is fitted to: joint names, parents (each before its
// children, the root first) and rest positions in model space, from
//
//   - a rigged .glb (a path, or its bytes in base64): its first skin's joints;
//   - Rigsmith's get_skeleton answer ({ joints: [{ name, parent, position }] });
//   - { names, parents, positions } (UniMate's form).
//
// A .glb's own nodes are kept too (`glb`), for writing the clip back onto it:
// each joint's node, rest local and world turns. Rigsmith's rigs have every
// rest turn the identity; any other file's are kept as they are.

import { llm } from '../lib/llm.js';
import { bytesOf, bad } from './request.js';

// UTF-8 bytes as a string (QuickJS has no TextDecoder).
function utf8(bytes) {
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

// Quaternions [x, y, z, w]; `qmul(a, b)` turns by b, then a.
export function qmul(a, b) {
	return [
		a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
		a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
		a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
		a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
	];
}
export const qconj = (q) => [-q[0], -q[1], -q[2], q[3]];
export function qrotate(q, v) {
	const [x, y, z, w] = q;
	const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
	return [v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx];
}

// A .glb's JSON and binary chunk.
export function parseGlb(bytes) {
	const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	if (bytes.length < 20 || dv.getUint32(0, true) !== 0x46546c67) throw bad('rig: not a .glb');
	const len = dv.getUint32(12, true);
	const gltf = JSON.parse(utf8(bytes.subarray(20, 20 + len)));
	let bin = null;
	const at = 20 + len;
	if (at + 8 <= bytes.length && dv.getUint32(at + 4, true) === 0x004e4942) bin = bytes.subarray(at + 8, at + 8 + dv.getUint32(at, true));
	return { gltf, bin };
}

function fromGlb(bytes) {
	const { gltf, bin } = parseGlb(bytes);
	const skin = gltf.skins?.[0];
	if (!skin) throw bad('rig: the .glb has no skin, so no skeleton to animate - rig it first (Rigsmith: apply_plan or edit_rig, then save)');
	const nodes = gltf.nodes;
	const parentOf = new Map();
	nodes.forEach((n, i) => (n.children ?? []).forEach((c) => parentOf.set(c, i)));
	const trs = (n) => {
		if (n.matrix) throw bad(`rig: node ${n.name ?? ''} has a matrix; only translation/rotation/scale nodes are read`);
		return { t: n.translation ?? [0, 0, 0], r: n.rotation ?? [0, 0, 0, 1], s: n.scale ?? [1, 1, 1] };
	};
	// World translation and turn of every node (scale only through positions).
	const world = new Map();
	const worldOf = (i) => {
		if (world.has(i)) return world.get(i);
		const { t, r, s } = trs(nodes[i]);
		let w;
		if (parentOf.has(i)) {
			const p = worldOf(parentOf.get(i));
			const st = [t[0] * p.s[0], t[1] * p.s[1], t[2] * p.s[2]];
			const o = qrotate(p.r, st);
			w = { p: [p.p[0] + o[0], p.p[1] + o[1], p.p[2] + o[2]], r: qmul(p.r, r), s: [p.s[0] * s[0], p.s[1] * s[1], p.s[2] * s[2]] };
		} else {
			w = { p: t.slice(), r: r.slice(), s: s.slice() };
		}
		world.set(i, w);
		return w;
	};
	const joints = skin.joints;
	const index = new Map(joints.map((n, k) => [n, k]));
	const parents = joints.map((n) => {
		let p = parentOf.get(n);
		while (p !== undefined && !index.has(p)) p = parentOf.get(p);
		return p === undefined ? -1 : index.get(p);
	});
	const rig = {
		names: joints.map((n, k) => nodes[n].name ?? `joint${k}`),
		parents,
		positions: joints.map((n) => worldOf(n).p),
	};
	rig.glb = {
		bytes, gltf, bin,
		nodes: joints.slice(),
		restLocal: joints.map((n) => trs(nodes[n]).r.slice()),
		restWorld: joints.map((n) => worldOf(n).r),
		// Where the node the skeleton's root hangs from is ({ p, r, s }).
		above: parentOf.has(joints[parents.indexOf(-1)]) ? worldOf(parentOf.get(joints[parents.indexOf(-1)])) : { p: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1] },
	};
	return rig;
}

// Parents before children, the one root first; `extra` per-joint arrays go along.
function ordered(rig) {
	const J = rig.names.length;
	const roots = rig.parents.map((p, j) => (p < 0 ? j : -1)).filter((j) => j >= 0);
	if (roots.length !== 1) throw bad(`rig: ${roots.length} root joints (${roots.map((j) => rig.names[j]).join(', ')}); a rig hangs from one`);
	const order = [roots[0]];
	for (let i = 0; i < order.length; i++) for (let j = 0; j < J; j++) if (rig.parents[j] === order[i]) order.push(j);
	if (order.length !== J) throw bad('rig: some joints do not hang from the root');
	const at = new Map(order.map((j, k) => [j, k]));
	const out = {
		names: order.map((j) => rig.names[j]),
		parents: order.map((j) => (rig.parents[j] < 0 ? -1 : at.get(rig.parents[j]))),
		positions: order.map((j) => rig.positions[j].slice(0, 3).map(Number)),
	};
	if (new Set(out.names).size !== J) throw bad('rig: two joints share a name');
	if (rig.glb) out.glb = { ...rig.glb, nodes: order.map((j) => rig.glb.nodes[j]), restLocal: order.map((j) => rig.glb.restLocal[j]), restWorld: order.map((j) => rig.glb.restWorld[j]) };
	const ys = out.positions.map((p) => p[1]);
	out.height = Math.max(...ys) - Math.min(...ys) || 1;
	return out;
}

export function readRig(value) {
	if (!value) throw bad('rig: a rigged .glb (path or base64), Rigsmith\'s get_skeleton JSON, or { names, parents, positions }');
	let rig;
	if (typeof value === 'string') {
		const text = value.trim();
		if (text.startsWith('{')) return readRig(JSON.parse(text));
		if (/\.json$/i.test(text)) return readRig(JSON.parse(llm.readText(text)));
		rig = fromGlb(bytesOf(text));
	} else if (Array.isArray(value.joints)) {
		rig = { names: value.joints.map((j) => j.name), parents: value.joints.map((j) => j.parent ?? -1), positions: value.joints.map((j) => j.position) };
	} else if (Array.isArray(value.names)) {
		rig = { names: value.names, parents: value.parents, positions: value.positions };
	} else {
		throw bad('rig: expected a .glb, { joints: [...] } or { names, parents, positions }');
	}
	const out = ordered(rig);
	llm.print(`  rig: ${out.names.length} joints, root ${out.names[0]}, height ${out.height.toFixed(3)}`);
	return out;
}
