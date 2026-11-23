// A triangle mesh as a .glb: one node, one mesh, one primitive with
// POSITION (and NORMAL when given) and 32-bit indices, and a plain material.
// What an image-to-3D model hands back; any glTF reader opens it.

import { llm } from '/core/llm.js';

const FLOAT = 5126, UINT = 5125, ARRAY_BUFFER = 34962, ELEMENT_ARRAY_BUFFER = 34963;

function utf8(text) {
	const bytes = [];
	for (const ch of text) {
		const c = ch.codePointAt(0);
		if (c < 0x80) bytes.push(c);
		else if (c < 0x800) bytes.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
		else if (c < 0x10000) bytes.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
		else bytes.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 0x3f), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
	}
	return bytes;
}

// Smooth vertex normals: each face's (area-weighted) normal added to its corners.
export function vertexNormals(positions, indices) {
	const n = new Float32Array(positions.length);
	for (let f = 0; f < indices.length; f += 3) {
		const [a, b, c] = [indices[f], indices[f + 1], indices[f + 2]];
		const ux = positions[3 * b] - positions[3 * a], uy = positions[3 * b + 1] - positions[3 * a + 1], uz = positions[3 * b + 2] - positions[3 * a + 2];
		const vx = positions[3 * c] - positions[3 * a], vy = positions[3 * c + 1] - positions[3 * a + 1], vz = positions[3 * c + 2] - positions[3 * a + 2];
		const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
		for (const i of [a, b, c]) { n[3 * i] += nx; n[3 * i + 1] += ny; n[3 * i + 2] += nz; }
	}
	for (let i = 0; i < n.length; i += 3) {
		const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
		n[i] /= l; n[i + 1] /= l; n[i + 2] /= l;
	}
	return n;
}

// `positions` Float32Array [V * 3] (metres, y up), `indices` [F * 3];
// `normals` optional [V * 3]; `doubleSided` for faces of no set winding.
export function meshGlb({ name = 'mesh', positions, indices, normals = null, doubleSided = true, color = [0.8, 0.8, 0.8, 1], generator = 'llm-runner', extras = null }) {
	const V = positions.length / 3;
	const idx = indices instanceof Uint32Array ? indices : Uint32Array.from(indices);
	const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
	for (let i = 0; i < V; i++) for (let d = 0; d < 3; d++) {
		min[d] = Math.min(min[d], positions[3 * i + d]);
		max[d] = Math.max(max[d], positions[3 * i + d]);
	}
	const parts = [new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength)];
	const views = [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: ARRAY_BUFFER }];
	let at = positions.byteLength;
	const attributes = { POSITION: 0 };
	const accessors = [{ bufferView: 0, componentType: FLOAT, count: V, type: 'VEC3', min, max }];
	if (normals) {
		parts.push(new Uint8Array(normals.buffer, normals.byteOffset, normals.byteLength));
		views.push({ buffer: 0, byteOffset: at, byteLength: normals.byteLength, target: ARRAY_BUFFER });
		attributes.NORMAL = accessors.length;
		accessors.push({ bufferView: views.length - 1, componentType: FLOAT, count: V, type: 'VEC3' });
		at += normals.byteLength;
	}
	parts.push(new Uint8Array(idx.buffer, idx.byteOffset, idx.byteLength));
	views.push({ buffer: 0, byteOffset: at, byteLength: idx.byteLength, target: ELEMENT_ARRAY_BUFFER });
	accessors.push({ bufferView: views.length - 1, componentType: UINT, count: idx.length, type: 'SCALAR' });
	at += idx.byteLength;
	const gltf = {
		asset: { version: '2.0', generator },
		scene: 0,
		scenes: [{ nodes: [0] }],
		nodes: [{ name, mesh: 0 }],
		meshes: [{ name, primitives: [{ attributes, indices: accessors.length - 1, material: 0, mode: 4 }] }],
		materials: [{ name: `${name} material`, doubleSided, pbrMetallicRoughness: { baseColorFactor: color, metallicFactor: 0, roughnessFactor: 0.8 } }],
		accessors,
		bufferViews: views,
		buffers: [{ byteLength: at }],
	};
	if (extras) gltf.extras = extras;
	const json = utf8(JSON.stringify(gltf));
	while (json.length % 4) json.push(0x20);
	const binLength = (at + 3) & ~3;
	const total = 12 + 8 + json.length + 8 + binLength;
	const out = new Uint8Array(total), dv = new DataView(out.buffer);
	dv.setUint32(0, 0x46546c67, true); // glTF
	dv.setUint32(4, 2, true);
	dv.setUint32(8, total, true);
	dv.setUint32(12, json.length, true);
	dv.setUint32(16, 0x4e4f534a, true); // JSON
	out.set(json, 20);
	let o = 20 + json.length;
	dv.setUint32(o, binLength, true);
	dv.setUint32(o + 4, 0x004e4942, true); // BIN
	o += 8;
	for (const p of parts) { out.set(p, o); o += p.length; }
	return out;
}

export function writeMeshGlb(path, mesh) {
	llm.writeBytes(path, meshGlb(mesh));
}
