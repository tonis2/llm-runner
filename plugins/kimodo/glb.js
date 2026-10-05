// A generated motion as a .glb: the skeleton as a node tree (rest offsets as
// translations, rest rotations identity), a skin naming its joints, and one
// animation - a rotation channel a joint and a translation channel on the root.
// No mesh; any tool that reads glTF clips reads this one.

import { llm } from '../lib/llm.js';

const FLOAT = 5126;

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

// `rotations` [frames, J, 4] parent-local xyzw, `root` [frames, 3] metres.
// `rest` is where the root stands at rest (frame 0's root when not given),
// `generator` the asset's generator and `extras` the scene-wide glTF extras.
export function writeMotionGlb(path, clip) {
	llm.writeBytes(path, motionGlb(clip));
}

// The same .glb as bytes, for a caller that keeps it in memory.
export function motionGlb({ name, skeleton, fps, frames, rotations, root, rest = null, generator = 'llm-runner kimodo', extras = null }) {
	const J = skeleton.parents.length;
	const views = [], accessors = [], chunks = [];
	let offset = 0;
	const add = (floats, type, count, minmax = false) => {
		const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
		views.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length });
		const accessor = { bufferView: views.length - 1, componentType: FLOAT, count, type };
		if (minmax) {
			accessor.min = [Math.min(...floats)];
			accessor.max = [Math.max(...floats)];
		}
		accessors.push(accessor);
		chunks.push(bytes);
		offset += bytes.length;
		return accessors.length - 1;
	};

	const times = new Float32Array(frames);
	for (let t = 0; t < frames; t++) times[t] = t / fps;
	const input = add(times, 'SCALAR', frames, true);

	const samplers = [], channels = [];
	samplers.push({ input, output: add(Float32Array.from(root), 'VEC3', frames), interpolation: 'LINEAR' });
	channels.push({ sampler: 0, target: { node: 0, path: 'translation' } });
	for (let j = 0; j < J; j++) {
		const track = new Float32Array(frames * 4);
		for (let t = 0; t < frames; t++) track.set(rotations.subarray((t * J + j) * 4, (t * J + j + 1) * 4), t * 4);
		samplers.push({ input, output: add(track, 'VEC4', frames), interpolation: 'LINEAR' });
		channels.push({ sampler: samplers.length - 1, target: { node: j, path: 'rotation' } });
	}

	const nodes = skeleton.names.map((n, j) => ({
		name: n,
		translation: j === 0 ? (rest ? [...rest] : [root[0], root[1], root[2]]) : skeleton.offsets[j].slice(),
	}));
	for (let j = 1; j < J; j++) {
		const p = nodes[skeleton.parents[j]];
		(p.children ??= []).push(j);
	}

	const gltf = {
		asset: { version: '2.0', generator },
		...(extras ? { extras } : {}),
		scene: 0,
		scenes: [{ nodes: [0] }],
		nodes,
		skins: [{ joints: skeleton.names.map((_, j) => j), skeleton: 0 }],
		animations: [{ name, samplers, channels }],
		accessors,
		bufferViews: views,
		buffers: [{ byteLength: offset }],
	};

	const json = utf8(JSON.stringify(gltf));
	while (json.length % 4) json.push(0x20);
	const total = 12 + 8 + json.length + 8 + offset;
	const out = new Uint8Array(total);
	const dv = new DataView(out.buffer);
	dv.setUint32(0, 0x46546c67, true); // glTF
	dv.setUint32(4, 2, true);
	dv.setUint32(8, total, true);
	dv.setUint32(12, json.length, true);
	dv.setUint32(16, 0x4e4f534a, true); // JSON
	out.set(json, 20);
	let at = 20 + json.length;
	dv.setUint32(at, offset, true);
	dv.setUint32(at + 4, 0x004e4942, true); // BIN
	at += 8;
	for (const c of chunks) {
		out.set(c, at);
		at += c.length;
	}
	return out;
}
