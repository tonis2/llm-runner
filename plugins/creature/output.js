// A fitted rig (fit.js) as a clip: the motion in Rigsmith's form (rest turns
// identity, each joint's turn against its parent, the root's place in model
// space), and that written out as
//
//   glb           the skeleton and its animation, no mesh, extras.facing
//                 'as-is' (Rigsmith keeps the rig's own names and facing)
//   bvh           BVH, rig units times `scale` (100: metres to centimetres)
//   clip          apply_clip's arguments, for Rigsmith over MCP
//   animated glb  the rig's own .glb with the animation added to its nodes
//
// The fit holds the root in place (the camera followed the animal); `travel`
// puts the walk back: the root moves by what a planted foot slides back.

import { motionGlb } from '../kimodo/glb.js';
import { motionBvh } from '../lib/bvh.js';
import { utf8 } from '../lib/graph/plugins.js';
import { qmul, qconj, qrotate } from './rig.js';

const FLOAT = 5126;
const round = (v, d = 5) => Number(v.toFixed(d));

// [x, y, z, w] of the rotation vector θ.
function quat(x, y, z) {
	const t = Math.hypot(x, y, z);
	if (t < 1e-12) return [0, 0, 0, 1];
	const s = Math.sin(t / 2) / t;
	return [x * s, y * s, z * s, Math.cos(t / 2)];
}

/**
 * The fit's motion: { skeleton { names, parents, offsets, restRoot }, fps,
 * frames, rotations [frames, J, 4], root [frames, 3], travel (model units) }.
 */
export function fittedMotion(fit, rig, { fps = 24, travel = true } = {}) {
	const p = fit.problem, T = p.T, J = p.J;
	p.forward();
	const rotations = new Float32Array(T * J * 4), root = new Float32Array(T * 3);
	for (let t = 0; t < T; t++) {
		for (let j = 0; j < J; j++) {
			const o = (t * J + j) * 3;
			rotations.set(quat(p.theta[o], p.theta[o + 1], p.theta[o + 2]), (t * J + j) * 4);
		}
		root.set(p.root.subarray(t * 3, t * 3 + 3), t * 3);
	}
	// The floor: the clip's lowest foot point on y = 0.
	let floor = Infinity;
	for (let t = 0; t < T; t++) for (const f of p.feet) floor = Math.min(floor, p.pos[(t * J + f) * 3 + 1]);
	if (Number.isFinite(floor)) for (let t = 0; t < T; t++) root[t * 3 + 1] -= floor;

	let moved = 0;
	if (travel && p.footLeaves.length && T > 1) {
		// Each foot as its leaf and the joint above; planted when low and
		// sliding back against where the body faces; of two, the lower.
		const feet = p.footLeaves.map((leaf) => [leaf, p.parents[leaf]]);
		const at = (t, j, d) => p.pos[(t * J + j) * 3 + d];
		const high = 0.1 * rig.height;
		let step = [0, 0], sum = [0, 0];
		for (let t = 1; t < T; t++) {
			const g = (t * J) * 9;
			const fx = p.G[g + 2], fz = p.G[g + 8], fn = Math.hypot(fx, fz) || 1;
			let best = null;
			for (const [leaf, up] of feet) {
				const dx = (at(t, leaf, 0) + at(t, up, 0) - at(t - 1, leaf, 0) - at(t - 1, up, 0)) / 2;
				const dz = (at(t, leaf, 2) + at(t, up, 2) - at(t - 1, leaf, 2) - at(t - 1, up, 2)) / 2;
				const h = Math.min(at(t, leaf, 1), at(t, up, 1)) - floor;
				if ((dx * fx + dz * fz) / fn < 0 && h < high && (!best || h < best.h)) best = { dx, dz, h };
			}
			if (best) step = [-best.dx, -best.dz];
			sum = [sum[0] + step[0], sum[1] + step[1]];
			root[t * 3] += sum[0];
			root[t * 3 + 2] += sum[1];
		}
		moved = Math.hypot(sum[0], sum[1]);
	}
	const offsets = rig.positions.map((q, j) => (rig.parents[j] < 0 ? [0, 0, 0] : q.map((v, d) => v - rig.positions[rig.parents[j]][d])));
	return {
		skeleton: { names: rig.names, parents: rig.parents, offsets, restRoot: rig.positions[0].slice() },
		fps, frames: T, rotations, root, travel: moved,
	};
}

export function clipGlb(motion, name) {
	return motionGlb({
		name, skeleton: motion.skeleton, fps: motion.fps, frames: motion.frames, rotations: motion.rotations, root: motion.root,
		rest: motion.skeleton.restRoot, generator: 'llm-runner creature', extras: { facing: 'as-is', source: 'creature' },
	});
}

export function clipBvh(motion, scale = 100) {
	return motionBvh({
		skeleton: motion.skeleton, fps: motion.fps, frames: motion.frames, rotations: motion.rotations, root: motion.root,
		rest: motion.skeleton.restRoot, scale,
	});
}

// apply_clip's arguments: every frame a keyframe, each joint's turn against
// its parent, the root's slide from rest as a fraction of rig height.
export function clipJson(motion, name, height) {
	const { names } = motion.skeleton, J = names.length, rest = motion.skeleton.restRoot;
	const keyframes = [];
	for (let t = 0; t < motion.frames; t++) {
		const joints = names.map((n, j) => {
			const q = motion.rotations.subarray((t * J + j) * 4, (t * J + j) * 4 + 4);
			return { name: n, rotation: { x: round(q[0]), y: round(q[1]), z: round(q[2]), w: round(q[3]) } };
		});
		joints[0].move = [0, 1, 2].map((d) => round((motion.root[t * 3 + d] - rest[d]) / height));
		keyframes.push({ time: round(t / motion.fps, 6), joints });
	}
	return { name, duration: round((motion.frames - 1) / motion.fps, 6), loop: false, blend: 'linear', keyframes };
}

// The rig's .glb with the motion as one more animation on its own nodes: a
// joint's node turns by W_parentᵀ R W (its rest world turn W, its parent's,
// the fitted turn R against rest), so a rig with any rest turns plays it, and
// the skeleton's root node also moves.
export function animatedGlb(motion, rig, name) {
	const g = rig.glb;
	if (!g) throw new Error('animated_glb: the rig was not given as a .glb');
	const gltf = JSON.parse(JSON.stringify(g.gltf));
	const J = rig.names.length, T = motion.frames;
	const chunks = [];
	let offset = g.bin ? g.bin.length : 0;
	offset += (4 - (offset % 4)) % 4;
	const start = offset;
	gltf.bufferViews ??= [];
	gltf.accessors ??= [];
	const add = (floats, type, minmax = false) => {
		const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
		gltf.bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length });
		const accessor = { bufferView: gltf.bufferViews.length - 1, componentType: FLOAT, count: T, type };
		if (minmax) {
			accessor.min = [floats[0]];
			accessor.max = [floats[floats.length - 1]];
		}
		gltf.accessors.push(accessor);
		chunks.push(bytes);
		offset += bytes.length;
		return gltf.accessors.length - 1;
	};
	const times = new Float32Array(T);
	for (let t = 0; t < T; t++) times[t] = t / motion.fps;
	const input = add(times, 'SCALAR', true);
	const samplers = [], channels = [];
	const above = g.above;
	for (let j = 0; j < J; j++) {
		const W = g.restWorld[j], Wp = rig.parents[j] < 0 ? above.r : g.restWorld[rig.parents[j]];
		const track = new Float32Array(T * 4);
		for (let t = 0; t < T; t++) {
			const R = Array.from(motion.rotations.subarray((t * J + j) * 4, (t * J + j) * 4 + 4));
			let q = qmul(qconj(Wp), qmul(R, W));
			const n = Math.hypot(...q);
			q = q.map((v) => v / n);
			track.set(q, t * 4);
		}
		samplers.push({ input, output: add(track, 'VEC4'), interpolation: 'LINEAR' });
		channels.push({ sampler: samplers.length - 1, target: { node: g.nodes[j], path: 'rotation' } });
	}
	const moves = new Float32Array(T * 3);
	for (let t = 0; t < T; t++) {
		const w = [0, 1, 2].map((d) => motion.root[t * 3 + d] - above.p[d]);
		const l = qrotate(qconj(above.r), w);
		moves.set([l[0] / above.s[0], l[1] / above.s[1], l[2] / above.s[2]], t * 3);
	}
	samplers.push({ input, output: add(moves, 'VEC3'), interpolation: 'LINEAR' });
	channels.push({ sampler: samplers.length - 1, target: { node: g.nodes[0], path: 'translation' } });
	(gltf.animations ??= []).push({ name, samplers, channels });
	gltf.buffers ??= [{}];
	gltf.buffers[0] = { ...gltf.buffers[0], byteLength: offset };
	delete gltf.buffers[0].uri;

	const json = Array.from(utf8(JSON.stringify(gltf)));
	while (json.length % 4) json.push(0x20);
	const binLength = offset + ((4 - (offset % 4)) % 4);
	const total = 12 + 8 + json.length + 8 + binLength;
	const out = new Uint8Array(total);
	const dv = new DataView(out.buffer);
	dv.setUint32(0, 0x46546c67, true);
	dv.setUint32(4, 2, true);
	dv.setUint32(8, total, true);
	dv.setUint32(12, json.length, true);
	dv.setUint32(16, 0x4e4f534a, true);
	out.set(json, 20);
	let at = 20 + json.length;
	dv.setUint32(at, binLength, true);
	dv.setUint32(at + 4, 0x004e4942, true);
	at += 8;
	if (g.bin) out.set(g.bin, at);
	let cursor = at + start;
	for (const c of chunks) {
		out.set(c, cursor);
		cursor += c.length;
	}
	return out;
}
