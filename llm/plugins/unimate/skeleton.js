// A rig as UniMate reads it: the skeleton put in its canonical frame, the
// joint order it expects, and the graph features of the joint tree.
//
// The input is the rig at rest: joint names, parents (-1 for the one root) and
// model-space positions, in any order and any units, Y up. The same steps as
// the reference's preprocessing (data_process/utils/{skeleton,topology,
// motion_features}.py) follow:
//
//   - joints in breadth-first order from the root, a joint's children
//     largest subtree first, shorter bone first between equals;
//   - turned about Y to face +Z (the facing pair: the best symmetric
//     "Right X"/"Left X" joints by cleaned name, as face_select_rule.py picks
//     them, or ones the caller names), root over the origin, scaled so the
//     longest leaf-to-leaf path along the bones is 2, lowest joint on Y = 0;
//   - per joint: depth, relation to every other joint (self, parent, child,
//     sibling, none, or end effector on the diagonal), tree distance capped at
//     5, and 8 eigenvectors of the normalised graph Laplacian.
//
// The rest pose is the model's reference: its motion is rotations of these
// bones (world-space vectors, rest rotations identity) and a root trajectory.

import { modelJointName } from './names.js';
import { STATS } from './stats.js';

export const MAX_JOINTS = 71;
export const MIN_JOINTS = 5;
const FEAT = 12;

// face_select_rule.py's preference order for the facing pair.
const PAIR_PRIORITY = [
	'Thigh', 'Shoulder', 'Front Shoulder', 'Back Hip', 'Hip', 'Scapula',
	'Upper Arm', 'Arm', 'Front Leg', 'Hind Leg', 'Middle Leg', 'Back Leg', 'Wing', 'Leg',
	'Pectoral Fin', 'Pelvic Fin', 'Fin', 'Gill', 'Pincer', 'Mandible', 'Large Mandible', 'Lower Mandible',
	'Stinger', 'Claw', 'Hand Claw', 'Antenna',
	'Forearm', 'Shin', 'Knee', 'Elbow', 'Ankle', 'Wrist',
	'Hand', 'Palm', 'Foot', 'Heel', 'Front Paw', 'Back Paw', 'Paw', 'Front Hoof', 'Rear Hoof', 'Hoof',
	'Fetlock', 'Cannon', 'Metacarpus', 'Pastern', 'Toe',
	'Thumb Finger', 'Index Finger', 'Middle Finger', 'Ring Finger', 'Pinky Finger', 'Finger',
	'Neck', 'Eye', 'Eyeball', 'Eyelid', 'Eyebrow', 'Ear', 'Horn', 'Cheek', 'Whisker', 'Fang', 'Barbel',
	'Tentacle', 'Feather', 'Tail',
];
const TAIL_WORDS = new Set(['Tail', 'Tail Twist']);
const HEAD_WORDS = new Set(['Head', 'Skull', 'Skull Base', 'Head End', 'Jaw', 'Upper Jaw', 'Lower Jaw', 'Tongue', 'Muzzle', 'Nose', 'Chin']);

const stripNum = (s) => s.replace(/\s+\d+$/, '');
const digitRuns = (s) => (s.match(/\d+/g) ?? []).join(',');

// The facing pair [right, left] (indices into `clean`), whether it is a body
// axis (head, tail) rather than a left-right pair, and what it is.
export function facingPair(clean, raw) {
	const right = new Map(), left = new Map();
	clean.forEach((n, i) => {
		if (n.startsWith('Right ')) (right.get(stripNum(n.slice(6))) ?? right.set(stripNum(n.slice(6)), []).get(stripNum(n.slice(6)))).push(i);
		else if (n.startsWith('Left ')) (left.get(stripNum(n.slice(5))) ?? left.set(stripNum(n.slice(5)), []).get(stripNum(n.slice(5)))).push(i);
	});
	const common = [...right.keys()].filter((k) => left.has(k));
	if (common.length) {
		const ordered = PAIR_PRIORITY.filter((s) => common.includes(s));
		ordered.push(...common.filter((s) => !PAIR_PRIORITY.includes(s)).sort());
		const suffix = ordered[0], rs = right.get(suffix), ls = left.get(suffix);
		if (rs.length === 1 && ls.length === 1) return { pair: [rs[0], ls[0]], bodyAxis: false, what: suffix };
		for (const key of [(r) => r, (r) => r.split(',').slice(0, -1).join(',')]) {
			const bySig = new Map();
			for (const li of ls) if (!bySig.has(key(digitRuns(raw[li])))) bySig.set(key(digitRuns(raw[li])), li);
			for (const ri of rs) {
				const li = bySig.get(key(digitRuns(raw[ri])));
				if (li !== undefined) return { pair: [ri, li], bodyAxis: false, what: suffix };
			}
		}
		return { pair: [rs[0], ls[0]], bodyAxis: false, what: suffix };
	}
	let tail = -1, head = -1;
	clean.forEach((n, i) => {
		if (TAIL_WORDS.has(stripNum(n))) tail = i;
		if (HEAD_WORDS.has(stripNum(n))) head = i;
	});
	if (tail >= 0 && head >= 0) return { pair: [head, tail], bodyAxis: true, what: 'body axis' };
	return null;
}

// Breadth-first order: children by subtree size, larger first, then bone length.
function bfsOrder(parents, lengths) {
	const n = parents.length, children = Array.from({ length: n }, () => []);
	let root = -1;
	parents.forEach((p, j) => (p === -1 ? (root = j) : children[p].push(j)));
	const size = new Array(n).fill(1), topo = [root];
	for (let i = 0; i < topo.length; i++) topo.push(...children[topo[i]]);
	for (let i = topo.length - 1; i >= 0; i--) for (const c of children[topo[i]]) size[topo[i]] += size[c];
	for (const c of children) c.sort((a, b) => size[b] - size[a] || lengths[a] - lengths[b]);
	const order = [root];
	for (let i = 0; i < order.length; i++) order.push(...children[order[i]]);
	return order;
}

// Symmetric eigen-decomposition (cyclic Jacobi): eigenvalues ascending and
// the eigenvectors as columns of an n x n row-major array.
function eigh(A, n) {
	const a = Float64Array.from(A), v = new Float64Array(n * n);
	for (let i = 0; i < n; i++) v[i * n + i] = 1;
	for (let sweep = 0; sweep < 100; sweep++) {
		let off = 0;
		for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
		if (off < 1e-30) break;
		for (let p = 0; p < n; p++) {
			for (let q = p + 1; q < n; q++) {
				const apq = a[p * n + q];
				if (Math.abs(apq) < 1e-300) continue;
				const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
				const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
				const c = 1 / Math.sqrt(t * t + 1), s = t * c;
				for (let k = 0; k < n; k++) {
					const akp = a[k * n + p], akq = a[k * n + q];
					a[k * n + p] = c * akp - s * akq;
					a[k * n + q] = s * akp + c * akq;
				}
				for (let k = 0; k < n; k++) {
					const apk = a[p * n + k], aqk = a[q * n + k];
					a[p * n + k] = c * apk - s * aqk;
					a[q * n + k] = s * apk + c * aqk;
				}
				for (let k = 0; k < n; k++) {
					const vkp = v[k * n + p], vkq = v[k * n + q];
					v[k * n + p] = c * vkp - s * vkq;
					v[k * n + q] = s * vkp + c * vkq;
				}
			}
		}
	}
	const idx = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[x * n + x] - a[y * n + y]);
	const values = idx.map((i) => a[i * n + i]);
	const vectors = new Float64Array(n * n);
	idx.forEach((src, dst) => { for (let k = 0; k < n; k++) vectors[k * n + dst] = v[k * n + src]; });
	return { values, vectors };
}

function laplacianEigenvectors(parents, K = 8) {
	const n = parents.length, k = Math.min(n - 1, K);
	const adj = new Float64Array(n * n), deg = new Float64Array(n);
	parents.forEach((p, c) => {
		if (p < 0 || p === c) return;
		adj[c * n + p] = adj[p * n + c] = 1;
	});
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += adj[i * n + j];
	const L = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			const di = deg[i] > 0 ? deg[i] ** -0.5 : 0, dj = deg[j] > 0 ? deg[j] ** -0.5 : 0;
			L[i * n + j] = (i === j ? 1 : 0) - di * adj[i * n + j] * dj;
		}
	}
	const { values, vectors } = eigh(L, n);
	const out = new Float32Array(n * K);
	// How many of the frequencies come first with an eigenvalue of their own:
	// past that, a repeated one (identical chains - a hand's fingers) has no
	// unique basis, and any is as good as the reference's.
	let distinct = 0;
	while (distinct < k) {
		const i = distinct + 1;
		if (Math.abs(values[i] - values[i - 1]) < 1e-7 || (i + 1 < n && Math.abs(values[i + 1] - values[i]) < 1e-7)) break;
		distinct++;
	}
	for (let f = 0; f < k; f++) {
		let norm = 0;
		for (let i = 0; i < n; i++) norm += vectors[i * n + f + 1] ** 2;
		norm = Math.sqrt(norm);
		for (let i = 0; i < n; i++) out[i * K + f] = norm > 1e-12 ? vectors[i * n + f + 1] / norm : vectors[i * n + f + 1];
	}
	return { vectors: out, distinct };
}

function topology(parents, maxPath = 5) {
	const n = parents.length, children = Array.from({ length: n }, () => []), adj = Array.from({ length: n }, () => []);
	parents.forEach((p, c) => {
		if (p < 0) return;
		children[p].push(c);
		adj[c].push(p);
		adj[p].push(c);
	});
	const relations = new Int32Array(n * n).fill(4), dist = new Int32Array(n * n).fill(maxPath), depths = new Int32Array(n);
	for (let i = 0; i < n; i++) {
		const pi = parents[i];
		for (let j = 0; j < n; j++) {
			const pj = parents[j];
			if (i === j) relations[i * n + j] = 0;
			else if (pj === i) relations[i * n + j] = 2;
			else if (j === pi && pi !== -1) relations[i * n + j] = 1;
			else if (pi !== -1 && pj === pi) relations[i * n + j] = 3;
		}
		if (children[i].length === 0) relations[i * n + i] = 5;
		const d = new Int32Array(n).fill(32767);
		d[i] = 0;
		const q = [i];
		for (let h = 0; h < q.length; h++) {
			const u = q[h];
			if (d[u] >= maxPath) continue;
			for (const v of adj[u]) if (d[v] > d[u] + 1) { d[v] = d[u] + 1; q.push(v); }
		}
		for (let j = 0; j < n; j++) dist[i * n + j] = Math.min(d[j], maxPath);
	}
	for (const j of Array.from({ length: n }, (_, i) => i)) {
		let d = 0;
		for (let p = parents[j]; p !== -1; p = parents[p]) d++;
		depths[j] = d;
	}
	return { relations, dist, depths };
}

// The longest leaf-to-leaf path along the bones (two farthest-first searches).
function diameter(parents, lengths) {
	const n = parents.length, adj = Array.from({ length: n }, () => []);
	parents.forEach((p, i) => { if (p !== -1) { adj[p].push([i, lengths[i]]); adj[i].push([p, lengths[i]]); } });
	const farthest = (s) => {
		const d = new Map([[s, 0]]), q = [s];
		let far = s, max = 0;
		for (let h = 0; h < q.length; h++) {
			for (const [v, w] of adj[q[h]]) {
				if (d.has(v)) continue;
				const dv = d.get(q[h]) + w;
				d.set(v, dv);
				q.push(v);
				if (dv > max) { max = dv; far = v; }
			}
		}
		return [far, max];
	};
	return farthest(farthest(0)[0])[1];
}

// The rotation about Y taking the rig's facing to +Z, as a quaternion [w, x, y, z]
// (Quaternions.between of the reference).
function facingTurn(p, pair, bodyAxis) {
	const norm = (v) => { const l = Math.max(Math.hypot(...v), 1e-8); return v.map((x) => x / l); };
	const across = norm([0, 1, 2].map((k) => p[pair[0]][k] - p[pair[1]][k]));
	// Y x across
	let fwd = norm([across[2], 0, -across[0]]);
	if (bodyAxis) fwd = [-fwd[2], fwd[1], fwd[0]]; // -90 degrees about Y
	const to = [0, 0, 1];
	const a = [fwd[1] * to[2] - fwd[2] * to[1], fwd[2] * to[0] - fwd[0] * to[2], fwd[0] * to[1] - fwd[1] * to[0]];
	const w = Math.sqrt(fwd.reduce((s, x) => s + x * x, 0)) + fwd[2];
	if (w < 1e-9) return [0, 0, 1, 0]; // facing -Z: half a turn about Y
	const q = [w, ...a], l = Math.hypot(...q);
	return q.map((x) => x / l);
}

export function rotate(q, v) {
	const [w, x, y, z] = q;
	const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
	return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

// Everything UniMate needs of a rig. `rig`: { names, parents, positions
// ([J][3]) }; `face`: [rightName, leftName] to override the facing pair;
// `stats`: 'mixamo' | 'truebones' | 'objaverse'.
export function skeletonCondition(rig, { face = null, stats = 'objaverse' } = {}) {
	const J = rig.parents.length;
	if (J < MIN_JOINTS || J > MAX_JOINTS) throw new Error(`UniMate animates skeletons of ${MIN_JOINTS} to ${MAX_JOINTS} joints; this one has ${J}`);
	if (rig.names.length !== J || rig.positions.length !== J) throw new Error('rig names, parents and positions differ in length');
	const roots = rig.parents.filter((p) => p === -1).length;
	if (roots !== 1) throw new Error(`the skeleton has ${roots} roots; UniMate needs one`);
	const set = STATS[stats];
	if (!set) throw new Error(`no statistics '${stats}'; there are ${Object.keys(STATS).join(', ')}`);

	const pos = rig.positions.map((p) => [p[0], p[1], p[2]]);
	const lengths = rig.parents.map((p, j) => (p === -1 ? Math.hypot(...pos[j]) : Math.hypot(pos[j][0] - pos[p][0], pos[j][1] - pos[p][1], pos[j][2] - pos[p][2])));
	const clean0 = rig.names.map((n) => modelJointName(n));
	let facing = null;
	if (face) {
		const r = rig.names.indexOf(face[0]), l = rig.names.indexOf(face[1]);
		if (r < 0 || l < 0) throw new Error(`facing joints ${face.join(', ')} are not in the rig`);
		facing = { pair: [r, l], bodyAxis: false, what: 'given' };
	} else {
		facing = facingPair(clean0, rig.names);
	}

	const order = bfsOrder(rig.parents, lengths);
	const at = new Array(J);
	order.forEach((old, k) => { at[old] = k; });
	const parents = order.map((old) => (rig.parents[old] === -1 ? -1 : at[rig.parents[old]]));
	const names = order.map((old) => rig.names[old]);
	const clean = order.map((old) => clean0[old]);
	let p = order.map((old) => pos[old]);

	let turn = facing ? facingTurn(pos, facing.pair, facing.bodyAxis) : [1, 0, 0, 0];
	// A rig whose sides are named the wrong way round for where its head is -
	// its "Left" legs on its right - reads as facing backwards and would move
	// tail first. When its head stands well behind the root, the way the names
	// face it, it is turned round to face its head. A standing figure's head is
	// over its hips, so this never touches one.
	if (facing && !face) {
		const head = clean0.findIndex((n) => /\bHead\b/.test(n));
		if (head >= 0) {
			const root = rig.parents.indexOf(-1);
			const h = rotate(turn, pos[head]), r = rotate(turn, pos[root]);
			const extent = Math.max(...[0, 1, 2].map((k) => Math.max(...pos.map((v) => v[k])) - Math.min(...pos.map((v) => v[k]))));
			const dx = h[0] - r[0], dz = h[2] - r[2];
			if (-dz > 0.15 * extent && -dz > 2 * Math.abs(dx)) {
				// Half a turn about Y after `turn`: [0, 0, 1, 0] * turn.
				turn = [-turn[2], turn[3], turn[0], -turn[1]];
				facing.what += ', turned round to face its head';
			}
		}
	}
	p = p.map((v) => rotate(turn, v));
	const rootXZ = [p[0][0], 0, p[0][2]];
	p = p.map((v) => [v[0] - rootXZ[0], v[1], v[2] - rootXZ[2]]);
	const diam = diameter(parents, order.map((old) => lengths[old]));
	if (!(diam > 1e-6)) throw new Error('the skeleton has no length (all its joints are in one place)');
	const scale = 2 / diam;
	p = p.map((v) => v.map((x) => x * scale));
	const ground = Math.min(...p.map((v) => v[1]));
	p = p.map((v) => [v[0], v[1] - ground, v[2]]);

	const offsets = p.map((v, j) => (parents[j] === -1 ? v.slice() : [0, 1, 2].map((k) => v[k] - p[parents[j]][k])));
	const mean = new Float32Array(J * FEAT), std = new Float32Array(J * FEAT);
	for (let j = 0; j < J; j++) {
		mean.set(j === 0 ? set.mean_root : set.mean_local, j * FEAT);
		std.set(j === 0 ? set.std_root : set.std_local, j * FEAT);
	}
	const identity6d = [1, 0, 0, 0, 1, 0];
	const tpos = new Float32Array(J * FEAT);
	for (let j = 0; j < J; j++) {
		const row = [...p[j], ...identity6d, 0, 0, 0];
		for (let d = 0; d < FEAT; d++) {
			const x = (row[d] - mean[j * FEAT + d]) / std[j * FEAT + d];
			tpos[j * FEAT + d] = Number.isFinite(x) ? x : 0;
		}
	}
	const tposParents = new Float32Array(J * FEAT);
	for (let j = 0; j < J; j++) {
		const src = parents[j] === -1 ? j : parents[j];
		tposParents.set(tpos.subarray(src * FEAT, (src + 1) * FEAT), j * FEAT);
	}
	const topo = topology(parents);
	const spectral = laplacianEigenvectors(parents);
	return {
		J, names, clean, parents, order, offsets, rest: p, scale, turn, rootXZ, ground,
		facing: facing ? { right: names[at[facing.pair[0]]], left: names[at[facing.pair[1]]], bodyAxis: facing.bodyAxis, what: facing.what } : null,
		tpos, tposParents, mean, std,
		spectral: spectral.vectors, spectralDistinct: spectral.distinct,
		depths: topo.depths, graphDist: topo.dist, relations: topo.relations,
	};
}
