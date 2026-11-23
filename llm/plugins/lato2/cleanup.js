// What LATO.2's raw faces have that a modelled mesh never does, taken out in
// this order:
//
//   crowded edges  an edge with three or more faces keeps the two that fit
//                  best (fewest open edges of their own, then the better shape)
//   folds          a face turned back against two or more of its neighbours
//                  (the crumpled look)
//   flaps          faces hanging on by one edge, `flapPasses` times over (none
//                  by default: thin parts such as legs and barrels come as
//                  strips of such faces and would be eaten)
//   specks         loose pieces with fewer than `minPiece` of the faces that
//                  are also small: under `minSize` of the mesh's diagonal
//   holes          rings of open edges up to `maxHole` long, closed by the
//                  same minimum-area triangulation LATO.2's ring filling uses
//                  (a long crack between two sheets zips shut that way)
//
// then the vertices no face uses are dropped and the winding made consistent.
// `smooth` > 0 adds that many Taubin steps (a shrink and an inflate each),
// which even out bumps without shrinking the shape.

import { orientFaces, minAreaTriangulation } from './faces.js';

export function cleanMesh(positions, faces, { minPiece = 0.01, minSize = 0.02, flapPasses = 0, maxHole = 150, foldCos = -0.5, smooth = 0 } = {}) {
	const n = positions.length / 3;
	const stats = { faces_in: faces.length / 3 };
	let tris = [];
	const seen = new Set();
	for (let f = 0; f < faces.length; f += 3) {
		const t = [faces[f], faces[f + 1], faces[f + 2]];
		if (t[0] === t[1] || t[1] === t[2] || t[0] === t[2]) continue;
		const s = [...t].sort((x, y) => x - y), k = (s[0] * n + s[1]) * n + s[2];
		if (seen.has(k)) continue;
		seen.add(k);
		tris.push(t);
	}

	const ek = (a, b) => (a < b ? a * n + b : b * n + a);
	const edgeFaces = (list) => {
		const m = new Map();
		list.forEach((t, f) => {
			for (let e = 0; e < 3; e++) {
				const k = ek(t[e], t[(e + 1) % 3]);
				let l = m.get(k);
				if (!l) m.set(k, (l = []));
				l.push(f);
			}
		});
		return m;
	};
	const openEdges = (t, m) => {
		let c = 0;
		for (let e = 0; e < 3; e++) if (m.get(ek(t[e], t[(e + 1) % 3])).length === 1) c++;
		return c;
	};
	const corner = (i) => [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
	// unit normal and shape (1 for an equilateral triangle, 0 for a sliver)
	const geometry = (t) => {
		const [a, b, c] = t.map(corner);
		const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
		const nx = u[1] * v[2] - u[2] * v[1], ny = u[2] * v[0] - u[0] * v[2], nz = u[0] * v[1] - u[1] * v[0];
		const len = Math.hypot(nx, ny, nz) || 1e-30;
		let sq = 0;
		for (const [p, q] of [[a, b], [b, c], [c, a]]) sq += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
		return { normal: [nx / len, ny / len, nz / len], shape: (2 * Math.sqrt(3) * len) / (sq || 1e-30) };
	};
	const orient = (list) => {
		const o = orientFaces(flatten(list), positions);
		const out = [];
		for (let f = 0; f < o.length; f += 3) out.push([o[f], o[f + 1], o[f + 2]]);
		return out;
	};

	// crowded edges
	{
		const m = edgeFaces(tris), drop = new Set();
		for (const l of m.values()) {
			if (l.length <= 2) continue;
			const ranked = l.filter((f) => !drop.has(f)).map((f) => ({ f, open: openEdges(tris[f], m), shape: geometry(tris[f]).shape }));
			ranked.sort((x, y) => x.open - y.open || y.shape - x.shape);
			for (const r of ranked.slice(2)) drop.add(r.f);
		}
		stats.crowded = drop.size;
		tris = tris.filter((_, f) => !drop.has(f));
	}

	// folds
	{
		tris = orient(tris);
		const m = edgeFaces(tris), normals = tris.map((t) => geometry(t).normal), drop = new Set();
		tris.forEach((t, f) => {
			let back = 0;
			for (let e = 0; e < 3; e++) {
				const l = m.get(ek(t[e], t[(e + 1) % 3]));
				if (l.length !== 2) continue;
				const g = l[0] === f ? l[1] : l[0], a = normals[f], b = normals[g];
				if (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] < foldCos) back++;
			}
			if (back >= 2) drop.add(f);
		});
		stats.folds = drop.size;
		tris = tris.filter((_, f) => !drop.has(f));
	}

	// flaps
	stats.flaps = 0;
	for (let pass = 0; pass < flapPasses; pass++) {
		const m = edgeFaces(tris), before = tris.length;
		tris = tris.filter((t) => openEdges(t, m) < 2);
		stats.flaps += before - tris.length;
		if (tris.length === before) break;
	}

	// specks
	{
		const m = edgeFaces(tris), parent = tris.map((_, f) => f);
		const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
		for (const l of m.values()) for (let i = 1; i < l.length; i++) { const a = find(l[0]), b = find(l[i]); if (a !== b) parent[a] = b; }
		// each piece's face count and bounding box
		const pieces = new Map();
		tris.forEach((t, f) => {
			const r = find(f);
			let p = pieces.get(r);
			if (!p) pieces.set(r, (p = { faces: 0, lo: [Infinity, Infinity, Infinity], hi: [-Infinity, -Infinity, -Infinity] }));
			p.faces++;
			for (const i of t) for (let d = 0; d < 3; d++) { p.lo[d] = Math.min(p.lo[d], positions[3 * i + d]); p.hi[d] = Math.max(p.hi[d], positions[3 * i + d]); }
		});
		const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
		for (const p of pieces.values()) for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], p.lo[d]); hi[d] = Math.max(hi[d], p.hi[d]); }
		const diag = (p) => Math.hypot(p.hi[0] - p.lo[0], p.hi[1] - p.lo[1], p.hi[2] - p.lo[2]);
		const whole = diag({ lo, hi }), fewest = minPiece * tris.length, before = tris.length;
		const speck = (p) => p.faces < 4 || (p.faces < fewest && diag(p) < minSize * whole);
		tris = tris.filter((_, f) => !speck(pieces.get(find(f))));
		stats.specks = before - tris.length;
	}

	// holes: walk the open edges as rings, either way round (the winding can
	// disagree along a crack); at a pinch, where four open edges meet, any
	// unused one goes on
	{
		const m = edgeFaces(tris), open = new Map(), existing = new Set();
		const tk = (t) => { const s = [...t].sort((x, y) => x - y); return (s[0] * n + s[1]) * n + s[2]; };
		for (const t of tris) existing.add(tk(t));
		for (const [k, l] of m) {
			if (l.length !== 1) continue;
			const a = Math.floor(k / n), b = k % n;
			for (const [x, y] of [[a, b], [b, a]]) { let o = open.get(x); if (!o) open.set(x, (o = [])); o.push(y); }
		}
		const used = new Set(), uses = new Map();
		for (const [k, l] of m) uses.set(k, l.length);
		let filled = 0;
		for (const [start, outs] of open) {
			for (const first of outs) {
				if (used.has(ek(start, first))) continue;
				const ring = [start];
				let cur = first, ok = false;
				used.add(ek(start, first));
				while (ring.length <= maxHole) {
					if (cur === start) { ok = true; break; }
					ring.push(cur);
					const nxt = open.get(cur).find((x) => !used.has(ek(cur, x)));
					if (nxt === undefined) break;
					used.add(ek(cur, nxt));
					cur = nxt;
				}
				if (!ok || ring.length < 3) continue;
				// a triangle that would give an edge a third face is left out
				// (the ring's own chords can already be edges elsewhere)
				for (const t of minAreaTriangulation(ring, positions, (x, y) => uses.has(ek(x, y)))) {
					if (t[0] === t[1] || t[1] === t[2] || t[0] === t[2] || existing.has(tk(t))) continue;
					const keys = [ek(t[0], t[1]), ek(t[1], t[2]), ek(t[2], t[0])];
					if (keys.some((k) => (uses.get(k) ?? 0) >= 2)) continue;
					for (const k of keys) uses.set(k, (uses.get(k) ?? 0) + 1);
					existing.add(tk(t));
					tris.push(t);
				}
				filled++;
			}
		}
		stats.holes_filled = filled;
	}

	// the vertices still used, renumbered
	const remap = new Int32Array(n).fill(-1);
	let V = 0;
	for (const t of tris) for (const i of t) if (remap[i] < 0) remap[i] = V++;
	let out = new Float32Array(V * 3);
	for (let i = 0; i < n; i++) if (remap[i] >= 0) out.set(positions.subarray(3 * i, 3 * i + 3), 3 * remap[i]);
	tris = tris.map((t) => t.map((i) => remap[i]));

	if (smooth > 0) out = taubin(out, tris, smooth);
	const result = orientFaces(flatten(tris), out);
	stats.vertices = V;
	stats.faces = result.length / 3;
	return { positions: out, faces: result, stats };
}

function flatten(list) {
	const out = new Int32Array(list.length * 3);
	list.forEach((t, f) => { out[3 * f] = t[0]; out[3 * f + 1] = t[1]; out[3 * f + 2] = t[2]; });
	return out;
}

// Taubin smoothing over the edge graph: each step moves every vertex toward
// the mean of its neighbours by 0.5, then away by 0.53.
function taubin(positions, tris, steps) {
	const V = positions.length / 3, nbr = Array.from({ length: V }, () => new Set());
	for (const [a, b, c] of tris) { nbr[a].add(b).add(c); nbr[b].add(a).add(c); nbr[c].add(a).add(b); }
	let p = Float32Array.from(positions);
	for (let s = 0; s < steps; s++) {
		for (const k of [0.5, -0.53]) {
			const q = new Float32Array(p.length);
			for (let i = 0; i < V; i++) {
				const nb = nbr[i];
				if (nb.size === 0) { q.set(p.subarray(3 * i, 3 * i + 3), 3 * i); continue; }
				let x = 0, y = 0, z = 0;
				for (const j of nb) { x += p[3 * j]; y += p[3 * j + 1]; z += p[3 * j + 2]; }
				x /= nb.size; y /= nb.size; z /= nb.size;
				q[3 * i] = p[3 * i] + k * (x - p[3 * i]);
				q[3 * i + 1] = p[3 * i + 1] + k * (y - p[3 * i + 1]);
				q[3 * i + 2] = p[3 * i + 2] + k * (z - p[3 * i + 2]);
			}
			p = q;
		}
	}
	return p;
}
