// Edges to triangles, as LATO.2's `utils/inference.py` does it: every
// triangle of the predicted edge graph is a face, and (optionally) the rings
// of open edges - edges with fewer than two faces - up to `maxRing` long are
// closed by a minimum-area triangulation, smallest rings first.

// edges Int32Array [E * 2] (i < j, sorted), n vertices, verts [n * 3] (the
// integer voxel coordinates, for the fill's areas) -> Int32Array [F * 3].
export function edgesToFaces(edges, n, verts, { fill = true, maxRing = 12 } = {}) {
	const nbr = Array.from({ length: n }, () => new Set());
	for (let e = 0; e < edges.length; e += 2) { nbr[edges[e]].add(edges[e + 1]); nbr[edges[e + 1]].add(edges[e]); }
	const faces = [];
	for (let e = 0; e < edges.length; e += 2) {
		const u = edges[e], v = edges[e + 1];
		const [small, big] = nbr[u].size < nbr[v].size ? [nbr[u], nbr[v]] : [nbr[v], nbr[u]];
		const common = [];
		for (const w of small) if (w > v && big.has(w)) common.push(w);
		common.sort((a, b) => a - b);
		for (const w of common) faces.push(u, v, w);
	}
	if (fill) faces.push(...fillOpenRings(nbr, faces, n, verts, maxRing));
	return Int32Array.from(faces);
}

const key = (a, b, n) => (a < b ? a * n + b : b * n + a);

function fillOpenRings(nbr, faces, n, verts, maxRing) {
	if (n < 4 || maxRing < 4) return [];
	const count = new Map();
	const bump = (a, b) => { const k = key(a, b, n); const c = (count.get(k) ?? 0) + 1; count.set(k, c); return c; };
	for (let f = 0; f < faces.length; f += 3) {
		bump(faces[f], faces[f + 1]); bump(faces[f + 1], faces[f + 2]); bump(faces[f + 2], faces[f]);
	}
	const open = Array.from({ length: n }, () => new Set());
	const openEdges = [];
	for (let u = 0; u < n; u++) {
		for (const v of [...nbr[u]].sort((a, b) => a - b)) {
			if (v > u && (count.get(key(u, v, n)) ?? 0) < 2) { openEdges.push([u, v]); open[u].add(v); open[v].add(u); }
		}
	}
	const added = [];
	for (let size = 4; size <= maxRing; size++) {
		for (const [u, v] of openEdges) {
			// a face-less edge can border two rings
			while ((count.get(key(u, v, n)) ?? 0) < 2) {
				const ring = shortestOpenRing(u, v, open, nbr, size);
				if (!ring || !chordless(ring, nbr)) break;
				for (const [a, b, c] of minAreaTriangulation(ring, verts)) {
					added.push(a, b, c);
					for (const [x, y] of [[a, b], [b, c], [c, a]]) {
						const k = bump(x, y);
						nbr[x].add(y);
						nbr[y].add(x);
						if (k >= 2) { open[x].delete(y); open[y].delete(x); }
					}
				}
			}
		}
	}
	return added;
}

// The shortest cycle of 4 .. maxSize vertices through open edge (u, v) that
// walks only open edges; vertices next to both u and v would be a chord of
// any such cycle and are left out.
function shortestOpenRing(u, v, open, nbr, maxSize) {
	const blocked = new Set();
	for (const w of nbr[u]) if (nbr[v].has(w)) blocked.add(w);
	const parent = new Map([[v, -1]]);
	let frontier = [v];
	for (let step = 0; step < maxSize - 1; step++) {
		const reached = [];
		for (let x of frontier) {
			for (const y of [...open[x]].sort((a, b) => a - b)) {
				if (parent.has(y) || blocked.has(y)) continue;
				if (y === u) {
					if (x === v) continue; // the edge itself
					const ring = [u];
					while (x !== -1) { ring.push(x); x = parent.get(x); }
					return ring;
				}
				parent.set(y, x);
				reached.push(y);
			}
		}
		if (!reached.length) break;
		frontier = reached;
	}
	return null;
}

function chordless(ring, nbr) {
	const n = ring.length;
	for (let i = 0; i < n - 2; i++) {
		for (let j = i + 2; j < (i === 0 ? n - 1 : n); j++) if (nbr[ring[i]].has(ring[j])) return false;
	}
	return true;
}

// Minimum-area triangulation of a closed 3D ring (O(n^3)): the cost is (the
// number of zero-area triangles, the total area), so collinear vertices give
// degenerate faces only when nothing else can. `forbidden(a, b)`, when given,
// rules out a chord between ring vertices a and b; with every triangulation
// ruled out there are none ([]).
export function minAreaTriangulation(ring, verts, forbidden = null) {
	const n = ring.length, p = ring.map((r) => [verts[3 * r], verts[3 * r + 1], verts[3 * r + 2]]);
	let ext = 0;
	for (let d = 0; d < 3; d++) { const vals = p.map((q) => q[d]); ext += (Math.max(...vals) - Math.min(...vals)) ** 2; }
	const eps = 1e-9 * ext;
	const area = (i, k, j) => {
		const a = [p[k][0] - p[i][0], p[k][1] - p[i][1], p[k][2] - p[i][2]], b = [p[j][0] - p[i][0], p[j][1] - p[i][1], p[j][2] - p[i][2]];
		const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
		return 0.5 * Math.hypot(c[0], c[1], c[2]);
	};
	const zeros = Array.from({ length: n }, () => new Array(n).fill(0));
	const sums = Array.from({ length: n }, () => new Array(n).fill(0));
	const split = Array.from({ length: n }, () => new Array(n).fill(-1));
	const bad = forbidden ? (x, y) => y - x >= 2 && !(x === 0 && y === n - 1) && forbidden(ring[x], ring[y]) : () => false;
	for (let span = 2; span < n; span++) {
		for (let i = 0; i + span < n; i++) {
			const j = i + span;
			let bz = Infinity, bs = Infinity;
			for (let k = i + 1; k < j; k++) {
				if (bad(i, k) || bad(k, j)) continue;
				const a = area(i, k, j);
				const z = zeros[i][k] + zeros[k][j] + (a <= eps ? 1 : 0), s = sums[i][k] + sums[k][j] + a;
				if (z < bz || (z === bz && s < bs)) { bz = z; bs = s; split[i][j] = k; }
			}
			zeros[i][j] = bz;
			sums[i][j] = bs;
		}
	}
	if (zeros[0][n - 1] === Infinity) return [];
	const tris = [], stack = [[0, n - 1]];
	while (stack.length) {
		const [i, j] = stack.pop();
		if (j - i < 2) continue;
		const k = split[i][j];
		tris.push([ring[i], ring[k], ring[j]]);
		stack.push([i, k], [k, j]);
	}
	return tris;
}

// Faces turned so that neighbours agree (a shared edge runs opposite ways in
// the two), each connected piece then flipped to enclose positive volume:
// LATO.2's faces come with no winding, and a .glb's normals follow it.
export function orientFaces(faces, positions) {
	const F = faces.length / 3, out = Int32Array.from(faces);
	const byEdge = new Map();
	const ek = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
	for (let f = 0; f < F; f++) {
		for (let e = 0; e < 3; e++) {
			const k = ek(faces[3 * f + e], faces[3 * f + (e + 1) % 3]);
			let l = byEdge.get(k);
			if (!l) byEdge.set(k, (l = []));
			l.push(f);
		}
	}
	// does face f run a -> b?
	const runs = (f, a, b) => {
		for (let e = 0; e < 3; e++) if (out[3 * f + e] === a && out[3 * f + (e + 1) % 3] === b) return true;
		return false;
	};
	const seen = new Uint8Array(F);
	for (let start = 0; start < F; start++) {
		if (seen[start]) continue;
		const piece = [start], queue = [start];
		seen[start] = 1;
		while (queue.length) {
			const f = queue.pop();
			for (let e = 0; e < 3; e++) {
				const a = out[3 * f + e], b = out[3 * f + (e + 1) % 3];
				const l = byEdge.get(ek(a, b));
				if (l.length !== 2) continue; // a boundary, or more than two faces
				const g = l[0] === f ? l[1] : l[0];
				if (seen[g]) continue;
				// g must run b -> a; flip it when it runs a -> b
				if (runs(g, a, b)) { const t = out[3 * g + 1]; out[3 * g + 1] = out[3 * g + 2]; out[3 * g + 2] = t; }
				seen[g] = 1;
				queue.push(g);
				piece.push(g);
			}
		}
		let vol = 0;
		for (const f of piece) {
			const [a, b, c] = [out[3 * f], out[3 * f + 1], out[3 * f + 2]].map((i) => [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]]);
			vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
		}
		if (vol < 0) for (const f of piece) { const t = out[3 * f + 1]; out[3 * f + 1] = out[3 * f + 2]; out[3 * f + 2] = t; }
	}
	return out;
}
