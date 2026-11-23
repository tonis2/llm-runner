<*
 Hi3D's mesh from its decoder's cells, on the host (`llm/plugins/hi3d/extract.js`):
 sparc3d's point_to_mesh (sparcubes/extract.py, sparse_cube.py) and its vendored
 FlexiCubes (ops/flexicubes/flexicubes.py, dual marching cubes without the
 weights), then fill_mesh_holes (sparcubes/fill_hole.py).

 `meshFlexiCubes(coords, cells, res, maxHole, activated)`: coords an Int32Array
 [n, 3] of cells on a res^3 grid, cells a Float32Array [n, 40] as the decoder
 makes them: 8 per-corner sign logits, 8 distances, 8 corners' xyz
 deformations (corner c = x + 2y + 4z), the last 32 before their tanh unless
 `activated`.

 - Each grid corner the cells share gets the mean of their sigmoid(sign), distance
   and half deformation; its value is +-|distance| by whether the mean sign is
   >= 0.5, its position the corner plus the deformation.
 - A cell with corners on both sides of zero is a surface cube; its case (the
   corners below zero as bits) picks dual vertices from FC_DMC, each the mean of
   its edges' zero crossings. An ambiguous case whose FC_CHECK neighbour is also
   ambiguous takes the table's resolved case.
 - Each crossed edge four surface cubes share makes a quad of their dual
   vertices (cubes in x, y, z order), turned by the sign at the edge's first
   corner, split [0, 1, 3], [3, 1, 2].
 - Holes (closed loops of edges only one face has, up to `maxHole` long) are
   ear-clipped as fill_hole.py does.

 Returns { positions: Float32Array (grid units), indices: Uint32Array, cubes,
 quads, holes }.
*>
module llm::host;

import fastjs::js;
import std::math;
import std::sort;

// FlexiCubes' cube_edges: each edge's two corners, in its own orientation.
const usz[2][12] FC_EDGES = {
	{ 0, 1 }, { 1, 5 }, { 4, 5 }, { 0, 4 }, { 2, 3 }, { 3, 7 },
	{ 6, 7 }, { 2, 6 }, { 2, 0 }, { 3, 1 }, { 7, 5 }, { 6, 4 },
};

const uint FC_NONE = uint::max;

<*
 Open-addressing map from a grid key to an index.
*>
struct KeyMap
{
	ulong[] keys;
	uint[] vals;
	ulong mask;
}

fn void KeyMap.init(&self, usz count)
{
	usz size = 1;
	while (size < 2 * count + 2) size <<= 1;
	self.keys = mem::alloc_array(ulong, (sz)size);
	self.vals = mem::alloc_array(uint, (sz)size);
	self.keys[..] = 0;
	self.mask = (ulong)size - 1;
}

fn void KeyMap.free(&self)
{
	mem::free(self.keys);
	mem::free(self.vals);
}

<*
 The slot of `key` (stored as key + 1, 0 being empty).
*>
fn usz KeyMap.slot(&self, ulong key) @inline
{
	ulong k = key + 1;
	ulong h = (k * 0x9E3779B97F4A7C15ul) >> 20;
	while (true)
	{
		usz i = (usz)(h & self.mask);
		if (self.keys[i] == 0 || self.keys[i] == k) return i;
		h++;
	}
}

<*
 The index for `key`, `fresh` (written into the map) when it has none.
*>
fn uint KeyMap.get_or_put(&self, ulong key, uint fresh, bool* added) @inline
{
	usz i = self.slot(key);
	if (self.keys[i] != 0)
	{
		*added = false;
		return self.vals[i];
	}
	self.keys[i] = key + 1;
	self.vals[i] = fresh;
	*added = true;
	return fresh;
}

fn uint KeyMap.find(&self, ulong key) @inline
{
	usz i = self.slot(key);
	return self.keys[i] == 0 ? FC_NONE : self.vals[i];
}

<*
 Ear-clip one hole, fill_hole.py's _ear_clip: repeatedly cut the vertex whose
 turning angle (signed against the loop's own normal estimate) gives the largest
 1 / angle. Appends the triangles to `out` from `at`; returns the new end.
*>
fn usz ear_clip(float[] pos, uint[] loop_in, usz len, uint[] out, usz at)
{
	uint[] idx = mem::alloc_array(uint, (sz)len);
	defer mem::free(idx);
	double[<3>][] p = mem::alloc_array(double[<3>], (sz)len);
	defer mem::free(p);
	double[<3>][] t = mem::alloc_array(double[<3>], (sz)len);
	defer mem::free(t);
	for (usz i = 0; i < len; i++)
	{
		idx[i] = loop_in[i];
		usz v = 3 * (usz)loop_in[i];
		p[i] = { (double)pos[v], (double)pos[v + 1], (double)pos[v + 2] };
	}
	usz n = len;
	while (true)
	{
		if (n == 3)
		{
			out[at++] = idx[2]; out[at++] = idx[1]; out[at++] = idx[0];
			return at;
		}
		double[<3>] center = { 0, 0, 0 };
		for (usz i = 0; i < n; i++) center += p[i];
		center /= (double)n;
		for (usz i = 0; i < n; i++)
		{
			double[<3>] d = p[(i + 1) % n] - p[i];
			t[i] = d / (math::sqrt(d.dot(d)) + 1e-6);
		}
		usz ear = 0;
		double best = -double::max;
		for (usz i = 0; i < n; i++)
		{
			double[<3>] tp = t[(i + n - 1) % n];
			double[<3>] nrm = (-tp).cross(t[i]);
			double[<3>] pp = p[(i + 1) % n] - center;
			double[<3>] q = p[i] - center;
			double s = pp.cross(q).dot(nrm);
			double sign = s > 0 ? 1.0 : (s < 0 ? -1.0 : 0.0);
			double angle = math::atan2(math::sqrt(nrm.dot(nrm)) * sign, -tp.dot(t[i]));
			double score = angle == 0.0 ? double::inf : 1.0 / angle;
			if (score > best)
			{
				best = score;
				ear = i;
			}
		}
		usz i0 = (ear + 1) % n;
		usz i2 = (ear + n - 1) % n;
		out[at++] = idx[i0]; out[at++] = idx[ear]; out[at++] = idx[i2];
		for (usz i = ear; i + 1 < n; i++)
		{
			idx[i] = idx[i + 1];
			p[i] = p[i + 1];
		}
		n--;
	}
}

<*
 fill_mesh_holes over faces [0, nf): returns the new face count; `out` must hold
 room for nf + the added faces (at most the boundary edges' count).
*>
fn usz fill_holes(float[] pos, usz nv, uint[] faces, usz nf, usz max_hole, usz* holes)
{
	// directed edges and how many faces use each undirected one
	KeyMap count_of;
	count_of.init(3 * nf);
	defer count_of.free();
	uint[] counts = mem::alloc_array(uint, (sz)(3 * nf + 1));
	defer mem::free(counts);
	counts[..] = 0;
	uint distinct = 0;
	for (usz f = 0; f < nf; f++)
	{
		for (usz k = 0; k < 3; k++)
		{
			ulong a = faces[3 * f + k];
			ulong b = faces[3 * f + (k + 1) % 3];
			ulong key = a < b ? a * (ulong)nv + b : b * (ulong)nv + a;
			bool added;
			uint id = count_of.get_or_put(key, distinct, &added);
			if (added) distinct++;
			counts[id]++;
		}
	}
	// boundary edges in the order torch lists them: all (v0, v1), then (v1, v2), then (v2, v0)
	usz nb = 0;
	for (usz k = 0; k < 3; k++)
	{
		for (usz f = 0; f < nf; f++)
		{
			ulong a = faces[3 * f + k];
			ulong b = faces[3 * f + (k + 1) % 3];
			ulong key = a < b ? a * (ulong)nv + b : b * (ulong)nv + a;
			if (counts[count_of.find(key)] == 1) nb++;
		}
	}
	if (nb == 0) return nf;
	uint[2][] boundary = mem::alloc_array(uint[2], (sz)nb);
	defer mem::free(boundary);
	usz bi = 0;
	for (usz k = 0; k < 3; k++)
	{
		for (usz f = 0; f < nf; f++)
		{
			uint a = faces[3 * f + k];
			uint b = faces[3 * f + (k + 1) % 3];
			ulong key = a < b ? (ulong)a * (ulong)nv + b : (ulong)b * (ulong)nv + a;
			if (counts[count_of.find(key)] == 1) boundary[bi++] = { a, b };
		}
	}
	// outgoing boundary edges per vertex, in order (CSR)
	uint[] first = mem::alloc_array(uint, (sz)(nv + 1));
	defer mem::free(first);
	first[..] = 0;
	foreach (e : boundary) first[e[0] + 1]++;
	for (usz v = 0; v < nv; v++) first[v + 1] += first[v];
	uint[] fill = mem::alloc_array(uint, (sz)nv);
	defer mem::free(fill);
	fill[..] = 0;
	uint[] outgoing = mem::alloc_array(uint, (sz)nb);
	defer mem::free(outgoing);
	for (usz i = 0; i < nb; i++)
	{
		uint s = boundary[i][0];
		outgoing[first[s] + fill[s]++] = (uint)i;
	}
	bool[] visited = mem::alloc_array(bool, (sz)nb);
	defer mem::free(visited);
	visited[..] = false;
	uint[] loop = mem::alloc_array(uint, (sz)(nb + 2));
	defer mem::free(loop);
	usz at = 3 * nf;
	for (usz i = 0; i < nb; i++)
	{
		if (visited[i]) continue;
		visited[i] = true;
		uint start = boundary[i][0];
		usz len = 0;
		loop[len++] = start;
		loop[len++] = boundary[i][1];
		while (loop[len - 1] != start)
		{
			uint v = loop[len - 1];
			bool moved = false;
			for (uint j = first[v]; j < first[v + 1]; j++)
			{
				uint e = outgoing[j];
				if (visited[e]) continue;
				visited[e] = true;
				loop[len++] = boundary[e][1];
				moved = true;
				break;
			}
			if (!moved) break;
		}
		if (loop[0] != loop[len - 1] || len <= 3) continue;
		len--;
		if (len < 3 || len > max_hole) continue;
		at = ear_clip(pos, loop[:len], len, faces, at);
		(*holes)++;
	}
	return at / 3;
}

fn js::Value? js_mesh_flexicubes(js::Engine* js, int[] coords, float[] cells, long res_in, long max_hole, long activated)
{
	usz n = coords.len / 3;
	if (coords.len % 3 || cells.len != 40 * n) return js.fail("meshFlexiCubes: coords [n, 3] and cells [n, 40] expected")~;
	if (n == 0) return js.fail("meshFlexiCubes: no cells")~;
	if (res_in < 1 || res_in > 4096) return js.fail("meshFlexiCubes: resolution %d is out of range", res_in)~;
	ulong res = (ulong)res_in;
	ulong side = res + 1;

	// the cells in x, y, z order (FlexiCubes' sparse ambiguity lookup assumes it)
	ulong[] order = mem::alloc_array(ulong, (sz)n);
	defer mem::free(order);
	for (usz i = 0; i < n; i++)
	{
		ulong key = ((ulong)coords[3 * i] * res + (ulong)coords[3 * i + 1]) * res + (ulong)coords[3 * i + 2];
		order[i] = key << 32 | (ulong)i;
	}
	sort::quicksort(order);

	// the corners: one vertex each, the mean of the cells' values there
	KeyMap corner;
	corner.init(4 * n);
	defer corner.free();
	uint[] cube = mem::alloc_array(uint, (sz)(8 * n));
	defer mem::free(cube);
	uint nv = 0;
	for (usz o = 0; o < n; o++)
	{
		usz i = (usz)(order[o] & 0xFFFF_FFFF);
		for (usz c = 0; c < 8; c++)
		{
			ulong x = (ulong)coords[3 * i] + (c & 1);
			ulong y = (ulong)coords[3 * i + 1] + ((c >> 1) & 1);
			ulong z = (ulong)coords[3 * i + 2] + ((c >> 2) & 1);
			bool added;
			cube[8 * o + c] = corner.get_or_put((x * side + y) * side + z, nv, &added);
			if (added) nv++;
		}
	}
	float[] acc = mem::alloc_array(float, (sz)(5 * nv));
	defer mem::free(acc);
	acc[..] = 0;
	uint[] uses = mem::alloc_array(uint, (sz)nv);
	defer mem::free(uses);
	uses[..] = 0;
	for (usz o = 0; o < n; o++)
	{
		usz i = (usz)(order[o] & 0xFFFF_FFFF);
		for (usz c = 0; c < 8; c++)
		{
			usz v = cube[8 * o + c];
			float[] row = cells[40 * i : 40];
			acc[5 * v] += 1.0f / (1.0f + math::exp(-row[c]));
			float d = row[8 + c];
			acc[5 * v + 1] += activated ? d : math::tanh(d);
			for (usz k = 0; k < 3; k++)
			{
				float e = row[16 + 3 * c + k];
				acc[5 * v + 2 + k] += (activated ? e : math::tanh(e)) * 0.5f;
			}
			uses[v]++;
		}
	}
	float[] sdf = mem::alloc_array(float, (sz)nv);
	defer mem::free(sdf);
	float[] vpos = mem::alloc_array(float, (sz)(3 * nv));
	defer mem::free(vpos);
	for (usz o = 0; o < n; o++)
	{
		usz i = (usz)(order[o] & 0xFFFF_FFFF);
		for (usz c = 0; c < 8; c++)
		{
			usz v = cube[8 * o + c];
			if (uses[v] == 0) continue;
			float u = (float)uses[v];
			float s = acc[5 * v] / u;
			float d = math::abs(acc[5 * v + 1] / u);
			sdf[v] = s >= 0.5f ? d : -d;
			vpos[3 * v] = (float)(coords[3 * i] + (int)(c & 1)) + acc[5 * v + 2] / u;
			vpos[3 * v + 1] = (float)(coords[3 * i + 1] + (int)((c >> 1) & 1)) + acc[5 * v + 3] / u;
			vpos[3 * v + 2] = (float)(coords[3 * i + 2] + (int)((c >> 2) & 1)) + acc[5 * v + 4] / u;
			uses[v] = 0;
		}
	}

	// surface cubes and their cases
	short[] cases = mem::alloc_array(short, (sz)n);
	defer mem::free(cases);
	usz surf = 0;
	for (usz o = 0; o < n; o++)
	{
		int id = 0;
		int below = 0;
		for (usz c = 0; c < 8; c++)
		{
			if (sdf[cube[8 * o + c]] < 0)
			{
				id |= 1 << c;
				below++;
			}
		}
		cases[o] = below > 0 && below < 8 ? (short)id : (short)-1;
		if (cases[o] >= 0) surf++;
	}
	if (surf == 0) return js.fail("meshFlexiCubes: the isosurface is empty")~;
	// ambiguous cases next to an ambiguous neighbour take the resolved case;
	// the neighbours are looked up among the ambiguous cubes by their original case
	KeyMap cell_of;
	cell_of.init(n);
	defer cell_of.free();
	for (usz o = 0; o < n; o++)
	{
		if (cases[o] < 0 || FC_CHECK[cases[o]][0] != 1) continue;
		bool added;
		cell_of.get_or_put(order[o] >> 32, (uint)o, &added);
	}
	short[] resolved = mem::alloc_array(short, (sz)n);
	defer mem::free(resolved);
	resolved[..] = cases[..];
	for (usz o = 0; o < n; o++)
	{
		if (cases[o] < 0 || FC_CHECK[cases[o]][0] != 1) continue;
		usz i = (usz)(order[o] & 0xFFFF_FFFF);
		long ax = (long)coords[3 * i] + FC_CHECK[cases[o]][1];
		long ay = (long)coords[3 * i + 1] + FC_CHECK[cases[o]][2];
		long az = (long)coords[3 * i + 2] + FC_CHECK[cases[o]][3];
		if (ax < 0 || ay < 0 || az < 0 || ax >= res_in || ay >= res_in || az >= res_in) continue;
		uint other = cell_of.find(((ulong)ax * res + (ulong)ay) * res + (ulong)az);
		if (other == FC_NONE) continue;
		if (FC_CHECK[cases[other]][0] == 1) resolved[o] = FC_CHECK[cases[o]][4];
	}

	// crossed edges, by their lower corner and axis: how many surface cubes have each
	usz edges = 3 * (usz)nv;
	uint[] edge_uses = mem::alloc_array(uint, (sz)edges);
	defer mem::free(edge_uses);
	edge_uses[..] = 0;
	for (usz o = 0; o < n; o++)
	{
		if (cases[o] < 0) continue;
		for (usz e = 0; e < 12; e++)
		{
			uint a = cube[8 * o + FC_EDGES[e][0]];
			uint b = cube[8 * o + FC_EDGES[e][1]];
			if ((sdf[a] < 0) == (sdf[b] < 0)) continue;
			usz ca = FC_EDGES[e][0];
			usz cb = FC_EDGES[e][1];
			usz axis = (ca ^ cb) == 1 ? 0 : ((ca ^ cb) == 2 ? 1 : 2);
			uint low = cube[8 * o + (ca < cb ? ca : cb)];
			edge_uses[3 * (usz)low + axis]++;
		}
	}

	// dual vertices: per surface cube and group, the mean of its edges' crossings
	usz nvd = 0;
	for (usz o = 0; o < n; o++) if (cases[o] >= 0) nvd += FC_NUM_VD[resolved[o]];
	float[] vd = mem::alloc_array(float, (sz)(3 * nvd + 3));
	defer mem::free(vd);
	// each crossed edge's four dual vertices, in cube order
	uint[4][] quad = mem::alloc_array(uint[4], (sz)edges);
	defer mem::free(quad);
	char[] filled = mem::alloc_array(char, (sz)edges);
	defer mem::free(filled);
	filled[..] = 0;
	usz at = 0;
	for (usz o = 0; o < n; o++)
	{
		if (cases[o] < 0) continue;
		int cs = resolved[o];
		for (usz g = 0; g < FC_NUM_VD[cs]; g++)
		{
			double[<3>] sum = { 0, 0, 0 };
			int count = 0;
			for (usz k = 0; k < 7; k++)
			{
				int e = FC_DMC[cs][g][k];
				if (e < 0) continue;
				uint a = cube[8 * o + FC_EDGES[e][0]];
				uint b = cube[8 * o + FC_EDGES[e][1]];
				float sa = sdf[a];
				float sb = sdf[b];
				// _linear_interp: (x_a s_b - x_b s_a) / (s_b - s_a)
				for (usz d = 0; d < 3; d++)
				{
					float ue = (vpos[3 * (usz)a + d] * sb + vpos[3 * (usz)b + d] * -sa) / (sb + -sa);
					sum[d] += (double)ue;
				}
				count++;
				usz ca = FC_EDGES[e][0];
				usz cb = FC_EDGES[e][1];
				usz axis = (ca ^ cb) == 1 ? 0 : ((ca ^ cb) == 2 ? 1 : 2);
				usz slot = 3 * (usz)cube[8 * o + (ca < cb ? ca : cb)] + axis;
				if (filled[slot] < 4) quad[slot][filled[slot]] = (uint)at;
				filled[slot]++;
			}
			for (usz d = 0; d < 3; d++) vd[3 * at + d] = (float)(sum[d] / count);
			at++;
		}
	}

	// quads: crossed edges four surface cubes share
	usz nq = 0;
	for (usz s = 0; s < edges; s++) if (edge_uses[s] == 4 && filled[s] == 4) nq++;
	// the quads' triangles, and room for ear-clipping: a hole adds at most one
	// face per boundary edge, and there are at most 3 per face
	uint[] faces = mem::alloc_array(uint, (sz)(6 * nq + 18 * nq + 16));
	defer mem::free(faces);
	usz nf = 0;
	// the edge's first corner as cube_edges orients it: the upper one for y edges
	for (usz o = 0; o < n; o++)
	{
		if (cases[o] < 0) continue;
		for (usz e = 0; e < 12; e++)
		{
			usz ca = FC_EDGES[e][0];
			usz cb = FC_EDGES[e][1];
			uint a = cube[8 * o + ca];
			uint b = cube[8 * o + cb];
			if ((sdf[a] < 0) == (sdf[b] < 0)) continue;
			usz axis = (ca ^ cb) == 1 ? 0 : ((ca ^ cb) == 2 ? 1 : 2);
			usz slot = 3 * (usz)cube[8 * o + (ca < cb ? ca : cb)] + axis;
			if (edge_uses[slot] != 4 || filled[slot] != 4) continue;
			uint[4] q = quad[slot];
			uint[4] r;
			if (sdf[a] > 0)
			{
				r = { q[0], q[1], q[3], q[2] };
			}
			else
			{
				r = { q[2], q[3], q[1], q[0] };
			}
			faces[nf++] = r[0]; faces[nf++] = r[1]; faces[nf++] = r[3];
			faces[nf++] = r[3]; faces[nf++] = r[1]; faces[nf++] = r[2];
			filled[slot] = 5; // once per edge
		}
	}
	usz holes = 0;
	usz total = fill_holes(vd, at, faces, nf / 3, (usz)max_hole, &holes);

	js::Value result = js.object()!;
	js.set(result, "positions", js.typed_copy(FLOAT32, ((char*)vd.ptr)[:at * 12])!)!;
	js.set(result, "indices", js.typed_copy(UINT32, ((char*)faces.ptr)[:total * 12])!)!;
	js.set(result, "cubes", js::number(surf))!;
	js.set(result, "quads", js::number(nq))!;
	js.set(result, "holes", js::number(holes))!;
	return result;
}
