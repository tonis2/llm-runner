<*
 Dense meshes cleaned up and decimated on the host, for the image-to-mesh
 plugins (`llm/core/mesh/simplify.js`): a JS interpreter would take minutes
 over the millions of faces of a 1024^3 surface.

 `meshSimplify(positions, indices, params)`: positions a Float32Array [n, 3],
 indices a Uint32Array [f, 3], params a Float64Array laid out as simplify.js
 lays it: target faces, part gap, part keep, separate, smooth, crease. In
 order:

 - parts: pieces (faces joined by shared vertices) with less than `keep` of
   the faces, none of whose vertices comes within about `gap` of a bigger
   piece kept, are dropped. That takes floaters and leaves the strands of
   hair a voxel or two from the head.
 - separate: vertices sharing their position with another (the two sides of
   a one-voxel plate) each move `separate` along their own normal, so the
   sides no longer coincide.
 - smooth: that many rounds of Taubin smoothing (a step in, a step out, so
   the volume stays): a voxel surface's sub-voxel staircase, which
   decimation would otherwise keep as creases along its contours, goes.
 - decimation to `target` faces (0: none): quadric-error edge collapses
   (Garland and Heckbert) in passes, as meshoptimizer's simplifier makes
   them. Every edge is ranked by the error of moving one end onto the other,
   then the cheapest are taken in order, no vertex twice in a pass. A
   collapse keeps the far end where it is, and is refused when it would
   turn a face over, join two surfaces (the ends sharing more than the
   edge's two opposite vertices), or leave faces around the kept end facing
   further apart than they did at either end (a quadric's planes have no
   side, so a thin plate's two sides look like one plane). Vertices on open
   or non-manifold edges stay.
 - normals, when `crease` (degrees) is above 0: at each vertex its faces
   grouped by direction, faces more than the crease apart in different
   groups, and the vertex split into one per group with that group's
   (area-weighted) normal. A thin plate's rim and an armour's edge shade
   sharp, instead of averaging faces that point apart into no direction.

 Returns { positions, indices, normals (or null), parts, dropped, separated,
 passes, seconds }: seconds per step, [parts, separate, smooth, decimate,
 normals].

 `meshShellSurface(coords, fields, r, table, outside, g, words)`: the outer
 surface of a voxel shell, pixal3d/mesh.js shellSurface's surface nets (that
 file has the JS and the description): coords an Int32Array [n, 3] on an r^3
 grid, fields the shape decoder's Float32Array [n, 7] (the dual vertex in
 each voxel from its first three), table the voxels' neighbour table
 [n, 27], outside the flood's bits on the padded g^3 grid, `words` words to
 a line along z. Returns { positions, indices }.
*>
module llm::host;

import fastjs::js;
import std::math;
import std::sort;
import std::time::clock;

const uint NO_VERTEX = uint::max;
const usz MAX_FAN = 64;
const double NEVER = 1e300;
// A collapse may turn a face by up to about 75 degrees.
const double MIN_TURN_COS = 0.25;

<*
 The squared distance to a set of weighted planes, as a symmetric 4x4 matrix.
*>
struct Quadric
{
	double xx, xy, xz, xw, yy, yz, yw, zz, zw, ww;
}

fn void Quadric.add_plane(&self, double nx, double ny, double nz, double d, double w)
{
	self.xx += w * nx * nx;
	self.xy += w * nx * ny;
	self.xz += w * nx * nz;
	self.xw += w * nx * d;
	self.yy += w * ny * ny;
	self.yz += w * ny * nz;
	self.yw += w * ny * d;
	self.zz += w * nz * nz;
	self.zw += w * nz * d;
	self.ww += w * d * d;
}

fn void Quadric.add(&self, Quadric* q)
{
	self.xx += q.xx;
	self.xy += q.xy;
	self.xz += q.xz;
	self.xw += q.xw;
	self.yy += q.yy;
	self.yz += q.yz;
	self.yw += q.yw;
	self.zz += q.zz;
	self.zw += q.zw;
	self.ww += q.ww;
}

fn double Quadric.error(&self, double x, double y, double z)
{
	double e = self.xx * x * x + 2 * (self.xy * x * y + self.xz * x * z + self.xw * x)
		+ self.yy * y * y + 2 * (self.yz * y * z + self.yw * y)
		+ self.zz * z * z + 2 * self.zw * z + self.ww;
	return e > 0 ? e : 0;
}

// --- Parts -------------------------------------------------------------------------

fn uint find_root(uint[] parent, uint x)
{
	while (parent[x] != x)
	{
		parent[x] = parent[parent[x]];
		x = parent[x];
	}
	return x;
}

<*
 Drop the small pieces far from bigger ones, in place: idx keeps its first
 `faces` faces' order among the faces left.

 @param [in] pos : "Positions [n * 3]"
 @param [inout] idx : "Faces [faces * 3]"
 @require gap > 0 : "A gap is needed"
 @return "The faces left"
*>
fn usz drop_far_parts(float[] pos, uint[] idx, usz faces, usz n, double gap, double keep, usz* parts, usz* dropped)
{
	uint[] parent = mem::alloc_array(uint, (sz)n);
	defer mem::free(parent);
	for (usz i = 0; i < n; i++) parent[i] = (uint)i;
	for (usz f = 0; f < faces; f++)
	{
		uint a = find_root(parent, idx[3 * f]);
		for (usz k = 1; k < 3; k++)
		{
			uint b = find_root(parent, idx[3 * f + k]);
			if (a == b) continue;
			if (a < b)
			{
				parent[b] = a;
			}
			else
			{
				parent[a] = b;
				a = b;
			}
		}
	}
	// pieces numbered by root, their faces counted
	uint[] piece = mem::alloc_array(uint, (sz)n);
	defer mem::free(piece);
	for (usz i = 0; i < n; i++) piece[i] = NO_VERTEX;
	uint count = 0;
	for (usz f = 0; f < faces; f++)
	{
		uint r = find_root(parent, idx[3 * f]);
		if (piece[r] == NO_VERTEX) piece[r] = count++;
	}
	*parts = count;
	if (count < 2) return faces;
	uint[] face_count = mem::new_array(uint, (sz)count);
	defer mem::free(face_count);
	for (usz f = 0; f < faces; f++) face_count[piece[find_root(parent, idx[3 * f])]]++;

	// each piece's vertices (each once), grouped
	char[] seen = mem::new_array(char, (sz)n);
	defer mem::free(seen);
	uint[] start = mem::new_array(uint, (sz)count + 1);
	defer mem::free(start);
	usz used = 0;
	for (usz i = 0; i < 3 * faces; i++)
	{
		uint v = idx[i];
		if (seen[v]) continue;
		seen[v] = 1;
		start[piece[find_root(parent, v)] + 1]++;
		used++;
	}
	for (uint p = 0; p < count; p++) start[p + 1] += start[p];
	uint[] verts = mem::alloc_array(uint, (sz)used);
	defer mem::free(verts);
	uint[] fill = mem::alloc_array(uint, (sz)count);
	defer mem::free(fill);
	for (uint p = 0; p < count; p++) fill[p] = start[p];
	for (usz i = 0; i < 3 * faces; i++)
	{
		uint v = idx[i];
		if (seen[v] != 1) continue;
		seen[v] = 2;
		verts[fill[piece[find_root(parent, v)]]++] = v;
	}

	// the grid the kept pieces mark, a cell `gap` wide
	float[3] lo = { float::max, float::max, float::max };
	float[3] hi = { -float::max, -float::max, -float::max };
	foreach (v : verts)
	{
		for (usz a = 0; a < 3; a++)
		{
			lo[a] = math::min(lo[a], pos[3 * (usz)v + a]);
			hi[a] = math::max(hi[a], pos[3 * (usz)v + a]);
		}
	}
	double cell = gap;
	usz[3] dim;
	for (;;)
	{
		for (usz a = 0; a < 3; a++) dim[a] = (usz)((double)(hi[a] - lo[a]) / cell) + 1;
		if (dim[0] * dim[1] * dim[2] <= 1 << 26) break;
		cell *= 2;
	}
	char[] marked = mem::new_array(char, (sz)(dim[0] * dim[1] * dim[2]));
	defer mem::free(marked);

	// biggest first: kept when big, or near a piece already kept
	ulong[] order = mem::alloc_array(ulong, (sz)count);
	defer mem::free(order);
	for (uint p = 0; p < count; p++) order[p] = (ulong)face_count[p] << 32 | p;
	sort::quicksort(order, fn int(ulong x, ulong y) => x > y ? -1 : (x < y ? 1 : 0));
	char[] drop = mem::new_array(char, (sz)count);
	defer mem::free(drop);
	usz dropped_faces = 0;
	foreach (rank, key : order)
	{
		uint p = (uint)(key & 0xFFFFFFFF);
		uint[] mine = verts[start[p]:start[p + 1] - start[p]];
		bool kept = rank == 0 || face_count[p] >= keep * faces;
		for (usz i = 0; !kept && i < mine.len; i++)
		{
			usz[3] c;
			for (usz a = 0; a < 3; a++) c[a] = (usz)((double)(pos[3 * (usz)mine[i] + a] - lo[a]) / cell);
			for (int dx = -1; !kept && dx <= 1; dx++)
			{
				for (int dy = -1; !kept && dy <= 1; dy++)
				{
					for (int dz = -1; !kept && dz <= 1; dz++)
					{
						long x = (long)c[0] + dx;
						long y = (long)c[1] + dy;
						long z = (long)c[2] + dz;
						if (x < 0 || y < 0 || z < 0 || x >= (long)dim[0] || y >= (long)dim[1] || z >= (long)dim[2]) continue;
						if (marked[((usz)x * dim[1] + (usz)y) * dim[2] + (usz)z]) kept = true;
					}
				}
			}
		}
		if (!kept)
		{
			drop[p] = 1;
			dropped_faces += face_count[p];
			(*dropped)++;
			continue;
		}
		foreach (v : mine)
		{
			usz[3] c;
			for (usz a = 0; a < 3; a++) c[a] = (usz)((double)(pos[3 * (usz)v + a] - lo[a]) / cell);
			marked[(c[0] * dim[1] + c[1]) * dim[2] + c[2]] = 1;
		}
	}
	if (!dropped_faces) return faces;
	usz live = 0;
	for (usz f = 0; f < faces; f++)
	{
		if (drop[piece[find_root(parent, idx[3 * f])]]) continue;
		for (usz k = 0; k < 3; k++) idx[3 * live + k] = idx[3 * f + k];
		live++;
	}
	return live;
}

// --- Coincident vertices -------------------------------------------------------------

<*
 Move every vertex that shares its position with another `amount` along its
 own normal (area-weighted).

 @param [inout] pos : "Positions [n * 3]"
 @param [in] idx : "Faces [faces * 3]"
 @return "How many vertices moved"
*>
fn usz separate_coincident(float[] pos, uint[] idx, usz faces, usz n, double amount)
{
	float[] normal = mem::new_array(float, (sz)(n * 3));
	defer mem::free(normal);
	char[] used = mem::new_array(char, (sz)n);
	defer mem::free(used);
	for (usz f = 0; f < faces; f++)
	{
		uint a = idx[3 * f];
		uint b = idx[3 * f + 1];
		uint c = idx[3 * f + 2];
		double[<3>] pa = { pos[3 * a], pos[3 * a + 1], pos[3 * a + 2] };
		double[<3>] pb = { pos[3 * b], pos[3 * b + 1], pos[3 * b + 2] };
		double[<3>] pc = { pos[3 * c], pos[3 * c + 1], pos[3 * c + 2] };
		double[<3>] nn = (pb - pa).cross(pc - pa);
		foreach (v : (uint[3]){ a, b, c })
		{
			used[v] = 1;
			for (usz k = 0; k < 3; k++) normal[3 * (usz)v + k] += (float)nn[k];
		}
	}
	// positions hashed by their bits: a vertex finding its position taken is shared, and so is the first
	usz size = 1;
	while (size < 2 * n) size <<= 1;
	uint[] slot = mem::alloc_array(uint, (sz)size);
	defer mem::free(slot);
	for (usz i = 0; i < size; i++) slot[i] = NO_VERTEX;
	char[] shared = mem::new_array(char, (sz)n);
	defer mem::free(shared);
	for (usz v = 0; v < n; v++)
	{
		if (!used[v]) continue;
		uint x = bitcast(pos[3 * v], uint);
		uint y = bitcast(pos[3 * v + 1], uint);
		uint z = bitcast(pos[3 * v + 2], uint);
		usz h = (usz)((x * 0x9E3779B1u) ^ (y * 0x85EBCA77u) ^ (z * 0xC2B2AE3Du)) & (size - 1);
		while (slot[h] != NO_VERTEX)
		{
			uint u = slot[h];
			if (bitcast(pos[3 * u], uint) == x && bitcast(pos[3 * u + 1], uint) == y && bitcast(pos[3 * u + 2], uint) == z)
			{
				shared[u] = 1;
				shared[v] = 1;
				break;
			}
			h = (h + 1) & (size - 1);
		}
		if (slot[h] == NO_VERTEX) slot[h] = (uint)v;
	}
	usz moved = 0;
	for (usz v = 0; v < n; v++)
	{
		if (!shared[v]) continue;
		double nx = normal[3 * v];
		double ny = normal[3 * v + 1];
		double nz = normal[3 * v + 2];
		double len = math::sqrt(nx * nx + ny * ny + nz * nz);
		if (!(len > 0)) continue;
		pos[3 * v] += (float)(amount * nx / len);
		pos[3 * v + 1] += (float)(amount * ny / len);
		pos[3 * v + 2] += (float)(amount * nz / len);
		moved++;
	}
	return moved;
}

// --- Smoothing ---------------------------------------------------------------------

<*
 Taubin smoothing, `rounds` times a step toward the mean of each vertex's
 face neighbours (0.5) and one back out (-0.53).

 @param [inout] pos : "Positions [n * 3]"
 @param [in] idx : "Faces [faces * 3]"
*>
fn void taubin(float[] pos, uint[] idx, usz faces, usz n, usz rounds)
{
	double[] sum = mem::alloc_array(double, (sz)(n * 3));
	defer mem::free(sum);
	uint[] count = mem::alloc_array(uint, (sz)n);
	defer mem::free(count);
	for (usz step = 0; step < 2 * rounds; step++)
	{
		double rate = step % 2 == 0 ? 0.5 : -0.53;
		for (usz i = 0; i < n * 3; i++) sum[i] = 0;
		for (usz i = 0; i < n; i++) count[i] = 0;
		for (usz f = 0; f < faces; f++)
		{
			for (usz k = 0; k < 3; k++)
			{
				usz v = idx[3 * f + k];
				usz u = idx[3 * f + (k + 1) % 3];
				usz w = idx[3 * f + (k + 2) % 3];
				for (usz a = 0; a < 3; a++) sum[3 * v + a] += (double)pos[3 * u + a] + pos[3 * w + a];
				count[v] += 2;
			}
		}
		for (usz v = 0; v < n; v++)
		{
			if (!count[v]) continue;
			for (usz a = 0; a < 3; a++)
			{
				double p = pos[3 * v + a];
				pos[3 * v + a] = (float)(p + rate * (sum[3 * v + a] / count[v] - p));
			}
		}
	}
}

// --- Decimation ----------------------------------------------------------------------

<*
 Whether the faces around v make one closed fan: each edge out of v is run
 the other way by exactly one face, and following them goes round all the
 faces once.
*>
fn bool closed_fan(uint[] idx, uint[] around, uint v)
{
	usz k = around.len;
	if (k < 3 || k > MAX_FAN) return false;
	uint[MAX_FAN] outs;
	uint[MAX_FAN] ins;
	char[MAX_FAN] next;
	foreach (i, f : around)
	{
		usz c = 3 * (usz)f;
		if (idx[c] == v)
		{
			outs[i] = idx[c + 1];
			ins[i] = idx[c + 2];
		}
		else if (idx[c + 1] == v)
		{
			outs[i] = idx[c + 2];
			ins[i] = idx[c];
		}
		else
		{
			outs[i] = idx[c];
			ins[i] = idx[c + 1];
		}
	}
	for (usz i = 0; i < k; i++)
	{
		int found = -1;
		for (usz j = 0; j < k; j++)
		{
			if (ins[j] != outs[i]) continue;
			if (found >= 0) return false;
			found = (int)j;
		}
		if (found < 0) return false;
		next[i] = (char)found;
	}
	usz steps = 0;
	usz at = 0;
	do
	{
		at = next[at];
		steps++;
	}
	while (at != 0 && steps <= k);
	return steps == k;
}

<*
 The vertices on the faces in `around` (remapped, degenerate faces skipped),
 other than v, each once, into `out`.

 @return "How many, or MAX_FAN + 1 when they do not fit"
*>
fn usz ring(uint[] idx, uint[] around, uint[] remap, uint v, uint[] out)
{
	usz m = 0;
	foreach (f : around)
	{
		uint[3] r = { remap[idx[3 * f]], remap[idx[3 * f + 1]], remap[idx[3 * f + 2]] };
		if (r[0] == r[1] || r[1] == r[2] || r[0] == r[2]) continue;
		foreach (w : r)
		{
			if (w == v) continue;
			bool have = false;
			for (usz i = 0; i < m; i++) if (out[i] == w) have = true;
			if (have) continue;
			if (m == out.len) return MAX_FAN + 1;
			out[m++] = w;
		}
	}
	return m;
}

<*
 The least dot product between the unit normals in normals[:count].
*>
fn double spread(double[<3>][] normals, usz count)
{
	double least = 1;
	for (usz i = 0; i < count; i++)
	{
		for (usz j = i + 1; j < count; j++) least = math::min(least, normals[i].dot(normals[j]));
	}
	return least;
}

<*
 Whether moving a onto b is refused: a face of a's would turn by more than
 the limit or become degenerate, or b's faces afterwards would face further
 apart than a's or b's did before. Quadrics do not know which way a plane
 faces, so without that the two sides of a thin plate look like one plane
 and their vertices get shared, which leaves them no normal.
*>
fn bool refused(float[] pos, uint[] idx, uint[] fa, uint[] fb, uint[] remap, uint a, uint b)
{
	double[<3>][MAX_FAN] before_a;
	double[<3>][MAX_FAN] before_b;
	double[<3>][2 * MAX_FAN] after;
	usz na = 0;
	usz nb = 0;
	usz m = 0;
	double[<3>] pb = { pos[3 * b], pos[3 * b + 1], pos[3 * b + 2] };
	foreach (f : fa)
	{
		uint[3] r = { remap[idx[3 * f]], remap[idx[3 * f + 1]], remap[idx[3 * f + 2]] };
		if (r[0] == r[1] || r[1] == r[2] || r[0] == r[2]) continue;
		double[<3>][3] p;
		double[<3>][3] q;
		for (usz k = 0; k < 3; k++)
		{
			p[k] = { pos[3 * r[k]], pos[3 * r[k] + 1], pos[3 * r[k] + 2] };
			q[k] = r[k] == a ? pb : p[k];
		}
		double[<3>] old = (p[1] - p[0]).cross(p[2] - p[0]);
		double lo = old.length();
		if (lo > 0 && na < MAX_FAN) before_a[na++] = old / lo;
		if (r[0] == b || r[1] == b || r[2] == b) continue;
		double[<3>] moved = (q[1] - q[0]).cross(q[2] - q[0]);
		double lm = moved.length();
		if (!(lm > 1e-30)) return true;
		if (old.dot(moved) <= MIN_TURN_COS * lo * lm) return true;
		after[m++] = moved / lm;
	}
	foreach (f : fb)
	{
		uint[3] r = { remap[idx[3 * f]], remap[idx[3 * f + 1]], remap[idx[3 * f + 2]] };
		if (r[0] == r[1] || r[1] == r[2] || r[0] == r[2]) continue;
		double[<3>][3] p;
		for (usz k = 0; k < 3; k++) p[k] = { pos[3 * r[k]], pos[3 * r[k] + 1], pos[3 * r[k] + 2] };
		double[<3>] nn = (p[1] - p[0]).cross(p[2] - p[0]);
		double len = nn.length();
		if (!(len > 0)) continue;
		if (nb < MAX_FAN) before_b[nb++] = nn / len;
		if (r[0] == a || r[1] == a || r[2] == a) continue;
		if (m < 2 * MAX_FAN) after[m++] = nn / len;
	}
	double now = spread(after[..], m);
	if (now >= 0) return false;
	return now < math::min(spread(before_a[..], na), spread(before_b[..], nb)) - 0.1;
}

<*
 Order of `keys` ascending (stable LSD radix, two 16-bit passes) into `order`.
*>
fn void sort_keys(uint[] keys, uint[] order)
{
	usz m = keys.len;
	uint[] tmp = mem::alloc_array(uint, (sz)m);
	defer mem::free(tmp);
	uint[] count = mem::alloc_array(uint, 65537);
	defer mem::free(count);
	for (usz pass = 0; pass < 2; pass++)
	{
		uint shift = pass == 0 ? 0 : 16;
		for (usz i = 0; i < count.len; i++) count[i] = 0;
		for (usz i = 0; i < m; i++)
		{
			uint key = pass == 0 ? keys[i] : keys[tmp[i]];
			count[((key >> shift) & 0xFFFF) + 1]++;
		}
		for (usz d = 0; d < 65536; d++) count[d + 1] += count[d];
		for (usz i = 0; i < m; i++)
		{
			uint item = pass == 0 ? (uint)i : tmp[i];
			uint key = keys[item];
			uint[] dst = pass == 0 ? tmp : order;
			dst[count[(key >> shift) & 0xFFFF]++] = item;
		}
	}
}

<*
 Decimate idx [faces * 3] in place toward `target` faces.

 @return "The faces left"
*>
fn usz decimate(float[] pos, uint[] idx, usz faces, usz n, usz target, usz* passes)
{
	Quadric[] quadric = mem::new_array(Quadric, (sz)n);
	defer mem::free(quadric);
	for (usz f = 0; f < faces; f++)
	{
		uint[3] v = { idx[3 * f], idx[3 * f + 1], idx[3 * f + 2] };
		double[<3>] a = { pos[3 * v[0]], pos[3 * v[0] + 1], pos[3 * v[0] + 2] };
		double[<3>] b = { pos[3 * v[1]], pos[3 * v[1] + 1], pos[3 * v[1] + 2] };
		double[<3>] c = { pos[3 * v[2]], pos[3 * v[2] + 1], pos[3 * v[2] + 2] };
		double[<3>] nn = (b - a).cross(c - a);
		double len = nn.length();
		if (!(len > 0)) continue;
		nn /= len;
		double d = -nn.dot(a);
		foreach (w : v) quadric[w].add_plane(nn[0], nn[1], nn[2], d, len / 2);
	}
	uint[] offset = mem::alloc_array(uint, (sz)n + 1);
	defer mem::free(offset);
	uint[] remap = mem::alloc_array(uint, (sz)n);
	defer mem::free(remap);
	char[] fixed = mem::alloc_array(char, (sz)n);
	defer mem::free(fixed);
	char[] locked = mem::alloc_array(char, (sz)n);
	defer mem::free(locked);
	uint[] around = mem::alloc_array(uint, (sz)(3 * faces));
	defer mem::free(around);
	usz most = 3 * faces / 2 + 3;
	uint[] from = mem::alloc_array(uint, (sz)most);
	defer mem::free(from);
	uint[] to = mem::alloc_array(uint, (sz)most);
	defer mem::free(to);
	uint[] key = mem::alloc_array(uint, (sz)most);
	defer mem::free(key);
	uint[] order = mem::alloc_array(uint, (sz)most);
	defer mem::free(order);
	double[] cost = mem::alloc_array(double, (sz)most);
	defer mem::free(cost);
	uint[MAX_FAN + 1] ring_a;
	uint[MAX_FAN + 1] ring_b;

	while (faces > target)
	{
		(*passes)++;
		// faces around each vertex
		for (usz i = 0; i <= n; i++) offset[i] = 0;
		for (usz i = 0; i < 3 * faces; i++) offset[idx[i] + 1]++;
		for (usz i = 0; i < n; i++) offset[i + 1] += offset[i];
		for (usz f = 0; f < faces; f++)
		{
			for (usz k = 0; k < 3; k++) around[offset[idx[3 * f + k]]++] = (uint)f;
		}
		for (usz i = n; i > 0; i--) offset[i] = offset[i - 1];
		offset[0] = 0;
		for (usz v = 0; v < n; v++)
		{
			fixed[v] = offset[v + 1] > offset[v] && !closed_fan(idx, around[offset[v]..offset[v + 1] - 1], (uint)v) ? 1 : 0;
		}
		// each edge once (its a < b half), in its cheaper allowed direction
		usz m = 0;
		for (usz f = 0; f < faces; f++)
		{
			for (usz k = 0; k < 3; k++)
			{
				uint a = idx[3 * f + k];
				uint b = idx[3 * f + (k + 1) % 3];
				if (a > b) continue;
				double ax = pos[3 * a];
				double ay = pos[3 * a + 1];
				double az = pos[3 * a + 2];
				double bx = pos[3 * b];
				double by = pos[3 * b + 1];
				double bz = pos[3 * b + 2];
				double onto_b = fixed[a] ? NEVER : quadric[a].error(bx, by, bz) + quadric[b].error(bx, by, bz);
				double onto_a = fixed[b] ? NEVER : quadric[a].error(ax, ay, az) + quadric[b].error(ax, ay, az);
				if (onto_b >= NEVER && onto_a >= NEVER) continue;
				if (m == most) break;
				bool keep_b = onto_b <= onto_a;
				from[m] = keep_b ? a : b;
				to[m] = keep_b ? b : a;
				cost[m] = keep_b ? onto_b : onto_a;
				key[m] = bitcast((float)cost[m], uint);
				m++;
			}
		}
		if (m == 0) break;
		sort_keys(key[:m], order[:m]);
		// meshoptimizer's pace: about the edges this pass needs, and those not much dearer
		usz goal = faces - target;
		usz edge_goal = goal / 2;
		double limit = edge_goal < m ? 1.5 * cost[order[edge_goal]] : NEVER;
		usz collapses = 0;
		for (usz attempt = 0; attempt < 2 && collapses == 0; attempt++)
		{
			if (attempt == 1) limit = NEVER;
			for (usz v = 0; v < n; v++) remap[v] = (uint)v;
			for (usz v = 0; v < n; v++) locked[v] = 0;
			usz removed = 0;
			for (usz i = 0; i < m && removed < goal; i++)
			{
				uint e = order[i];
				if (cost[e] > limit) break;
				uint a = from[e];
				uint b = to[e];
				if (locked[a] || locked[b]) continue;
				uint[] fa = around[offset[a]..offset[a + 1] - 1];
				uint[] fb = around[offset[b]..offset[b + 1] - 1];
				usz na = ring(idx, fa, remap, a, ring_a[..]);
				usz nb = ring(idx, fb, remap, b, ring_b[..]);
				if (na > MAX_FAN || nb > MAX_FAN) continue;
				usz common = 0;
				for (usz x = 0; x < na; x++)
				{
					if (ring_a[x] == b) continue;
					for (usz y = 0; y < nb; y++) if (ring_b[y] == ring_a[x]) common++;
				}
				if (common != 2) continue;
				if (refused(pos, idx, fa, fb, remap, a, b)) continue;
				foreach (f : fa)
				{
					uint[3] r = { remap[idx[3 * f]], remap[idx[3 * f + 1]], remap[idx[3 * f + 2]] };
					if (r[0] == r[1] || r[1] == r[2] || r[0] == r[2]) continue;
					if (r[0] == b || r[1] == b || r[2] == b) removed++;
				}
				remap[a] = b;
				locked[a] = 1;
				locked[b] = 1;
				quadric[b].add(&quadric[a]);
				collapses++;
			}
		}
		if (collapses == 0) break;
		usz live = 0;
		for (usz f = 0; f < faces; f++)
		{
			uint[3] r = { remap[idx[3 * f]], remap[idx[3 * f + 1]], remap[idx[3 * f + 2]] };
			if (r[0] == r[1] || r[1] == r[2] || r[0] == r[2]) continue;
			for (usz k = 0; k < 3; k++) idx[3 * live + k] = r[k];
			live++;
		}
		faces = live;
	}
	return faces;
}

// --- Normals -----------------------------------------------------------------------

const usz MAX_GROUPS = 32;

<*
 Normals split at creases: each vertex's faces grouped greedily (a face joins
 the group whose normal is nearest, if within `cos_crease`), each group a
 vertex of its own. idx is rewritten to the new vertices.

 @param [in] pos : "Positions [m * 3]"
 @param [inout] idx : "Faces [faces * 3]"
 @return "Positions and normals [count * 3] each, which the caller frees"
*>
fn float[][2] creased_normals(float[] pos, uint[] idx, usz faces, usz m, double cos_crease)
{
	double[<3>][] area_normal = mem::alloc_array(double[<3>], (sz)faces);
	defer mem::free(area_normal);
	for (usz f = 0; f < faces; f++)
	{
		double[<3>][3] p;
		for (usz k = 0; k < 3; k++) p[k] = { pos[3 * (usz)idx[3 * f + k]], pos[3 * (usz)idx[3 * f + k] + 1], pos[3 * (usz)idx[3 * f + k] + 2] };
		area_normal[f] = (p[1] - p[0]).cross(p[2] - p[0]);
	}
	uint[] offset = mem::new_array(uint, (sz)m + 1);
	defer mem::free(offset);
	uint[] around = mem::alloc_array(uint, (sz)(3 * faces));
	defer mem::free(around);
	for (usz i = 0; i < 3 * faces; i++) offset[idx[i] + 1]++;
	for (usz i = 0; i < m; i++) offset[i + 1] += offset[i];
	for (usz i = 0; i < 3 * faces; i++) around[offset[idx[i]]++] = (uint)i;
	for (usz i = m; i > 0; i--) offset[i] = offset[i - 1];
	offset[0] = 0;

	// each corner's new vertex; groups numbered as they are made
	uint[] corner = mem::alloc_array(uint, (sz)(3 * faces));
	defer mem::free(corner);
	usz count = 0;
	usz room = m + m / 4 + 16;
	double[<3>][] sum = mem::alloc_array(double[<3>], (sz)room);
	defer mem::free(sum);
	uint[] origin = mem::alloc_array(uint, (sz)room);
	defer mem::free(origin);
	for (usz v = 0; v < m; v++)
	{
		usz first = count;
		usz groups = 0;
		for (usz j = offset[v]; j < offset[v + 1]; j++)
		{
			uint c = around[j];
			double[<3>] an = area_normal[c / 3];
			double len = an.length();
			usz best = first;
			double best_dot = -2;
			for (usz g = first; g < first + groups; g++)
			{
				double sl = sum[g].length();
				double d = len > 0 && sl > 0 ? an.dot(sum[g]) / (len * sl) : 1;
				if (d > best_dot)
				{
					best_dot = d;
					best = g;
				}
			}
			if (groups == 0 || (best_dot < cos_crease && groups < MAX_GROUPS))
			{
				if (count == room)
				{
					usz bigger = room * 2;
					double[<3>][] s2 = mem::alloc_array(double[<3>], (sz)bigger);
					s2[:count] = sum[:count];
					mem::free(sum);
					sum = s2;
					uint[] o2 = mem::alloc_array(uint, (sz)bigger);
					o2[:count] = origin[:count];
					mem::free(origin);
					origin = o2;
					room = bigger;
				}
				best = count++;
				groups++;
				sum[best] = { 0, 0, 0 };
				origin[best] = (uint)v;
			}
			sum[best] += an;
			corner[c] = (uint)best;
		}
	}
	float[] out_pos = mem::alloc_array(float, (sz)(3 * count));
	float[] out_nrm = mem::alloc_array(float, (sz)(3 * count));
	for (usz g = 0; g < count; g++)
	{
		double len = sum[g].length();
		for (usz k = 0; k < 3; k++)
		{
			out_pos[3 * g + k] = pos[3 * (usz)origin[g] + k];
			out_nrm[3 * g + k] = len > 0 ? (float)(sum[g][k] / len) : (k == 2 ? 1f : 0f);
		}
	}
	for (usz i = 0; i < 3 * faces; i++) idx[i] = corner[i];
	return { out_pos, out_nrm };
}

// --- The verb ------------------------------------------------------------------------

fn js::Value? js_mesh_simplify(js::Engine* js, float[] positions, uint[] indices, double[] params)
{
	if (positions.len % 3 || indices.len % 3) return js.fail("meshSimplify: positions and indices come in threes")~;
	if (params.len < 6) return js.fail("meshSimplify: params are target faces, part gap, part keep, separate, smooth, crease")~;
	usz n = positions.len / 3;
	usz faces = indices.len / 3;
	foreach (v : indices) if (v >= n) return js.fail("meshSimplify: an index (%d) past the %d vertices", v, n)~;
	usz target = params[0] > 0 ? (usz)params[0] : 0;
	float[] pos = mem::alloc_array(float, (sz)positions.len);
	defer mem::free(pos);
	pos[..] = positions[..];
	uint[] idx = mem::alloc_array(uint, (sz)indices.len);
	defer mem::free(idx);
	idx[..] = indices[..];

	usz parts = 0;
	usz dropped = 0;
	usz separated = 0;
	usz passes = 0;
	double[5] seconds;
	Clock t = clock::now();
	if (params[1] > 0) faces = drop_far_parts(pos, idx, faces, n, params[1], params[2], &parts, &dropped);
	seconds[0] = t.mark().to_sec();
	if (params[3] > 0) separated = separate_coincident(pos, idx, faces, n, params[3]);
	seconds[1] = t.mark().to_sec();
	if (params[4] > 0) taubin(pos, idx, faces, n, (usz)params[4]);
	seconds[2] = t.mark().to_sec();
	if (target > 0 && faces > target) faces = decimate(pos, idx, faces, n, target, &passes);
	seconds[3] = t.mark().to_sec();

	// the vertices still used, renumbered in order of first use
	uint[] renumber = mem::alloc_array(uint, (sz)n);
	defer mem::free(renumber);
	for (usz v = 0; v < n; v++) renumber[v] = NO_VERTEX;
	uint m = 0;
	for (usz i = 0; i < 3 * faces; i++)
	{
		if (renumber[idx[i]] == NO_VERTEX) renumber[idx[i]] = m++;
		idx[i] = renumber[idx[i]];
	}
	float[] out = mem::alloc_array(float, (sz)m * 3);
	defer mem::free(out);
	for (usz v = 0; v < n; v++)
	{
		uint w = renumber[v];
		if (w == NO_VERTEX) continue;
		for (usz k = 0; k < 3; k++) out[3 * (usz)w + k] = pos[3 * v + k];
	}

	js::Value result = js.object()!;
	if (params[5] > 0)
	{
		float[][2] split = creased_normals(out, idx, faces, m, math::cos(params[5] * math::PI / 180));
		defer mem::free(split[0]);
		defer mem::free(split[1]);
		seconds[4] = t.mark().to_sec();
		js.set(result, "positions", js.typed_copy(FLOAT32, ((char*)split[0].ptr)[:split[0].len * 4])!)!;
		js.set(result, "normals", js.typed_copy(FLOAT32, ((char*)split[1].ptr)[:split[1].len * 4])!)!;
	}
	else
	{
		js.set(result, "positions", js.typed_copy(FLOAT32, ((char*)out.ptr)[:out.len * 4])!)!;
		js.set(result, "normals", js::NULL)!;
	}
	js.set(result, "indices", js.typed_copy(UINT32, ((char*)idx.ptr)[:faces * 12])!)!;
	js.set(result, "parts", js::number(parts))!;
	js.set(result, "dropped", js::number(dropped))!;
	js.set(result, "separated", js::number(separated))!;
	js.set(result, "passes", js::number(passes))!;
	js::Value times = js.array(5)!;
	foreach (sec : seconds) js.push(times, js::number(sec));
	js.set(result, "seconds", times)!;
	return result;
}

// --- The shell's surface -----------------------------------------------------------

fn js::Value? js_mesh_shell_surface(js::Engine* js, int[] coords, float[] fields, long r, uint[] table, uint[] outside, long g_in, long words_in)
{
	usz n = coords.len / 3;
	usz g = (usz)g_in;
	usz words = (usz)words_in;
	if (coords.len % 3 || fields.len < 7 * n || table.len < 27 * n) return js.fail("meshShellSurface: coords [n, 3], fields [n, 7] and table [n, 27] disagree")~;
	if (outside.len < g * g * words || g * g * g >= (usz)uint::max) return js.fail("meshShellSurface: the outside bits do not cover a %d^3 grid", g)~;
	// each voxel's predicted vertex, in [-0.5, 0.5]^3
	float[] vertex = mem::alloc_array(float, (sz)(3 * n));
	defer mem::free(vertex);
	for (usz i = 0; i < n; i++)
	{
		for (usz a = 0; a < 3; a++)
		{
			double sig = 1 / (1 + math::exp(-(double)fields[7 * i + a]));
			vertex[3 * i + a] = (float)((coords[3 * i + a] + 2 * sig - 0.5) / r - 0.5);
		}
	}
	// faces toward the outside, counted first
	usz quads = 0;
	for (usz v = 0; v < n; v++)
	{
		for (usz a = 0; a < 3; a++)
		{
			for (int s = -1; s <= 1; s += 2)
			{
				if (cell_outside(outside, g, words, coords, v, a, s)) quads++;
			}
		}
	}
	// cubes by their min-corner cell on the padded grid: key -> vertex
	usz size = 1;
	while (size < 4 * n) size <<= 1;
	uint[] keys = mem::alloc_array(uint, (sz)size);
	defer mem::free(keys);
	uint[] ids = mem::alloc_array(uint, (sz)size);
	defer mem::free(ids);
	for (usz i = 0; i < size; i++) keys[i] = uint::max;
	float[] positions = mem::alloc_array(float, (sz)(3 * 4 * quads + 3));
	defer mem::free(positions);
	uint[] faces = mem::alloc_array(uint, (sz)(6 * quads + 1));
	defer mem::free(faces);
	usz nv = 0;
	usz nf = 0;
	int[2][4] around_edge = { { -1, -1 }, { 0, -1 }, { 0, 0 }, { -1, 0 } };
	for (usz v = 0; v < n; v++)
	{
		for (usz a = 0; a < 3; a++)
		{
			usz b = (a + 1) % 3;
			usz c = (a + 2) % 3;
			for (int s = -1; s <= 1; s += 2)
			{
				if (!cell_outside(outside, g, words, coords, v, a, s)) continue;
				// the cubes around the edge v -> v + s e_a, counter-clockwise about +e_a
				uint[4] q;
				for (usz k = 0; k < 4; k++)
				{
					int[3] off;
					off[a] = s > 0 ? 0 : -1;
					off[b] = around_edge[k][0];
					off[c] = around_edge[k][1];
					ulong key = ((ulong)(coords[3 * v] + off[0] + 1) * g + (ulong)(coords[3 * v + 1] + off[1] + 1)) * g + (ulong)(coords[3 * v + 2] + off[2] + 1);
					usz h = (usz)(((uint)key * 0x9E3779B1u) & (uint)(size - 1));
					while (keys[h] != uint::max && keys[h] != (uint)key) h = (h + 1) & (size - 1);
					if (keys[h] == (uint)key)
					{
						q[k] = ids[h];
						continue;
					}
					keys[h] = (uint)key;
					ids[h] = (uint)nv;
					// the mean of the shell voxels' vertices at the cube's corners
					double[3] sum;
					usz count = 0;
					for (int i = 0; i <= 1; i++)
					{
						for (int j = 0; j <= 1; j++)
						{
							for (int l = 0; l <= 1; l++)
							{
								uint w = table[27 * v + (usz)((off[0] + i + 1) * 9 + (off[1] + j + 1) * 3 + off[2] + l + 1)];
								if (w == uint::max) continue;
								for (usz d = 0; d < 3; d++) sum[d] += vertex[3 * (usz)w + d];
								count++;
							}
						}
					}
					for (usz d = 0; d < 3; d++) positions[3 * nv + d] = (float)(sum[d] / count);
					q[k] = (uint)nv++;
				}
				// facing +e_a as it stands; reversed when the outside is at -e_a
				if (s > 0)
				{
					faces[nf++] = q[0]; faces[nf++] = q[1]; faces[nf++] = q[2];
					faces[nf++] = q[0]; faces[nf++] = q[2]; faces[nf++] = q[3];
				}
				else
				{
					faces[nf++] = q[0]; faces[nf++] = q[2]; faces[nf++] = q[1];
					faces[nf++] = q[0]; faces[nf++] = q[3]; faces[nf++] = q[2];
				}
			}
		}
	}
	js::Value result = js.object()!;
	js.set(result, "positions", js.typed_copy(FLOAT32, ((char*)positions.ptr)[:nv * 12])!)!;
	js.set(result, "indices", js.typed_copy(UINT32, ((char*)faces.ptr)[:nf * 4])!)!;
	return result;
}

<*
 Whether the cell beside voxel v, s along axis a, is outside (the padded
 grid's bits).
*>
fn bool cell_outside(uint[] outside, usz g, usz words, int[] coords, usz v, usz a, int s) @inline
{
	long[3] p = { (long)coords[3 * v] + 1, (long)coords[3 * v + 1] + 1, (long)coords[3 * v + 2] + 1 };
	p[a] += s;
	usz z = (usz)p[2];
	return (outside[((usz)p[0] * g + (usz)p[1]) * words + (z >> 5)] >> (z & 31)) & 1 == 1;
}
