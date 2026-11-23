<*
 Hi3D's surface voxels of a mesh, on the host (`llm/plugins/hi3d/voxels.js`):
 sparc3d's `adaptive_vox`, every cell of a res^3 grid that a triangle touches.

 `meshVoxelize(positions, indices, params)`: positions a Float32Array [n, 3],
 indices a Uint32Array [f, 3], params a Float64Array [resolution, budget].
 The mesh's bounding box (grown by 1e-6) is centred in the grid and its longest
 side scaled to resolution - 6 (3 empty cells each side). While more than
 `budget` cells are hit, the grid shrinks to int(resolution * 0.95^k) and the
 mesh is voxelized again. The arithmetic is float32 in torch's order, so the
 cells are sparc3d's; the triangle-box test is collision.c3's separating-axis
 one (the same 13 axes as sparc3d's triton kernel, touching counts).

 Returns { coords: Int32Array [m, 3] (x, y, z, ordered by z, then y, then x,
 as sparc3d's sorted keys), grid, center: Float32Array [3], scale, tries }:
 a vertex p is at voxel coordinates (p - center) * scale + grid / 2.
*>
module llm::host;

import fastjs::js;
import std::math;
import collision;

const int VOXEL_PAD = 3;
const float VOXEL_EPS = 1e-6f;

<*
 The hit cells of one grid try as bits, x fastest; returns how many.

 @require res > 0 && res <= 2048 : "Grid out of range"
*>
fn usz voxelize_bits(float[] norm, uint[] indices, int res, ulong[] bits)
{
	bits[..] = 0;
	usz count = 0;
	usz faces = indices.len / 3;
	ulong r = (ulong)res;
	for (usz f = 0; f < faces; f++)
	{
		TriangleVerts tri;
		for (usz k = 0; k < 3; k++)
		{
			usz v = 3 * (usz)indices[3 * f + k];
			tri[k] = { norm[v], norm[v + 1], norm[v + 2] };
		}
		Aabb3 bounds = tri.aabb();
		int[3] lo;
		int[3] hi;
		for (usz a = 0; a < 3; a++)
		{
			lo[a] = math::max((int)math::floor(bounds.min[a]), 0);
			hi[a] = math::min((int)math::floor(bounds.max[a]), res - 1);
		}
		for (int z = lo[2]; z <= hi[2]; z++)
		{
			for (int y = lo[1]; y <= hi[1]; y++)
			{
				for (int x = lo[0]; x <= hi[0]; x++)
				{
					ulong cell = ((ulong)z * r + (ulong)y) * r + (ulong)x;
					ulong bit = 1ul << (cell & 63);
					if (bits[cell >> 6] & bit) continue;
					Aabb3 box = { .min = { (float)x, (float)y, (float)z }, .max = { (float)(x + 1), (float)(y + 1), (float)(z + 1) } };
					if (!tri.overlaps_aabb(box)) continue;
					bits[cell >> 6] |= bit;
					count++;
				}
			}
		}
	}
	return count;
}

<*
 The vertices in voxel coordinates for a grid of `res`, as MeshVoxelizer's
 float32 tensors compute them; `center` and `scale` are written out.
*>
fn void voxel_frame(float[] positions, int res, float[] norm, float[<3>]* center, float* scale)
{
	usz n = positions.len / 3;
	float[<3>] lo = { float::max, float::max, float::max };
	float[<3>] hi = { -float::max, -float::max, -float::max };
	for (usz v = 0; v < n; v++)
	{
		for (usz a = 0; a < 3; a++)
		{
			lo[a] = math::min(lo[a], positions[3 * v + a]);
			hi[a] = math::max(hi[a], positions[3 * v + a]);
		}
	}
	float[<3>] elo = lo - VOXEL_EPS;
	float[<3>] ehi = hi + VOXEL_EPS;
	*center = (elo + ehi) / 2.0f;
	float[<3>] range = ehi - elo;
	float longest = math::max(range.x, math::max(range.y, range.z));
	// torch's scalar / tensor is tensor.reciprocal() * scalar
	*scale = (1.0f / longest) * (float)((double)res - (double)(VOXEL_PAD * 2) - 2e-6);
	float half = (float)((double)res / 2.0);
	for (usz v = 0; v < n; v++)
	{
		for (usz a = 0; a < 3; a++) norm[3 * v + a] = ((positions[3 * v + a] - (*center)[a]) * *scale + half) + 0.0f;
	}
}

fn js::Value? js_mesh_voxelize(js::Engine* js, float[] positions, uint[] indices, double[] params)
{
	if (positions.len % 3 || indices.len % 3) return js.fail("meshVoxelize: positions and indices come in threes")~;
	if (params.len < 2) return js.fail("meshVoxelize: params are resolution, budget")~;
	usz n = positions.len / 3;
	if (n == 0 || indices.len == 0) return js.fail("meshVoxelize: an empty mesh")~;
	foreach (v : indices) if (v >= n) return js.fail("meshVoxelize: an index (%d) past the %d vertices", v, n)~;
	int resolution = (int)params[0];
	usz budget = (usz)params[1];
	if (resolution < 2 * VOXEL_PAD + 1 || resolution > 2048) return js.fail("meshVoxelize: resolution %d is outside [7, 2048]", resolution)~;
	if (budget < 1) return js.fail("meshVoxelize: the budget must be positive")~;

	float[] norm = mem::alloc_array(float, (sz)positions.len);
	defer mem::free(norm);
	usz words = ((usz)resolution * (usz)resolution * (usz)resolution + 63) / 64;
	ulong[] bits = mem::alloc_array(ulong, (sz)words);
	defer mem::free(bits);

	double shrink = 1.0;
	int grid;
	usz count;
	usz tries = 0;
	float[<3>] center;
	float scale;
	while (true)
	{
		grid = (int)((double)resolution * shrink);
		if (grid < 2 * VOXEL_PAD + 1) return js.fail("meshVoxelize: no grid down to %d holds the mesh in %d voxels", grid, budget)~;
		voxel_frame(positions, grid, norm, &center, &scale);
		count = voxelize_bits(norm, indices, grid, bits[:((usz)grid * (usz)grid * (usz)grid + 63) / 64]);
		tries++;
		if (count <= budget) break;
		shrink *= 0.95;
	}

	int[] coords = mem::alloc_array(int, (sz)(3 * count + 3));
	defer mem::free(coords);
	usz g = (usz)grid;
	usz m = 0;
	for (usz cell = 0; cell < g * g * g; cell++)
	{
		if (!(bits[cell >> 6] & (1ul << (cell & 63)))) continue;
		coords[m++] = (int)(cell % g);
		coords[m++] = (int)((cell / g) % g);
		coords[m++] = (int)(cell / (g * g));
	}
	float[3] c = { center.x, center.y, center.z };
	js::Value result = js.object()!;
	js.set(result, "coords", js.typed_copy(INT32, ((char*)coords.ptr)[:m * 4])!)!;
	js.set(result, "grid", js::number(grid))!;
	js.set(result, "center", js.typed_copy(FLOAT32, ((char*)&c)[:12])!)!;
	js.set(result, "scale", js::number(scale))!;
	js.set(result, "tries", js::number(tries))!;
	return result;
}
