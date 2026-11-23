// The voxels Hi3D refines inside: sparc3d's `adaptive_vox` of a mesh, in native
// code (lib/host/mesh_voxelize.c3). Every cell of a resolution^3 grid that a
// triangle touches, with the mesh's bounding box centred and its longest side
// filling all but 3 cells each side; while more than `budget` cells are hit,
// the grid shrinks to int(resolution * 0.95^k) and the mesh goes again.
//
// Returns { coords: Int32Array [m, 3] (x, y, z; z slowest, x fastest, as
// sparc3d orders them), grid, center: Float32Array [3], scale, tries }: a
// vertex p lands at (p - center) * scale + grid / 2 in voxel units.

import { llm } from '/core/llm.js';

export function surfaceVoxels(positions, indices, { resolution = 128, budget = 90000 } = {}) {
	if (typeof llm.meshVoxelize !== 'function') throw new Error('this llm-runner has no meshVoxelize (lib/host/mesh_voxelize.c3): rebuild it');
	const p = positions instanceof Float32Array ? positions : Float32Array.from(positions);
	const i = indices instanceof Uint32Array ? indices : Uint32Array.from(indices);
	return llm.meshVoxelize(p, i, Float64Array.of(resolution, budget));
}
