// Dense meshes cleaned up and decimated in native code
// (lib/host/mesh_simplify.c3): the interpreter would take minutes over the
// millions of faces a 1024^3 surface has. In order:
//
//   partGap    pieces with less than `partKeep` of the faces, none of whose
//              vertices comes within about partGap (model units) of a
//              bigger piece kept, are dropped: floaters go, a strand of
//              hair beside the head stays (0: every piece stays)
//   separate   vertices sharing a position with another (the two sides of a
//              one-voxel plate) each move this far along their own normal,
//              so double-sided viewers do not show the sides fighting
//   smooth     rounds of Taubin smoothing (in, then back out: the volume
//              stays), so a voxel surface's sub-voxel staircase does not
//              come out of decimation as creases along its contours
//   faces      quadric-error edge collapses down to about this many faces
//              (0: no decimation); vertices only ever move onto neighbours
//   crease     normals (degrees; 0: none): each vertex split where its faces
//              turn by more than this, so sharp edges and thin rims shade
//              sharp instead of averaging faces that point apart
//
// Returns { positions, indices, normals (or null), parts, dropped, separated,
// passes, seconds } (seconds per step: parts, separate, smooth, decimate,
// normals).

import { llm } from '/core/llm.js';

export function simplifyMesh(positions, indices, { faces = 0, partGap = 0, partKeep = 0.01, separate = 0, smooth = 0, crease = 0 } = {}) {
	if (!llm.meshNative) throw new Error('this llm-runner has no meshSimplify (lib/host/mesh_simplify.c3): rebuild it');
	const p = positions instanceof Float32Array ? positions : Float32Array.from(positions);
	const i = indices instanceof Uint32Array ? indices : Uint32Array.from(indices);
	return llm.meshSimplify(p, i, Float64Array.of(faces, partGap, partKeep, separate, smooth, crease));
}
