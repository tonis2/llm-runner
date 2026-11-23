// Hi3D v1.0 (github.com/hi3d-official/Hi3Dv1.0, a Sparc3D refiner): a picture
// and a coarse mesh of it to a finer mesh, regenerated inside the mesh's voxels.
// One pass of sparc3d's refinement ladder, the 128^3 -> 1024^3 one
// (`refine_voxels(image, coords, 128, 1024)`):
//
//   mesh -> its surface voxels on a 128^3 grid (shrunk until at most `budget`)
//   picture -> squared up on white -> DINOv2-large at 756 (all 2917 tokens)
//      beside CLIP ViT-L/14's pooled embedding, each L2-normalised
//   -> the flow (Heun, 25 steps) on those voxels -> the latent
//   -> the decoder: signs, distances and deformations on a 1024^3 grid's cells
//   -> FlexiCubes, holes ear-clipped (native, lib/host/mesh_flexicubes.c3)
//   -> back into the coarse mesh's frame, cleaned up and decimated
//      (core/mesh/simplify.js)
//
// Sparc3D keeps the geometry only; the mesh out has no texture.
//
// The weights folder is Hi3D's Hugging Face layout (transformer_1024/,
// vae/, scheduler/) with dinov2-large/ and clip-vit-large-patch14/ (the
// vision tower's tensors) beside them.

import { llm } from '/core/llm.js';
import { submit } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { DinoV2 } from '/core/models/dinov2.js';
import { ClipVision } from '/core/models/clip_vision.js';
import { simplifyMesh } from '/core/mesh/simplify.js';
import { upload, readFloats, disposeAll } from '/plugins/lato2/nn.js';
import { prepare, dinoPixels, clipPixels, DINO_SIZE } from './image.js';
import { surfaceVoxels } from './voxels.js';
import { Hi3DFlow } from './flow.js';
import { Hi3DDecoder } from './decoder.js';

export const DEFAULTS = {
	seed: 42,
	// Heun updates (each two or three model calls)
	steps: 25,
	// voxels at most on the 128^3 grid (sparc3d's 90000 for this pass); the
	// grid shrinks to fit
	budget: 90000,
	// faces in the mesh written; 0 keeps them all
	faces: 1000000,
	// pieces with less than this share of the faces, and nothing within
	// `part_gap` (the mesh's units) of a kept piece, are dropped (sparc3d drops
	// every piece under 0.5%; the gap keeps strands of hair)
	part_keep: 0.005,
	part_gap: 0.02,
	// 'f16' (the transformer's stored halves, on the matrix cores) or 'f32'
	linear: 'f16',
	// the decoder's weights: 'f32' (as stored) or 'q8'
	decoder: 'f32',
};

const unit = (v) => {
	let s = 0;
	for (let i = 0; i < v.length; i++) s += v[i] * v[i];
	s = Math.max(Math.sqrt(s), 1e-12);
	for (let i = 0; i < v.length; i++) v[i] /= s;
	return v;
};

export class Hi3D {
	constructor(weights) {
		this.w = weights;
		this.config = JSON.parse(llm.readText(`${weights}/transformer_1024/config.json`));
	}

	dispose() {}

	// The conditioning [2917, 2048] (a tensor): DINOv2's tokens beside CLIP's
	// embedding, each row L2-normalised.
	async condition(pre) {
		const t0 = llm.now();
		const dino = new DinoV2(openSafetensors(`${this.w}/dinov2-large/model.safetensors`), { linear: 'f32' });
		await dino.load();
		const tokens = await dino.encode(dinoPixels(pre), DINO_SIZE, DINO_SIZE, { norm: 'model' });
		const D = dino.dim, L = tokens.elements / D, d = readFloats(tokens, L * D);
		tokens.dispose();
		dino.dispose();
		const clip = new ClipVision(openSafetensors(`${this.w}/clip-vit-large-patch14/model.safetensors`));
		await clip.load();
		const c = unit(await clip.pooled(clipPixels(pre)));
		clip.dispose();
		const rows = new Float32Array(L * (D + c.length));
		for (let t = 0; t < L; t++) {
			rows.set(unit(d.subarray(t * D, (t + 1) * D)), t * (D + c.length));
			rows.set(c, t * (D + c.length) + D);
		}
		llm.print(`  conditioning: ${L} tokens x ${D + c.length} in ${llm.since(t0)}`);
		return { cond: upload(rows), L };
	}

	// img an 8-bit image (RGBA cutout or RGB), mesh { positions, indices } in any
	// frame. Returns { positions, indices, normals, stats, pre } in the mesh's
	// frame. `options.progress(done, steps)`, if given, is awaited before the
	// flow and after each of its steps (a graph node's ctx.progress).
	async refine(img, mesh, options = {}) {
		const o = { ...DEFAULTS, ...options }, t0 = llm.now(), stats = {};
		const pre = prepare(img);
		const vox = surfaceVoxels(mesh.positions, mesh.indices, { resolution: 128, budget: o.budget });
		const n = vox.coords.length / 3;
		stats.voxels = n;
		stats.grid = vox.grid;
		llm.print(`  voxels: ${n} on a ${vox.grid}^3 grid (${vox.tries} tries)`);
		const { cond } = await this.condition(pre);
		let latent = null, cells;
		try {
			try {
				if (o.progress) await o.progress(0, o.steps);
				const flow = new Hi3DFlow(openSafetensors(`${this.w}/transformer_1024/model.safetensors`), this.config, { linear: o.linear });
				let st = null;
				try {
					await flow.load();
					st = flow.structure(vox.coords, n);
					stats.tokens = st.m;
					latent = upload(llm.randomNormal(n * flow.outC, o.seed));
					await flow.sample(latent, st, cond, { steps: o.steps, onStep: o.progress ?? null });
				} finally {
					submit();
					st?.dispose();
					flow.dispose();
				}
			} finally {
				cond.dispose();
			}
			const dec = new Hi3DDecoder(`${this.w}/vae/model.safetensors`, { linear: o.decoder });
			dec.load();
			try {
				cells = await dec.run(latent, vox.coords);
			} finally {
				dec.dispose();
			}
		} finally {
			latent?.dispose();
		}
		stats.cells = cells.coords.length / 3;
		const t1 = llm.now();
		const surf = llm.meshFlexiCubes(cells.coords, cells.raw, cells.r, 10000, 0);
		llm.print(`  FlexiCubes: ${surf.cubes} surface cubes, ${surf.positions.length / 3} vertices, ${surf.indices.length / 3} faces (${surf.holes} holes filled) in ${llm.since(t1)}`);
		// cell corners at 1024 are eighths of the 128 voxels: back through the
		// voxelizer's frame into the mesh's
		const p = surf.positions, k = cells.r / 128, half = vox.grid / 2;
		for (let i = 0; i < p.length; i += 3) {
			for (let a = 0; a < 3; a++) p[i + a] = (p[i + a] / k - half) / vox.scale + vox.center[a];
		}
		const t2 = llm.now();
		const out = simplifyMesh(p, surf.indices, { faces: o.faces, partGap: o.part_gap, partKeep: o.part_keep, crease: 60 });
		llm.print(`  cleanup: ${out.dropped} of ${out.parts} pieces dropped, to ${out.indices.length / 3} faces in ${llm.since(t2)}`);
		stats.surface_faces = surf.indices.length / 3;
		stats.vertices = out.positions.length / 3;
		stats.faces = out.indices.length / 3;
		stats.seconds = (llm.now() - t0) / 1000;
		return { positions: out.positions, indices: out.indices, normals: out.normals, stats, pre };
	}
}
