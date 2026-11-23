// LATO.2 image -> mesh, stage by stage (scripts/image_e2e_inference.py):
//
//   cutout -> conditioning picture -> DINOv2 tokens
//   -> TRELLIS sparse-structure flow + decoder: occupied voxels of 64^3
//   -> V-Flow: a vertex latent on those voxels
//   -> V-VAE decoder + offset head: vertices on 1024^3
//   -> voxel encoder + T-Flow: a topology latent per vertex
//   -> T-VAE decoder: edges -> faces
//
// The weights folder holds LATO.2's .pt files (vflow, vvae, offset_head,
// tflow, tvae, voxel_encoder), trellis/ (TRELLIS-image-large's
// ss_flow_img_dit_L_16l8_fp16 and ss_dec_conv3d_16l8_fp16, .json and
// .safetensors) and dinov2_vitl14_reg4_pretrain.pth.

import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';
import { openTorch } from '/core/weights/torch.js';
import { DinoV2 } from '/core/models/dinov2.js';
import { vertexNormals } from '/core/mesh/glb.js';
import { upload, readFloats, disposeAll } from './nn.js';
import { conditioningImage, dinoPixels } from './image.js';
import { SparseStructureFlow, SparseStructureDecoder } from './trellis.js';
import { VertexFlow } from './vflow.js';
import { VertexDecoder } from './vvae.js';
import { VoxelEncoder, TopologyFlow, TopologyDecoder } from './topology.js';
import { edgesToFaces, orientFaces } from './faces.js';
import { cleanMesh } from './cleanup.js';

export const DEFAULTS = {
	seed: 42,
	padding: 1.2,
	ss_steps: 25, ss_cfg: 5.0, ss_rescale: 3.0,
	vertices: 2000, vflow_steps: 24, cfg: 3.0,
	threshold: 0.5,
	tflow_steps: 50, edge_threshold: 0.0,
	fill_holes: true, max_ring: 12,
	orient: true,
	// cleanup.js after the faces; `smooth` Taubin steps on top (rounded shapes
	// gain, thin hard-surface parts shrink, so it is off by default)
	cleanup: true, smooth: 0,
	linear: 'f32',
};

// The TRELLIS grid is z up and LATO.2's meshes y up: occupancy logits
// [64^3] (row (x * 64 + y) * 64 + z) -> voxels (x, z, 63 - y), sorted.
export function occupiedVoxels(logits, R) {
	const keys = [];
	for (let x = 0; x < R; x++) for (let y = 0; y < R; y++) for (let z = 0; z < R; z++) {
		if (logits[(x * R + y) * R + z] > 0) keys.push((x * R + z) * R + (R - 1 - y));
	}
	keys.sort((a, b) => a - b);
	const coords = new Int32Array(keys.length * 3);
	keys.forEach((k, i) => { coords[3 * i] = Math.floor(k / (R * R)); coords[3 * i + 1] = Math.floor(k / R) % R; coords[3 * i + 2] = k % R; });
	return coords;
}

export class Lato2 {
	// `keep` holds every model on the GPU between runs (a server); otherwise
	// each stage loads its weights and lets them go when it is done.
	constructor(weights, { keep = false, linear = 'f32' } = {}) {
		this.w = weights;
		this.keep = keep;
		this.linear = linear;
		this.models = {};
	}

	async model(name) {
		if (this.models[name]) return this.models[name];
		const w = this.w, linear = this.linear;
		let m;
		switch (name) {
			case 'dino': m = new DinoV2(openTorch(`${w}/dinov2_vitl14_reg4_pretrain.pth`), { linear: 'f32' }); await m.load(); break;
			case 'ss': m = new SparseStructureFlow(`${w}/trellis`, { linear }); await m.load(); break;
			case 'ssdec': m = new SparseStructureDecoder(`${w}/trellis`); m.load(); break;
			case 'vflow': m = new VertexFlow(`${w}/vflow.pt`, { linear }); await m.load(); break;
			case 'vvae': m = new VertexDecoder(`${w}/vvae.pt`, `${w}/offset_head.pt`); m.load(); break;
			case 'voxel': m = new VoxelEncoder(`${w}/voxel_encoder.pt`); m.load(); break;
			case 'tflow': m = new TopologyFlow(`${w}/tflow.pt`, { linear }); await m.load(); break;
			case 'tvae': m = new TopologyDecoder(`${w}/tvae.pt`); m.load(); break;
			default: throw new Error(`no model ${name}`);
		}
		this.models[name] = m;
		return m;
	}

	release(name) {
		if (this.keep || !this.models[name]) return;
		submit();
		this.models[name].dispose();
		delete this.models[name];
	}

	dispose() {
		submit();
		for (const m of Object.values(this.models)) m.dispose();
		this.models = {};
	}

	// TRELLIS's sparse-structure flow and decoder: occupied 64^3 voxels.
	async sparseStructure(c, u, L, seed, o) {
		const ss = await this.model('ss');
		const x = upload(llm.randomNormal(ss.R ** 3 * ss.inC, seed));
		try {
			await ss.sample(x, c, u, L, { steps: o.ss_steps, cfg: o.ss_cfg, rescale: o.ss_rescale });
			this.release('ss');
			const ssdec = await this.model('ssdec');
			const { logits, R } = ssdec.run(x, ss.R);
			const voxels = occupiedVoxels(readFloats(logits, R ** 3), R);
			logits.dispose();
			this.release('ssdec');
			return voxels;
		} finally {
			submit();
			x.dispose();
		}
	}

	// An RGBA cutout -> { positions Float32Array [V * 3] (y up, in -0.5 .. 0.5),
	// indices Uint32Array [F * 3], normals, cond (the picture), stats }.
	async generate(img, options = {}) {
		const o = { ...DEFAULTS, ...options };
		const t0 = llm.now(), stats = {};
		const cond = conditioningImage(img, { padding: o.padding });
		const keep = [];
		try {
			// DINOv2 tokens, and none for the unconditional half
			const dino = await this.model('dino');
			const tokens = await dino.encode(dinoPixels(cond), 518, 518);
			const L = tokens.elements / dino.dim;
			const c = f32(L * dino.dim), u = f32(L * dino.dim);
			keep.push(tokens, c, u);
			op.layerNorm(tokens, c, dino.dim, L, 1e-5);
			submit();
			this.release('dino');

			// sparse structure, unless the caller has one: `voxels` [n * 3] of
			// 64^3 in LATO.2's frame, sorted (a real mesh's, as in its mesh-to-mesh)
			const voxels = o.voxels ?? await this.sparseStructure(c, u, L, o.seed, o);
			const nv = voxels.length / 3;
			stats.voxels = nv;
			if (nv === 0) throw new Error('the sparse structure came out empty');
			llm.print(`  sparse structure: ${nv} voxels of 64^3${o.voxels ? ' (given)' : ''}`);

			// vertices
			const vflow = await this.model('vflow');
			const z = upload(llm.randomNormal(nv * vflow.latentDim, o.seed + 1));
			keep.push(z);
			await vflow.sample(z, voxels, nv, c, u, L, { steps: o.vflow_steps, cfg: o.cfg, vertices: o.vertices });
			this.release('vflow');
			const vvae = await this.model('vvae');
			const verts = await vvae.decode(z, voxels, nv, { threshold: o.threshold });
			this.release('vvae');
			const V = verts.coords.length / 3;
			stats.vertices = V;
			if (V < 3) throw new Error(`only ${V} vertices came out`);

			// topology
			const venc = await this.model('voxel');
			const { tokens: vt, count: T } = venc.run(voxels, nv);
			keep.push(vt);
			this.release('voxel');
			const tflow = await this.model('tflow');
			if (V > tflow.maxVertices) throw new Error(`${V} vertices; T-Flow takes at most ${tflow.maxVertices} (ask for fewer)`);
			const zt = upload(llm.randomNormal(V * tflow.zDim, o.seed + 2));
			keep.push(zt);
			await tflow.sample(zt, verts.coords, V, vt, T, { steps: o.tflow_steps });
			this.release('tflow');
			const tvae = await this.model('tvae');
			const t1 = llm.now();
			const feats = tvae.features(zt, verts.coords, V);
			keep.push(feats);
			const edges = tvae.edges(feats, V, o.edge_threshold);
			this.release('tvae');
			stats.edges = edges.length / 2;
			let faces = edgesToFaces(edges, V, verts.coords, { fill: o.fill_holes, maxRing: o.max_ring });
			llm.print(`  topology: ${stats.edges} edges, ${faces.length / 3} faces in ${llm.since(t1)}`);

			const res = verts.resolution;
			let positions = new Float32Array(V * 3);
			for (let i = 0; i < V * 3; i++) positions[i] = verts.coords[i] / res - 0.5 + verts.offsets[i] / (2 * res);
			if (o.cleanup) {
				const c = cleanMesh(positions, faces, { smooth: o.smooth });
				({ positions, faces } = c);
				const { faces_in, vertices, faces: _, ...removed } = c.stats;
				stats.cleanup = removed;
				stats.vertices = vertices;
				llm.print(`  cleanup: ${JSON.stringify(removed)}`);
			} else if (o.orient) {
				faces = orientFaces(faces, positions);
			}
			stats.faces = faces.length / 3;
			stats.seconds = (llm.now() - t0) / 1000;
			const indices = Uint32Array.from(faces);
			return { positions, indices, normals: vertexNormals(positions, indices), cond, stats };
		} finally {
			submit();
			disposeAll(keep);
		}
	}
}
