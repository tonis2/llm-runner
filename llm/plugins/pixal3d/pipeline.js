// Pixal3D image -> mesh (github.com/TencentARC/Pixal3D), the cascade:
//
//   cutout -> squared-up picture -> DINOv3 ViT-L/16 at 512 (+ NAF at 512)
//   -> sparse-structure flow on the 16^3 grid's proj features -> decoder:
//      64^3 occupancy, max-pooled to 32^3 voxels
//   -> shape flow (512) on those voxels' proj features -> the shape latent
//   at 1024:
//   -> the shape decoder's subdivisions alone take that latent's voxels to
//      512^3, quantised to the 64^3 grid: the fine stage's tokens
//   -> shape flow (1024) on their proj features from the 1024 picture
//      (DINOv3 at 1024, NAF at 512)
//   then
//   -> shape decoder: FlexiDualGrid fields on the voxels of a 16x finer grid
//   -> the voxel shell's outer surface (mesh.js): a closed, outward mesh
//   -> cleaned up and decimated (core/mesh/simplify.js, in native code):
//      floaters dropped, a thin plate's two sides parted, smoothed a little,
//      decimated to `faces`, normals split at creases
//
// The weights folder holds Pixal3D's ckpts/ (decoders, .json + .safetensors),
// pipeline.json, dinov3-vitl16/ (Hugging Face format) and naf/naf_release.pth;
// the four flows are one GGUF (pixal3d-base-flow-f16.gguf, there or beside it).

import { llm } from '/core/llm.js';
import { submit } from '/core/gpu.js';
import { SparseStructureDecoder } from '/plugins/lato2/trellis.js';
import { upload, readFloats, disposeAll } from '/plugins/lato2/nn.js';
import { preprocess } from './image.js';
import { Conditioner, camera } from './condition.js';
import { Flow } from './flow.js';
import { ShapeDecoder } from './decoder.js';
import { simplifyMesh } from '/core/mesh/simplify.js';
import { floodOutside, shellSurfaceNative } from './mesh.js';

export const DEFAULTS = {
	seed: 42,
	// the mesh's grid: 512 (one stage) or 1024 (the cascade). Pixal3D's 1536
	// cascade (the same 1024 flow on a 96^3 grid) would need more than 16 GB
	// in the decoder's last levels.
	resolution: 1024,
	// faces in the mesh written (Pixal3D's own export decimates to a million too);
	// 0 keeps them all (17M at 1024)
	faces: 1000000,
	// horizontal field of view, radians (inference.py's --fov; 0.2 is near orthographic)
	fov: 0.2,
	ss_steps: 12, ss_guidance: 7.5, ss_rescale: 0.7, ss_rescale_t: 5,
	shape_steps: 12, shape_guidance: 7.5, shape_rescale: 0.5, shape_rescale_t: 3,
	// 'f16' (the GGUF's own, on the matrix cores), 'q8' or 'f32'
	linear: 'f16',
};

// 64^3 occupancy logits (row (x * 64 + y) * 64 + z) max-pooled by 2: the
// occupied 32^3 cells [n, 3] in x, y, z order.
export function pooledCells(logits, R) {
	const h = R / 2, hit = new Uint8Array(h * h * h);
	for (let i = 0; i < R * R * R; i++) {
		if (!(logits[i] > 0)) continue;
		const x = Math.floor(i / (R * R)), y = Math.floor(i / R) % R, z = i % R;
		hit[((x >> 1) * h + (y >> 1)) * h + (z >> 1)] = 1;
	}
	let n = 0;
	for (let i = 0; i < hit.length; i++) n += hit[i];
	const coords = new Int32Array(n * 3);
	for (let i = 0, j = 0; i < hit.length; i++) {
		if (!hit[i]) continue;
		coords[3 * j] = Math.floor(i / (h * h));
		coords[3 * j + 1] = Math.floor(i / h) % h;
		coords[3 * j + 2] = i % h;
		j++;
	}
	return coords;
}

// The cascade's fine tokens: voxels `hr` [n, 3] of a 512^3 grid quantised to
// a g^3 one (Pixal3DImageTo3DPipeline.run: round((c + 0.5) / 512 (g - 1))),
// each once, sorted with x slowest as torch's unique.
export function cascadeTokens(hr, g) {
	const hit = new Uint8Array(g * g * g), n = hr.length / 3;
	const q = (c) => Math.round(((c + 0.5) / 512) * (g - 1));
	for (let i = 0; i < n; i++) hit[(q(hr[3 * i]) * g + q(hr[3 * i + 1])) * g + q(hr[3 * i + 2])] = 1;
	let m = 0;
	for (let i = 0; i < hit.length; i++) m += hit[i];
	const out = new Int32Array(m * 3);
	for (let i = 0, j = 0; i < hit.length; i++) {
		if (!hit[i]) continue;
		out[3 * j] = Math.floor(i / (g * g));
		out[3 * j + 1] = Math.floor(i / g) % g;
		out[3 * j + 2] = i % g;
		j++;
	}
	return out;
}

function gridCoords(R) {
	const c = new Int32Array(R * R * R * 3);
	for (let t = 0; t < R * R * R; t++) { c[3 * t] = Math.floor(t / (R * R)); c[3 * t + 1] = Math.floor(t / R) % R; c[3 * t + 2] = t % R; }
	return c;
}

export class Pixal3D {
	constructor(weights, { gguf = null, linear = 'f16' } = {}) {
		this.w = weights;
		this.gguf = gguf ?? Pixal3D.findGguf(weights);
		this.linear = linear;
		this.norm = JSON.parse(llm.readText(`${weights}/pipeline.json`)).args;
	}

	static findGguf(w) {
		for (const p of [`${w}/pixal3d-base-flow-f16.gguf`, `${w}/../pixal3d-base-flow-f16.gguf`]) if (llm.exists(p)) return p;
		throw new Error(`no pixal3d-base-flow-f16.gguf in ${w} or beside it: give gguf=`);
	}

	dispose() { this.m?.close(); this.m = null; }

	async flow(which) {
		this.m ??= llm.open(this.gguf);
		const f = new Flow(this.m, which, { linear: this.linear });
		await f.load();
		return f;
	}

	decoder() {
		const sd = new ShapeDecoder(`${this.w}/ckpts`, { linear: this.linear === 'f16' ? 'f16' : 'f32' });
		sd.load();
		return sd;
	}

	// A shape flow's latent on `coords` (n voxels), denormalised: a tensor [n, 32].
	async shapeLatent(which, coords, global, proj, o, seed, onStep = null) {
		const n = coords.length / 3, flow = await this.flow(which);
		const z = upload(llm.randomNormal(n * flow.outC, seed));
		try {
			await flow.sample(z, coords, { global, proj }, n, {
				steps: o.shape_steps, guidance: o.shape_guidance, rescale: o.shape_rescale, interval: [0.6, 1], rescaleT: o.shape_rescale_t, onStep,
			});
		} catch (e) {
			z.dispose();
			throw e;
		} finally {
			flow.dispose();
		}
		const C = flow.outC, { mean, std } = this.norm.shape_slat_normalization;
		const lat = readFloats(z, n * C);
		for (let i = 0; i < lat.length; i++) lat[i] = lat[i] * std[i % C] + mean[i % C];
		z.buffer.write(lat);
		return z;
	}

	// An 8-bit RGBA cutout to { positions, indices, normals, stats }.
	// `options.progress(done, steps)`, if given, is awaited after each flow
	// step, the three flows' steps counted together (a graph node's
	// ctx.progress).
	async generate(img, options = {}) {
		const o = { ...DEFAULTS, ...options }, t0 = llm.now(), stats = {};
		if (o.resolution !== 512 && o.resolution !== 1024) throw new Error(`resolution ${o.resolution}: 512 or 1024`);
		let done = 0;
		const total = o.ss_steps + o.shape_steps * (o.resolution > 512 ? 2 : 1);
		const onStep = o.progress ? () => o.progress(++done, total) : null;
		const pre = preprocess(img);
		const cam = camera(o.fov);
		const owned = [];
		let cond = new Conditioner(this.w), pic = null, fine = null, lat = null;
		const own = (t) => { owned.push(t); return t; };
		try {
			// both pictures first, so DINOv3 and NAF are gone before the flows load
			await cond.load();
			pic = await cond.picture(pre, 512, { naf: 512 });
			if (o.resolution > 512) fine = await cond.picture(pre, 1024, { naf: 512 });
			cond.dispose();
			const global = own(cond.global(pic));

			// sparse structure: the 16^3 latent, then 32^3 cells
			const ssProj = own(cond.proj(pic, 16, cam, null, { naf: false }));
			const flow = await this.flow('ss');
			const x = own(upload(llm.randomNormal(16 ** 3 * flow.outC, o.seed)));
			try {
				await flow.sample(x, gridCoords(16), { global, proj: ssProj }, 16 ** 3, {
					steps: o.ss_steps, guidance: o.ss_guidance, rescale: o.ss_rescale, interval: [0.6, 1], rescaleT: o.ss_rescale_t, onStep,
				});
			} finally {
				flow.dispose();
			}
			const dec = new SparseStructureDecoder(`${this.w}/ckpts`);
			dec.load();
			const { logits, R } = dec.run(x, 16);
			submit();
			let coords = pooledCells(readFloats(logits, R ** 3), R);
			disposeAll([logits]);
			dec.dispose();
			stats.cells = coords.length / 3;
			if (!stats.cells) throw new Error('the sparse structure came out empty');

			// the shape latent on those cells
			const proj = own(cond.proj(pic, 32, cam, coords));
			cond.disposePicture(pic);
			pic = null;
			lat = await this.shapeLatent('shape512', coords, global, proj, o, o.seed + 1, onStep);
			let r = 32;

			if (fine) {
				// its voxels at 512^3, quantised to the fine stage's grid
				const sd = this.decoder();
				let hr;
				try {
					hr = (await sd.run(lat, coords, 32, { upsampleOnly: 4 })).coords;
				} finally {
					sd.dispose();
					lat.dispose();
					lat = null;
				}
				r = o.resolution / 16;
				coords = cascadeTokens(hr, r);
				stats.tokens = coords.length / 3;
				llm.print(`  cascade: ${hr.length / 3} voxels at 512^3 to ${stats.tokens} tokens at ${r}^3`);
				const fineGlobal = own(cond.global(fine)), fineProj = own(cond.proj(fine, r, cam, coords));
				cond.disposePicture(fine);
				fine = null;
				lat = await this.shapeLatent('shape1024', coords, fineGlobal, fineProj, o, o.seed + 2, onStep);
			}
			disposeAll(owned.splice(0));

			// decode and mesh
			const sd = this.decoder();
			let out;
			try {
				out = await sd.run(lat, coords, r);
			} finally {
				sd.dispose();
				lat.dispose();
				lat = null;
			}
			stats.resolution = out.r;
			stats.voxels = out.coords.length / 3;
			const t1 = llm.now();
			const outside = floodOutside(out.coords, out.r);
			const flooded = llm.since(t1), t2 = llm.now();
			const surf = shellSurfaceNative(out.coords, out.fields, out.r, out.table, outside);
			llm.print(`  surface: outside flooded in ${outside.rounds} rounds, ${flooded}; ${surf.positions.length / 3} vertices, ${surf.faces.length / 3} faces in ${llm.since(t2)}`);
			// floaters beyond about 20 voxels of 1024 go; plate sides part by a quarter voxel
			const t3 = llm.now();
			const mesh = simplifyMesh(surf.positions, surf.faces, { faces: o.faces, partGap: 0.02, separate: 0.25 / out.r, smooth: 3, crease: 60 });
			llm.print(`  cleanup: ${mesh.dropped} of ${mesh.parts} pieces dropped, ${mesh.separated} vertices parted, to ${mesh.indices.length / 3} faces in ${llm.since(t3)}`);
			stats.surface_faces = surf.faces.length / 3;
			stats.vertices = mesh.positions.length / 3;
			stats.faces = mesh.indices.length / 3;
			stats.seconds = (llm.now() - t0) / 1000;
			return { positions: mesh.positions, indices: mesh.indices, normals: mesh.normals, stats, pre };
		} finally {
			submit();
			cond.disposePicture(pic);
			cond.disposePicture(fine);
			cond.dispose();
			disposeAll(owned);
			lat?.dispose();
		}
	}
}
