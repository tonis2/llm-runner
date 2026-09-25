// LoRA and LoKr adapters, folded into the DiT's Q8_0 weights at load time.
//
// Each adapted linear gets W += scale * delta, dequantised, added and
// requantised in place by one kernel pass, and the adapter is dropped: a merged
// step costs what an unmerged one does. Adapters fold in order, each reading the
// weights the previous one left. The merge cannot be undone, which is why a
// resident DiT with a different adapter set is reloaded rather than re-merged.
//
// Naming schemes and scale rules are `lib/model/lora.c3`'s; the merge kernels
// are `lora_merge_q8` and `lokr_merge_q8`.

import { llm, GGML } from '../lib/llm.js';
import { dispatch, pc, submit } from '../lib/gpu.js';

// Where each site lives in a file, per naming scheme. An absent name means the
// scheme has no such site.
const SCHEMES = [
	{
		dual: 'double_blocks.', single: 'single_blocks.', sep: '.',
		img_qkv: 'img_attn.qkv', img_proj: 'img_attn.proj', img_mlp_up: 'img_mlp.0', img_mlp_down: 'img_mlp.2',
		txt_qkv: 'txt_attn.qkv', txt_proj: 'txt_attn.proj', txt_mlp_up: 'txt_mlp.0', txt_mlp_down: 'txt_mlp.2',
		linear1: 'linear1', linear2: 'linear2',
	},
	{
		dual: 'diffusion_model.double_blocks.', single: 'diffusion_model.single_blocks.', sep: '.',
		img_qkv: 'img_attn.qkv', img_proj: 'img_attn.proj', img_mlp_up: 'img_mlp.0', img_mlp_down: 'img_mlp.2',
		txt_qkv: 'txt_attn.qkv', txt_proj: 'txt_attn.proj', txt_mlp_up: 'txt_mlp.0', txt_mlp_down: 'txt_mlp.2',
		linear1: 'linear1', linear2: 'linear2',
	},
	{
		// Diffusers: Q/K/V unfused, landing in row ranges of the fused weights.
		// No MLP sites - Flux 1's GELU FFN has no place in Klein's SwiGLU.
		dual: 'transformer.transformer_blocks.', single: 'transformer.single_transformer_blocks.', sep: '.',
		img_q: 'attn.to_q', img_k: 'attn.to_k', img_v: 'attn.to_v', img_proj: 'attn.to_out.0',
		txt_q: 'attn.add_q_proj', txt_k: 'attn.add_k_proj', txt_v: 'attn.add_v_proj', txt_proj: 'attn.to_add_out',
		single_q: 'attn.to_q', single_k: 'attn.to_k', single_v: 'attn.to_v', linear2: 'proj_out',
	},
	{
		dual: 'lora_unet_double_blocks_', single: 'lora_unet_single_blocks_', sep: '_',
		img_qkv: 'img_attn_qkv', img_proj: 'img_attn_proj', img_mlp_up: 'img_mlp_0', img_mlp_down: 'img_mlp_2',
		txt_qkv: 'txt_attn_qkv', txt_proj: 'txt_attn_proj', txt_mlp_up: 'txt_mlp_0', txt_mlp_down: 'txt_mlp_2',
		linear1: 'linear1', linear2: 'linear2',
	},
];

// Every site: [scheme key, block list, weight key, row offset in units of dim].
const DUAL_SITES = [
	['img_qkv', 'imgQkv', 0], ['img_q', 'imgQkv', 0], ['img_k', 'imgQkv', 1], ['img_v', 'imgQkv', 2],
	['img_proj', 'imgProj', 0], ['img_mlp_up', 'imgUp', 0], ['img_mlp_down', 'imgDown', 0],
	['txt_qkv', 'txtQkv', 0], ['txt_q', 'txtQkv', 0], ['txt_k', 'txtQkv', 1], ['txt_v', 'txtQkv', 2],
	['txt_proj', 'txtProj', 0], ['txt_mlp_up', 'txtUp', 0], ['txt_mlp_down', 'txtDown', 0],
];
const SINGLE_SITES = [
	['linear1', 'linear1', 0], ['single_q', 'linear1', 0], ['single_k', 'linear1', 1], ['single_v', 'linear1', 2],
	['linear2', 'linear2', 0],
];

function modulePath(prefix, layer, sep, module) { return `${prefix}${layer}${sep}${module}`; }

// The first scheme with any site at block 0 in this file, for `suffixes`
// (the LoRA pair names, or `.lokr_w1`).
function detect(file, suffixes) {
	for (const s of SCHEMES) {
		for (const [key] of DUAL_SITES) {
			if (!s[key]) continue;
			for (const suffix of suffixes) if (file.has(modulePath(s.dual, 0, s.sep, s[key]) + suffix)) return s;
		}
		for (const [key] of SINGLE_SITES) {
			if (!s[key]) continue;
			for (const suffix of suffixes) if (file.has(modulePath(s.single, 0, s.sep, s[key]) + suffix)) return s;
		}
	}
	return null;
}

function scalar(file, name) {
	if (!file.has(name)) return -1;
	return file.floats(name)[0];
}

function globalAlpha(file) {
	for (const key of ['lora_alpha', 'ss_network_alpha', 'alpha']) {
		const v = parseFloat(file.meta(key));
		if (Number.isFinite(v) && v !== 0) return v;
	}
	return 0;
}

// Every (site, weight) pair of the file under `scheme`, as { module, weight, rowOffset }.
function* sites(dit, scheme) {
	const dim = dit.config.dim;
	for (let l = 0; l < dit.dual.length; l++) {
		for (const [key, weight, slot] of DUAL_SITES) {
			if (scheme[key]) yield { module: modulePath(scheme.dual, l, scheme.sep, scheme[key]), weight: dit.dual[l][weight], rowOffset: slot * dim };
		}
	}
	for (let l = 0; l < dit.single.length; l++) {
		for (const [key, weight, slot] of SINGLE_SITES) {
			if (scheme[key]) yield { module: modulePath(scheme.single, l, scheme.sep, scheme[key]), weight: dit.single[l][weight], rowOffset: slot * dim };
		}
	}
}

function mergeLora(dit, file, scheme, strength) {
	const alpha = globalAlpha(file);
	let merged = 0, skipped = 0;
	for (const site of sites(dit, scheme)) {
		let aName = `${site.module}.lora_A.weight`, bName = `${site.module}.lora_B.weight`;
		if (!file.has(aName)) { aName = `${site.module}.lora_down.weight`; bName = `${site.module}.lora_up.weight`; }
		if (!file.has(aName) || !file.has(bName)) continue;
		const w = site.weight;
		const inDim = w.shape[0];
		const [rank] = file.info(aName).shape;             // A: [rank, in]
		const [bRows] = file.info(bName).shape;            // B: [rows, rank]
		if (w.type !== GGML.Q8_0 || inDim % 256 !== 0 || rank > 256) { skipped++; continue; }
		let effective = scalar(file, `${site.module}.alpha`);
		if (effective < 0) effective = alpha;
		if (effective <= 0) effective = rank;
		const scale = (strength * effective) / rank;
		const a = file.upload(aName, 'f32');
		const b = file.upload(bName, 'f32');
		dispatch('lora_merge_q8', [w, a, b], [inDim / 256, bRows], pc('uuuufuuu', inDim, bRows, site.rowOffset, rank, scale, 0, 0, 0));
		submit();
		a.dispose();
		b.dispose();
		merged++;
	}
	return { merged, skipped };
}

function mergeLokr(dit, file, scheme, strength) {
	let merged = 0, skipped = 0;
	for (const site of sites(dit, scheme)) {
		const n1 = `${site.module}.lokr_w1`, n2 = `${site.module}.lokr_w2`;
		if (!file.has(n1) || !file.has(n2)) continue;
		const w = site.weight;
		const inDim = w.shape[0];
		const [w1r, w1c] = file.info(n1).shape;
		const [w2r, w2c] = file.info(n2).shape;
		if (w.type !== GGML.Q8_0 || inDim % 256 !== 0 || w1r * w1c > 256 || w1c * w2c !== inDim) { skipped++; continue; }
		// ai-toolkit's LoKr files carry an alpha the trainer has already folded
		// in; the scale that looks right is the strength itself.
		const w1 = file.upload(n1, 'f32');
		const w2 = file.upload(n2, 'f32');
		dispatch('lokr_merge_q8', [w, w1, w2], [inDim / 256, w1r * w2r], pc('uuuuuuuf', inDim, w1r * w2r, site.rowOffset, w1r, w1c, w2r, w2c, strength));
		submit();
		w1.dispose();
		w2.dispose();
		merged++;
	}
	return { merged, skipped };
}

// Fold `loras` ([{ path, strength }]) into `dit`'s resident weights, in order.
export function mergeLoras(dit, loras) {
	for (const { path, strength } of loras) {
		const t0 = llm.now();
		const file = llm.open(path);
		const s = strength > 0 ? strength : 1;
		let result;
		let scheme = detect(file, ['.lokr_w1']);
		if (scheme) {
			result = mergeLokr(dit, file, scheme, s);
			llm.print(`  LoKr ${path.split('/').pop()}: ${result.merged} sites merged${result.skipped ? `, ${result.skipped} skipped` : ''} (${llm.since(t0)})`);
		} else if ((scheme = detect(file, ['.lora_A.weight', '.lora_down.weight']))) {
			result = mergeLora(dit, file, scheme, s);
			llm.print(`  LoRA ${path.split('/').pop()}: ${result.merged} sites merged${result.skipped ? `, ${result.skipped} skipped` : ''} (${llm.since(t0)})`);
		} else {
			llm.print(`  ${path}: no LoRA or LoKr naming this knows - skipped`);
		}
		file.close();
	}
}
