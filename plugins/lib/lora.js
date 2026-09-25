// LoRA and LoKr adapters, folded into a DiT's Q8_0 weights at load time.
//
// Each adapted linear gets W += scale * delta, dequantised, added and
// requantised in place by one kernel pass, and the adapter is dropped: a merged
// step costs what an unmerged one does. Adapters fold in order, each reading the
// weights the previous one left. The merge cannot be undone, which is why a
// resident DiT with a different adapter set is reloaded rather than re-merged.
//
// A model describes where its sites are: `sites(scheme)` yields every
// { module, weight, rowOffset } a naming scheme names, where `module` is the
// file's name for the site (or a list of names to try) and `rowOffset` is where
// the site's rows start in `weight` (unfused Q/K/V land in a fused QKV). The
// schemes are the model's; the first one naming any site at its first block is
// the file's. Scale rules are the old C3 loader's; the merge kernels are
// `lora_merge_q8` and `lokr_merge_q8`.

import { llm, GGML } from './llm.js';
import { dispatch, pc, submit } from './gpu.js';

export const MERGE_KERNELS = ['lora_merge_q8', 'lokr_merge_q8'];

const names = (module) => (Array.isArray(module) ? module : [module]);

// The first scheme with any site at block 0 in this file, for `suffixes` (the
// LoRA pair names, or `.lokr_w1`). `firstSites(scheme)` is that scheme's block 0.
function detect(file, schemes, firstSites, suffixes) {
	for (const s of schemes) {
		for (const site of firstSites(s)) {
			for (const m of names(site.module)) for (const suffix of suffixes) if (file.has(m + suffix)) return s;
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

// The name under which `file` holds `site`, trying each of its spellings with
// `suffix`, or null.
function present(file, site, suffix) {
	for (const m of names(site.module)) if (file.has(m + suffix)) return m;
	return null;
}

function mergeLora(file, sites, strength) {
	const alpha = globalAlpha(file);
	let merged = 0, skipped = 0;
	for (const site of sites) {
		let module = present(file, site, '.lora_A.weight');
		let aName, bName;
		if (module) { aName = `${module}.lora_A.weight`; bName = `${module}.lora_B.weight`; }
		else if ((module = present(file, site, '.lora_down.weight'))) { aName = `${module}.lora_down.weight`; bName = `${module}.lora_up.weight`; }
		else continue;
		if (!file.has(bName)) continue;
		const w = site.weight;
		const inDim = w.shape[0];
		const [rank] = file.info(aName).shape;             // A: [rank, in]
		const [bRows] = file.info(bName).shape;            // B: [rows, rank]
		if (w.type !== GGML.Q8_0 || inDim % 256 !== 0 || rank > 256) { skipped++; continue; }
		let effective = scalar(file, `${module}.alpha`);
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

function mergeLokr(file, sites, strength) {
	let merged = 0, skipped = 0;
	for (const site of sites) {
		const module = present(file, site, '.lokr_w1');
		if (!module) continue;
		const n1 = `${module}.lokr_w1`, n2 = `${module}.lokr_w2`;
		if (!file.has(n2)) continue;
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

// Fold `loras` ([{ path, strength }]) into a model's resident weights, in
// order. `model` is { schemes, sites(scheme), firstSites(scheme) }.
export function mergeLoras(model, loras) {
	for (const { path, strength } of loras) {
		const t0 = llm.now();
		const file = llm.open(path);
		const s = strength > 0 ? strength : 1;
		const name = path.split('/').pop();
		let scheme = detect(file, model.schemes, model.firstSites, ['.lokr_w1']);
		if (scheme) {
			const r = mergeLokr(file, model.sites(scheme), s);
			llm.print(`  LoKr ${name}: ${r.merged} sites merged${r.skipped ? `, ${r.skipped} skipped` : ''} (${llm.since(t0)})`);
		} else if ((scheme = detect(file, model.schemes, model.firstSites, ['.lora_A.weight', '.lora_down.weight']))) {
			const r = mergeLora(file, model.sites(scheme), s);
			llm.print(`  LoRA ${name}: ${r.merged} sites merged${r.skipped ? `, ${r.skipped} skipped` : ''} (${llm.since(t0)})`);
		} else {
			llm.print(`  ${path}: no LoRA or LoKr naming this knows - skipped`);
		}
		file.close();
	}
}

// `config.lora` / `config.loras` as [{ path, strength }]: a path, a list of
// paths, or a list of { path, strength }. The C3 CLI's `lora_path` and
// `lora_strength` are read too.
export function loraList(config) {
	let l = config.lora ?? config.loras;
	if (l == null && config.lora_path) l = [{ path: config.lora_path, strength: config.lora_strength ?? 1 }];
	if (l == null) return [];
	if (!Array.isArray(l)) l = [l];
	return l.map((e) => (typeof e === 'string' ? { path: e, strength: 1 } : { path: e.path, strength: e.strength ?? 1 }));
}
