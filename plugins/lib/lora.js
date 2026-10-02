// LoRA and LoKr adapters, folded into a DiT's Q8_0 weights at load time.
//
// Each adapted linear gets W += scale * delta, dequantised, added and
// requantised in place by one kernel pass, and the adapter is dropped: a merged
// step costs what an unmerged one does. Adapters fold in order, each reading the
// weights the previous one left. The merge cannot be undone, which is why a
// resident DiT with a different adapter set is reloaded rather than re-merged.
//
// The requantisation is lossy for a LoRA whose delta is about one Q8_0 step:
// a block whose largest value does not move keeps its scale, and every
// element's delta under half a step rounds back to where it was. A model can
// instead keep LoRAs unmerged (`{ unmerged: true }`): A and B (B scaled) stay
// on the weight as `adapters` and `ops.matmul` adds x @ A^T @ B^T, exactly.
// Weights that read one input (the model's `inputs()`, as lists of weights)
// share one stacked A, so x @ A^T is one matmul for all of them.
// LoKr always merges.
//
// A model describes where its sites are: `sites(scheme)` yields every
// { module, weight, rowOffset } a naming scheme names, where `module` is the
// file's name for the site (or a list of names to try) and `rowOffset` is where
// the site's rows start in `weight` (unfused Q/K/V land in a fused QKV). The
// schemes are the model's; the first one naming any site at its first block is
// the file's. Scale rules are the old C3 loader's; the merge kernels are
// `lora_merge_q8` and `lokr_merge_q8`.

import { llm, GGML, f32 } from './llm.js';
import { dispatch, pc, submit, copy } from './gpu.js';

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
	// PEFT writes its adapter config as JSON under `lora_adapter_metadata`.
	const peft = file.meta('lora_adapter_metadata');
	if (peft) {
		try {
			const meta = JSON.parse(peft);
			for (const key of Object.keys(meta)) {
				if (key === 'lora_alpha' || key.endsWith('.lora_alpha')) {
					const a = parseFloat(meta[key]);
					if (Number.isFinite(a) && a !== 0) return a;
				}
			}
		} catch {
			// not the JSON we know
		}
	}
	return 0;
}

// The name under which `file` holds `site`, trying each of its spellings with
// `suffix`, or null.
function present(file, site, suffix) {
	for (const m of names(site.module)) if (file.has(m + suffix)) return m;
	return null;
}

function mergeLora(file, sites, strength, unmerged) {
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
		if (!unmerged && (w.type !== GGML.Q8_0 || inDim % 256 !== 0 || rank > 256)) { skipped++; continue; }
		let effective = scalar(file, `${module}.alpha`);
		if (effective < 0) effective = alpha;
		if (effective <= 0) effective = rank;
		const scale = (strength * effective) / rank;
		if (unmerged) {
			const bHost = file.floats(bName);
			for (let i = 0; i < bHost.length; i++) bHost[i] *= scale;
			const b = f32(bHost.length);
			b.buffer.write(bHost);
			// A goes up as Q8_0 when its rows are whole blocks: x @ A^T then runs on
			// the Q8_0 kernels (the matrix cores), at A's own precision - a rounding
			// of A is small against A, where a merge's is a step of W.
			const a = file.upload(aName, inDim % 32 === 0 ? 'q8' : 'f32');
			(w.adapters ??= []).push({ a, b, rank, rowOffset: site.rowOffset, rows: bRows, inDim });
			merged++;
			continue;
		}
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

// One group for the adapters of `weights`, which all read one input: their A
// matrices stacked into one [sum of ranks, in] tensor (rows are whole in either
// type, so stacking is appending bytes), each adapter's rows of it at `col`.
// The adapter's own A goes; it disposes with its weight.
function groupAdapters(weights) {
	const adapters = weights.flatMap((w) => w.adapters ?? []);
	if (adapters.length === 0) return;
	const { inDim, a: { type } } = adapters[0];
	if (adapters.some((ad) => ad.inDim !== inDim || ad.a.type !== type)) throw new Error('adapters grouped on one input must share its width');
	const rank = adapters.reduce((sum, ad) => sum + ad.rank, 0);
	const rowBytes = type === GGML.Q8_0 ? (inDim / 32) * 34 : inDim * 4;
	// A word of slack: the Q8_0 kernels read one word past a row's last block.
	const a = f32(rank * rowBytes / 4 + 4, [inDim, rank]);
	a.type = type;
	const g = { a, rank, ready: null, refs: adapters.length };
	let col = 0;
	for (const ad of adapters) {
		copy(ad.a, g.a, ad.rank * rowBytes, 0, col * rowBytes);
		ad.col = col;
		col += ad.rank;
	}
	submit();
	for (const ad of adapters) {
		ad.a.dispose();
		ad.a = null;
		ad.group = g;
		ad.dispose = () => {
			ad.b.dispose();
			if (--g.refs === 0) g.a.dispose();
		};
	}
}

// Fold `loras` ([{ path, strength }]) into a model's resident weights, in
// order. `model` is { schemes, sites(scheme), firstSites(scheme), inputs()? };
// with `unmerged`, LoRAs are attached to the weights instead of folded in.
export function mergeLoras(model, loras, { unmerged = false } = {}) {
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
			const r = mergeLora(file, model.sites(scheme), s, unmerged);
			llm.print(`  LoRA ${name}: ${r.merged} sites ${unmerged ? 'attached' : 'merged'}${r.skipped ? `, ${r.skipped} skipped` : ''} (${llm.since(t0)})`);
		} else {
			llm.print(`  ${path}: no LoRA or LoKr naming this knows - skipped`);
		}
		file.close();
	}
	if (!unmerged) return;
	// Group what shares an input; every other adapted weight is its own group.
	const grouped = new Set();
	for (const weights of model.inputs?.() ?? []) {
		groupAdapters(weights);
		for (const w of weights) grouped.add(w);
	}
	for (const scheme of model.schemes) {
		for (const { weight } of model.sites(scheme)) {
			if (grouped.has(weight)) continue;
			groupAdapters([weight]);
			grouped.add(weight);
		}
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
