// Qwen-Image 2.1 as graph nodes: the prompt encoding (Qwen3-VL-8B's last layer),
// the DiT loader and the sampler. Its latents are its own VAE's ('qwen'), which
// the core VAE nodes decode and encode.

import { llm, f32, image } from '../lib/llm.js';
import { submit, copy } from '../lib/gpu.js';
import { eulerStep, add, scale } from '../lib/ops.js';
import { TextEncoder } from '../lib/qwen3.js';
import { Qwen3Vision } from '../lib/qwen3_vision.js';
import { makeLatent, encodeImage } from '../lib/latents.js';
import { defineNodes } from '../lib/graph/registry.js';
import { QwenImageDiT } from './dit.js';
import { sigmas } from './sched.js';

const SYSTEM = '<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n';

// The prompt as the model was trained on it: a raw template, not the chat
// template, and never empty.
function template(prompt) {
	return `${SYSTEM}<|im_start|>user\n${prompt || ' '}<|im_end|>\n<|im_start|>assistant\n`;
}

// `slots` are where each reference image's latent tokens go in the text rows
// ({ at, h, w } in latent tokens), `images` those images at the size they were
// read, for the sampler to put through the VAE.
function conditioning(tensor, n, dim, slots = [], images = []) {
	return { family: 'qwenimage', tensor, n, dim, slots, images, dispose() { tensor.dispose(); } };
}

// The pipeline's size for a reference image: about `resolution` squared pixels
// at the image's aspect, the sides multiples of 32.
function referenceSize(resolution, img) {
	const ratio = img.width / img.height;
	const w = Math.sqrt(resolution * resolution * ratio);
	return { w: Math.max(32, Math.round(w / 32) * 32), h: Math.max(32, Math.round(w / ratio / 32) * 32) };
}

// Prompt tokens with each image's pad tokens in place, and Qwen3-VL's 3D
// positions: text counts up on all three axes; an image's tokens hold the time
// axis at the position reached and lay out rows and columns from it, and the
// text after goes on from there plus the grid's longer side.
function tokenizeWithImages(tokenizer, prompt, grids) {
	const pad = tokenizer.find('<|image_pad|>');
	const ids = [];
	const blocks = [];
	const text = (t) => { for (const id of tokenizer.encode(t, true)) ids.push(id); };
	text(`${SYSTEM}<|im_start|>user\n`);
	grids.forEach((g, i) => {
		text(`${i > 0 ? ' ' : ''}<image${i + 1}><|vision_start|>`);
		blocks.push({ at: ids.length, rows: g.gh * g.gw, gh: g.gh, gw: g.gw });
		for (let j = 0; j < g.gh * g.gw; j++) ids.push(pad);
		text('<|vision_end|>');
	});
	text(`${prompt || ' '}<|im_end|>\n<|im_start|>assistant\n`);
	const positions = new Int32Array(3 * ids.length);
	let pos = 0, t = 0;
	for (const b of blocks) {
		for (; t < b.at; t++, pos++) positions.fill(pos, 3 * t, 3 * t + 3);
		for (let r = 0; r < b.gh; r++) {
			for (let c = 0; c < b.gw; c++, t++) {
				positions[3 * t] = pos;
				positions[3 * t + 1] = pos + r;
				positions[3 * t + 2] = pos + c;
			}
		}
		pos += Math.max(b.gh, b.gw);
	}
	for (; t < ids.length; t++, pos++) positions.fill(pos, 3 * t, 3 * t + 3);
	return { ids: Uint32Array.from(ids), positions, blocks };
}

async function encodeWithImages(encoder, prompt, vision, images, resolution, t0) {
	if (!vision) throw new Error('a prompt with images needs the vision encoder: wire an mmproj file into `vision`');
	const sized = images.map((img) => {
		const { w, h } = referenceSize(resolution, img);
		return img.width === w && img.height === h ? img : image.resize(img, w, h);
	});
	const vis = new Qwen3Vision(vision);
	const seen = [];
	try {
		await vis.load();
		for (const img of sized) seen.push(await vis.encode(image.toTensor(img), img.height, img.width));
	} finally {
		vis.close();
	}
	const enc = new TextEncoder(encoder.path);
	try {
		const { ids, positions, blocks } = tokenizeWithImages(enc.tokenizer, prompt, seen);
		const drop = enc.tokenizer.encode(SYSTEM, true).length;
		const all = await enc.encodeLayers(ids, [enc.config.nLayers - 1], {
			positions,
			inject: blocks.map((b, i) => ({ at: b.at, rows: b.rows, tensor: seen[i].main })),
			deepstack: blocks.map((b, i) => ({ at: b.at, rows: b.rows, tensors: seen[i].deepstack })),
		});
		// The text rows alone (the image pads' rows are where the DiT puts the
		// references' latent tokens), and where each image goes among them.
		const dim = enc.config.dim;
		const n = ids.length - drop - blocks.reduce((sum, b) => sum + b.rows, 0);
		const out = f32(n * dim);
		const slots = [];
		let from = drop, to = 0;
		for (const b of blocks) {
			copy(all, out, (b.at - from) * dim * 4, from * dim * 4, to * dim * 4);
			to += b.at - from;
			slots.push({ at: to, h: 2 * b.gh, w: 2 * b.gw });
			from = b.at + b.rows;
		}
		copy(all, out, (ids.length - from) * dim * 4, from * dim * 4, to * dim * 4);
		submit();
		all.dispose();
		llm.print(`  [phase] text_encode: ${n} tokens and ${blocks.length} image(s) in ${llm.since(t0)}`);
		return conditioning(out, n, dim, slots, sized);
	} finally {
		for (const s of seen) {
			s.main.dispose();
			for (const d of s.deepstack) d.dispose();
		}
		enc.close();
	}
}

defineNodes('qwenimage', {
	'qwenimage.load': {
		title: 'Qwen-Image DiT',
		category: 'loaders',
		description: 'A Qwen-Image 2.1 GGUF.',
		inputs: { path: 'PATH(dit)' },
		outputs: { model: 'MODEL' },
		async run({ path }) {
			const t0 = llm.now();
			const file = llm.open(path);
			const dit = new QwenImageDiT(file);
			await dit.load();
			llm.print(`  [phase] dit_load: ${llm.since(t0)}`);
			return {
				model: {
					family: 'qwenimage',
					dit,
					dispose() { dit.release(); dit.unload(); file.close(); },
				},
			};
		},
	},

	'qwenimage.text_encode': {
		title: 'Qwen-Image prompt',
		category: 'conditioning',
		description: 'Qwen3-VL-8B hidden states from the last layer, the system prompt dropped. Images to edit from (up to three) are read by the vision encoder (`vision`, an mmproj file), resized to about resolution^2 pixels.',
		inputs: {
			encoder: 'TEXT_ENCODER',
			prompt: 'STRING*=',
			vision: 'PATH(mmproj)?',
			image1: 'IMAGE?',
			image2: 'IMAGE?',
			image3: 'IMAGE?',
			resolution: 'INT=1024',
		},
		outputs: { cond: 'CONDITIONING' },
		async run({ encoder, prompt, vision, image1, image2, image3, resolution }) {
			const t0 = llm.now();
			const images = [image1, image2, image3].filter(Boolean);
			if (images.length > 0) return { cond: await encodeWithImages(encoder, prompt, vision, images, resolution, t0) };
			const enc = new TextEncoder(encoder.path);
			try {
				const tokens = enc.tokenizer.encode(template(prompt), true);
				const drop = enc.tokenizer.encode(SYSTEM, true).length;
				const all = await enc.encodeLayers(tokens, [enc.config.nLayers - 1]);
				const dim = enc.config.dim;
				const n = tokens.length - drop;
				const out = f32(n * dim);
				copy(all, out, n * dim * 4, drop * dim * 4);
				submit();
				all.dispose();
				llm.print(`  [phase] text_encode: ${n} tokens in ${llm.since(t0)}`);
				return { cond: conditioning(out, n, dim) };
			} finally {
				enc.close();
			}
		},
	},

	'qwenimage.sample': {
		title: 'Qwen-Image sampler',
		category: 'sampling',
		description: 'Flow-matching Euler steps (40 is what Qwen recommends), with true CFG when cfg > 1 and a negative prompt. With a latent it is img2img. A prompt with images needs the VAE to encode them; a width or height of 0 then takes the last image\'s.',
		inputs: {
			positive: 'CONDITIONING',
			negative: 'CONDITIONING?',
			model: 'MODEL',
			vae: 'VAE?',
			latent: 'LATENT(qwen)?',
			width: 'INT=1024',
			height: 'INT=1024',
			steps: 'INT=40',
			seed: 'INT=42',
			cfg: 'FLOAT=1',
			strength: 'FLOAT=0.6',
		},
		outputs: { latent: 'LATENT(qwen)' },
		async run({ positive, negative, model, vae, latent: init, width, height, steps, seed, cfg, strength }, ctx) {
			for (const c of [positive, negative]) {
				if (c && c.family !== 'qwenimage') throw new Error(`qwenimage.sample needs Qwen-Image prompts, not ${c.family}`);
			}
			if (model.family !== 'qwenimage') throw new Error(`qwenimage.sample needs a Qwen-Image DiT, not ${model.family}`);
			if (init && init.format !== 'qwen') throw new Error(`qwenimage.sample starts from qwen latents; this one is ${init.format}`);
			const useCfg = cfg > 1;
			if (useCfg && !negative) throw new Error('cfg > 1 needs a negative prompt wired in');
			const refsOf = async (c) => {
				if (!c || c.images.length === 0) return [];
				if (!vae || vae.format !== 'qwen') throw new Error('a prompt with images needs the Qwen-Image VAE wired into `vae`');
				const out = [];
				for (const img of c.images) out.push(await encodeImage(vae.path, img));
				return out;
			};
			const t0 = llm.now();
			const condRefs = await refsOf(positive);
			const uncondRefs = useCfg ? await refsOf(negative) : [];
			if (condRefs.length + uncondRefs.length > 0) llm.print(`  [phase] reference_encode: ${llm.since(t0)}`);
			const last = condRefs[condRefs.length - 1];
			if (last) {
				width = width || last.w * 16;
				height = height || last.h * 16;
			}
			// Sides are multiples of 32: a latent token is 16 pixels, and the
			// vision encoder's tokens are 2x2 of them.
			const latentH = init ? init.h : Math.max(2, Math.round(height / 32) * 2);
			const latentW = init ? init.w : Math.max(2, Math.round(width / 32) * 2);
			const dit = model.dit;
			if (positive.dim !== dit.textDim) throw new Error(`Qwen-Image reads ${dit.textDim}-wide text states (Qwen3-VL-8B); this encoder gives ${positive.dim}`);

			const t1 = llm.now();
			dit.prepare(latentH, latentW);
			let cond = null, uncond = null;
			try {
				cond = await dit.encodePrefix(positive, condRefs);
				uncond = useCfg ? await dit.encodePrefix(negative, uncondRefs) : null;
				llm.print(`  [phase] dit_setup: ${llm.since(t1)}`);

				const t2 = llm.now();
				const count = dit.channels * latentH * latentW;
				const sched = sigmas(steps, latentH * latentW, init ? strength : 1);
				const n = sched.length - 1;
				const x0 = llm.randomNormal(count, seed);
				if (init) {
					const s0 = sched[0];
					for (let i = 0; i < count; i++) x0[i] = s0 * x0[i] + (1 - s0) * init.data[i];
				}
				dit.a.latent.buffer.write(x0);
				const condV = useCfg ? f32(count) : null;
				try {
					for (let s = 0; s < n; s++) {
						const ts = llm.now();
						if (useCfg) {
							// v = uncond + cfg * (cond - uncond), on the GPU.
							await dit.forward(cond, sched[s]);
							copy(dit.a.velocity, condV, count * 4);
							await dit.forward(uncond, sched[s]);
							scale(dit.a.velocity, count, -1);
							add(condV, dit.a.velocity, count);
							scale(condV, count, cfg);
							scale(dit.a.velocity, count, -1);
							add(dit.a.velocity, condV, count);
						} else {
							await dit.forward(cond, sched[s]);
						}
						eulerStep(dit.a.latent, dit.a.velocity, count, sched[s + 1] - sched[s]);
						submit();
						llm.print(`  step ${s + 1}/${n}: sigma ${sched[s].toFixed(4)}  ${(llm.now() - ts).toFixed(0)}ms`);
						await ctx.progress(s + 1, n, {});
					}
				} finally {
					if (condV) condV.dispose();
				}
				const out = new Float32Array(dit.a.latent.buffer.readBytes().buffer);
				llm.print(`  [phase] denoise: ${llm.since(t2)}`);
				return { latent: makeLatent('qwen', out, latentH, latentW) };
			} finally {
				for (const p of [cond, uncond]) if (p) p.dispose();
				dit.release();
			}
		},
	},
});
