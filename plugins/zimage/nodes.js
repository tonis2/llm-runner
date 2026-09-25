// Z-Image Turbo as graph nodes: the prompt encoding, the DiT loader and the
// sampler. Its latents are the Flux 1 VAE's ('flux1'), so the core VAE nodes
// decode them with ae.safetensors or taef1.

import { llm, f32 } from '../lib/llm.js';
import { submit, copy } from '../lib/gpu.js';
import { eulerStep, add, scale } from '../lib/ops.js';
import { TextEncoder } from '../lib/qwen3.js';
import { makeLatent } from '../lib/latents.js';
import { defineNodes } from '../lib/graph/registry.js';
import { ZImageDiT } from './dit.js';
import { mergeLoras } from './lora.js';

// Linear sigmas from 1 to 1/steps, a static shift of 3, then 0. Below
// `strength` 1 (img2img) the linear ramp starts lower, at the t whose shifted
// sigma is `strength`.
export function sigmas(steps, strength = 1, shift = 3) {
	const s = Math.min(Math.max(strength, 0), 1);
	const top = s >= 1 ? 1 : s / (shift * (1 - s) + s);
	const out = new Float32Array(steps + 1);
	for (let i = 0; i < steps; i++) {
		const t = Math.fround(top * (1 - (i * (1 - 1 / steps)) / (steps > 1 ? steps - 1 : 1)));
		out[i] = (shift * t) / (1 + (shift - 1) * t);
	}
	out[steps] = 0;
	return out;
}

function chat(prompt) { return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`; }

function conditioning(tensor, n, dim) {
	return { family: 'zimage', tensor, n, dim, dispose() { tensor.dispose(); } };
}

defineNodes('zimage', {
	'zimage.load': {
		title: 'Z-Image DiT',
		category: 'loaders',
		description: 'A Z-Image Turbo GGUF, with any LoRAs folded in.',
		inputs: { path: 'PATH(dit)', lora: 'LORA?' },
		outputs: { model: 'MODEL' },
		async run({ path, lora }) {
			const t0 = llm.now();
			const file = llm.open(path);
			const dit = new ZImageDiT(file);
			await dit.load();
			if (lora && lora.length > 0) mergeLoras(dit, lora);
			llm.print(`  [phase] dit_load: ${llm.since(t0)}`);
			return {
				model: {
					family: 'zimage',
					dit,
					dispose() { dit.release(); dit.unload(); file.close(); },
				},
			};
		},
	},

	'zimage.text_encode': {
		title: 'Z-Image prompt',
		category: 'conditioning',
		description: 'Qwen3-4B hidden states from the second-to-last layer.',
		inputs: { encoder: 'TEXT_ENCODER', prompt: 'STRING*=' },
		outputs: { cond: 'CONDITIONING' },
		async run({ encoder, prompt }) {
			const t0 = llm.now();
			const enc = new TextEncoder(encoder.path);
			const tokens = enc.tokenizer.encode(chat(prompt), true);
			const out = await enc.encodeLayers(tokens, [enc.config.nLayers - 2]);
			enc.close();
			llm.print(`  [phase] text_encode: ${llm.since(t0)}`);
			return { cond: conditioning(out, tokens.length, enc.config.dim) };
		},
	},

	'zimage.embeddings': {
		title: 'Z-Image embeddings file',
		category: 'conditioning',
		description: 'Precomputed embeddings in the C3 pipeline\'s format: u32 tokens, u32 dim, then floats.',
		inputs: { path: 'PATH(embeddings)' },
		outputs: { cond: 'CONDITIONING' },
		run({ path }) {
			const bytes = llm.readBytes(path);
			const [n, dim] = new Uint32Array(bytes.buffer, 0, 2);
			const t = f32(n * dim);
			t.buffer.write(new Uint8Array(bytes.buffer, 8, n * dim * 4));
			llm.print(`  embeddings from ${path}: [${n}, ${dim}]`);
			return { cond: conditioning(t, n, dim) };
		},
	},

	'zimage.sample': {
		title: 'Z-Image sampler',
		category: 'sampling',
		description: 'Flow-matching Euler steps, with classifier-free guidance when cfg > 1 (then `negative` is the unconditional prompt). With a latent it is img2img.',
		inputs: {
			positive: 'CONDITIONING',
			negative: 'CONDITIONING?',
			latent: 'LATENT(flux1)?',
			model: 'MODEL',
			width: 'INT=1024',
			height: 'INT=1024',
			steps: 'INT=4',
			seed: 'INT=42',
			cfg: 'FLOAT=0',
			strength: 'FLOAT=0.6',
		},
		outputs: { latent: 'LATENT(flux1)' },
		async run({ positive, negative, latent: init, model, width, height, steps, seed, cfg, strength }, ctx) {
			for (const c of [positive, negative]) {
				if (c && c.family !== 'zimage') throw new Error(`zimage.sample needs Z-Image prompts, not ${c.family}`);
			}
			if (model.family !== 'zimage') throw new Error(`zimage.sample needs a Z-Image DiT, not ${model.family}`);
			if (init && init.format !== 'flux1') throw new Error(`zimage.sample starts from flux1 latents; this one is ${init.format}`);
			const useCfg = cfg > 1;
			if (useCfg && !negative) throw new Error('cfg > 1 needs a negative prompt wired in (an empty prompt will do)');
			const latentH = init ? init.h : Math.round(height / 16) * 2;
			const latentW = init ? init.w : Math.round(width / 16) * 2;
			const dit = model.dit;

			const t1 = llm.now();
			dit.prepare(latentH, latentW, Math.max(positive.n, useCfg ? negative.n : 0));
			if (positive.dim !== dit.textDim) {
				dit.release();
				throw new Error(`Z-Image reads ${dit.textDim}-wide text embeddings (Qwen3-4B); the text model gives ${positive.dim}`);
			}
			const condText = dit.encodeText(positive.tensor, positive.n);
			const uncondText = useCfg ? dit.encodeText(negative.tensor, negative.n) : null;
			llm.print(`  [phase] dit_setup: ${llm.since(t1)}`);

			const t2 = llm.now();
			const sched = sigmas(steps, init ? strength : 1);
			const count = dit.channels * latentH * latentW;
			const x0 = llm.randomNormal(count, seed);
			if (init) {
				// x = sigma * noise + (1 - sigma) * image, at the first sigma.
				const s0 = sched[0];
				for (let i = 0; i < count; i++) x0[i] = s0 * x0[i] + (1 - s0) * init.data[i];
			}
			dit.a.latent.buffer.write(x0);
			const saved = useCfg ? f32(count) : null;
			const condV = useCfg ? f32(count) : null;
			try {
				for (let s = 0; s < steps; s++) {
					const ts = llm.now();
					if (useCfg) {
						// v = uncond + cfg * (cond - uncond), on the GPU.
						copy(dit.a.latent, saved, count * 4);
						await dit.forward(condText, sched[s]);
						copy(dit.a.velocity, condV, count * 4);
						copy(saved, dit.a.latent, count * 4);
						await dit.forward(uncondText, sched[s]);
						scale(dit.a.velocity, count, -1);
						add(condV, dit.a.velocity, count);        // cond - uncond
						scale(condV, count, cfg);
						scale(dit.a.velocity, count, -1);
						add(dit.a.velocity, condV, count);
						copy(saved, dit.a.latent, count * 4);
					} else {
						await dit.forward(condText, sched[s]);
					}
					eulerStep(dit.a.latent, dit.a.velocity, count, sched[s + 1] - sched[s]);
					submit();
					llm.print(`  step ${s + 1}/${steps}: sigma ${sched[s].toFixed(4)}  ${(llm.now() - ts).toFixed(0)}ms`);
					await ctx.progress(s + 1, steps, {
						latent: () => makeLatent('flux1', new Float32Array(dit.a.latent.buffer.readBytes().buffer), latentH, latentW),
					});
				}
				const out = new Float32Array(dit.a.latent.buffer.readBytes().buffer);
				llm.print(`  [phase] denoise: ${llm.since(t2)}`);
				return { latent: makeLatent('flux1', out, latentH, latentW) };
			} finally {
				for (const t of [saved, condV, condText, uncondText]) if (t) t.dispose();
				dit.release();
			}
		},
	},
});
