// Z-Image Turbo as a plugin.
//
//   llm-runner zimage --config zimage.json
//
// Qwen3 hidden states (the second-to-last layer) as the text conditioning, a
// flow-matching Euler loop through the Lumina2 DiT with optional CFG, and the
// Flux 1 VAE. `text_embeddings` may name a file of precomputed embeddings in
// the C3 pipeline's format (u32 tokens, u32 dim, then floats) instead of a
// text model. Settings are the C3 CLI's JSON keys.

import { llm, image, f32 } from '../lib/llm.js';
import { submit, copy } from '../lib/gpu.js';
import { eulerStep, add, scale } from '../lib/ops.js';
import { TextEncoder } from '../lib/qwen3.js';
import { FluxVAEDecoder } from '../lib/flux_vae.js';
import { ZImageDiT } from './dit.js';

const SCALE_FACTOR = 0.3611;
const SHIFT_FACTOR = 0.1159;

// Linear sigmas from 1 to 1/steps, a static shift of 3, then 0.
function sigmas(steps, shift = 3) {
	const out = new Float32Array(steps + 1);
	for (let i = 0; i < steps; i++) {
		const t = Math.fround(1 - (i * (1 - 1 / steps)) / (steps > 1 ? steps - 1 : 1));
		out[i] = (shift * t) / (1 + (shift - 1) * t);
	}
	out[steps] = 0;
	return out;
}

function chat(prompt) { return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`; }

// [n, dim] embeddings as a Tensor, from the text model or a file.
function embed(config, prompt, fromFile) {
	if (fromFile) {
		const bytes = llm.readBytes(fromFile);
		const header = new Uint32Array(bytes.buffer, 0, 2);
		const [n, dim] = header;
		const t = f32(n * dim);
		t.buffer.write(new Uint8Array(bytes.buffer, 8, n * dim * 4));
		llm.print(`  embeddings from ${fromFile}: [${n}, ${dim}]`);
		return { tensor: t, n, dim };
	}
	const enc = new TextEncoder(config.text_model);
	const tokens = enc.tokenizer.encode(chat(prompt), true);
	const out = enc.encodeLayers(tokens, [enc.config.nLayers - 2]);
	enc.close();
	return { tensor: out, n: tokens.length, dim: enc.config.dim };
}

llm.plugin({
	name: 'zimage',
	async generate(config) {
		const start = llm.now();
		const size = Math.round((config.size ?? 1024) / 16) * 16;
		const steps = config.steps ?? 4;
		const seed = config.seed ?? 42;
		const cfg = config.cfg_scale ?? 0;
		const useCfg = cfg > 1;
		const latentH = size / 8, latentW = size / 8;
		llm.print(`\n=== Z-Image: ${size}x${size}, ${steps} steps, seed ${seed}${useCfg ? ', cfg ' + cfg : ''} ===`);

		const t0 = llm.now();
		const cond = embed(config, config.prompt ?? '', config.text_embeddings);
		const uncond = useCfg ? embed(config, '', null) : null;
		llm.print(`  [phase] text_encode: ${llm.since(t0)}`);

		const t1 = llm.now();
		const model = llm.open(config.model);
		const dit = new ZImageDiT(model);
		dit.load();
		dit.prepare(latentH, latentW, Math.max(cond.n, uncond ? uncond.n : 0));
		if (cond.dim !== dit.textDim) {
			throw new Error(`Z-Image reads ${dit.textDim}-wide text embeddings (Qwen3-4B); the text model gives ${cond.dim}`);
		}
		const condText = dit.encodeText(cond.tensor, cond.n);
		cond.tensor.dispose();
		let uncondText = null;
		if (uncond) {
			uncondText = dit.encodeText(uncond.tensor, uncond.n);
			uncond.tensor.dispose();
		}
		llm.print(`  [phase] dit_setup: ${llm.since(t1)}`);

		const t2 = llm.now();
		const sched = sigmas(steps);
		const count = dit.channels * latentH * latentW;
		dit.a.latent.buffer.write(llm.randomNormal(count, seed));
		const saved = useCfg ? f32(count) : null;
		const condV = useCfg ? f32(count) : null;
		for (let s = 0; s < steps; s++) {
			const ts = llm.now();
			if (useCfg) {
				// v = uncond + cfg * (cond - uncond), on the GPU.
				copy(dit.a.latent, saved, count * 4);
				dit.forward(condText, sched[s]);
				copy(dit.a.velocity, condV, count * 4);
				copy(saved, dit.a.latent, count * 4);
				dit.forward(uncondText, sched[s]);
				scale(dit.a.velocity, count, -1);
				add(condV, dit.a.velocity, count);        // cond - uncond
				scale(condV, count, cfg);
				scale(dit.a.velocity, count, -1);
				add(dit.a.velocity, condV, count);
				copy(saved, dit.a.latent, count * 4);
			} else {
				dit.forward(condText, sched[s]);
			}
			eulerStep(dit.a.latent, dit.a.velocity, count, sched[s + 1] - sched[s]);
			submit();
			llm.print(`  step ${s + 1}/${steps}: sigma ${sched[s].toFixed(4)}  ${(llm.now() - ts).toFixed(0)}ms`);
		}
		const latent = new Float32Array(dit.a.latent.buffer.readBytes().buffer);
		for (const t of [saved, condV]) if (t) t.dispose();
		condText.dispose();
		if (uncondText) uncondText.dispose();
		dit.release();
		dit.unload();
		model.close();
		llm.print(`  [phase] denoise: ${llm.since(t2)}`);

		const t3 = llm.now();
		for (let i = 0; i < latent.length; i++) latent[i] = latent[i] / SCALE_FACTOR + SHIFT_FACTOR;
		const vae = llm.open(config.vae);
		const decoder = new FluxVAEDecoder(vae);
		const pixels = decoder.decode(latent, latentH, latentW);
		decoder.dispose();
		vae.close();
		const img = image.fromTensor(pixels, size, size, 3);
		llm.print(`  [phase] vae_decode: ${llm.since(t3)}`);

		const out = config.output ?? 'output.png';
		image.savePng(out, img);
		llm.print(`=== done in ${llm.since(start)}, saved ${out} ===`);
		return { output: out, width: size, height: size };
	},
});
