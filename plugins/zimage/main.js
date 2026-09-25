// Z-Image Turbo as a plugin.
//
//   llm-runner zimage --config zimage.json
//   llm-runner zimage --config zimage.json input=photo.png strength=0.6
//
// Qwen3 hidden states (the second-to-last layer) as the text conditioning, a
// flow-matching Euler loop through the Lumina2 DiT with optional CFG, and the
// Flux 1 VAE, or TAESD (`taesd`) for a fast, rough decode. `input` starts from
// an image instead of noise (img2img, `strength` of the way back to noise).
// `lora` / `lora_path` fold adapters into the DiT. `text_embeddings` may name a
// file of precomputed embeddings in the C3 pipeline's format (u32 tokens, u32
// dim, then floats) instead of a text model. Settings are the C3 CLI's JSON
// keys; `width` and `height` may stand in for `size`.

import { llm, image, f32 } from '../lib/llm.js';
import { submit, copy } from '../lib/gpu.js';
import { eulerStep, add, scale, useMatrixCores } from '../lib/ops.js';
import { TextEncoder } from '../lib/qwen3.js';
import { FluxVAEDecoder, FluxVAEEncoder } from '../lib/flux_vae.js';
import { TAESDDecoder } from '../lib/taesd.js';
import { loraList } from '../lib/lora.js';
import { ZImageDiT } from './dit.js';
import { mergeLoras } from './lora.js';

const SCALE_FACTOR = 0.3611;
const SHIFT_FACTOR = 0.1159;

// Linear sigmas from 1 to 1/steps, a static shift of 3, then 0. Below
// `strength` 1 (img2img) the linear ramp starts lower, at the t whose shifted
// sigma is `strength`.
function sigmas(steps, strength = 1, shift = 3) {
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

// The image as a DiT latent [16, h/8, w/8]: the VAE encoder's mean, normalised.
function encodeImage(config, input, width, height) {
	const src = typeof input === 'string' ? image.load(input) : input;
	const img = image.cropTo(src, width, height);
	const vae = llm.open(config.vae);
	const encoder = new FluxVAEEncoder(vae);
	const enc = encoder.encode(image.toTensor(img), height, width);
	encoder.dispose();
	vae.close();
	const n = (enc.channels / 2) * enc.h * enc.w;
	const latent = new Float32Array(n);
	for (let i = 0; i < n; i++) latent[i] = (enc.data[i] - SHIFT_FACTOR) * SCALE_FACTOR;
	return latent;
}

// Latent to [3, H, W] pixels in [0, 1], through TAESD or the Flux VAE.
function decode(config, latent, latentH, latentW) {
	if (config.taesd) {
		const tae = llm.open(config.taesd);
		const decoder = new TAESDDecoder(tae);
		const pixels = decoder.decode(latent, latentH, latentW);
		decoder.dispose();
		tae.close();
		return pixels;
	}
	for (let i = 0; i < latent.length; i++) latent[i] = latent[i] / SCALE_FACTOR + SHIFT_FACTOR;
	const vae = llm.open(config.vae);
	const decoder = new FluxVAEDecoder(vae);
	const pixels = decoder.decode(latent, latentH, latentW);
	decoder.dispose();
	vae.close();
	return pixels;
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
		useMatrixCores(config.matrix_cores ?? true);
		const width = Math.round((config.width ?? config.size ?? 1024) / 16) * 16;
		const height = Math.round((config.height ?? config.size ?? 1024) / 16) * 16;
		const steps = config.steps ?? 4;
		const seed = config.seed ?? 42;
		const cfg = config.cfg_scale ?? 0;
		const useCfg = cfg > 1;
		const input = config.input_image ?? config.input ?? null;
		const strength = input ? (config.strength ?? 0.6) : 1;
		const latentH = height / 8, latentW = width / 8;
		llm.print(`\n=== Z-Image: ${width}x${height}, ${steps} steps, seed ${seed}${useCfg ? ', cfg ' + cfg : ''}${input ? ', img2img strength ' + strength : ''} ===`);

		const initLatent = input ? encodeImage(config, input, width, height) : null;

		const t0 = llm.now();
		const cond = embed(config, config.prompt ?? '', config.text_embeddings);
		const uncond = useCfg ? embed(config, '', null) : null;
		llm.print(`  [phase] text_encode: ${llm.since(t0)}`);

		const t1 = llm.now();
		const model = llm.open(config.model);
		const dit = new ZImageDiT(model);
		dit.load();
		const loras = loraList(config);
		if (loras.length > 0) mergeLoras(dit, loras);
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
		const sched = sigmas(steps, strength);
		const count = dit.channels * latentH * latentW;
		const x0 = llm.randomNormal(count, seed);
		if (initLatent) {
			// x = sigma * noise + (1 - sigma) * image, at the first sigma.
			const s0 = sched[0];
			for (let i = 0; i < count; i++) x0[i] = s0 * x0[i] + (1 - s0) * initLatent[i];
		}
		dit.a.latent.buffer.write(x0);
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
		const pixels = decode(config, latent, latentH, latentW);
		const img = image.fromTensor(pixels, width, height, 3);
		llm.print(`  [phase] vae_decode: ${llm.since(t3)}`);

		const out = config.output ?? 'output.png';
		image.savePng(out, img);
		llm.print(`=== done in ${llm.since(start)}, saved ${out} ===`);
		return { output: out, width, height };
	},
});
