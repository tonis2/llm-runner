// The latent's life outside the DiT: the noise it starts as, the sigma schedule
// it is stepped along, the reference images encoded into it, and the decode
// back to pixels.

import { llm, image } from '../lib/llm.js';
import { FluxVAEDecoder, FluxVAEEncoder } from '../lib/flux_vae.js';

// Flux 2's schedule: an empirical shift fitted to the image's token count,
// applied to a linear ramp, ending at 0. `strength` < 1 starts partway down
// (img2img).
export function sigmas(steps, imageSeqLen, strength = 1) {
	const a1 = 8.73809524e-5, b1 = 1.89833333, a2 = 0.00016927, b2 = 0.45666666;
	let mu;
	if (imageSeqLen > 4300) {
		mu = a2 * imageSeqLen + b2;
	} else {
		const m200 = a2 * imageSeqLen + b2;
		const m10 = a1 * imageSeqLen + b1;
		const a = (m200 - m10) / 190;
		const b = m200 - 200 * a;
		mu = a * steps + b;
	}
	const shift = Math.exp(mu);
	let top = 1;
	if (strength < 1) {
		const s = Math.min(Math.max(strength, 0), 1);
		top = s / (shift * (1 - s) + s);
	}
	const out = new Float32Array(steps + 1);
	for (let i = 0; i < steps; i++) {
		const frac = (i * (1 - 1 / steps)) / (steps > 1 ? steps - 1 : 1);
		const t = Math.fround(top * (1 - frac));
		out[i] = shift === 1 ? t : (shift * t) / (1 + (shift - 1) * t);
	}
	out[steps] = 0;
	llm.print(`  schedule: mu ${mu.toFixed(3)}, shift ${shift.toFixed(3)}, sigmas ${Array.from(out).map((x) => x.toFixed(4)).join(' ')}`);
	return out;
}

export function noise(count, seed) {
	return llm.randomNormal(count, seed);
}

// [C, H, W] -> [C / r^2, H r, W r]
function pixelShuffle(input, channels, h, w, r) {
	const oc = channels / (r * r), oh = h * r, ow = w * r;
	const out = new Float32Array(oc * oh * ow);
	for (let c = 0; c < oc; c++) {
		for (let y = 0; y < oh; y++) {
			for (let x = 0; x < ow; x++) {
				const ic = c * r * r + (y % r) * r + (x % r);
				out[c * oh * ow + y * ow + x] = input[ic * h * w + Math.floor(y / r) * w + Math.floor(x / r)];
			}
		}
	}
	return out;
}

// [C, H r, W r] -> [C r^2, H, W], the inverse.
function pixelUnshuffle(input, channels, h, w, r) {
	const oh = h / r, ow = w / r;
	const out = new Float32Array(channels * r * r * oh * ow);
	for (let c = 0; c < channels; c++) {
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				const dc = c * r * r + (y % r) * r + (x % r);
				out[dc * oh * ow + Math.floor(y / r) * ow + Math.floor(x / r)] = input[c * h * w + y * w + x];
			}
		}
	}
	return out;
}

// out[oc, s] = bias[oc] + sum_ic W[oc, ic] * in[ic, s]
function conv1x1(input, weight, bias, inC, outC, spatial) {
	const out = new Float32Array(outC * spatial);
	for (let oc = 0; oc < outC; oc++) {
		const row = oc * inC;
		const base = oc * spatial;
		for (let s = 0; s < spatial; s++) out[base + s] = bias[oc];
		for (let ic = 0; ic < inC; ic++) {
			const wv = weight[row + ic];
			const src = ic * spatial;
			for (let s = 0; s < spatial; s++) out[base + s] += wv * input[src + s];
		}
	}
	return out;
}

// DiT latent [128, h, w] -> image floats [3, 16h, 16w] in [0, 1]: undo the
// batch norm, pixel-shuffle to the VAE's 32 channels, post_quant_conv, decode.
export function decodeLatent(config, latent, latentH, latentW) {
	const vae = llm.open(config.vae);
	const mean = vae.floats('bn.running_mean');
	const variance = vae.floats('bn.running_var');
	const spatial = latentH * latentW;
	const channels = mean.length;
	for (let c = 0; c < channels; c++) {
		const std = Math.sqrt(variance[c] + 1e-5);
		for (let j = 0; j < spatial; j++) latent[c * spatial + j] = latent[c * spatial + j] * std + mean[c];
	}
	const vaeCh = channels / 4;
	const h = latentH * 2, w = latentW * 2;
	let x = pixelShuffle(latent, channels, latentH, latentW, 2);
	if (vae.has('post_quant_conv.weight')) {
		x = conv1x1(x, vae.floats('post_quant_conv.weight'), vae.floats('post_quant_conv.bias'), vaeCh, vaeCh, h * w);
	}
	const decoder = new FluxVAEDecoder(vae);
	const pixels = decoder.decode(x, h, w);
	decoder.dispose();
	vae.close();
	return pixels;
}

// An image to the DiT's latent space: [3, H, W] in [-1, 1] through the encoder,
// quant_conv, the mean half, unshuffled to 128 channels, then the batch norm
// the decode undoes. Returns [128, H/16, W/16].
function encodeToLatent(vae, encoder, img) {
	const H = img.height, W = img.width;
	const enc = encoder.encode(image.toTensor(img), H, W);
	const spatial = enc.h * enc.w;
	let x = enc.data;
	if (vae.has('quant_conv.weight')) {
		x = conv1x1(x, vae.floats('quant_conv.weight'), vae.floats('quant_conv.bias'), enc.channels, enc.channels, spatial);
	}
	const half = enc.channels / 2;
	const latent = pixelUnshuffle(x.subarray(0, half * spatial), half, enc.h, enc.w, 2);
	const mean = vae.floats('bn.running_mean');
	const variance = vae.floats('bn.running_var');
	const ls = (enc.h / 2) * (enc.w / 2);
	for (let c = 0; c < mean.length; c++) {
		const std = Math.sqrt(variance[c] + 1e-5);
		for (let j = 0; j < ls; j++) latent[c * ls + j] = (latent[c * ls + j] - mean[c]) / std;
	}
	return latent;
}

function loadInput(input) {
	return typeof input === 'string' ? image.load(input) : input;
}

// Reference images for img2img (one, cropped to the output, the starting
// latent) or kontext (up to four, each at its own aspect ratio, packed
// token-major after the noise tokens and told apart by rope's T axis).
export function encodeReference(config, inputs, mode, width, height) {
	const vae = llm.open(config.vae);
	const encoder = new FluxVAEEncoder(vae);
	try {
		if (mode === 'img2img') {
			const img = image.cropTo(loadInput(inputs[0]), width, height);
			return { refs: [], refPatches: null, initLatent: encodeToLatent(vae, encoder, img) };
		}
		if (inputs.length > 4) throw new Error(`kontext takes at most 4 reference images, not ${inputs.length}`);
		// References are only ever scaled down: to the output's long side, or to
		// max_ref_long_side when that is set lower.
		let ceiling = Math.max(width, height);
		if (config.max_ref_long_side) ceiling = Math.floor(config.max_ref_long_side / 16) * 16;
		const refs = [];
		const latents = [];
		for (const input of inputs) {
			const src = loadInput(input);
			const native = Math.max(src.width, src.height);
			const img = image.fit16(src, Math.min(ceiling, native));
			const latent = encodeToLatent(vae, encoder, img);
			refs.push({ w: img.width / 16, h: img.height / 16 });
			latents.push(latent);
		}
		const channels = latents[0].length / (refs[0].w * refs[0].h);
		const total = refs.reduce((a, r) => a + r.w * r.h, 0);
		const packed = new Float32Array(total * channels);
		let at = 0;
		for (let i = 0; i < refs.length; i++) {
			const n = refs[i].w * refs[i].h;
			const chw = latents[i];
			for (let p = 0; p < n; p++) for (let c = 0; c < channels; c++) packed[(at + p) * channels + c] = chw[c * n + p];
			at += n;
		}
		llm.print(`  kontext: ${refs.length} reference image(s), ${total} tokens`);
		return { refs, refPatches: packed, initLatent: null };
	} finally {
		encoder.dispose();
		vae.close();
	}
}

export { pixelShuffle, pixelUnshuffle, conv1x1 };
