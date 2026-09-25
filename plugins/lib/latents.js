// Latent formats: the space a denoiser works in, and the VAE that maps it to
// pixels and back.
//
//   flux2  Flux 2's VAE: 32 channels at /8, pixel-unshuffled to 128 at /16 and
//          batch-normalised with the VAE's own running stats (`bn.*`).
//   flux1  The Flux 1 VAE (`ae.safetensors`, Z-Image's): 16 channels at /8,
//          shifted by 0.1159 and scaled by 0.3611. taef1 decodes it too.
//
// A LATENT value is { format, data: Float32Array [C, h, w], channels, h, w }.
// A VAE decodes only its own format; that check is what lets a graph swap one
// VAE for another (the Flux 1 VAE for TAESD) and refuse the swaps that cannot
// work.

import { llm, image } from './llm.js';
import { FluxVAEDecoder, FluxVAEEncoder } from './flux_vae.js';
import { TAESDDecoder } from './taesd.js';

export const FORMATS = {
	flux2: { channels: 128, factor: 16 },
	flux1: { channels: 16, factor: 8 },
};

const FLUX1_SCALE = 0.3611;
const FLUX1_SHIFT = 0.1159;

// A VAE file's kind, from its tensor names: 'flux2', 'flux1' or 'taesd'.
export function vaeKind(model) {
	if (model.has('bn.running_mean')) return 'flux2';
	if (model.has('decoder.layers.0.weight')) return 'taesd';
	return 'flux1';
}

export const KIND_FORMAT = { flux2: 'flux2', flux1: 'flux1', taesd: 'flux1' };

export function makeLatent(format, data, h, w) {
	return { format, data, channels: data.length / (h * w), h, w };
}

// [C, H, W] -> [C / r^2, H r, W r]
export function pixelShuffle(input, channels, h, w, r) {
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
export function pixelUnshuffle(input, channels, h, w, r) {
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
export function conv1x1(input, weight, bias, inC, outC, spatial) {
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

// Flux 2: undo the batch norm, pixel-shuffle to the VAE's 32 channels,
// post_quant_conv, decode. [128, h, w] -> [3, 16h, 16w].
function decodeFlux2(vae, data, latentH, latentW) {
	const latent = data.slice();
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
	return pixels;
}

// Flux 2: [3, H, W] in [-1, 1] through the encoder, quant_conv, the mean half,
// unshuffled to 128 channels, then the batch norm the decode undoes.
function encodeFlux2(vae, encoder, img) {
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
	return makeLatent('flux2', latent, enc.h / 2, enc.w / 2);
}

// Flux 1: the encoder's mean, shifted and scaled.
function encodeFlux1(encoder, img) {
	const enc = encoder.encode(image.toTensor(img), img.height, img.width);
	const n = (enc.channels / 2) * enc.h * enc.w;
	const latent = new Float32Array(n);
	for (let i = 0; i < n; i++) latent[i] = (enc.data[i] - FLUX1_SHIFT) * FLUX1_SCALE;
	return makeLatent('flux1', latent, enc.h, enc.w);
}

// An image through a VAE file into its latent format. The image's sides must
// be multiples of the format's factor.
export function encodeImage(vaePath, img) {
	const vae = llm.open(vaePath);
	const kind = vaeKind(vae);
	if (kind === 'taesd') {
		vae.close();
		throw new Error(`${vaePath} is a TAESD decoder; it has no encoder`);
	}
	const encoder = new FluxVAEEncoder(vae);
	try {
		return kind === 'flux2' ? encodeFlux2(vae, encoder, img) : encodeFlux1(encoder, img);
	} finally {
		encoder.dispose();
		vae.close();
	}
}

// A latent through a VAE file to [3, H, W] floats in [0, 1].
export function decodeLatent(vaePath, latent) {
	const vae = llm.open(vaePath);
	try {
		const kind = vaeKind(vae);
		const format = KIND_FORMAT[kind];
		if (latent.format !== format) {
			throw new Error(`${vaePath.split('/').pop()} decodes ${format} latents; this one is ${latent.format}`);
		}
		if (kind === 'flux2') return decodeFlux2(vae, latent.data, latent.h, latent.w);
		if (kind === 'taesd') {
			const decoder = new TAESDDecoder(vae);
			const pixels = decoder.decode(latent.data, latent.h, latent.w);
			decoder.dispose();
			return pixels;
		}
		const data = latent.data.slice();
		for (let i = 0; i < data.length; i++) data[i] = data[i] / FLUX1_SCALE + FLUX1_SHIFT;
		const decoder = new FluxVAEDecoder(vae);
		const pixels = decoder.decode(data, latent.h, latent.w);
		decoder.dispose();
		return pixels;
	} finally {
		vae.close();
	}
}
