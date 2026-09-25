// The latent's schedule and the noise it starts as. Encoding to and decoding
// from the latent is the VAE's: `lib/latents.js`, format 'flux2'.

import { llm } from '../lib/llm.js';

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
