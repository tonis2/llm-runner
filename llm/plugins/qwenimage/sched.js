// Qwen-Image 2.1's sigmas: diffusers' FlowMatchEulerDiscreteScheduler as the
// model ships it - a linear ramp from 1 to 1/steps, an exponential time shift
// whose strength grows with the image's token count, stretched so the last
// sigma lands on 0.02, then 0.

const BASE_SEQ = 256, MAX_SEQ = 8192, BASE_SHIFT = 0.5, MAX_SHIFT = 0.9;
const TERMINAL = 0.02;

export function shiftFor(tokens) {
	const m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ);
	return tokens * m + (BASE_SHIFT - m * BASE_SEQ);
}

// `steps` + 1 sigmas. Below `strength` 1 (img2img) the schedule is cut to its
// tail: the steps that start at or below `strength`.
export function sigmas(steps, tokens, strength = 1) {
	const mu = Math.exp(shiftFor(tokens));
	const s = new Float64Array(steps);
	for (let i = 0; i < steps; i++) {
		const t = steps > 1 ? 1 + (i * (1 / steps - 1)) / (steps - 1) : 1;
		s[i] = mu / (mu + (1 / t - 1));
	}
	const scale = (1 - s[steps - 1]) / (1 - TERMINAL);
	const out = [];
	for (let i = 0; i < steps; i++) out.push(1 - (1 - s[i]) / scale);
	const cut = strength >= 1 ? out : out.filter((x) => x <= strength + 1e-9);
	return Float32Array.from([...cut, 0]);
}
