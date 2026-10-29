// The host side of Kimodo's sampling: the DDIM schedule and step, the
// global-to-local root conversion between the two denoiser stages, and the
// decode of a normalised motion into joint rotations. Ported from kimodo.cpp
// (diffusion.cpp, motion_rep.cpp, motion_decode.cpp), which follows upstream.

// Upstream's Stats: a feature is (x - mean) / sqrt(std^2 + 1e-5).
const scale = (std) => Math.sqrt(std * std + 1e-5);

// The cosine schedule over `baseSteps`, respaced to `steps` evenly spaced
// timesteps: { timesteps, alpha, alphaPrev }, index 0 the cleanest.
export function cosineSchedule(baseSteps, steps) {
	const alphaBar = (t) => Math.cos(((t + 0.008) / 1.008) * Math.PI / 2) ** 2;
	const base = new Float32Array(baseSteps);
	let cumulative = 1;
	for (let i = 0; i < baseSteps; i++) {
		const beta = Math.min(1 - alphaBar((i + 1) / baseSteps) / alphaBar(i / baseSteps), 0.999);
		cumulative *= 1 - beta;
		base[i] = cumulative;
	}
	const stride = (baseSteps - 1) / Math.max(1, steps - 1);
	const timesteps = [], alpha = [], alphaPrev = [];
	let previous = 1;
	for (let i = 0; i < steps; i++) {
		const t = Math.min(Math.floor(i * stride + 0.5), baseSteps - 1);
		const a = Math.max(base[t], Math.fround(1e-9));
		timesteps.push(t);
		alpha.push(a);
		alphaPrev.push(previous);
		previous = a;
	}
	return { timesteps, alpha, alphaPrev };
}

// One eta = 0 DDIM step from x_t and the predicted clean motion, into `out`.
export function ddimStep(schedule, index, x, clean, out) {
	const f = Math.fround;
	const alpha = schedule.alpha[index], previous = schedule.alphaPrev[index];
	const reciprocal = f(1 / f(Math.sqrt(alpha)));
	const reciprocalM1 = f(Math.sqrt(f(f(1 - alpha) / alpha)));
	const keep = f(Math.sqrt(previous)), noise = f(Math.sqrt(f(1 - previous)));
	for (let i = 0; i < x.length; i++) {
		const epsilon = f(f(f(reciprocal * x[i]) - clean[i]) / reciprocalM1);
		out[i] = f(clean[i] * keep) + f(noise * epsilon);
	}
}

// The root stage's normalised global root [batch, frames, 5] (x, y, z, heading
// cos, sin) as the body stage's normalised local root [batch, frames, 4]:
// heading rate, x and z speed (per second, at `fps`) and height.
export function globalToLocalRoot(root, batch, frames, stats, fps) {
	const { globalMean: gm, globalStd: gs, localMean: lm, localStd: ls } = stats;
	const out = new Float32Array(batch * frames * 4);
	const angle = new Float64Array(frames), x = new Float64Array(frames), y = new Float64Array(frames), z = new Float64Array(frames);
	for (let b = 0; b < batch; b++) {
		for (let t = 0; t < frames; t++) {
			const p = (b * frames + t) * 5;
			x[t] = root[p] * scale(gs[0]) + gm[0];
			y[t] = root[p + 1] * scale(gs[1]) + gm[1];
			z[t] = root[p + 2] * scale(gs[2]) + gm[2];
			angle[t] = Math.atan2(root[p + 4] * scale(gs[4]) + gm[4], root[p + 3] * scale(gs[3]) + gm[3]);
		}
		for (let t = 0; t < frames; t++) {
			const next = t + 1 < frames ? t + 1 : frames - 1, prev = t + 1 < frames ? t : frames - 2;
			const cosDiff = Math.cos(angle[next]) * Math.cos(angle[prev]) + Math.sin(angle[next]) * Math.sin(angle[prev]);
			const sinDiff = Math.sin(angle[next]) * Math.cos(angle[prev]) - Math.cos(angle[next]) * Math.sin(angle[prev]);
			const raw = [fps * Math.atan2(sinDiff, cosDiff), fps * (x[next] - x[prev]), fps * (z[next] - z[prev]), y[t]];
			for (let d = 0; d < 4; d++) out[(b * frames + t) * 4 + d] = (raw[d] - lm[d]) / scale(ls[d]);
		}
	}
	return out;
}

// A 6D rotation (two columns) as a row-major 3x3, Gram-Schmidt.
function sixD(v, o) {
	let n = Math.hypot(v[o], v[o + 1], v[o + 2]);
	const a = [v[o] / n, v[o + 1] / n, v[o + 2] / n];
	const zc = [a[1] * v[o + 5] - a[2] * v[o + 4], a[2] * v[o + 3] - a[0] * v[o + 5], a[0] * v[o + 4] - a[1] * v[o + 3]];
	n = Math.hypot(zc[0], zc[1], zc[2]);
	const z = [zc[0] / n, zc[1] / n, zc[2] / n];
	const b = [z[1] * a[2] - z[2] * a[1], z[2] * a[0] - z[0] * a[2], z[0] * a[1] - z[1] * a[0]];
	return [a[0], b[0], z[0], a[1], b[1], z[1], a[2], b[2], z[2]];
}

// transpose(p) * m
function localOf(p, m) {
	const r = new Array(9).fill(0);
	for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) r[i * 3 + j] += p[k * 3 + i] * m[k * 3 + j];
	return r;
}

function quaternion(m, q, o) {
	let w, x, y, z;
	const t = m[0] + m[4] + m[8];
	if (t > 0) {
		const s = 2 * Math.sqrt(t + 1);
		w = 0.25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s;
	} else if (m[0] > m[4] && m[0] > m[8]) {
		const s = 2 * Math.sqrt(1 + m[0] - m[4] - m[8]);
		w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s;
	} else if (m[4] > m[8]) {
		const s = 2 * Math.sqrt(1 + m[4] - m[0] - m[8]);
		w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s;
	} else {
		const s = 2 * Math.sqrt(1 + m[8] - m[0] - m[4]);
		w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s;
	}
	q[o] = x; q[o + 1] = y; q[o + 2] = z; q[o + 3] = w;
}

// A normalised motion [frames, 9 + 12 J] as parent-local joint rotations
// [frames, J, 4] (x, y, z, w), root positions [frames, 3] in metres and foot
// contacts [frames, 4] (left heel, left toe, right heel, right toe; 1 down, 0
// not, at upstream's 0.5).
export function decodeMotion(motion, frames, skeleton, stats) {
	const J = skeleton.parents.length;
	const D = 9 + 12 * J, body = D - 5, rotations = 5 + 3 * J;
	const { globalMean: gm, globalStd: gs, bodyMean: bm, bodyStd: bs } = stats;
	const local = new Float32Array(frames * J * 4);
	const root = new Float32Array(frames * 3);
	const contacts = new Float32Array(frames * 4);
	const f = new Float64Array(D);
	const global = new Array(J);
	for (let t = 0; t < frames; t++) {
		for (let i = 0; i < 5; i++) f[i] = motion[t * D + i] * scale(gs[i]) + gm[i];
		for (let i = 0; i < body; i++) f[5 + i] = motion[t * D + 5 + i] * scale(bs[i]) + bm[i];
		root[t * 3] = f[0] + f[5];
		root[t * 3 + 1] = f[6];
		root[t * 3 + 2] = f[2] + f[7];
		for (let k = 0; k < 4; k++) contacts[t * 4 + k] = f[D - 4 + k] > 0.5 ? 1 : 0;
		for (let j = 0; j < J; j++) global[j] = sixD(f, rotations + j * 6);
		for (let j = 0; j < J; j++) {
			const p = skeleton.parents[j];
			quaternion(p < 0 ? global[j] : localOf(global[p], global[j]), local, (t * J + j) * 4);
		}
	}
	return { rotations: local, root, contacts };
}
