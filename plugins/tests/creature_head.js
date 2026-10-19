// The joint finder's head (creature/detect.js) and its training step
// (creature/train.js) on the GPU, against the same sums in float64 on the
// host, on small random numbers: no model, no download.
//
//   llm-runner tests/creature_head.js
//
// 1. The forward pass: the PCA, the hidden layer and the scores.
// 2. The loss: each joint's scores softmaxed over a picture's patches against a
//    Gaussian about its target, a joint the picture does not have left out.
// 3. Every gradient (w1, b1, w2, b2) against central differences of that loss.
// 4. peaks(): a joint's heatmap with two bumps gives both, the bigger first,
//    and the refined peak between patches where the mass is.
import { llm, f32 } from '../lib/llm.js';
import { Head, peaks } from '../creature/detect.js';
import { stepBuffers, headGradients } from '../creature/train.js';

let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

let seed = 7;
const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
const fill = (n, s = 1) => Float32Array.from({ length: n }, () => random() * s);

const B = 2, GW = 4, GH = 4, N = GW * GH, D = 12, K = 8, H = 6, J = 3, SIGMA = 1;
const w = { pca: fill(K * D, 0.6), pca_bias: fill(K, 0.2), w1: fill(H * K, 0.8), b1: fill(H, 0.3), w2: fill(J * H, 1.2), b2: fill(J, 0.3) };
const head = new Head(w, { dim: D, rank: K, hidden: H, joints: J });
const tokens = fill(B * N * D, 2);
// Targets [image, joint, (x, y, weight, -)]: image 1's last joint missing.
const target = new Float32Array(B * J * 4);
for (let b = 0; b < B; b++) for (let j = 0; j < J; j++) {
	const o = (b * J + j) * 4;
	target[o] = 0.5 + (random() + 0.5) * 2.5; target[o + 1] = 0.5 + (random() + 0.5) * 2.5; target[o + 2] = b === 1 && j === J - 1 ? 0 : 1;
}

// The host's sums, in float64.
function forward(wt) {
	const F = new Float64Array(B * N * K), A = new Float64Array(B * N * H), S = new Float64Array(B * N * J);
	for (let r = 0; r < B * N; r++) {
		for (let k = 0; k < K; k++) {
			let v = w.pca_bias[k];
			for (let d = 0; d < D; d++) v += w.pca[k * D + d] * tokens[r * D + d];
			F[r * K + k] = v;
		}
		for (let h = 0; h < H; h++) {
			let v = wt.b1[h];
			for (let k = 0; k < K; k++) v += wt.w1[h * K + k] * F[r * K + k];
			A[r * H + h] = Math.max(0, v);
		}
		for (let j = 0; j < J; j++) {
			let v = wt.b2[j];
			for (let h = 0; h < H; h++) v += wt.w2[j * H + h] * A[r * H + h];
			S[r * J + j] = v;
		}
	}
	return { F, S };
}
function loss(wt) {
	const { S } = forward(wt);
	let total = 0;
	for (let b = 0; b < B; b++) for (let j = 0; j < J; j++) {
		const o = (b * J + j) * 4;
		if (!(target[o + 2] > 0)) continue;
		let m = -Infinity;
		for (let n = 0; n < N; n++) m = Math.max(m, S[(b * N + n) * J + j]);
		let z = 0, zt = 0;
		for (let n = 0; n < N; n++) {
			z += Math.exp(S[(b * N + n) * J + j] - m);
			zt += Math.exp(-((n % GW - target[o]) ** 2 + (Math.floor(n / GW) - target[o + 1]) ** 2) / (2 * SIGMA * SIGMA));
		}
		for (let n = 0; n < N; n++) {
			const t = Math.exp(-((n % GW - target[o]) ** 2 + (Math.floor(n / GW) - target[o + 1]) ** 2) / (2 * SIGMA * SIGMA)) / zt;
			total -= t * (S[(b * N + n) * J + j] - m - Math.log(z));
		}
	}
	return total / B;
}

// 1. Forward.
{
	const t = f32(B * N * D);
	t.buffer.write(tokens);
	const gpu = head.scores(t, B * N);
	t.dispose();
	const { S } = forward(w);
	let worst = 0;
	for (let i = 0; i < S.length; i++) worst = Math.max(worst, Math.abs(gpu[i] - S[i]));
	check(worst < 1e-3, `the head's scores match the host's (worst ${worst.toExponential(2)})`);
}

// 2 and 3. Loss and gradients: the features as the trainer keeps them.
{
	const g = stepBuffers({ B, N, J, rank: K, hidden: H });
	const t = f32(B * N * D);
	t.buffer.write(tokens);
	head.project(t, g.x, B * N);
	t.dispose();
	g.target.buffer.write(target);
	const w2t = new Float32Array(H * J);
	for (let j = 0; j < J; j++) for (let h = 0; h < H; h++) w2t[h * J + j] = w.w2[j * H + h];
	g.w2t.buffer.write(w2t);
	const { grads, loss: parts } = headGradients(head, g, { B, N, J, G: GW, rank: K, hidden: H, sigma: SIGMA });
	for (const x of Object.values(g)) x.dispose();
	const gpuLoss = parts.reduce((a, b) => a + b, 0) / B;
	const hostLoss = loss(w);
	check(Math.abs(gpuLoss - hostLoss) < 1e-3 * Math.max(1, Math.abs(hostLoss)), `the heatmap loss matches the host's (${gpuLoss.toFixed(5)} vs ${hostLoss.toFixed(5)})`);
	check(parts[(1 * J + J - 1)] === 0, 'a joint the picture does not have costs nothing');
	for (const name of ['w1', 'b1', 'w2', 'b2']) {
		let worst = 0, scale = 0;
		for (let i = 0; i < w[name].length; i++) {
			const plus = { ...w, [name]: Float64Array.from(w[name]) }, minus = { ...w, [name]: Float64Array.from(w[name]) };
			plus[name][i] += 1e-4;
			minus[name][i] -= 1e-4;
			const fd = (loss(plus) - loss(minus)) / 2e-4;
			worst = Math.max(worst, Math.abs(fd - grads[name][i]));
			scale = Math.max(scale, Math.abs(fd));
		}
		check(worst < 2e-3 * Math.max(1, scale), `d loss / d ${name} matches central differences (worst ${worst.toExponential(2)} of ${scale.toFixed(3)})`);
	}
}

// 4. peaks().
{
	const gw = 8, gh = 6, scores = new Float32Array(gw * gh);
	for (let n = 0; n < gw * gh; n++) {
		const x = n % gw, y = Math.floor(n / gw);
		scores[n] = Math.log(Math.exp(-((x - 2.3) ** 2 + (y - 2) ** 2) / 1.2) * 3 + Math.exp(-((x - 6) ** 2 + (y - 4) ** 2) / 1.2) * 1.5 + 1e-4);
	}
	const found = peaks(scores, 0, 1, gw, gh);
	check(found.length === 2, `two bumps, two peaks (${found.length})`);
	check(Math.abs(found[0].at[0] - 2.3) < 0.35 && Math.abs(found[0].at[1] - 2) < 0.2, `the bigger first, refined between patches (${found[0].at.map((v) => v.toFixed(2))})`);
	check(found[1] && Math.abs(found[1].at[0] - 6) < 0.3 && found[1].mass < found[0].mass, `the smaller second (${found[1]?.at.map((v) => v.toFixed(2))}, mass ${found[1]?.mass.toFixed(2)} of ${found[0].mass.toFixed(2)})`);
}

head.dispose();
llm.print(failures ? `${failures} failed` : 'all passed');
