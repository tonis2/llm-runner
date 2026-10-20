// The joint finder's head (creature/detect.js) and its training step
// (creature/train.js) on the GPU, against the same sums in float64 on the
// host, on small random numbers: no model, no download.
//
//   llm-runner tests/creature_head.js
//
// 1. The forward pass: the PCA, the hidden layer and the scores.
// 2. The loss: each joint's scores softmaxed over a picture's patches against a
//    Gaussian about its target, a joint the picture does not have left out, a
//    target with a width of its own (a flat one: a hidden landmark) taking it.
// 3. Every gradient (w1, b1, w2, b2) against central differences of that loss.
// 4. peaks(): a joint's heatmap with two bumps gives both, the bigger first,
//    and the refined peak between patches where the mass is.
// 5. peaks() places a Gaussian's peak between patches exactly, moves smoothly as
//    the bump crosses from one patch to the next, and keeps to a few hundredths
//    of a patch under noise.
// 6. Looking mirrored: each point's partner on the other side, and the mirrored
//    crop's column i is the crop's column S - 1 - i.
// 7. The PCA when the features' variances span eight orders of magnitude (a
//    middle block's do): orthonormal directions, the top ones found, whitened
//    to unit variance, nothing lost to float32.
import { llm, f32 } from '../lib/llm.js';
import { Head, peaks, partner, fittedLandmarks } from '../creature/detect.js';
import { cropGeometry } from '../lib/warp.js';
import { stepBuffers, headGradients, principal, orthonormalRows, generator } from '../creature/train.js';

let failures = 0;
const check = (ok, msg) => { llm.print(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

let seed = 7;
const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
const fill = (n, s = 1) => Float32Array.from({ length: n }, () => random() * s);

const B = 2, GW = 4, GH = 4, N = GW * GH, D = 12, K = 8, H = 6, J = 3, SIGMA = 1;
const w = { pca: fill(K * D, 0.6), pca_bias: fill(K, 0.2), w1: fill(H * K, 0.8), b1: fill(H, 0.3), w2: fill(J * H, 1.2), b2: fill(J, 0.3) };
const head = new Head(w, { dim: D, rank: K, hidden: H, joints: J });
const tokens = fill(B * N * D, 2);
// Targets [image, joint, (x, y, weight, width)]: image 1's last joint missing,
// image 0's second flat (a hidden landmark's, at a third of the weight).
const target = new Float32Array(B * J * 4);
for (let b = 0; b < B; b++) for (let j = 0; j < J; j++) {
	const o = (b * J + j) * 4;
	target[o] = 0.5 + (random() + 0.5) * 2.5; target[o + 1] = 0.5 + (random() + 0.5) * 2.5; target[o + 2] = b === 1 && j === J - 1 ? 0 : 1;
	if (b === 0 && j === 1) { target[o + 2] = 0.3; target[o + 3] = 1000; }
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
		const sg = target[o + 3] > 0 ? target[o + 3] : SIGMA;
		let m = -Infinity;
		for (let n = 0; n < N; n++) m = Math.max(m, S[(b * N + n) * J + j]);
		let z = 0, zt = 0;
		for (let n = 0; n < N; n++) {
			z += Math.exp(S[(b * N + n) * J + j] - m);
			zt += Math.exp(-((n % GW - target[o]) ** 2 + (Math.floor(n / GW) - target[o + 1]) ** 2) / (2 * sg * sg));
		}
		let part = 0;
		for (let n = 0; n < N; n++) {
			const t = Math.exp(-((n % GW - target[o]) ** 2 + (Math.floor(n / GW) - target[o + 1]) ** 2) / (2 * sg * sg)) / zt;
			part -= t * (S[(b * N + n) * J + j] - m - Math.log(z));
		}
		total += target[o + 2] * part;
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

// 5. Between patches.
{
	const gw = 10, gh = 9;
	// Log scores of a Gaussian of `sigma` patches at (cx, cy), plus `noise`.
	const bump = (cx, cy, sigma, noise = () => 0) => Float32Array.from({ length: gw * gh }, (_, n) => -((n % gw - cx) ** 2 + (Math.floor(n / gw) - cy) ** 2) / (2 * sigma * sigma) + noise());
	let worst = 0;
	for (const sigma of [0.8, 1, 1.4]) {
		for (let o = -0.45; o <= 0.451; o += 0.05) {
			const [f] = peaks(bump(4 + o, 4 - o * 0.7, sigma), 0, 1, gw, gh);
			worst = Math.max(worst, Math.abs(f.at[0] - (4 + o)), Math.abs(f.at[1] - (4 - o * 0.7)));
		}
	}
	check(worst < 0.02, `a Gaussian's peak is placed between patches (worst ${worst.toFixed(4)} of a patch)`);
	let jump = 0, before = null;
	for (let cx = 3.3; cx <= 3.701; cx += 0.01) {
		const [f] = peaks(bump(cx, 4.2, 1), 0, 1, gw, gh);
		if (before !== null) jump = Math.max(jump, Math.abs(f.at[0] - before - 0.01));
		before = f.at[0];
	}
	check(jump < 0.02, `the peak moves smoothly across a patch's edge (worst step off by ${jump.toFixed(4)})`);
	seed = 11;
	let noisy = 0;
	for (let k = 0; k < 40; k++) {
		const cx = 3 + (random() + 0.5) * 3, cy = 3 + (random() + 0.5) * 2;
		const [f] = peaks(bump(cx, cy, 1, () => random() * 0.2), 0, 1, gw, gh);
		noisy = Math.max(noisy, Math.hypot(f.at[0] - cx, f.at[1] - cy));
	}
	check(noisy < 0.12, `with noise of 0.1 in the log scores the peak stays within ${noisy.toFixed(3)} of a patch`);
}

// 6. Mirrored.
{
	const pairs = [['thigh_L', 'thigh_R'], ['ankle_R.heel', 'ankle_L.heel'], ['head.left', 'head.right'], ['LeftArm', 'RightArm'], ['l_eye', 'r_eye'], ['head.front', 'head.front'], ['hips', 'hips'], ['tail_5', 'tail_5']];
	const wrong = pairs.filter(([a, b]) => partner(a) !== b || partner(b) !== a);
	check(!wrong.length, `each point's partner is on the other side${wrong.length ? ` (not ${wrong.map(([a]) => `${a} -> ${partner(a)}`).join(', ')})` : ''}`);
	const marks = [['head.front', 'head'], ['head.left', 'head'], ['head.right', 'head'], ['ankle_L.heel', 'ankle_L'], ['ankle_R.heel', 'ankle_R']].map(([name, joint]) => ({ name, joint }));
	const kept = fittedLandmarks(marks).map((m) => m.name).join();
	check(kept === 'head.front,ankle_L.heel,ankle_R.heel', `a fit leaves out a mirror pair on one joint, not the heels (${kept})`);
	// A 640-wide frame, a 448 crop of the box [101, 40, 401, 340] and its mirror.
	const img = { width: 640, height: 480, channels: 3 }, S = 448, a = S / 300, tx = -a * 101 + (a - 1) / 2, d = a, ty = -d * 40 + (d - 1) / 2;
	const crop = cropGeometry(img, [a, 0, tx, 0, d, ty], { size: S });
	const mirrored = cropGeometry(img, [a, 0, S - 1 - tx - a * (img.width - 1), 0, d, ty], { size: S, mirror: true });
	let worst = 0;
	// Column i's two source columns and blend, against column S - 1 - i's taken the other way round.
	for (let i = 1; i < S - 1; i++) {
		const j = S - 1 - i;
		const at = (g, w, k) => (g.geo[k] / 3) * (1 - w.wts[k]) + (g.geo[S + k] / 3) * w.wts[k];
		worst = Math.max(worst, Math.abs(at(mirrored, mirrored, i) - at(crop, crop, j)));
	}
	check(worst < 0.01, `the mirrored crop's columns are the crop's, the other way round (worst ${worst.toFixed(4)} of a pixel)`);
}

// 7. The PCA across a wide spread.
{
	// Orthonormal rows out of rows whose lengths span 1e-6 to 1e6, nearly alike.
	const n = 12, D = 40, Z = new Float64Array(n * D);
	seed = 21;
	for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) Z[i * D + d] = 10 ** (6 - i) * ((d === 0 ? 5 : 0) + random());
	const Q = orthonormalRows(Z, n, D);
	let off = 0;
	for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
		let s = 0;
		for (let d = 0; d < D; d++) s += Q[i * D + d] * Q[j * D + d];
		off = Math.max(off, Math.abs(s - (i === j ? 1 : 0)));
	}
	check(off < 1e-9, `rows a million times apart in length come out orthonormal (worst ${off.toExponential(2)})`);

	// Features [N, D] with variance 10^(4 - 8 d / (D - 1)) along axis d, on the GPU.
	const N = 900, F = 48, rank = 6, x = new Float32Array(N * F), noise = llm.randomNormal(N * F, 31);
	const scale = Array.from({ length: F }, (_, d) => 10 ** (2 - (4 * d) / (F - 1)));
	for (let k = 0; k < N; k++) for (let d = 0; d < F; d++) x[k * F + d] = 3 + scale[d] * noise[k * F + d];
	const t = f32(N * F);
	t.buffer.write(x);
	const { pca, pca_bias } = await principal([t], N, F, rank, generator(5));
	t.dispose();
	let finite = true, outside = 0, worstVar = 0;
	for (let k = 0; k < rank; k++) {
		let inTop = 0, all = 0;
		for (let d = 0; d < F; d++) { finite = finite && Number.isFinite(pca[k * F + d]); all += pca[k * F + d] ** 2; if (d < rank) inTop += pca[k * F + d] ** 2; }
		outside = Math.max(outside, 1 - inTop / all);
		let m = 0, v = 0;
		for (let i = 0; i < N; i++) { let y = pca_bias[k]; for (let d = 0; d < F; d++) y += pca[k * F + d] * x[i * F + d]; m += y / N; v += (y * y) / N; }
		worstVar = Math.max(worstVar, Math.abs(v - m * m - 1), Math.abs(m));
	}
	check(finite && outside < 0.01, `the top directions are the top axes (at most ${(outside * 100).toFixed(2)}% outside them)`);
	check(worstVar < 0.05, `each component comes out with mean 0 and variance 1 (worst ${worstVar.toFixed(4)} off)`);
}

head.dispose();
llm.print(failures ? `${failures} failed` : 'all passed');
