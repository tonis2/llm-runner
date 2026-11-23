// Pixal3D's flows against the torch reference: the first two model calls
// (conditioned, then the zero-conditioned negative) and the finished sample,
// from the reference's own noise and conditioning.
//
//   llm-runner tests/pixal3d_flow.js golden=build/pixal3d/golden/example4 gguf="/path/pixal3d-base-flow-f16.gguf" weights="/path/pixal3d" [stage=ss|shape512] [linear=f32|q8]
//
// stage=ss also decodes the sample to occupancy and compares the 32^3 cells.
import { llm, f32 } from '/core/llm.js';
import { submit } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { Flow } from '/plugins/pixal3d/flow.js';
import { SparseStructureDecoder } from '/plugins/lato2/trellis.js';
import { upload, readFloats } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config, stage = cfg.stage ?? 'ss';
function compare(name, got, want) {
	let worst = 0, sum = 0, dot = 0, gg = 0, ww = 0;
	for (let i = 0; i < want.length; i++) {
		const d = Math.abs(got[i] - want[i]);
		worst = Math.max(worst, d);
		sum += d;
		dot += got[i] * want[i];
		gg += got[i] * got[i];
		ww += want[i] * want[i];
	}
	llm.print(`${name}: worst ${worst.toFixed(4)}, mean ${(sum / want.length).toExponential(2)}, cosine ${(dot / Math.sqrt(gg * ww)).toFixed(6)}, rms ${Math.sqrt(ww / want.length).toFixed(3)}`);
}
// [C, R^3] channel-first (torch's dense layout) to rows [R^3, C].
function toRows(a, C) {
	const n = a.length / C, out = new Float32Array(a.length);
	for (let c = 0; c < C; c++) for (let t = 0; t < n; t++) out[t * C + c] = a[c * n + t];
	return out;
}

const gguf = llm.open(cfg.gguf);
const flow = new Flow(gguf, stage, { linear: cfg.linear ?? 'f32' });
await flow.load();
const C = flow.outC;
let g, coords, noise, want, cond, params;
if (stage === 'ss') {
	g = openSafetensors(`${cfg.golden}/ss.safetensors`);
	const prep = llm.open(`${cfg.golden}/prep.safetensors`);
	const R = 16, n = R ** 3;
	coords = new Int32Array(n * 3);
	for (let t = 0; t < n; t++) { coords[3 * t] = Math.floor(t / (R * R)); coords[3 * t + 1] = Math.floor(t / R) % R; coords[3 * t + 2] = t % R; }
	noise = toRows(g.floats('noise'), C);
	want = { v0: toRows(g.floats('v_cond0'), C), v1: toRows(g.floats('v_uncond0'), C), z: toRows(g.floats('z'), C) };
	cond = { global: upload(prep.floats('ss_global')), proj: upload(prep.floats('ss_proj')) };
	params = { guidance: 7.5, rescale: 0.7, interval: [0.6, 1], rescaleT: 5 };
} else {
	g = openSafetensors(`${cfg.golden}/shape512.safetensors`);
	const c4 = g.ints('coords'), n = c4.length / 4;
	coords = new Int32Array(n * 3);
	for (let t = 0; t < n; t++) for (let a = 0; a < 3; a++) coords[3 * t + a] = c4[4 * t + 1 + a];
	noise = g.floats('noise');
	// the reference keeps the denormalised latent
	const norm = JSON.parse(llm.readText(`${cfg.weights}/pipeline.json`)).args.shape_slat_normalization;
	const slat = g.floats('slat'), z = new Float32Array(slat.length);
	for (let i = 0; i < slat.length; i++) z[i] = (slat[i] - norm.mean[i % C]) / norm.std[i % C];
	want = { v0: g.floats('v_cond0'), v1: g.floats('v_uncond0'), z };
	cond = { global: upload(g.floats('global')), proj: upload(g.floats('proj')) };
	params = { guidance: 7.5, rescale: 0.5, interval: [0.6, 1], rescaleT: 3 };
}
const n = coords.length / 3;
const steps = g.ints("steps")[0];
llm.print(`${stage}: ${n} tokens, ${steps} steps, t0 ${g.floats('t0')[0]}`);
const x = upload(noise);
await flow.sample(x, coords, cond, n, {
	steps, ...params,
	onCall: (i, t, v) => {
		if (i === 0) compare(`call 0 (t ${(1000 * t).toFixed(1)}), conditioned`, readFloats(v, n * C), want.v0);
		if (i === 1) compare('call 1, negative', readFloats(v, n * C), want.v1);
	},
});
const z = readFloats(x, n * C);
compare('sample', z, want.z);
flow.dispose();

if (stage === 'ss') {
	const dec = new SparseStructureDecoder(`${cfg.weights}/ckpts`);
	dec.load();
	const { logits, R } = dec.run(x, 16);
	submit();
	const occ = readFloats(logits, R ** 3);
	const wantLogits = g.floats('logits');
	compare('decoder logits (ours vs reference, from our sample)', occ, wantLogits);
	// 64^3 occupancy max-pooled to 32^3, against the reference's cells
	const cells = new Set();
	for (let i = 0; i < R ** 3; i++) {
		if (occ[i] <= 0) continue;
		const xx = Math.floor(i / (R * R)), yy = Math.floor(i / R) % R, zz = i % R;
		cells.add(((xx >> 1) * 32 + (yy >> 1)) * 32 + (zz >> 1));
	}
	const ref = g.ints('coords'), refSet = new Set();
	for (let t = 0; t < ref.length / 4; t++) refSet.add((ref[4 * t + 1] * 32 + ref[4 * t + 2]) * 32 + ref[4 * t + 3]);
	let both = 0;
	for (const c of cells) if (refSet.has(c)) both++;
	llm.print(`32^3 cells: ours ${cells.size}, reference ${refSet.size}, shared ${both}`);
	logits.dispose();
	dec.dispose();
}
x.dispose();
cond.global.dispose();
cond.proj.dispose();
