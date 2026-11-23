// Pixal3D's cascade against the torch reference's (shape1024.safetensors):
// the 512 latent's voxels after the decoder's four subdivisions and their
// 64^3 tokens, the 1024 picture's conditioning at the reference's tokens,
// and the 1024 shape flow's first two calls (conditioned, then negative)
// from the reference's noise and conditioning.
//
//   llm-runner tests/pixal3d_cascade.js golden=build/pixal3d/golden/example4 image=build/pixal3d/inputs/example4.png weights="/path/pixal3d" gguf="/path/pixal3d-base-flow-f16.gguf" [linear=f16|q8|f32]
import { llm, image } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { preprocess } from '/plugins/pixal3d/image.js';
import { Conditioner, camera } from '/plugins/pixal3d/condition.js';
import { ShapeDecoder } from '/plugins/pixal3d/decoder.js';
import { Flow } from '/plugins/pixal3d/flow.js';
import { cascadeTokens } from '/plugins/pixal3d/pipeline.js';
import { upload, readFloats } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config, linear = cfg.linear ?? 'f16';
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
	llm.print(`${name}: ${got.length === want.length ? '' : `LENGTH ${got.length} vs ${want.length}, `}worst ${worst.toFixed(4)}, mean ${(sum / want.length).toExponential(2)}, cosine ${(dot / Math.sqrt(gg * ww)).toFixed(6)}, rms ${Math.sqrt(ww / want.length).toFixed(3)}`);
}
// [n, 4] (batch, x, y, z) to [n, 3]
function xyz(c4) {
	const n = c4.length / 4, out = new Int32Array(n * 3);
	for (let t = 0; t < n; t++) for (let a = 0; a < 3; a++) out[3 * t + a] = c4[4 * t + 1 + a];
	return out;
}
// cells of a and b (each [n, 3] on an r^3 grid) both have
function shared(a, b, r) {
	const s = new Set();
	for (let i = 0; i < a.length / 3; i++) s.add((a[3 * i] * r + a[3 * i + 1]) * r + a[3 * i + 2]);
	let both = 0;
	for (let i = 0; i < b.length / 3; i++) if (s.has((b[3 * i] * r + b[3 * i + 1]) * r + b[3 * i + 2])) both++;
	return both;
}

const sh = openSafetensors(`${cfg.golden}/shape512.safetensors`);
const gold = openSafetensors(`${cfg.golden}/shape1024.safetensors`);
const coords = xyz(gold.ints('coords')), n = coords.length / 3;

// 1. the tokens
{
	const dec = new ShapeDecoder(`${cfg.weights}/ckpts`, { linear });
	dec.load();
	const latent = upload(sh.floats('slat'));
	const { coords: hr, r } = await dec.run(latent, xyz(sh.ints('coords')), 32, { upsampleOnly: 4 });
	latent.dispose();
	dec.dispose();
	const want = xyz(gold.ints('hr512'));
	llm.print(`voxels at ${r}^3: ours ${hr.length / 3}, reference ${want.length / 3}, shared ${shared(hr, want, r)}`);
	const tokens = cascadeTokens(hr, 64);
	let same = tokens.length === coords.length;
	for (let i = 0; same && i < coords.length; i++) same = tokens[i] === coords[i];
	llm.print(`tokens at 64^3: ours ${tokens.length / 3}, reference ${n}, shared ${shared(tokens, coords, 64)}${same ? ', same order' : ''}`);
	// from the reference's voxels exactly
	const fromRef = cascadeTokens(want, 64);
	let exact = fromRef.length === coords.length;
	for (let i = 0; exact && i < coords.length; i++) exact = fromRef[i] === coords[i];
	llm.print(`tokens from the reference's voxels: ${exact ? 'identical' : 'DIFFERENT'}`);
}

// 2. the 1024 picture's conditioning at the reference's tokens
const cond = new Conditioner(cfg.weights);
await cond.load();
const pic = await cond.picture(preprocess(image.load(cfg.image)), 1024, { naf: 512 });
cond.dispose();
const global = cond.global(pic), proj = cond.proj(pic, 64, camera(0.2), coords);
cond.disposePicture(pic);
compare('1024 global', readFloats(global, 5 * 1024), gold.floats('global'));
{
	const got = readFloats(proj, n * 2048), want = gold.floats('proj');
	const half = (a, h) => { const o = new Float32Array(n * 1024); for (let t = 0; t < n; t++) o.set(a.subarray(t * 2048 + h * 1024, t * 2048 + h * 1024 + 1024), t * 1024); return o; };
	compare('1024 proj, DINO half', half(got, 0), half(want, 0));
	compare('1024 proj, NAF half', half(got, 1), half(want, 1));
}
global.dispose();
proj.dispose();

// 3. the 1024 flow's first two calls on the reference's conditioning
{
	const gguf = llm.open(cfg.gguf);
	const flow = new Flow(gguf, 'shape1024', { linear });
	await flow.load();
	const C = flow.outC;
	const c = { global: upload(gold.floats('global')), proj: upload(gold.floats('proj')) };
	const x = upload(gold.floats('noise'));
	const want = { v0: gold.floats('v_cond0'), v1: gold.floats('v_uncond0') };
	const t0 = llm.now();
	await flow.sample(x, coords, c, n, {
		steps: 1, guidance: 7.5, rescale: 0.5, interval: [0.6, 1], rescaleT: 3,
		onCall: (i, t, v) => {
			if (i === 0) compare(`call 0 (t ${(1000 * t).toFixed(1)}, reference ${gold.floats('t0')[0].toFixed(1)}), conditioned`, readFloats(v, n * C), want.v0);
			if (i === 1) compare('call 1, negative', readFloats(v, n * C), want.v1);
		},
	});
	llm.print(`two calls on ${n} tokens in ${llm.since(t0)}`);
	x.dispose();
	c.global.dispose();
	c.proj.dispose();
	flow.dispose();
	gguf.close();
}
