// Hi3D end to end on a small voxel set against the torch reference's
// (small.safetensors from ref_hi3d.py small): the whole Heun sample from the
// reference's noise (the calls' times and the first four outputs, the final
// latent), then the decoder from the reference's latent (each level's cells
// and features, the output's signs and fields).
//
//   llm-runner tests/hi3d_small.js golden=build/hi3d/golden/example4 mesh=build/pixal3d/out/example4_1024.glb weights="/path/hi3d" [linear=f16|f32] [decoder=f32|q8]
import { llm } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { readMeshGlb } from '/core/mesh/glb.js';
import { Hi3DFlow } from '/plugins/hi3d/flow.js';
import { Hi3DDecoder } from '/plugins/hi3d/decoder.js';
import { surfaceVoxels } from '/plugins/hi3d/voxels.js';
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
	llm.print(`${name}: ${got.length === want.length ? '' : `LENGTH ${got.length} vs ${want.length}, `}worst ${worst.toExponential(3)}, mean ${(sum / want.length).toExponential(2)}, cosine ${(dot / Math.sqrt(gg * ww)).toFixed(6)}, rms ${Math.sqrt(ww / want.length).toFixed(4)}`);
}
// The rows of `got` (cells `gc`) in the order of `want`'s cells `wc`: the
// reference's cells, ours aligned to them (missing rows zero).
function aligned(gc, got, wc, width) {
	const at = new Map();
	for (let i = 0; i < gc.length / 3; i++) at.set(`${gc[3 * i]},${gc[3 * i + 1]},${gc[3 * i + 2]}`, i);
	const out = new Float32Array((wc.length / 3) * width);
	let shared = 0;
	for (let i = 0; i < wc.length / 3; i++) {
		const j = at.get(`${wc[3 * i]},${wc[3 * i + 1]},${wc[3 * i + 2]}`);
		if (j === undefined) continue;
		shared++;
		out.set(got.subarray(j * width, (j + 1) * width), i * width);
	}
	return { out, shared };
}

const gold = openSafetensors(`${cfg.golden}/small.safetensors`);
const coords = gold.ints('coords'), n = coords.length / 3;
{
	const mesh = readMeshGlb(cfg.mesh);
	const v = surfaceVoxels(mesh.positions, mesh.indices, { resolution: 128, budget: Number(gold.metadata.budget) });
	let same = v.coords.length === coords.length;
	for (let i = 0; same && i < coords.length; i++) same = v.coords[i] === coords[i];
	llm.print(`voxels: grid ${v.grid} (reference ${gold.metadata.grid}), ${v.coords.length / 3} (reference ${n})${same ? ', identical' : ', DIFFERENT'}`);
}

if (cfg.skipFlow !== 'true') {
	const config = JSON.parse(llm.readText(`${cfg.weights}/transformer_1024/config.json`));
	const flow = new Hi3DFlow(openSafetensors(`${cfg.weights}/transformer_1024/model.safetensors`), config, { linear });
	await flow.load();
	const st = flow.structure(coords, n);
	const prep = openSafetensors(`${cfg.golden}/prep.safetensors`);
	const dino = prep.floats('dino'), clip = prep.floats('clip');
	const condRows = new Float32Array(2917 * 2048);
	for (let t = 0; t < 2917; t++) { condRows.set(dino.subarray(t * 1024, (t + 1) * 1024), t * 2048); condRows.set(clip, t * 2048 + 1024); }
	const cond = upload(condRows);
	const x = upload(gold.floats('noise'));
	const times = [], conds = [];
	let call = 0;
	await flow.sample(x, st, cond, {
		onCall: (t, c, v) => {
			times.push(t);
			conds.push(c);
			if (call < 4) compare(`call ${call} (t ${t.toFixed(6)}, ${c ? 'conditioned' : 'zero'})`, readFloats(v, n * 32), gold.floats(`call${call}_v`));
			call++;
		},
	});
	const wantT = gold.floats('call_t'), wantC = gold.ints('call_cond');
	let sameCalls = times.length === wantT.length;
	// the reference logs the model's time, 1000 t
	for (let i = 0; sameCalls && i < times.length; i++) sameCalls = Math.abs(Math.fround(1000 * times[i]) - wantT[i]) < 1e-3 && (conds[i] ? 1 : 0) === wantC[i];
	llm.print(`calls: ours ${times.length}, reference ${wantT.length}${sameCalls ? ', same times and conditions' : ', DIFFERENT'}`);
	compare('final latent (denormalised)', readFloats(x, n * 32), gold.floats('latents'));
	for (const d of [x, cond, st]) d.dispose();
	flow.dispose();
}

{
	const dec = new Hi3DDecoder(`${cfg.weights}/vae/model.safetensors`, { linear: cfg.decoder ?? 'f32' });
	dec.load();
	const latent = upload(gold.floats('latents'));
	const grab = (name, t, r, w, cells) => {
		const wc = gold.ints(`${name}_coord`);
		const { out, shared } = aligned(cells, readFloats(t, r * w), wc, w);
		llm.print(`${name}: ours ${r} cells, reference ${wc.length / 3}, shared ${shared}`);
		compare(`  ${name} features`, out, gold.floats(`${name}_feat`));
	};
	const d = await dec.run(latent, coords, { grab });
	latent.dispose();
	const wc = gold.ints('decoded_coord'), cells = d.coords.length / 3;
	const ds = new Float32Array(cells * 8), df = new Float32Array(cells * 32);
	for (let i = 0; i < cells; i++) {
		for (let c = 0; c < 8; c++) ds[i * 8 + c] = d.raw[i * 40 + c];
		for (let c = 0; c < 32; c++) df[i * 32 + c] = Math.tanh(d.raw[i * 40 + 8 + c]);
	}
	const sign = aligned(d.coords, ds, wc, 8), feat = aligned(d.coords, df, wc, 32);
	llm.print(`decoded: ours ${d.coords.length / 3} cells at ${d.r}^3, reference ${wc.length / 3}, shared ${sign.shared}`);
	compare('  sign logits', sign.out, gold.floats('decoded_sign'));
	compare('  fields (tanh)', feat.out, gold.floats('decoded_feat'));
	dec.dispose();
}
