// Hi3D's flow against the torch reference's (flow.safetensors from ref_hi3d.py
// flow, over voxels.safetensors' voxels and prep.safetensors' conditioning): the
// schedule, the tokens, the activations after the input blocks, the first
// block and all 24, and the first two calls (conditioned, then the zero
// condition) from the reference's noise.
//
//   llm-runner tests/hi3d_flow.js golden=build/hi3d/golden/example4 weights="/path/hi3d" [linear=f16|f32]
import { llm } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { Hi3DFlow, schedule } from '/plugins/hi3d/flow.js';
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

const gold = openSafetensors(`${cfg.golden}/flow.safetensors`);
const coords = openSafetensors(`${cfg.golden}/voxels.safetensors`).ints('coords'), n = coords.length / 3;

const sch = schedule(25, n);
compare('warped times', sch.warped, gold.floats('warped'));
compare('sigmas', sch.sigmas, gold.floats('sigmas'));
compare('evaluated times', sch.hats, gold.floats('sigma_hats'));

const config = JSON.parse(llm.readText(`${cfg.weights}/transformer_1024/config.json`));
const flow = new Hi3DFlow(openSafetensors(`${cfg.weights}/transformer_1024/model.safetensors`), config, { linear });
await flow.load();
const st = flow.structure(coords, n);
{
	const want = gold.ints('tokens');
	let same = want.length === st.tokens.length;
	for (let i = 0; same && i < want.length; i++) same = want[i] === st.tokens[i];
	llm.print(`tokens: ours ${st.m}, reference ${want.length / 3}${same ? ', same order' : ', DIFFERENT'}`);
}
const prep = openSafetensors(`${cfg.golden}/prep.safetensors`);
const dino = prep.floats('dino'), clip = prep.floats('clip');
const condRows = new Float32Array(2917 * 2048);
for (let t = 0; t < 2917; t++) { condRows.set(dino.subarray(t * 1024, (t + 1) * 1024), t * 2048); condRows.set(clip, t * 2048 + 1024); }
const cond = upload(condRows);
const pos = flow.context(cond), neg = flow.context(null);
const x = upload(gold.floats('noise')), v = upload(new Float32Array(n * 32));
const t = sch.hats[0];
const t0 = llm.now();
let debug = null;
try { debug = openSafetensors(`${cfg.golden}/in_debug.safetensors`); } catch { /* ref_hi3d's dbg_hi3d_in.py not run */ }
await flow.forward(x, t, st, pos, v, (name, tensor, r, w) => {
	const from = gold.has?.(name) || !debug ? gold : debug;
	if (from === debug || name.startsWith('h_')) compare(name, readFloats(tensor, r * w), (name.startsWith('h_') ? gold : debug).floats(name));
});
compare(`call 0 (t ${t}), conditioned`, readFloats(v, n * 32), gold.floats('v_cond0'));
llm.print(`  in ${llm.since(t0)}`);
const t1 = llm.now();
await flow.forward(x, t, st, neg, v);
compare('call 1, zero condition', readFloats(v, n * 32), gold.floats('v_uncond0'));
llm.print(`  in ${llm.since(t1)}`);
for (const d of [pos, neg, x, v, cond, st]) d.dispose();
flow.dispose();
