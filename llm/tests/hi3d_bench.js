// Hi3D's flow, timed: one conditioned and one zero-condition call over the
// flow golden's voxels, then (with profile=true) a kernel table of a further
// conditioned call.
//
//   llm-runner tests/hi3d_bench.js golden=build/hi3d/golden/example4 weights="/path/hi3d" [profile=true]
import { llm } from '/core/llm.js';
import { submit, profileReport, profileReset } from '/core/gpu.js';
import { openSafetensors } from '/core/weights/torch.js';
import { Hi3DFlow } from '/plugins/hi3d/flow.js';
import { upload } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config;
const coords = openSafetensors(`${cfg.golden}/voxels.safetensors`).ints('coords'), n = coords.length / 3;
const config = JSON.parse(llm.readText(`${cfg.weights}/transformer_1024/config.json`));
const flow = new Hi3DFlow(openSafetensors(`${cfg.weights}/transformer_1024/model.safetensors`), config, { linear: cfg.linear ?? 'f16' });
await flow.load();
const st = flow.structure(coords, n);
const cond = upload(new Float32Array(2917 * 2048).map((_, i) => Math.sin(i) * 0.03));
const pos = flow.context(cond), neg = flow.context(null);
const x = upload(new Float32Array(n * 32).map((_, i) => Math.cos(i))), v = upload(new Float32Array(n * 32));
for (const [name, ctx] of [['warm-up', pos], ['conditioned', pos], ['zero condition', neg]]) {
	const t0 = llm.now();
	await flow.forward(x, 0.9, st, ctx, v);
	submit();
	v.buffer.readBytes(0, 4);
	llm.print(`${name}: ${llm.since(t0)}`);
}
if (cfg.profile) {
	profileReset();
	await flow.forward(x, 0.9, st, pos, v);
	submit();
	profileReport('Hi3D flow, conditioned');
}
for (const d of [pos, neg, x, v, cond, st]) d.dispose();
flow.dispose();
