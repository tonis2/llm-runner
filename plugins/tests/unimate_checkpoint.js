// `llm-runner tests/unimate_checkpoint.js ckpt=<.pt>`: the .pt reader against numbers from torch.
import { llm } from '../lib/llm.js';
import { UniMateCheckpoint } from '../unimate/checkpoint.js';
const path = globalThis.__llm_config?.ckpt ?? '/run/media/tonis/extra/AI models/UniMate/unimate_uniml3d_f60_v2/checkpoints/checkpoint_step_100000.pt';
const c = new UniMateCheckpoint(path);
for (const n of ['cond_embedder.weight', 'rope_j.spectral_encoder.phi.0.weight', 'transformer_blocks.9.mlp.w3.bias', 'final_layer.joint_out.2.bias', 'tpos_pool.pool.queries']) {
	const t = c.get(n);
	let s = 0, s2 = 0;
	for (const v of t.data) { s += v; s2 += v * v; }
	llm.print(`${n} [${t.shape}] sum ${s.toFixed(6)} sumsq ${s2.toFixed(6)} first ${t.data[0].toFixed(7)}`);
}
llm.print(`tensors: ${c.tensors.size}`);
