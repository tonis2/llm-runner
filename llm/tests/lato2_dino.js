// DINOv2 ViT-L/14 with registers against the torch reference's tokens for the
// same pixels (LATO.2's conditioning: x_prenorm, layer-normed without affine).
//
//   llm-runner tests/lato2_dino.js golden=build/lato2/golden/example0 weights="/path/lato2"
import { llm, f32 } from '/core/llm.js';
import * as op from '/core/ops.js';
import { submit } from '/core/gpu.js';
import { openTorch } from '/core/weights/torch.js';
import { DinoV2 } from '/core/models/dinov2.js';

const cfg = globalThis.__llm_config;
const gold = llm.open(`${cfg.golden}/dino.safetensors`);
const want = gold.floats('cond');
const read = (t, n) => new Float32Array(t.buffer.readBytes(0, n * 4).buffer);

for (const linear of ['f32', 'q8']) {
	const dino = new DinoV2(openTorch(`${cfg.weights}/dinov2_vitl14_reg4_pretrain.pth`), { linear });
	await dino.load();
	const tokens = await dino.encode(gold.floats('pixels'), 518, 518);
	const n = tokens.elements / dino.dim;
	const normed = f32(n * dino.dim);
	op.layerNorm(tokens, normed, dino.dim, n, 1e-5);
	submit();
	const got = read(normed, n * dino.dim);
	let worst = 0, sum = 0;
	for (let i = 0; i < want.length; i++) { const d = Math.abs(got[i] - want[i]); worst = Math.max(worst, d); sum += d; }
	llm.print(`${linear}: ${n} tokens, worst ${worst.toFixed(4)}, mean ${(sum / want.length).toExponential(2)} (values are layer-normed, ~1)`);
	tokens.dispose();
	normed.dispose();
	dino.dispose();
}
