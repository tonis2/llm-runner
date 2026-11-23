// Pixal3D's DINOv3 ViT-L/16 (the Hugging Face checkpoint) against the torch
// reference's tokens for the same 512 picture: every token, CLS and registers
// first, through a layer norm without affine (Pixal3D's extract_features).
//
//   llm-runner tests/pixal3d_dino.js golden=build/pixal3d/golden/example4 weights="/path/pixal3d"
import { llm } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { DinoV3 } from '/core/models/dinov3.js';

const cfg = globalThis.__llm_config;
const gold = llm.open(`${cfg.golden}/prep.safetensors`);
const want = gold.floats('dino512');
const pixels = gold.floats('image512');
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
for (let c = 0; c < 3; c++) {
	for (let i = c * 512 * 512; i < (c + 1) * 512 * 512; i++) pixels[i] = (pixels[i] - MEAN[c]) / STD[c];
}
const read = (t, n) => new Float32Array(t.buffer.readBytes(0, n * 4).buffer);

for (const linear of (cfg.linear ?? 'f32,q8').split(',')) {
	const dino = new DinoV3(openSafetensors(`${cfg.weights}/dinov3-vitl16/model.safetensors`), { prefix: '', linear });
	await dino.load();
	const tokens = await dino.encode(pixels, 512, 512, { all: true, norm: 'plain' });
	const n = tokens.elements / dino.dim;
	const got = read(tokens, n * dino.dim);
	const D = dino.dim;
	for (const [name, from, to] of [['prefix', 0, 5], ['patches', 5, n]]) {
		let worst = 0, sum = 0, dot = 0, gg = 0, ww = 0;
		for (let i = from * D; i < to * D; i++) {
			const d = Math.abs(got[i] - want[i]);
			worst = Math.max(worst, d);
			sum += d;
			dot += got[i] * want[i];
			gg += got[i] * got[i];
			ww += want[i] * want[i];
		}
		llm.print(`${linear} ${name}: worst ${worst.toFixed(4)}, mean ${(sum / ((to - from) * D)).toExponential(2)}, cosine ${(dot / Math.sqrt(gg * ww)).toFixed(6)}`);
	}
	tokens.dispose();
	dino.dispose();
}
