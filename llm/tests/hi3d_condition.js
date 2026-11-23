// Hi3D's conditioning against the torch reference's (prep.safetensors from
// ref_hi3d.py prep): the prepared picture, both encoders' pixels, DINOv2-large's
// tokens at 756 (Hugging Face checkpoint, final norm, positions resized) and
// CLIP ViT-L/14's pooled embedding.
//
//   llm-runner tests/hi3d_condition.js golden=build/hi3d/golden/example4 image=build/pixal3d/inputs/example4.png weights="/path/hi3d" [linear=f32|q8]
import { llm, image } from '/core/llm.js';
import { openSafetensors } from '/core/weights/torch.js';
import { DinoV2 } from '/core/models/dinov2.js';
import { ClipVision } from '/core/models/clip_vision.js';
import { prepare, dinoPixels, clipPixels, DINO_SIZE } from '/plugins/hi3d/image.js';
import { readFloats } from '/plugins/lato2/nn.js';

const cfg = globalThis.__llm_config, linear = cfg.linear ?? 'f32';
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
	llm.print(`${name}: ${got.length === want.length ? '' : `LENGTH ${got.length} vs ${want.length}, `}worst ${worst.toFixed(5)}, mean ${(sum / want.length).toExponential(2)}, cosine ${(dot / Math.sqrt(gg * ww)).toFixed(6)}, rms ${Math.sqrt(ww / want.length).toFixed(3)}`);
}
const unit = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.max(Math.sqrt(s), 1e-12); return v.map((x) => x / s); };

const gold = openSafetensors(`${cfg.golden}/prep.safetensors`);
const pre = prepare(image.load(cfg.image));
{
	const want = gold.ints('prepared'), [h, w] = gold.dims('prepared');
	let off = 0, worst = 0;
	if (pre.width === w && pre.height === h) {
		for (let i = 0; i < want.length; i++) {
			const d = Math.abs(pre.pixels[i] - want[i]);
			if (d) off++;
			worst = Math.max(worst, d);
		}
	}
	llm.print(`prepared: ours ${pre.width}x${pre.height}, reference ${w}x${h}, ${off} bytes differ (worst ${worst})`);
}
const t0 = llm.now();
const dpx = dinoPixels(pre), cpx = clipPixels(pre);
llm.print(`pixels in ${llm.since(t0)}`);
compare('DINOv2 pixels', dpx, gold.floats('dino_pixels'));
compare('CLIP pixels', cpx, gold.floats('clip_pixels'));

{
	const dino = new DinoV2(openSafetensors(`${cfg.weights}/dinov2-large/model.safetensors`), { linear });
	await dino.load();
	const t1 = llm.now();
	dino.prefixRows(DINO_SIZE / 14, DINO_SIZE / 14);
	llm.print(`positions resized in ${llm.since(t1)}`);
	const tokens = await dino.encode(gold.floats('dino_pixels'), DINO_SIZE, DINO_SIZE, { norm: 'model' });
	const n = tokens.elements / dino.dim, got = readFloats(tokens, n * dino.dim);
	tokens.dispose();
	compare(`DINOv2 ${n} tokens (reference pixels)`, got, gold.floats('dino_raw'));
	const D = dino.dim, normed = new Float32Array(got.length);
	for (let t = 0; t < n; t++) normed.set(unit(got.subarray(t * D, (t + 1) * D)), t * D);
	compare('DINOv2 tokens, L2-normalised', normed, gold.floats('dino'));
	const ours = await dino.encode(dpx, DINO_SIZE, DINO_SIZE, { norm: 'model', quiet: true });
	compare('DINOv2 tokens (our pixels)', readFloats(ours, n * D), gold.floats('dino_raw'));
	ours.dispose();
	dino.dispose();
}
{
	const clip = new ClipVision(openSafetensors(`${cfg.weights}/clip-vit-large-patch14/model.safetensors`), { linear });
	await clip.load();
	const pooled = await clip.pooled(gold.floats('clip_pixels'));
	compare('CLIP pooled (reference pixels)', pooled, gold.floats('clip_raw'));
	compare('CLIP pooled, L2-normalised', unit(pooled), gold.floats('clip'));
	compare('CLIP pooled (our pixels)', await clip.pooled(cpx, { quiet: true }), gold.floats('clip_raw'));
	clip.dispose();
}
