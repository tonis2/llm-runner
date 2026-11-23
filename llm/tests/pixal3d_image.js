// Pixal3D's preprocessing against the reference's: the squared-up cutout
// (preprocessed.png) and its 512 Lanczos resize (image512), byte for byte.
//
//   llm-runner tests/pixal3d_image.js golden=build/pixal3d/golden/example4 image=build/pixal3d/inputs/example4.png
import { llm, image } from '/core/llm.js';
import { preprocess, planes } from '/plugins/pixal3d/image.js';

const cfg = globalThis.__llm_config;
const pre = preprocess(image.load(cfg.image));
const want = image.load(`${cfg.golden}/preprocessed.png`);
let diff = 0, worst = 0;
if (pre.width !== want.width || pre.height !== want.height) llm.print(`size ${pre.width}x${pre.height}, reference ${want.width}x${want.height}`);
else {
	for (let i = 0; i < pre.pixels.length; i++) {
		const d = Math.abs(pre.pixels[i] - want.pixels[i]);
		if (d) diff++;
		worst = Math.max(worst, d);
	}
	llm.print(`preprocessed ${pre.width}x${pre.height}: ${diff} of ${pre.pixels.length} bytes differ, worst ${worst}`);
}
const p = planes(pre, 512), w = llm.open(`${cfg.golden}/prep.safetensors`).floats('image512');
diff = 0;
worst = 0;
for (let i = 0; i < p.length; i++) {
	const d = Math.abs(p[i] - w[i]) * 255;
	if (d > 0.01) diff++;
	worst = Math.max(worst, d);
}
llm.print(`512: ${diff} of ${p.length} values differ, worst ${worst.toFixed(2)} / 255`);
