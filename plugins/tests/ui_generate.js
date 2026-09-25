// A real generation inside the window: the Z-Image template, run by an
// Executor that awaits `three.nextFrame()`, with a TAESD preview of every step
// put on screen through `three.ui.texture`.
//
//   llm-runner ui --entry plugins/tests/ui_generate.js taesd=taef1.safetensors [size=512]
//
// Checks that the window kept drawing through the run (frames counted by an
// animation loop) and writes tests/out/ui_generate.png, then closes.

import { llm, image } from '../lib/llm.js';
import { FORMATS } from '../lib/latents.js';
import { Executor } from '../lib/graph/executor.js';
import { loadPlugins } from '../lib/graph/plugins.js';
import { decodeLatent } from '../lib/latents.js';
import { setBreather } from '../lib/gpu.js';

const config = globalThis.__llm_config ?? {};
const size = config.size ?? 512;

// The longest time between two frames is how long the window froze.
let frames = 0;
let lastFrame = 0;
let longest = 0;
three.setAnimationLoop(() => {
	frames++;
	const now = llm.now();
	if (lastFrame) longest = Math.max(longest, now - lastFrame);
	lastFrame = now;
});

let tex = 0;
let status = 'loading';
function show(image) {
	if (image) tex = three.ui.texture(image, tex);
	const ops = [{ op: 'text', at: [24, 20], text: status, size: 20, color: 0xe8e8e8 }];
	if (tex) ops.push({ op: 'image', at: [24, 60], size: [512, 512], texture: tex, radius: 6 });
	three.ui.set({ type: 'stack', children: [{ type: 'rect', color: 0x1b1d23 }, { type: 'draw', ops }] });
}

async function main() {
	show();
	await loadPlugins();
	const graph = JSON.parse(llm.readText(`${llm.root}/graph/templates/zimage-t2i.json`));
	for (const n of graph.nodes) {
		if (n.id === 'sample') Object.assign(n.params, { width: size, height: size });
		if (n.id === 'save') { n.type = 'core.preview'; n.params = { image: n.params.image }; }
	}

	setBreather(() => three.nextFrame());
	const executor = new Executor({ yieldFrame: () => three.nextFrame() });
	const t0 = llm.now();
	const framesBefore = frames;
	longest = 0;
	let previews = 0;
	const { results } = await executor.run(graph, {
		async onProgress(e) {
			if (e.done) {
				status = `${e.node} done in ${(e.ms / 1000).toFixed(2)}s`;
				show();
			} else if (e.latent && config.taesd) {
				const lat = e.latent();
				const f = FORMATS[lat.format].factor;
				show(image.fromTensor(await decodeLatent(config.taesd, lat), lat.w * f, lat.h * f, 3));
				previews++;
				status = `${e.node} ${e.step}/${e.total}`;
				show();
			}
		},
	});
	const final = results.save.image;
	status = `done: ${final.width}x${final.height} in ${((llm.now() - t0) / 1000).toFixed(1)}s, ${frames - framesBefore} frames drawn (longest gap ${longest.toFixed(0)}ms), ${previews} previews`;
	show(final);
	await three.nextFrame();
	await three.nextFrame();
	const shot = three.screenshot('tests/out/ui_generate.png');
	console.log(`ui_generate: ${status}; wrote ${shot.path}`);
	if (frames - framesBefore < 8) console.log('ui_generate: FAILED - the window did not draw during the run');
	three.quit();
}

main().catch((e) => {
	console.log(`ui_generate: FAILED - ${e.stack ?? e.message ?? e}`);
	three.quit();
});
