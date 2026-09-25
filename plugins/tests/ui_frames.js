// What the studio needs from the engine, checked in a window:
//
//   llm-runner ui --entry plugins/tests/ui_frames.js
//
// - `three.ui.texture` puts pixels on screen through an `image` op, and a
//   second call with the handle replaces them (same size: a copy; new size: a
//   new image under the same handle).
// - `await three.nextFrame()` hands the thread back to the frame loop: long
//   work split by it lets the window draw between the pieces.
// - `__llm` is installed on the loop's runtime.
//
// Writes tests/out/ui_frames.png (the last frame) and closes the window.

const W = 256;
const H = 192;

function gradient(width, height, phase) {
	const px = new Uint8Array(width * height * 3);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const i = (y * width + x) * 3;
			px[i] = (x * 255 / width + phase) & 255;
			px[i + 1] = (y * 255) / height;
			px[i + 2] = phase & 255;
		}
	}
	return { width, height, channels: 3, pixels: px };
}

function show(tex, label) {
	three.ui.set({
		type: 'stack',
		children: [
			{ type: 'rect', color: 0x1b1d23 },
			{
				type: 'draw',
				ops: [
					{ op: 'text', at: [24, 20], text: label, size: 20, color: 0xe8e8e8 },
					{ op: 'image', at: [24, 60], size: [W * 2, H * 2], texture: tex, radius: 8 },
				],
			},
		],
	});
}

async function main() {
	if (typeof globalThis.__llm?.print !== 'function') throw new Error('__llm is not installed on the loop runtime');
	let tex = three.ui.texture(gradient(W, H, 0));
	show(tex, 'frame 0');

	// Thirty frames: each await is one drawn frame, and a texture update each.
	const t0 = three.clock.wall ?? 0;
	let frames = 0;
	for (let i = 1; i <= 30; i++) {
		await three.nextFrame();
		three.ui.texture(gradient(W, H, i * 8), tex);
		frames++;
	}
	if (frames !== 30) throw new Error(`expected 30 frames, got ${frames}`);

	// A different size under the same handle.
	const again = three.ui.texture(gradient(W / 2, H / 2, 128), tex);
	if (again !== tex) throw new Error(`a resize gave a new handle ${again}, not ${tex}`);
	show(tex, `resized, ${frames} frames`);
	await three.nextFrame();
	await three.nextFrame();

	// Freed and made again: the spare is reused, not grown.
	three.ui.freeTexture(tex);
	const reused = three.ui.texture(gradient(W, H, 200));
	if (reused !== tex) throw new Error(`a freed handle was not reused (${reused} after ${tex})`);
	show(reused, 'reused handle');
	await three.nextFrame();

	const shot = three.screenshot('tests/out/ui_frames.png');
	console.log(`ui_frames: ok - ${frames} frames, texture ${tex}, wrote ${shot.path} ${shot.width}x${shot.height}`);
	three.quit();
}

main().catch((e) => {
	console.log(`ui_frames: FAILED - ${e.message ?? e}`);
	three.quit();
});
