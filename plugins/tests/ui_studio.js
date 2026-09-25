// The studio driven by a scripted pointer and keyboard: edits a number inside
// a node, types a prompt, drags a node and zooms, and checks the graph after
// each step. Leaves the clipboard as it found it (text only).
//
//   llm-runner ui --entry plugins/tests/ui_studio.js

import { llm } from '../lib/llm.js';
import { app } from '../studio/app.js';
import { layoutOf } from '../studio/layout.js';
import { PALETTE_W, BAR_H, LABEL_H } from '../studio/theme.js';
import '../studio/main.js';

const frames = async (n) => { for (let i = 0; i < n; i++) await three.nextFrame(); };
const fails = [];
const check = (ok, what) => { llm.print(`ui_studio: ${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) fails.push(what); };

// A point on the canvas, in graph units, to the window's pixels.
function pixel(g) {
	const c = app.widgets.canvas;
	const [x, y] = c.toScreen(g);
	const s = three.ui.scale;
	return [(x + PALETTE_W + 1) * s, (y + BAR_H) * s];
}
function fieldPoint(id, input, fx = 0.5) {
	const n = app.doc.node(id);
	const f = layoutOf(n, app.images[id]).fields.find((f) => f.input.name === input);
	return pixel([n.pos[0] + f.x + f.w * fx, n.pos[1] + f.y + LABEL_H + 10]);
}
async function click([x, y]) {
	three.input.movePointer(x, y);
	await frames(2);
	three.input.pressButton(0);
	await frames(2);
	three.input.releaseButton(0);
	await frames(2);
}
async function tap(...keys) {
	for (const k of keys) three.input.press(k);
	await frames(2);
	for (const k of [...keys].reverse()) three.input.release(k);
	await frames(2);
}
async function typeText(text) {
	three.clipboard.write(text);
	await tap('controlleft', 'v');
}

(async () => {
	while (!app.widgets.canvas || app.doc.nodes.length === 0) await frames(1);
	await frames(10);
	const saved = three.clipboard.read();

	// A number field inside the sampler.
	await click(fieldPoint('sample', 'steps'));
	await tap('controlleft', 'a');
	await typeText('9');
	check(app.doc.node('sample').params.steps === 9, `steps edited in the node: ${app.doc.node('sample').params.steps}`);

	// The prompt box: click at its end, add a line.
	await click(fieldPoint('prompt', 'prompt'));
	await tap('controlleft', 'a');
	await typeText('a fox\nin snow');
	const prompt = app.doc.node('prompt').params.prompt;
	check(prompt === 'a fox\nin snow', `prompt typed in the node: ${JSON.stringify(prompt)}`);
	await tap('escape');

	// Drag the decode node by its header.
	const n = app.doc.node('decode');
	const before = [...n.pos];
	const [hx, hy] = pixel([n.pos[0] + 60, n.pos[1] + 12]);
	three.input.movePointer(hx, hy);
	await frames(2);
	three.input.pressButton(0);
	await frames(2);
	for (let i = 1; i <= 5; i++) { three.input.movePointer(hx + i * 20, hy + i * 10); await frames(1); }
	three.input.releaseButton(0);
	await frames(2);
	const moved = [n.pos[0] - before[0], n.pos[1] - before[1]];
	check(moved[0] > 20 && moved[1] > 10, `decode dragged by ${moved.join(', ')} units`);

	// Zoom in with the wheel over the sampler, then edit again at the new zoom.
	const z0 = app.widgets.canvas.view.zoom;
	three.input.movePointer(...pixel([app.doc.node('sample').pos[0] + 20, app.doc.node('sample').pos[1] + 10]));
	await frames(2);
	three.input.scroll(3);
	await frames(3);
	const z1 = app.widgets.canvas.view.zoom;
	check(z1 > z0, `zoomed from ${z0.toFixed(2)} to ${z1.toFixed(2)}`);
	await click(fieldPoint('sample', 'cfg'));
	await tap('controlleft', 'a');
	await typeText('2.5');
	check(app.doc.node('sample').params.cfg === 2.5, `cfg edited after zooming: ${app.doc.node('sample').params.cfg}`);
	await tap('escape');

	three.clipboard.write(saved);
	await frames(3);
	three.ui.flush();
	const shot = three.screenshot('tests/out/ui_studio.png');
	llm.print(`ui_studio: ${fails.length ? `FAILED ${fails.length}` : 'all ok'}; wrote ${shot.path}`);
	three.quit();
})().catch((e) => { llm.print(`ui_studio: FAILED - ${e.stack ?? e}`); three.quit(); });
