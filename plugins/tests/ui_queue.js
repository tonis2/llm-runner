// The studio's run queue on a graph that needs no model (load an image, save
// it): items are snapshots, run in order, a failed one stops the queue and is
// fixed with Load / Update, reordering, removing, stop-after and cancel.
//
//   llm-runner ui --entry plugins/tests/ui_queue.js

import { llm, image } from '../lib/llm.js';
import { app, cancel } from '../studio/app.js';
import { Doc } from '../studio/doc.js';
import {
	addToQueue, runQueue, stopQueueAfterCurrent, loadItem, updateItem, moveItem, removeItem, clearDone,
} from '../studio/queue.js';
import '../studio/main.js';

const frames = async (n) => { for (let i = 0; i < n; i++) await three.nextFrame(); };
const fails = [];
const check = (ok, what) => { llm.print(`ui_queue: ${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) fails.push(what); };

const dir = `${llm.root}/tests/out/ui_queue`;
globalThis.__llm.makeDir(dir);

// Two small pictures to feed in, of different shapes.
function picture(path, w, h, rgb) {
	const f = new Float32Array(3 * w * h);
	for (let c = 0; c < 3; c++) for (let i = 0; i < w * h; i++) f[c * w * h + i] = rgb[c] * (0.3 + 0.7 * (i % w) / w);
	image.savePng(path, image.fromTensor(f, w, h, 3));
	return path;
}
const a = picture(`${dir}/in-a.png`, 64, 48, [1, 0.3, 0.2]);
const b = picture(`${dir}/in-b.png`, 48, 64, [0.2, 0.4, 1]);
const missing = `${dir}/not-there.png`;

const setInput = (path) => { app.doc.setParam('load', 'path', path); app.changed(); };
const statuses = () => app.queue.map((i) => i.status).join(',');

(async () => {
	while (!app.widgets.chrome) await frames(1);
	await frames(5);
	app.doc = new Doc({
		nodes: [
			{ id: 'load', type: 'core.load_image', params: { path: a } },
			{ id: 'save', type: 'core.save_image', params: { image: { from: 'load.image' }, path: `${dir}/out-#.png` } },
		],
	});
	app.leftTab = 'queue';
	app.refresh();
	await frames(3);

	// Three items, the doc edited between them.
	addToQueue();
	setInput(missing);
	addToQueue();
	setInput(b);
	addToQueue();
	const [i1, i2, i3] = app.queue;
	check(app.queue.length === 3, `three items queued (${app.queue.length})`);
	check(i1.inputs[0] === a && i2.inputs[0] === missing && i3.inputs[0] === b, 'each item kept the input it was queued with');
	check(!!i1.thumb && !i2.thumb && !!i3.thumb, 'thumbnails for the inputs that exist');
	check(!app.statusError, `no warning with a numbered save path: ${app.status}`);

	// A failure stops the queue; what came before is done, what comes after waits.
	await runQueue();
	await frames(3);
	check(statuses() === 'done,error,pending', `stopped at the failed item: ${statuses()}`);
	check(i1.outputs.length === 1 && llm.exists(i1.outputs[0]), `item 1 saved ${i1.outputs[0]}`);
	check(!!i2.error, `item 2's error kept: ${i2.error}`);
	check(!app.queueRunning && !app.running, 'nothing left running');
	three.ui.flush();
	three.screenshot(`tests/out/ui_queue/queue-stopped.png`);

	// Fix it: load it, set the path, update it, and run again.
	app.queueSelected = i2.id;
	loadItem();
	check(app.doc.node('load').params.path === missing, 'Load put the item in the editor');
	setInput(a);
	updateItem();
	check(i2.status === 'pending' && i2.inputs[0] === a, 'Update replaced the item and made it pending again');
	await runQueue();
	check(statuses() === 'done,done,done', `all done after the fix: ${statuses()}`);
	const outs = app.queue.map((i) => i.outputs[0]);
	check(new Set(outs).size === 3 && outs.every((o) => o && llm.exists(o)), `three distinct outputs: ${outs.map((o) => o?.split('/').pop()).join(', ')}`);
	check(i3.thumb && i3.thumb.height > i3.thumb.width, `item 3's thumbnail is its (tall) output: ${i3.thumb?.width}x${i3.thumb?.height}`);

	// A card click selects that item.
	app.widgets.queue.pointer({ phase: 'down', button: 0, x: 30, y: 70 + 20 });
	check(app.queueSelected === i2.id, 'clicking the second card selects it');

	// Order, remove, clear.
	app.queueSelected = i3.id;
	moveItem(-1);
	check(app.queue.map((i) => i.id).join() === [i1.id, i3.id, i2.id].join(), 'Up moved item 3 above item 2');
	removeItem();
	check(app.queue.length === 2 && !app.queue.includes(i3), 'Remove took item 3 out');
	clearDone();
	check(app.queue.length === 0, 'Clear done emptied a finished queue');

	// A fixed save path is queued, with a warning.
	app.doc.setParam('save', 'path', `${dir}/fixed.png`);
	addToQueue();
	check(app.statusError && app.status.includes('overwrite'), `warned about a fixed save path: ${app.status}`);
	removeItem();
	app.doc.setParam('save', 'path', `${dir}/out-#.png`);

	// Stop after this: the running item finishes, the next one waits.
	addToQueue();
	addToQueue();
	const stopping = runQueue();
	stopQueueAfterCurrent();
	await stopping;
	check(statuses() === 'done,pending', `stop-after ran one item: ${statuses()}`);
	clearDone();

	// Cancel: the item it stopped goes back to waiting.
	const cancelling = runQueue();
	cancel();
	await cancelling;
	check(statuses() === 'pending' && !app.queueRunning, `cancelled item waits again: ${statuses()}`);

	await frames(3);
	three.ui.flush();
	const shot = three.screenshot(`tests/out/ui_queue/queue.png`);
	llm.print(`ui_queue: ${fails.length ? `FAILED ${fails.length}` : 'all ok'}; wrote ${shot.path}`);
	three.quit();
})().catch((e) => { llm.print(`ui_queue: ${e.stack ?? e}`); three.quit(); });
