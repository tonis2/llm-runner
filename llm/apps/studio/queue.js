// The run queue: snapshots of the graph - prompt, images, every setting - put
// aside with "Add to queue" and run one after another with "Run queue". The
// models stay loaded between items, as they do between runs, so a queue of
// prompts on one graph only encodes and samples again.
//
// An item that fails stops the queue and stays first in line, so fixing it
// (Load, edit, Update) and pressing Run queue again carries on from it.
// Cancel stops the queue too, and the item it stopped goes back to waiting.

import { llm, image } from '/core/llm.js';
import { app, runGraph, safe, dropImages } from './app.js';
import { Doc, defOf, isWire } from './doc.js';
import { clip } from './canvas.js';
import { THEME, PALETTE_W } from './theme.js';

const { Padding, Column, Row, Clip, Drawing, Label, Button } = three.ui;

const CARD_H = 64;
const CARD_GAP = 6;
const THUMB = 56;
const LIST_W = PALETTE_W - 16;
const BUTTON_H = 30;
// Everything in the panel but the list: its padding, the title and the three
// rows of buttons, with the gaps between them.
const CONTROLS_H = 16 + 26 + 3 * BUTTON_H + 4 * 8;

app.queue = [];
app.queueRunning = false;
app.queueStopAfter = false;
app.queueSelected = null;
let nextId = 1;

const runnable = (item) => item.status === 'pending' || item.status === 'error';

export function queueCounts() {
	const q = app.queue;
	return { total: q.length, runnable: q.filter(runnable).length, done: q.filter((i) => i.status === 'done').length };
}

// ---------------------------------------------------------------------------
// What an item shows

function settingOf(n, name) {
	const v = n.params[name];
	if (v !== undefined) return v;
	return defOf(n.type)?.inputs.find((i) => i.name === name)?.default;
}

// The prompt the sampler is conditioned on: the text of the node wired into a
// `positive` input, else the first prompt in the graph.
function promptOf(nodes) {
	const byId = new Map(nodes.map((n) => [n.id, n]));
	for (const n of nodes) {
		const w = n.params.positive;
		if (!isWire(w)) continue;
		const src = byId.get(w.from.slice(0, w.from.lastIndexOf('.')));
		const text = src && settingOf(src, 'prompt');
		if (typeof text === 'string' && text.trim()) return text.trim();
	}
	for (const n of nodes) {
		const text = settingOf(n, 'prompt');
		if (typeof text === 'string' && text.trim()) return text.trim();
	}
	return '';
}

function describe(item) {
	const nodes = item.doc.nodes;
	item.prompt = promptOf(nodes);
	item.inputs = nodes.filter((n) => n.type === 'core.load_image' && typeof n.params.path === 'string').map((n) => n.params.path);
	const seeded = nodes.find((n) => typeof settingOf(n, 'seed') === 'number');
	item.seed = seeded ? settingOf(seeded, 'seed') : null;
}

function freeThumb(item) {
	if (item.thumb) three.ui.freeTexture(item.thumb.tex);
	item.thumb = null;
}

// A small picture for the card: `img`, or the image at a path.
function setThumb(item, img) {
	try {
		if (typeof img === 'string') img = llm.exists(img) ? image.load(img) : null;
		if (!img || !img.pixels) return;
		const s = THUMB / Math.max(img.width, img.height);
		const w = Math.max(1, Math.round(img.width * s));
		const h = Math.max(1, Math.round(img.height * s));
		const small = image.resize(img, w, h);
		const tex = three.ui.texture(small, item.thumb ? item.thumb.tex : 0);
		item.thumb = { tex, width: w, height: h };
	} catch {
		// No picture is fine; the card says what it is in words.
	}
}

// The open graph, frozen: its nodes with positions (so Load gives back the
// same layout), deep-copied because the doc hands out its live params.
function snapshot() {
	return JSON.parse(JSON.stringify(app.doc.toJSON()));
}

// What the executor runs: the nodes without their positions.
function graphOf(item) {
	return { nodes: item.doc.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params })) };
}

// Whether `doc` (a snapshot) is the open graph's nodes, so a run of it can
// light up the canvas.
function sameShape(doc) {
	const open = app.doc.nodes;
	return doc.nodes.length === open.length && doc.nodes.every((n) => app.doc.node(n.id)?.type === n.type);
}

function warnings(doc) {
	const fixed = doc.nodes.filter((n) => n.type === 'core.save_image' && !String(settingOf(n, 'path') ?? '').includes('#'));
	return fixed.length ? ` · ${fixed.map((n) => n.id).join(', ')} saves to one path, so each item will overwrite it (add # to number them)` : '';
}

// ---------------------------------------------------------------------------
// Actions

export function addToQueue() {
	if (app.doc.nodes.length === 0) return app.say('nothing to queue: the graph is empty', true);
	const item = {
		id: nextId++, doc: snapshot(), docPath: app.doc.path,
		status: 'pending', error: null, outputs: [], thumb: null,
	};
	describe(item);
	if (item.inputs.length) setThumb(item, item.inputs[0]);
	app.queue.push(item);
	app.queueSelected = item.id;
	const warn = warnings(item.doc);
	app.say(`queued #${app.queue.length}${warn}`, !!warn);
	app.refresh();
}

export async function runQueue() {
	if (app.running) return;
	app.queueRunning = true;
	app.queueStopAfter = false;
	try {
		let item;
		let ran = 0;
		// Looked up again each time round, so items added, moved or removed
		// while the queue runs are taken as they are then.
		while (!app.queueStopAfter && (item = app.queue.find(runnable))) {
			item.status = 'running';
			item.error = null;
			ran++;
			app.refresh();
			const n = app.queue.indexOf(item) + 1;
			const r = await runGraph(graphOf(item), { label: `queue ${n}/${app.queue.length}`, show: sameShape(item.doc) });
			if (r.status === 'cancelled') {
				item.status = 'pending';
				break;
			}
			item.status = r.status;
			item.error = r.error;
			item.outputs = r.outputs;
			if (r.status === 'error') {
				app.queueSelected = item.id;
				break;
			}
			const shown = Object.values(r.results ?? {}).find((v) => v && v.image && v.image.pixels);
			setThumb(item, shown ? shown.image : r.outputs[0]);
		}
		if (!ran) app.say('nothing in the queue to run');
		else if (!item) app.say(`${app.status} · queue finished`);
		else if (app.queueStopAfter && item.status === 'done') app.say(`${app.status} · queue stopped`);
	} finally {
		app.queueRunning = false;
		app.queueStopAfter = false;
		app.refresh();
	}
}

export function stopQueueAfterCurrent() {
	if (!app.queueRunning) return;
	app.queueStopAfter = true;
	app.say('the queue stops after this item');
	app.refresh();
}

function selected() {
	return app.queue.find((i) => i.id === app.queueSelected) ?? null;
}

// The selected item, unless it is the one running.
function editable() {
	const item = selected();
	if (!item) { app.say('select a queued item first', true); return null; }
	if (item.status === 'running') { app.say('that item is running', true); return null; }
	return item;
}

export function loadItem() {
	const item = editable();
	if (!item) return;
	const keep = sameShape(item.doc);
	if (!keep) dropImages();
	app.doc = new Doc(item.doc);
	app.doc.path = item.docPath;
	app.doc.dirty = true;
	app.nodeState = {};
	if (!keep) {
		app.selected = null;
		app.widgets.canvas.frame();
	}
	app.say(`loaded queued #${app.queue.indexOf(item) + 1} into the editor`);
	app.refresh();
}

export function updateItem() {
	const item = editable();
	if (!item) return;
	item.doc = snapshot();
	item.docPath = app.doc.path;
	item.status = 'pending';
	item.error = null;
	item.outputs = [];
	describe(item);
	freeThumb(item);
	if (item.inputs.length) setThumb(item, item.inputs[0]);
	const warn = warnings(item.doc);
	app.say(`updated queued #${app.queue.indexOf(item) + 1} from the editor${warn}`, !!warn);
	app.refresh();
}

export function moveItem(by) {
	const item = selected();
	if (!item) return app.say('select a queued item first', true);
	const q = app.queue;
	const from = q.indexOf(item);
	const to = Math.max(0, Math.min(q.length - 1, from + by));
	if (to === from) return;
	q.splice(from, 1);
	q.splice(to, 0, item);
	app.refresh();
}

export function removeItem() {
	const item = editable();
	if (!item) return;
	freeThumb(item);
	const q = app.queue;
	const at = q.indexOf(item);
	q.splice(at, 1);
	app.queueSelected = q[Math.min(at, q.length - 1)]?.id ?? null;
	app.refresh();
}

export function clearDone() {
	for (const item of app.queue) if (item.status === 'done') freeThumb(item);
	app.queue = app.queue.filter((i) => i.status !== 'done');
	if (!selected()) app.queueSelected = null;
	app.refresh();
}

// ---------------------------------------------------------------------------
// The panel

// One of `per` buttons sharing a row of the list's width.
function button(label, onClick, per, key) {
	const w = Math.floor((LIST_W - 6 * (per - 1)) / per);
	return new Button(label, safe(onClick), { key, size: [w, BUTTON_H], textSize: 12 });
}

const STATUS_COLOR = { pending: THEME.dim, running: THEME.running, done: THEME.cached, error: THEME.error };

export class QueuePanel extends three.Widget {
	constructor() {
		super();
		this.height = 400;
		this.scroll = 0;
	}

	listHeight() {
		return Math.max(this.height - CONTROLS_H, CARD_H);
	}

	contentHeight() {
		return app.queue.length * (CARD_H + CARD_GAP);
	}

	clampScroll() {
		this.scroll = Math.max(0, Math.min(this.scroll, this.contentHeight() - this.listHeight()));
	}

	pointer(e) {
		if (e.phase === 'wheel') {
			if (!e.wheel) return;
			// A notch is 1 on X11 and about 15 points on Wayland: a card a notch.
			this.scroll -= Math.max(-1, Math.min(1, e.wheel)) * (CARD_H + CARD_GAP);
			this.clampScroll();
		} else if (e.phase === 'down' && e.button === 0) {
			const y = e.y + this.scroll;
			const i = Math.floor(y / (CARD_H + CARD_GAP));
			if (i >= 0 && i < app.queue.length && y - i * (CARD_H + CARD_GAP) < CARD_H) {
				app.queueSelected = app.queue[i].id;
				this.update();
			}
		}
	}

	ops() {
		const W = LIST_W;
		const H = this.listHeight();
		const ops = [{ op: 'rect', at: [0, 0], size: [W, H], radius: THEME.radius, color: THEME.body }];
		if (app.queue.length === 0) {
			ops.push({ op: 'text', at: [10, 12], text: 'Nothing queued.', size: 12, color: THEME.dim });
			ops.push({ op: 'text', at: [10, 32], text: 'Set up a prompt and press', size: 11, color: THEME.dim });
			ops.push({ op: 'text', at: [10, 48], text: 'Add to queue (ctrl+e).', size: 11, color: THEME.dim });
			return ops;
		}
		const tx = 10 + THUMB + 8;
		const tw = W - tx - 6;
		app.queue.forEach((item, i) => {
			const y = i * (CARD_H + CARD_GAP) - this.scroll;
			if (y + CARD_H < 0 || y > H) return;
			const sel = item.id === app.queueSelected;
			const color = STATUS_COLOR[item.status] ?? THEME.dim;
			ops.push({
				op: 'rect', at: [0, y], size: [W, CARD_H], radius: THEME.radius,
				color: sel ? THEME.hover : THEME.node, borderColor: sel ? THEME.accent : THEME.border, borderWidth: 1,
			});
			ops.push({ op: 'rect', at: [0, y], size: [4, CARD_H], radius: 2, color });
			const t = item.thumb;
			if (t) {
				ops.push({ op: 'image', at: [10 + (THUMB - t.width) / 2, y + 4 + (THUMB - t.height) / 2], size: [t.width, t.height], texture: t.tex, radius: 3 });
			} else {
				ops.push({ op: 'rect', at: [10, y + 4], size: [THUMB, THUMB], radius: 3, color: THEME.body });
			}
			const head = `#${i + 1} · ${item.status === 'error' ? 'failed' : item.status}`;
			ops.push({ op: 'text', at: [tx, y + 5], text: head, size: 11, color: item.status === 'pending' ? THEME.dim : color });
			ops.push({ op: 'text', at: [tx, y + 23], text: clip(item.prompt || '(no prompt)', tw, 12), size: 12, color: THEME.text });
			const names = item.inputs.map((p) => p.split('/').pop()).join(', ');
			const last = item.status === 'error'
				? item.error ?? 'failed'
				: item.status === 'done' && item.outputs.length
					? `→ ${item.outputs[0].split('/').pop()}`
					: [names, item.seed !== null ? `seed ${item.seed}` : ''].filter(Boolean).join(' · ');
			ops.push({ op: 'text', at: [tx, y + 43], text: clip(last, tw, 10), size: 10, color: item.status === 'error' ? THEME.error : THEME.dim });
		});
		return ops;
	}

	render() {
		this.clampScroll();
		const { total, runnable: k, done } = queueCounts();
		const summary = total ? `${k} to run${done ? ` · ${done} done` : ''}` : '';
		return new Padding({ insets: 8 },
			new Column({ gap: 8 },
				new Row({ gap: 8, cross: 'center' },
					new Label('Queue', { textSize: 13 }),
					new Label(summary, { color: THEME.dim, textSize: 11 }),
				),
				new Clip({ size: [LIST_W, this.listHeight()] },
					new Drawing({ key: 'queueList', size: [LIST_W, this.listHeight()], ops: this.ops(), onPointer: (e) => this.pointer(e) })),
				new Row({ gap: 6 },
					app.queueRunning
						? button(app.queueStopAfter ? 'Stopping…' : 'Stop after this', stopQueueAfterCurrent, 2, 'queueRun')
						: button('Run queue', runQueue, 2, 'queueRun'),
					button('Clear done', clearDone, 2, 'queueClear'),
				),
				new Row({ gap: 6 },
					button('Load', loadItem, 3, 'queueLoad'),
					button('Update', updateItem, 3, 'queueUpdate'),
					button('Remove', removeItem, 3, 'queueRemove'),
				),
				new Row({ gap: 6 },
					button('Move up', () => moveItem(-1), 2, 'queueUp'),
					button('Move down', () => moveItem(1), 2, 'queueDown'),
				),
			),
		);
	}
}
