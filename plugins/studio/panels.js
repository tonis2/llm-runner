// Everything around the canvas: the node list, the inspector for the selected
// node, the status bar, the menu, and the dialogs.

import { llm } from '../lib/llm.js';
import { app, run, cancel, freeMemory, dropImages } from './app.js';
import { allTypes, defOf, isSetting, isWire } from './doc.js';
import {
	THEME, typeColor, BAR_H, STATUS_H, PALETTE_W, INSPECTOR_W,
} from './theme.js';

const {
	Stack, Row, Column, Padding, Anchored, Scroll,
	Rect, Label, Drawing, Select, TextField, Tree, Button, Checkbox,
	MenuBar, FileBrowser, Dialog,
} = three.ui;

// A handler that reports instead of throwing: a throw inside a widget
// callback stops that callback for good.
export function safe(fn) {
	return (...args) => {
		try {
			const r = fn(...args);
			if (r && typeof r.catch === 'function') r.catch((e) => app.say(e.message ?? String(e), true));
		} catch (e) {
			app.say(e && e.message ? e.message : String(e), true);
		}
	};
}

// Lines of at most `width` points.
function wrap(text, width, size) {
	const lines = [];
	let line = '';
	for (const word of String(text).split(/\s+/)) {
		const next = line ? `${line} ${word}` : word;
		if (line && three.ui.measure(next, { size })[0] > width) {
			lines.push(line);
			line = word;
		} else {
			line = next;
		}
	}
	if (line) lines.push(line);
	return lines;
}

// ---------------------------------------------------------------------------
// Node list

export class Palette extends three.Widget {
	constructor() {
		super();
		this.search = '';
		this.height = 400;
		this.collapsed = {};
	}

	items() {
		const q = this.search.trim().toLowerCase();
		const byCategory = new Map();
		for (const [type, d] of Object.entries(allTypes())) {
			if (q && !`${type} ${d.title} ${d.category}`.toLowerCase().includes(q)) continue;
			if (!byCategory.has(d.category)) byCategory.set(d.category, []);
			byCategory.get(d.category).push({ type, title: d.title, plugin: d.plugin });
		}
		const items = [];
		for (const cat of [...byCategory.keys()].sort()) {
			const open = q || !this.collapsed[cat];
			items.push({ label: cat, depth: 0, expandable: true, expanded: open, category: cat });
			if (!open) continue;
			for (const t of byCategory.get(cat).sort((a, b) => a.title.localeCompare(b.title))) {
				items.push({ label: t.title, trailing: t.plugin, depth: 1, type: t.type });
			}
		}
		return items;
	}

	pick(item) {
		if (!item) return;
		if (item.category) {
			this.collapsed = { ...this.collapsed, [item.category]: !this.collapsed[item.category] };
			return;
		}
		const canvas = app.widgets.canvas;
		const n = app.doc.add(item.type, canvas.center());
		app.select(n.id);
		app.changed();
	}

	render() {
		const items = this.items();
		return new Padding({ insets: 8 },
			new Column({ gap: 8 },
				new Label('Nodes', { textSize: 13 }),
				new TextField({
					key: 'paletteSearch', text: this.search, placeholder: 'Search', size: [PALETTE_W - 16, 0],
					onChange: safe((t) => { this.search = t; }),
				}),
				new Scroll({ key: 'paletteScroll', size: [PALETTE_W - 16, Math.max(this.height - 80, 60)] },
					new Tree({
						key: 'paletteTree', rows: items, selected: -1,
						onSelect: safe((i) => this.pick(items[i])),
					})),
			),
		);
	}
}

// ---------------------------------------------------------------------------
// Inspector: the selected node's settings and picture

export class Inspector extends three.Widget {
	constructor() {
		super();
		this.height = 400;
		// Text typed into a number field that does not read as a number yet
		// ("-", "0."), by field, so the field keeps it until it does.
		this.drafts = {};
	}

	field(n, input, width) {
		const key = `${n.id}:${input.name}`;
		const v = n.params[input.name] ?? input.default;
		const set = (value) => { app.doc.setParam(n.id, input.name, value); app.changed(); };
		switch (input.type) {
			case 'BOOL':
				return new Checkbox(input.name, !!v, safe(set), { key });
			case 'ENUM': {
				const options = input.options ?? [];
				return new Select(options, Math.max(0, options.indexOf(String(v))), safe((i) => set(options[i])), { key, size: [width, 0] });
			}
			case 'INT':
			case 'FLOAT': {
				const draft = this.drafts[key];
				const field = new TextField({
					key, text: draft ?? (v === undefined ? '' : String(v)), size: [input.name === 'seed' ? width - 80 : width, 0],
					onChange: safe((t) => {
						const x = Number(t);
						if (t.trim() !== '' && Number.isFinite(x)) {
							this.drafts = { ...this.drafts, [key]: undefined };
							set(input.type === 'INT' ? Math.trunc(x) : x);
						} else {
							this.drafts = { ...this.drafts, [key]: t };
						}
					}),
				});
				if (input.name !== 'seed') return field;
				return new Row({ gap: 6 }, field, new Button('Random', safe(() => set(Math.floor(Math.random() * 2 ** 31)))));
			}
			case 'PATH':
				return new Row({ gap: 6 },
					new TextField({ key, text: v ?? '', size: [width - 80, 0], onChange: safe((t) => set(t || undefined)) }),
					new Button('Browse', safe(() => app.widgets.dialogs.pickFile(n.id, input))),
				);
			default:
				return new TextField({ key, text: v ?? '', size: [width, 0], placeholder: input.multiline ? 'text' : '', onChange: safe((t) => set(t)) });
		}
	}

	render() {
		const w = INSPECTOR_W - 24;
		const n = app.selected ? app.doc.node(app.selected) : null;
		const rows = [];
		if (!n) {
			rows.push(new Label('Nothing selected', { textSize: 13 }));
			rows.push(new Label('Click a node to edit it.', { color: THEME.dim, textSize: 11 }));
			rows.push(new Label('Drag from a port to wire it; drag', { color: THEME.dim, textSize: 11 }));
			rows.push(new Label('the background to pan, wheel to zoom.', { color: THEME.dim, textSize: 11 }));
		} else {
			const def = defOf(n.type);
			rows.push(new Label(def.title, { textSize: 15 }));
			rows.push(new Label(`${n.id} · ${n.type}`, { color: THEME.dim, textSize: 11 }));
			for (const line of wrap(def.description ?? '', w, 11)) rows.push(new Label(line, { color: THEME.dim, textSize: 11 }));
			if (app.nodeState[n.id] === 'error' && app.statusError) {
				for (const line of wrap(app.status, w, 11)) rows.push(new Label(line, { color: THEME.error, textSize: 11 }));
			}
			for (const input of def.inputs) {
				const wire = app.doc.wire(n.id, input.name);
				const tag = input.kind && input.type !== 'ENUM' ? `${input.type}(${input.kind})` : input.type;
				rows.push(new Row({ gap: 6, cross: 'center' },
					new Rect({ size: [8, 8], radius: 4, color: typeColor(input.type) }),
					new Label(input.name, { textSize: 12 }),
					new Label(tag, { color: THEME.dim, textSize: 10 }),
				));
				if (wire) {
					rows.push(new Row({ gap: 6, cross: 'center' },
						new Label(`← ${wire.node}.${wire.port}`, { color: THEME.dim, textSize: 12 }),
						new Button('Unwire', safe(() => { app.doc.disconnect(n.id, input.name); app.changed(); }), { key: `unwire:${n.id}:${input.name}` }),
					));
				} else if (isSetting(input)) {
					rows.push(this.field(n, input, w));
				} else {
					rows.push(new Label(input.optional ? 'optional · not wired' : 'needs a wire', {
						color: input.optional ? THEME.dim : THEME.error, textSize: 11,
					}));
				}
			}
			const img = app.images[n.id];
			if (img) {
				const h = Math.round(w * img.height / img.width);
				rows.push(new Label(`${img.width} × ${img.height}`, { color: THEME.dim, textSize: 11 }));
				rows.push(new Drawing({ key: `preview:${n.id}`, size: [w, h], ops: [{ op: 'image', at: [0, 0], size: [w, h], texture: img.tex, radius: 4 }] }));
			}
			rows.push(new Row({ gap: 6 },
				new Button('Run to here', safe(() => run([n.id]))),
				new Button('Duplicate', safe(() => { const c = app.doc.duplicate(n.id); if (c) app.select(c.id); app.changed(); })),
				new Button('Delete', safe(() => removeSelected())),
			));
		}
		return new Scroll({ key: 'inspectorScroll', size: [INSPECTOR_W, this.height] },
			new Padding({ insets: 12 }, new Column({ gap: 7 }, ...rows)));
	}
}

export function removeSelected() {
	if (!app.selected || app.running) return;
	const id = app.selected;
	app.doc.remove(id);
	if (app.images[id]) {
		three.ui.freeTexture(app.images[id].tex);
		delete app.images[id];
	}
	app.selected = null;
	app.refresh();
}

// ---------------------------------------------------------------------------
// Status bar

export class StatusBar extends three.Widget {
	constructor() {
		super();
		this.width = 800;
	}

	render() {
		const p = app.progress;
		const frac = p && p.total ? p.step / p.total : app.running ? 0.02 : 0;
		const doc = app.doc;
		const name = `${doc.path ? doc.path.split('/').pop() : 'untitled'}${doc.dirty ? ' *' : ''}`;
		return new Stack({ size: [this.width, STATUS_H] },
			new Rect({ color: THEME.header, solid: true }),
			new Anchored({ h: 'start', v: 'center', margin: [8, 0] },
				new Row({ gap: 10, cross: 'center' },
					app.running
						? new Button('Cancel', safe(cancel), { key: 'runButton' })
						: new Button('Run', safe(() => run()), { key: 'runButton' }),
					new Drawing({
						key: 'progress', size: [140, 6], ops: [
							{ op: 'rect', at: [0, 0], size: [140, 6], radius: 3, color: THEME.body },
							{ op: 'rect', at: [0, 0], size: [140 * frac, 6], radius: 3, color: THEME.running },
						],
					}),
					new Label(app.status, { color: app.statusError ? THEME.error : THEME.text, textSize: 12 }),
				)),
			new Anchored({ h: 'end', v: 'center', margin: [10, 0] },
				new Label(name, { color: THEME.dim, textSize: 12 })),
		);
	}
}

// ---------------------------------------------------------------------------
// Menu and layout

function templates() {
	const dir = `${llm.root}/graph/templates`;
	return globalThis.__llm.listDir(dir).filter((e) => !e.dir && e.name.endsWith('.json')).map((e) => ({ name: e.name.replace(/\.json$/, ''), path: `${dir}/${e.name}` }));
}

function menuSpec() {
	return [
		{ title: 'File', items: ['New', 'Open…', 'Save', 'Save as…', '-', 'Quit'] },
		{ title: 'Templates', items: templates().map((t) => t.name) },
		{ title: 'Run', items: ['Run', 'Cancel', '-', 'Free memory'] },
		{ title: 'View', items: ['Frame all', '-', 'Plugins and settings…'] },
	];
}

function onMenu(menu, item) {
	const spec = menuSpec()[menu];
	const label = spec && spec.items[item];
	const d = app.widgets.dialogs;
	if (spec.title === 'Templates') return app.actions.open(templates()[item].path, { template: true });
	switch (label) {
		case 'New': return app.actions.newGraph();
		case 'Open…': d.openFile = true; return;
		case 'Save': return app.actions.save();
		case 'Save as…': d.saveAsText = app.doc.path ?? 'graphs/graph.json'; d.saveAs = true; return;
		case 'Quit': return three.quit();
		case 'Run': return run();
		case 'Cancel': return cancel();
		case 'Free memory': return freeMemory();
		case 'Frame all': return app.widgets.canvas.frame();
		case 'Plugins and settings…': d.settingsText = app.settings.preview_vae ?? ''; d.plugins = true; return;
	}
}

export class Chrome extends three.Widget {
	constructor() {
		super();
		this.size = [1200, 800];
	}

	render() {
		const [W, H] = this.size;
		const mid = Math.max(H - BAR_H - STATUS_H, 100);
		const { canvas, palette, inspector, status } = app.widgets;
		// Keyed by the size, so a new window size rebuilds the layout rather
		// than patching sizes into boxes that were laid out for the old one.
		const k = `${W}x${H}`;
		return new Stack({},
			new Column({ key: `layout:${k}` },
				new Rect({ size: [W, BAR_H], color: THEME.header }),
				new Row({},
					new Stack({ key: `left:${k}`, size: [PALETTE_W, mid] }, new Rect({ color: THEME.panel, solid: true }), palette),
					new Rect({ size: [1, mid], color: THEME.border }),
					canvas,
					new Rect({ size: [1, mid], color: THEME.border }),
					new Stack({ key: `right:${k}`, size: [INSPECTOR_W, mid] }, new Rect({ color: THEME.panel, solid: true }), inspector),
				),
				status,
			),
			new Anchored({ h: 'start', v: 'start' }, new MenuBar(menuSpec(), safe(onMenu))),
		);
	}

	// Called every frame with the interface's size; lays the panels out again
	// when the window changed.
	resize(W, H) {
		if (this.size[0] === W && this.size[1] === H) return;
		this.size = [W, H];
		const mid = Math.max(H - BAR_H - STATUS_H, 100);
		const { canvas, palette, inspector, status } = app.widgets;
		canvas.setSize([Math.max(W - PALETTE_W - INSPECTOR_W - 2, 100), mid]);
		palette.height = mid;
		inspector.height = mid;
		status.width = W;
	}
}

// ---------------------------------------------------------------------------
// Dialogs

const MODEL_MASK = ['*.gguf', '*.safetensors'];
const IMAGE_MASK = ['*.png', '*.jpg', '*.jpeg'];

function dirOf(path) {
	if (typeof path !== 'string') return '';
	const slash = path.lastIndexOf('/');
	return slash > 0 ? path.slice(0, slash) : '';
}

export class Dialogs extends three.Widget {
	static layer = 1;

	constructor() {
		super();
		this.openFile = false;
		this.saveAs = false;
		this.saveAsText = '';
		this.plugins = false;
		this.settingsText = '';
		this.pick = null;   // { node, input, kind, start }
	}

	pickFile(node, input) {
		const current = app.doc.node(node)?.params[input.name];
		const start = dirOf(current) || app.settings.model_dir || globalThis.__llm.homeDir();
		this.pick = { node, input: input.name, kind: input.kind ?? '', start };
	}

	render() {
		const pick = this.pick;
		const children = [
			new Dialog({
				key: 'openDialog', title: 'Open graph', open: this.openFile,
				modal: true, closeOutside: true, size: [460, 380],
				onDismiss: safe(() => { this.openFile = false; }),
			},
				new Scroll({ key: 'openScroll' },
					new FileBrowser({
						key: 'openBrowser', start: llm.exists('graphs') ? 'graphs' : '.', mask: ['*.json'],
						onChoose: safe((path) => { this.openFile = false; app.actions.open(path); }),
					})),
			),
			new Dialog({
				key: 'saveAsDialog', title: 'Save graph as', open: this.saveAs,
				modal: true, closeOutside: true, size: [420, 0],
				onDismiss: safe(() => { this.saveAs = false; }),
			},
				new Column({ gap: 8 },
					new TextField({ key: 'saveAsField', text: this.saveAsText, size: [396, 0], onChange: (t) => { this.saveAsText = t; } }),
					new Row({ gap: 8 },
						new Button('Cancel', safe(() => { this.saveAs = false; })),
						new Button('Save', safe(() => { this.saveAs = false; app.actions.save(this.saveAsText); })),
					),
				),
			),
			new Dialog({
				key: 'pickDialog', title: pick ? `Choose ${pick.input}` : 'Choose a file', open: !!pick,
				modal: true, closeOutside: true, size: [520, 420],
				onDismiss: safe(() => { this.pick = null; }),
			},
				pick && new Scroll({ key: `pickScroll:${pick.node}:${pick.input}` },
					new FileBrowser({
						key: `pickBrowser:${pick.node}:${pick.input}`, start: pick.start,
						mask: pick.kind === 'image' ? IMAGE_MASK : MODEL_MASK,
						onChoose: safe((path) => {
							app.doc.setParam(pick.node, pick.input, path);
							if (pick.kind !== 'image') app.settings.model_dir = dirOf(path);
							this.pick = null;
							app.changed();
						}),
					})),
			),
			new Dialog({
				key: 'pluginsDialog', title: 'Plugins and settings', open: this.plugins,
				modal: true, closeOutside: true, size: [520, 0],
				onDismiss: safe(() => { this.plugins = false; }),
			},
				new Column({ gap: 8 },
					new Label(`Plugins in ${llm.root}/ (changes apply on the next start)`, { color: THEME.dim, textSize: 11 }),
					...app.plugins.map((p) => new Column({ gap: 2 },
						new Row({ gap: 10, cross: 'center' },
							new Checkbox(p.name, p.enabled !== false, safe((on) => togglePlugin(p, on)), { key: `plugin:${p.name}` }),
							new Label(p.version ? `v${p.version}` : '', { color: THEME.dim, textSize: 11 }),
							new Label(p.loaded ? 'loaded' : p.error ? 'failed' : p.enabled === false ? 'disabled' : '', {
								color: p.error ? THEME.error : THEME.dim, textSize: 11,
							}),
						),
						p.description && new Label(p.description, { color: THEME.dim, textSize: 11 }),
						p.error && new Label(p.error, { color: THEME.error, textSize: 11 }),
					)),
					new Rect({ size: [496, 1], color: THEME.border }),
					new Label('Step preview decoder (taef1 for Z-Image; empty for none)', { color: THEME.dim, textSize: 11 }),
					new TextField({ key: 'previewVae', text: this.settingsText, size: [496, 0], onChange: (t) => { this.settingsText = t; } }),
					new Row({ gap: 8 },
						new Button('Close', safe(() => { this.plugins = false; })),
						new Button('Save', safe(() => {
							app.settings.preview_vae = this.settingsText.trim() || undefined;
							app.saveSettings();
							this.plugins = false;
							app.say('settings saved');
						})),
					),
				),
			),
		];
		return new Stack({}, ...children);
	}
}

function togglePlugin(p, on) {
	const disabled = new Set(app.settings.disabled_plugins ?? []);
	if (on) disabled.delete(p.name);
	else disabled.add(p.name);
	app.settings.disabled_plugins = [...disabled];
	p.enabled = on;
	app.saveSettings();
	app.widgets.dialogs.update();
}

export { dropImages };
