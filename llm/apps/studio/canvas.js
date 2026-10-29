// The graph, as cui's node editor: the document becomes nodes, ports and wires
// by index, each node's settings become real editors on its card, and what the
// user does comes back through the editor's callbacks as changes to the
// document. Panning, zooming, hit testing, wires and selection boxes are the
// editor's; this file only says what the graph is and what a gesture means.
//
// A setting that is not wired is an editor with its port level with it, so it
// can still be wired; a wired one, and every input that only takes a wire, is a
// port with a row of its own. A node's picture, when it has one, is under its
// settings.

import { app, run } from './app.js';
import { allTypes, defOf, isSetting, isWired } from './doc.js';
import { layoutOf } from './layout.js';
import { editor } from './fields.js';
import { accepts } from '/core/graph/types.js';
import { THEME, typeColor, NODE_W, NODE_PAD, PROMPT_H, PALETTE_W, BAR_H } from './theme.js';

const { NodeGraph, NodeCard, PortRow, ContextMenu, Column, Label, Drawing, Stack, Anchored } = three.ui;

const INNER = NODE_W - 2 * NODE_PAD;
const IMAGE_FILE = /\.(png|jpe?g)$/i;

// The node menu, in the order it is shown.
const NODE_ACTIONS = ['Run up to here', 'Duplicate', 'Delete'];

export class GraphCanvas extends three.Widget {
	constructor() {
		super();
		this.size = [800, 600];
		this.view = { pan: [0, 0], zoom: 1 };
		this.version = 0;       // bumped by anything that changes what is drawn
		// Frame the graph whenever the canvas changes size, until somebody
		// pans or zooms it themselves.
		this.autoFrame = true;
		// The open menu: { kind: 'add' | 'node', at, plane, from?, node? }.
		this.menu = null;
	}

	setSize(size) {
		this.size = size;
		if (this.autoFrame) this.frame();
	}

	// Canvas points to graph units, and back.
	toGraph([x, y]) { const v = this.view; return [x / v.zoom + v.pan[0], y / v.zoom + v.pan[1]]; }
	toScreen([x, y]) { const v = this.view; return [(x - v.pan[0]) * v.zoom, (y - v.pan[1]) * v.zoom]; }

	// Where a new node goes: the middle of what is on screen.
	center() { return this.toGraph([this.size[0] / 2 - NODE_W / 2, this.size[1] / 3]); }

	// Fit the whole graph on screen.
	frame() {
		const nodes = app.doc.nodes;
		if (nodes.length === 0) { this.view = { pan: [0, 0], zoom: 1 }; return; }
		let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
		for (const n of nodes) {
			const h = layoutOf(n, app.images[n.id]).h;
			x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1]);
			x1 = Math.max(x1, n.pos[0] + NODE_W); y1 = Math.max(y1, n.pos[1] + h);
		}
		const pad = 40;
		const zoom = Math.min(1.2, Math.max(0.25, Math.min(this.size[0] / (x1 - x0 + 2 * pad), this.size[1] / (y1 - y0 + 2 * pad))));
		this.view = {
			zoom,
			pan: [(x0 + x1) / 2 - this.size[0] / 2 / zoom, (y0 + y1) / 2 - this.size[1] / 2 / zoom],
		};
	}

	// ------------------------------------------------------------------
	// The graph by index

	// Nodes, ports and wires as the editor takes them, with the maps back to the
	// document: which port index is which node's which input or output.
	model() {
		const doc = app.doc;
		const index = new Map(doc.nodes.map((n, i) => [n.id, i]));
		const nodes = [];
		const ports = [];
		const info = [];        // per port: { node, input?, output?, type }
		const portOf = new Map(); // `${id}>${output}` / `${id}<${input}` -> port index
		const p = app.progress;
		const selection = app.selection ?? new Set();

		doc.nodes.forEach((n, i) => {
			const def = defOf(n.type);
			const state = app.nodeState[n.id];
			nodes.push({
				pos: n.pos,
				width: NODE_W,
				title: def.title,
				tag: state === 'cached' ? 'cached' : n.id,
				header: def.missing ? THEME.error : undefined,
				selected: selection.has(n.id),
				running: state === 'running',
				error: state === 'error',
				progress: state === 'running' && p && p.node === n.id && p.total ? p.step / p.total : 0,
			});
			for (const input of def.inputs) {
				const wired = isWired(n.params[input.name]);
				const setting = isSetting(input);
				portOf.set(`${n.id}<${input.name}`, ports.length);
				info.push({ node: n.id, input: input.name, type: input.type });
				ports.push({
					node: i,
					label: input.name + (input.optional || setting ? '' : ' *'),
					color: typeColor(input.type),
					hollow: setting && !wired,
				});
			}
			for (const output of def.outputs) {
				portOf.set(`${n.id}>${output.name}`, ports.length);
				info.push({ node: n.id, output: output.name, type: output.type });
				ports.push({
					node: i,
					output: true,
					label: output.kind ? `${output.name} (${output.kind})` : output.name,
					color: typeColor(output.type),
				});
			}
		});

		const wires = [];
		const wireInfo = [];
		const chosen = app.selectedWire;
		for (const n of doc.nodes) {
			for (const input of defOf(n.type).inputs) {
				for (const w of doc.wires(n.id, input.name)) {
					const from = portOf.get(`${w.node}>${w.port}`);
					const to = portOf.get(`${n.id}<${input.name}`);
					if (from === undefined || to === undefined) continue;
					const selected = !!chosen && chosen.node === n.id && chosen.input === input.name
						&& chosen.src.node === w.node && chosen.src.port === w.port;
					wires.push({ from, to, selected });
					wireInfo.push({ node: n.id, input: input.name, src: w });
				}
			}
		}
		return { nodes, ports, wires, info, portOf, wireInfo, index };
	}

	// What goes on a node's card: an editor per setting nothing feeds, each
	// with its port beside it, and the node's picture.
	body(n, portOf) {
		const def = defOf(n.type);
		const rows = [];
		for (const input of def.inputs) {
			if (!isSetting(input) || isWired(n.params[input.name])) continue;
			const field = editor(n, input, {
				key: `field:${n.id}:${input.name}`,
				width: INNER,
				height: input.multiline ? PROMPT_H : 0,
				textSize: 12,
			});
			const labelled = input.type === 'BOOL' ? field
				: new Column({ gap: 2 }, new Label(input.name, { textSize: 10, color: THEME.dim }), field);
			rows.push(new PortRow(portOf.get(`${n.id}<${input.name}`), labelled));
		}
		const img = app.images[n.id];
		if (img) {
			const h = Math.min(INNER * img.height / img.width, 360);
			rows.push(new Drawing({
				key: `thumb:${n.id}`,
				size: [INNER, h],
				ops: [{ op: 'image', at: [0, 0], size: [INNER, h], texture: img.tex, radius: 3 }],
			}));
		}
		return rows.length ? new Column({ gap: 6 }, ...rows) : null;
	}

	render() {
		const m = this.model();
		this.current = m;
		const doc = app.doc;
		const cards = doc.nodes.map((n, i) => new NodeCard(i, this.body(n, m.portOf)));
		const children = [
			new NodeGraph({
				key: 'graph',
				size: this.size,
				nodes: m.nodes,
				ports: m.ports,
				wires: m.wires,
				view: this.view,
				planeColor: THEME.canvas,
				gridColor: THEME.grid,
				cardColor: THEME.node,
				headerColor: THEME.nodeHeader,
				cardBorder: THEME.border,
				textColor: THEME.text,
				accent: THEME.accent,
				onView: (pan, zoom) => { this.autoFrame = false; this.view = { pan, zoom }; },
				onSelect: (node, mode) => this.selectNodes(node >= 0 ? [doc.nodes[node]?.id] : [], mode),
				onBox: (nodes, mode) => this.selectNodes(nodes.map((i) => doc.nodes[i]?.id), mode),
				onMove: (dx, dy, done) => this.moveSelection(dx, dy, done),
				canConnect: (from, to) => this.canConnect(from, to),
				onConnect: (from, to) => this.connect(from, to),
				onDisconnect: (wire) => this.disconnect(wire),
				onWireSelect: (wire) => this.selectWire(wire),
				onWireDropped: (port, plane, screen) => this.openAddMenu(screen, plane, port),
				onContext: (target, index, plane, screen) => this.context(target, index, plane, screen),
				onCommand: (name) => this.command(name),
				onAddRequest: (plane, screen) => this.openAddMenu(screen, plane),
			}, ...cards),
		];
		if (doc.nodes.length === 0) {
			children.push(new Anchored({ h: 'center', v: 'center' },
				new Label('Add nodes from the list, or open a template', { textSize: 14, color: THEME.dim })));
		}
		const menu = this.menu;
		children.push(new ContextMenu({
			key: 'graphMenu',
			items: menu ? menu.items : [],
			open: !!menu,
			at: menu ? menu.at : [0, 0],
			searchable: !!menu && menu.kind === 'add',
			placeholder: 'Add a node',
			onChoose: (i) => this.choose(i),
			onDismiss: () => { this.menu = null; },
		}));
		return new Stack({ size: this.size }, ...children);
	}

	// ------------------------------------------------------------------
	// What a gesture means

	selectNodes(ids, mode) {
		const selection = new Set(mode === 'replace' ? [] : app.selection ?? []);
		for (const id of ids) {
			if (!id) continue;
			if (mode === 'toggle' && selection.has(id)) selection.delete(id);
			else selection.add(id);
		}
		app.selectedWire = null;
		app.selection = selection;
		const last = ids.filter(Boolean).at(-1);
		app.selected = last && selection.has(last) ? last : (selection.values().next().value ?? null);
		app.refresh();
	}

	moveSelection(dx, dy, done) {
		const selection = app.selection ?? new Set();
		for (const n of app.doc.nodes) {
			if (!selection.has(n.id)) continue;
			n.pos = done ? [Math.round(n.pos[0] + dx), Math.round(n.pos[1] + dy)] : [n.pos[0] + dx, n.pos[1] + dy];
		}
		app.doc.dirty = true;
		this.version++;
	}

	canConnect(from, to) {
		const m = this.current;
		const a = m?.info[from], b = m?.info[to];
		return !!a && !!b && a.node !== b.node && accepts(b.type, a.type);
	}

	connect(from, to) {
		const m = this.current;
		const a = m.info[from], b = m.info[to];
		const problem = app.doc.connect(a.node, a.output, b.node, b.input);
		if (problem) app.say(problem, true);
		app.changed();
	}

	disconnect(wire) {
		const w = this.current.wireInfo[wire];
		if (!w) return;
		app.doc.disconnect(w.node, w.input, w.src);
		app.selectedWire = null;
		app.changed();
	}

	selectWire(wire) {
		const w = this.current.wireInfo[wire];
		app.selectedWire = w ? { node: w.node, input: w.input, src: w.src } : null;
		this.version++;
	}

	command(name) {
		switch (name) {
			case 'delete': return this.deleteSelection();
			case 'duplicate': {
				const copies = [...(app.selection ?? [])].map((id) => app.doc.duplicate(id)).filter(Boolean);
				if (copies.length) this.selectNodes(copies.map((c) => c.id), 'replace');
				return app.changed();
			}
			case 'selectAll': return this.selectNodes(app.doc.nodes.map((n) => n.id), 'replace');
		}
	}

	deleteSelection() {
		if (app.running) return;
		const w = app.selectedWire;
		if (w) {
			app.doc.disconnect(w.node, w.input, w.src);
			app.selectedWire = null;
			return app.changed();
		}
		for (const id of app.selection ?? []) {
			app.doc.remove(id);
			if (app.images[id]) {
				three.ui.freeTexture(app.images[id].tex);
				delete app.images[id];
			}
		}
		app.selection = new Set();
		app.selected = null;
		app.refresh();
	}

	// ------------------------------------------------------------------
	// The menus

	// The node types, as menu entries — only those that can take the wire being
	// dragged from `port`, when there is one.
	openAddMenu(at, plane, port = null) {
		const types = allTypes();
		const from = port === null ? null : this.current.info[port];
		const entries = [];
		for (const [type, def] of Object.entries(types)) {
			if (from) {
				const fits = from.output
					? def.inputs.some((i) => accepts(i.type, from.type))
					: def.outputs.some((o) => accepts(from.type, o.type));
				if (!fits) continue;
			}
			entries.push({ type, def });
		}
		entries.sort((a, b) => (a.def.category ?? '').localeCompare(b.def.category ?? '') || a.def.title.localeCompare(b.def.title));
		this.menu = {
			kind: 'add', at: this.menuPoint(at), plane, from, entries,
			items: entries.map(({ def }) => ({ label: def.title, category: def.category })),
		};
	}

	context(target, index, plane, screen) {
		if (target === 'node') {
			const n = app.doc.nodes[index];
			if (!n) return;
			if (!(app.selection ?? new Set()).has(n.id)) this.selectNodes([n.id], 'replace');
			this.menu = { kind: 'node', at: this.menuPoint(screen), node: n.id, items: NODE_ACTIONS };
			return;
		}
		if (target === 'wire') {
			this.selectWire(index);
			return this.deleteSelection();
		}
		this.openAddMenu(screen, plane);
	}

	// The editor reports points in the interface's coordinates, which is where
	// a menu opens.
	menuPoint(at) { return [at[0], at[1]]; }

	choose(i) {
		const menu = this.menu;
		this.menu = null;
		if (!menu) return;
		if (menu.kind === 'node') {
			const action = NODE_ACTIONS[i];
			if (action === 'Delete') return this.deleteSelection();
			if (action === 'Duplicate') return this.command('duplicate');
			if (action === 'Run up to here') return run([menu.node]);
			return;
		}
		const { type, def } = menu.entries[i];
		const n = app.doc.add(type, [Math.round(menu.plane[0]), Math.round(menu.plane[1])]);
		const from = menu.from;
		if (from) {
			// The new node takes the dropped wire on its first port that fits.
			const problem = from.output
				? (() => {
					const into = def.inputs.find((x) => accepts(x.type, from.type));
					return into ? app.doc.connect(from.node, from.output, n.id, into.name) : null;
				})()
				: (() => {
					const out = def.outputs.find((x) => accepts(from.type, x.type));
					return out ? app.doc.connect(n.id, out.name, from.node, from.input) : null;
				})();
			if (problem) app.say(problem, true);
		}
		this.selectNodes([n.id], 'replace');
		app.changed();
	}

	// ------------------------------------------------------------------
	// Files dropped on the window

	// An image dropped on a Load image node becomes its path; any other image
	// gets a new Load image node where it landed, the next one below it.
	// Anything else is named in the status bar.
	dropFiles(paths) {
		const p = three.input.pointer;
		const scale = Math.max(1, three.window.scale || 1);
		const local = [p.x / scale - PALETTE_W - 1, p.y / scale - BAR_H];
		const onCanvas = local[0] >= 0 && local[1] >= 0 && local[0] < this.size[0] && local[1] < this.size[1];
		let at = onCanvas ? this.toGraph(local) : this.center();
		const target = onCanvas ? this.nodeAt(at) : null;
		const images = paths.filter((f) => IMAGE_FILE.test(f));
		const skipped = paths.filter((f) => !IMAGE_FILE.test(f));
		let added = null;
		for (const path of images) {
			if (target?.type === 'core.load_image' && path === images[0]) {
				app.doc.setParam(target.id, 'path', path);
				added = target;
				continue;
			}
			added = app.doc.add('core.load_image', at);
			app.doc.setParam(added.id, 'path', path);
			at = [at[0], at[1] + layoutOf(added).h + 20];
		}
		if (added) {
			this.selectNodes([added.id], 'replace');
			app.changed();
		}
		const name = (f) => f.slice(f.lastIndexOf('/') + 1);
		if (skipped.length) app.say(`not an image (png or jpeg): ${skipped.map(name).join(', ')}`, true);
		else if (images.length) app.say(`dropped ${images.map(name).join(', ')}`);
	}

	// The topmost node whose card covers a graph point, by the layout's height.
	nodeAt(g) {
		const nodes = app.doc.nodes;
		for (let k = nodes.length - 1; k >= 0; k--) {
			const n = nodes[k];
			const h = layoutOf(n, app.images[n.id]).h;
			if (g[0] >= n.pos[0] && g[0] <= n.pos[0] + NODE_W && g[1] >= n.pos[1] && g[1] <= n.pos[1] + h) return n;
		}
		return null;
	}
}
