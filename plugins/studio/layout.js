// Where everything on a node card goes, in graph units (a point at zoom 1):
//
//   header
//   port rows   inputs that take a wire - and settings that have one - down the
//               left, outputs down the right
//   settings    an editor for each setting that is not wired: numbers, choices
//               and switches two to a row, text and paths the full width, a
//               prompt as a box of lines. Each keeps its port, level with its
//               box, so it can still be wired.
//   picture     the node's last image, if it has one
//
// The canvas draws from this, places the editors from it and hit-tests with it,
// so the three cannot disagree.

import { Doc, defOf, isSetting, isWire } from './doc.js';
import {
	NODE_W, NODE_HEAD, NODE_ROW, NODE_PAD, FIELD_X, FIELD_H, LABEL_H, PROMPT_H, FIELD_GAP,
} from './theme.js';

const NARROW = new Set(['INT', 'FLOAT', 'ENUM', 'BOOL']);

// `image` is the node's picture ({ width, height }) or undefined.
export function layoutOf(n, image) {
	const def = defOf(n.type);
	const sockets = [];
	const settings = [];
	for (const input of def.inputs) {
		if (isSetting(input) && !isWire(n.params[input.name])) settings.push(input);
		else sockets.push(input);
	}

	const L = { def, w: NODE_W, h: 0, inputs: new Map(), outputs: new Map(), sockets, fields: [], thumb: null };
	const rows = Math.max(sockets.length, def.outputs.length, settings.length ? 0 : 1);
	sockets.forEach((input, i) => L.inputs.set(input.name, NODE_HEAD + i * NODE_ROW + NODE_ROW / 2));
	def.outputs.forEach((output, i) => L.outputs.set(output.name, NODE_HEAD + i * NODE_ROW + NODE_ROW / 2));
	let y = NODE_HEAD + rows * NODE_ROW + (settings.length ? FIELD_GAP : 0);

	const inner = NODE_W - 2 * FIELD_X;
	const half = (inner - FIELD_GAP) / 2;
	let waiting = null; // a half-width editor with room beside it
	for (const input of settings) {
		const narrow = NARROW.has(input.type);
		const h = input.multiline ? PROMPT_H : FIELD_H;
		let f;
		if (narrow && waiting) {
			f = { input, x: FIELD_X + half + FIELD_GAP, y: waiting.y, w: half, h };
			waiting = null;
		} else {
			f = { input, x: FIELD_X, y, w: narrow ? half : inner, h };
			waiting = narrow ? f : null;
			y += LABEL_H + h + FIELD_GAP;
		}
		L.fields.push(f);
		L.inputs.set(input.name, f.y + LABEL_H + Math.min(h, FIELD_H) / 2);
	}

	if (image) {
		const w = NODE_W - 2 * NODE_PAD;
		L.thumb = { x: NODE_PAD, y: y + NODE_PAD / 2, w, h: Math.min(w * image.height / image.width, 360) };
		y += L.thumb.h + NODE_PAD;
	}
	L.h = y + NODE_PAD;
	return L;
}

// A new graph's columns are spaced by how tall its nodes are.
Doc.heightOf = (n) => layoutOf(n).h;
