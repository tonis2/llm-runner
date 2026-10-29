// How tall a node's card is, in graph units (a point at zoom 1), worked out the
// way the card lays itself out:
//
//   header
//   port rows   inputs that take a wire - and settings that have one - down the
//               left, outputs down the right
//   settings    an editor for each setting that is not wired, under its name, a
//               prompt as a box of lines
//   picture     the node's last image, if it has one
//
// The card measures itself; this is for the places that need a height before
// or without it - framing the graph, spacing a fresh layout, finding the node
// a file was dropped on.

import { Doc, defOf, isSetting, isWired } from './doc.js';
import { NODE_W, NODE_HEAD, NODE_ROW, NODE_PAD, FIELD_H, LABEL_H, PROMPT_H, FIELD_GAP } from './theme.js';

// `image` is the node's picture ({ width, height }) or undefined.
export function layoutOf(n, image) {
	const def = defOf(n.type);
	let sockets = 0;
	let y = NODE_HEAD + NODE_PAD / 2;
	const fields = [];
	for (const input of def.inputs) {
		if (isSetting(input) && !isWired(n.params[input.name])) fields.push(input);
		else sockets++;
	}
	y += Math.max(sockets, def.outputs.length) * NODE_ROW;
	for (const input of fields) {
		const box = input.multiline ? PROMPT_H : FIELD_H;
		y += (input.type === 'BOOL' ? box : LABEL_H + box) + FIELD_GAP;
	}
	if (image) {
		const w = NODE_W - 2 * NODE_PAD;
		y += Math.min(w * image.height / image.width, 360) + FIELD_GAP;
	}
	return { def, w: NODE_W, h: y + NODE_PAD };
}

// A new graph's columns are spaced by how tall its nodes are.
Doc.heightOf = (n) => layoutOf(n).h;
