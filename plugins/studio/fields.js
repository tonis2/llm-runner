// The editor for one setting, built the same way inside a node card and in the
// inspector: a switch for BOOL, a dropdown for ENUM, a number field (with a new
// random seed beside `seed`), a path with a file picker, and text - a box of
// lines when the input is multi-line.

import { app, safe } from './app.js';

const { Row, TextField, Select, Checkbox, Button } = three.ui;

// Text typed into a number field that does not read as a number yet ("-",
// "0."), by node and input, so the field keeps it until it does.
const drafts = new Map();

// `width` and `height` are in points on screen; `textSize` scales the whole
// editor, which is how a node's editors follow the zoom.
export function editor(n, input, { key, width, height = 0, textSize = 12 }) {
	const v = n.params[input.name] ?? input.default;
	const set = (value) => { app.doc.setParam(n.id, input.name, value); app.changed(); };
	const draftKey = `${n.id}:${input.name}`;
	const side = Math.round(textSize * 2.1); // a square button beside a field
	const gap = Math.max(2, Math.round(textSize / 3));
	const button = (label, onClick) => new Button(label, safe(onClick), {
		key: `${key}:button`, size: [side, height || side], textSize,
	});

	switch (input.type) {
		case 'BOOL':
			return new Checkbox(input.name, !!v, safe(set), { key, textSize });
		case 'ENUM': {
			const options = input.options ?? [];
			return new Select(options, Math.max(0, options.indexOf(String(v))), safe((i) => set(options[i])), {
				key, size: [width, 0], textSize,
			});
		}
		case 'INT':
		case 'FLOAT': {
			const seed = input.name === 'seed';
			const field = new TextField({
				key,
				text: drafts.get(draftKey) ?? (v === undefined ? '' : String(v)),
				size: [seed ? width - side - gap : width, 0],
				textSize,
				onChange: safe((t) => {
					const x = Number(t);
					if (t.trim() !== '' && Number.isFinite(x)) {
						drafts.delete(draftKey);
						set(input.type === 'INT' ? Math.trunc(x) : x);
					} else {
						drafts.set(draftKey, t);
						app.changed();
					}
				}),
			});
			if (!seed) return field;
			return new Row({ gap, cross: 'center' }, field, button('↻', () => {
				drafts.delete(draftKey);
				set(Math.floor(Math.random() * 2 ** 31));
			}));
		}
		case 'PATH':
			return new Row({ gap, cross: 'center' },
				new TextField({
					key, text: v ?? '', size: [width - side - gap, 0], textSize, placeholder: input.kind ?? 'path',
					onChange: safe((t) => set(t || undefined)),
				}),
				button('…', () => app.widgets.dialogs.pickFile(n.id, input)),
			);
		default:
			return new TextField({
				key, text: v ?? '', size: [width, input.multiline ? height : 0], textSize,
				multiline: !!input.multiline, placeholder: input.name,
				onChange: safe((t) => set(t)),
			});
	}
}
