// A multiline text field: wraps, takes Return as a newline, pastes lines.
//
//   llm-runner ui --entry plugins/tests/ui_textarea.js
//
// Writes tests/out/ui_textarea.png and prints the text after the edits.

import { llm } from '../lib/llm.js';

const frames = async (n) => { for (let i = 0; i < n; i++) await three.nextFrame(); };
let value = '';
three.ui.set({
	type: 'stack', children: [
		{ type: 'rect', color: 0x1b1d23 },
		{
			type: 'anchored', h: 'start', v: 'start', margin: [20, 20], children: [{
				type: 'textfield', key: 'prompt', multiline: true, size: [300, 110],
				text: 'A red fox sitting in fresh snow at dawn, soft golden light, shallow depth of field, photographed on film',
				onChange: (t) => { value = t; },
			}],
		},
	],
});

async function tap(...keys) {
	for (const k of keys) three.input.press(k);
	await frames(2);
	for (const k of keys.reverse()) three.input.release(k);
	await frames(2);
}

(async () => {
	await frames(5);
	three.input.movePointer(300, 40);
	three.input.pressButton(0);
	await frames(2);
	three.input.releaseButton(0);
	await frames(2);
	await tap('end');
	await tap('enter');
	const before = three.clipboard.read();
	three.clipboard.write('second line\nthird line');
	await tap('controlleft', 'v');
	three.clipboard.write(before);
	await tap('up');
	await frames(3);
	three.ui.flush();
	const shot = three.screenshot('tests/out/ui_textarea.png');
	llm.print(`ui_textarea: ${JSON.stringify(value)}; wrote ${shot.path}`);
	three.quit();
})().catch((e) => { llm.print(`ui_textarea: FAILED - ${e.stack ?? e}`); three.quit(); });
