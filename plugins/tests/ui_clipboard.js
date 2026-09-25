// The system clipboard from a window.
//
//   llm-runner ui --entry plugins/tests/ui_clipboard.js [write="text" hold=3000]
//
// Reads what another program put there, pastes it into a text field with a
// scripted ctrl+v, copies the field back out with ctrl+a ctrl+c, and with
// `write=` writes and stays up `hold` ms so another program can read it back.

import { llm } from '../lib/llm.js';

const config = globalThis.__llm_config ?? {};
const frames = async (n) => { for (let i = 0; i < n; i++) await three.nextFrame(); };
let typed = '';
three.ui.set({
	type: 'stack', children: [
		{ type: 'rect', color: 0x1b1d23 },
		{ type: 'textfield', key: 'field', text: 'before ', size: [400, 0], onChange: (t) => { typed = t; } },
	],
});

async function chord(...keys) {
	for (const k of keys) three.input.press(k);
	await frames(2);
	for (const k of keys.reverse()) three.input.release(k);
	await frames(2);
}

(async () => {
	await frames(10);
	llm.print(`ui_clipboard: read ${JSON.stringify(three.clipboard.read())}`);

	// Focus the field (it sits at the top left), put the caret at the end, paste.
	three.input.movePointer(390, 12);
	three.input.pressButton(0);
	await frames(2);
	three.input.releaseButton(0);
	await frames(2);
	await chord('controlleft', 'v');
	llm.print(`ui_clipboard: field after ctrl+v ${JSON.stringify(typed)}`);
	await chord('controlleft', 'a');
	await chord('controlleft', 'c');
	llm.print(`ui_clipboard: clipboard after ctrl+a ctrl+c ${JSON.stringify(three.clipboard.read())}`);

	if (config.write) {
		three.clipboard.write(config.write);
		llm.print(`ui_clipboard: wrote ${JSON.stringify(config.write)}`);
	}
	const until = llm.now() + (config.hold ?? 0);
	while (llm.now() < until) await three.nextFrame();
	three.quit();
})().catch((e) => { llm.print(`ui_clipboard: FAILED - ${e.stack ?? e}`); three.quit(); });
