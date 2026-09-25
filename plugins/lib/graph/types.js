// Port types: what flows along a graph's edges, and how a port is declared.
//
// A node's inputs and outputs are typed, so a wrong wire is refused before
// anything runs. The values themselves are plain objects; what each carries:
//
//   IMAGE         { width, height, channels, pixels: Uint8Array }  (host, 8-bit)
//   LATENT        { format, data: Float32Array [C, h, w], channels, h, w }
//                 `format` names the latent space ('flux2', 'flux1'); a VAE
//                 only decodes its own, which is what makes VAEs swappable.
//   CONDITIONING  { family, tensor (GPU, [n, dim]), n, dim, dispose() }
//   MODEL         { family, ..., dispose() }  a denoiser with its weights resident
//   TEXT_ENCODER  { kind, path }  weights stream in per layer when it runs
//   VAE           { kind, format, path }
//   LORA          [{ path, strength }]  adapters, in the order they are folded in
//   REFERENCES    [{ w, h, latent }]  encoded reference images (Flux kontext)
//
// Primitives are node settings - a widget in the editor, a param in a graph
// file - and can be wired from another node's output too.
//
// A port is declared as a string:
//
//   'INT=4'                 an int, default 4
//   'LATENT?'               optional
//   'ENUM(depth|height)=depth'
//   'PATH(vae)'             a file; the kind is a hint for the editor's picker
//   'STRING*='              multi-line text, default ''
//
// or as an object { type, default, optional, options, kind, min, max, multiline }.

export const TYPES = [
	'IMAGE', 'LATENT', 'CONDITIONING', 'MODEL', 'TEXT_ENCODER', 'VAE', 'LORA', 'REFERENCES',
	'STRING', 'INT', 'FLOAT', 'BOOL', 'ENUM', 'PATH', 'ANY',
];

export const PRIMITIVES = new Set(['STRING', 'INT', 'FLOAT', 'BOOL', 'ENUM', 'PATH']);

const SPEC = /^([A-Z_]+)(?:\(([^)]*)\))?(\*)?(\?)?(?:=(.*))?$/s;

function parseDefault(type, text) {
	switch (type) {
		case 'INT': return parseInt(text, 10);
		case 'FLOAT': return parseFloat(text);
		case 'BOOL': return text === 'true' || text === '1';
		default: return text;
	}
}

// A port declaration to { type, optional, default?, options?, kind?, ... }.
export function parsePort(spec, where = '') {
	let port;
	if (typeof spec === 'string') {
		const m = SPEC.exec(spec);
		if (!m) throw new Error(`${where}: cannot read the port type '${spec}'`);
		const [, type, args, star, question, def] = m;
		port = { type, optional: !!question };
		if (args !== undefined) {
			if (type === 'ENUM') port.options = args.split('|');
			else port.kind = args;
		}
		if (star) port.multiline = true;
		if (def !== undefined) port.default = parseDefault(type, def);
	} else {
		port = { optional: false, ...spec };
	}
	if (!TYPES.includes(port.type)) throw new Error(`${where}: unknown port type ${port.type}`);
	if (port.type === 'ENUM' && port.default === undefined && port.options) port.default = port.options[0];
	if (port.default !== undefined) port.optional = true;
	return port;
}

// A setting's value as its port's type, or a thrown error that says why not.
export function coerce(port, value, where) {
	switch (port.type) {
		case 'INT': {
			const n = typeof value === 'number' ? value : Number(value);
			if (!Number.isFinite(n)) throw new Error(`${where} wants an integer, not ${JSON.stringify(value)}`);
			return Math.trunc(n);
		}
		case 'FLOAT': {
			const n = typeof value === 'number' ? value : Number(value);
			if (!Number.isFinite(n)) throw new Error(`${where} wants a number, not ${JSON.stringify(value)}`);
			return n;
		}
		case 'BOOL':
			return value === true || value === 'true' || value === 1 || value === '1';
		case 'ENUM':
			if (port.options && !port.options.includes(String(value))) {
				throw new Error(`${where} is one of ${port.options.join(', ')}, not ${JSON.stringify(value)}`);
			}
			return String(value);
		case 'STRING':
			return value == null ? '' : String(value);
		default:
			// PATH may also carry an already-loaded value (an image handed over by
			// the server), which a node reads as it is.
			return value;
	}
}

// Can an output of type `from` feed an input of type `to`?
export function accepts(to, from) {
	return to === 'ANY' || from === 'ANY' || to === from || (to === 'PATH' && from === 'STRING') || (to === 'STRING' && from === 'PATH');
}
