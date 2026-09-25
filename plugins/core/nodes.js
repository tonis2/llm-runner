// The nodes every graph uses, whatever the model: images in and out, the
// shared loaders (text encoder, VAE, LoRA), and the VAE encode/decode that any
// denoiser's latent goes through.

import { llm, image } from '../lib/llm.js';
import { defineNodes } from '../lib/graph/registry.js';
import { vaeKind, KIND_FORMAT, FORMATS, encodeImage, decodeLatent } from '../lib/latents.js';

// `#` in a file name is the next number not yet taken: out-#.png is out-1.png,
// then out-2.png.
function numbered(path) {
	if (!path.includes('#')) return path;
	for (let n = 1; ; n++) {
		const p = path.replace('#', String(n));
		if (!llm.exists(p)) return p;
	}
}

function makeDirFor(path) {
	const slash = path.lastIndexOf('/');
	if (slash > 0) globalThis.__llm.makeDir(path.slice(0, slash));
}

defineNodes('core', {
	'core.load_image': {
		title: 'Load image',
		category: 'image',
		inputs: { path: 'PATH(image)' },
		outputs: { image: 'IMAGE' },
		run({ path }) {
			// An image may also arrive already decoded (the server's init_images).
			return { image: typeof path === 'string' ? image.load(path) : path };
		},
	},

	'core.resize': {
		title: 'Resize image',
		category: 'image',
		description: 'crop: cover width x height and centre-crop. exact: stretch. fit16: long side to `width`, sides rounded to 16.',
		inputs: { image: 'IMAGE', width: 'INT=1024', height: 'INT=1024', mode: 'ENUM(crop|exact|fit16)=crop' },
		outputs: { image: 'IMAGE' },
		run({ image: img, width, height, mode }) {
			if (mode === 'exact') return { image: image.resize(img, width, height) };
			if (mode === 'fit16') return { image: image.fit16(img, width) };
			return { image: image.cropTo(img, width, height) };
		},
	},

	'core.save_image': {
		title: 'Save image',
		category: 'image',
		description: 'Writes a PNG. A # in the name becomes the next free number.',
		output: true,
		inputs: { image: 'IMAGE', path: 'STRING=output/image-#.png' },
		outputs: {},
		run({ image: img, path }) {
			const out = numbered(path);
			makeDirFor(out);
			image.savePng(out, img);
			llm.print(`  saved ${out} (${img.width}x${img.height})`);
			return { output: out, width: img.width, height: img.height };
		},
	},

	'core.preview': {
		title: 'Preview',
		category: 'image',
		description: 'Hands the image back to whoever ran the graph (the editor, a server request) without saving it.',
		output: true,
		inputs: { image: 'IMAGE' },
		outputs: {},
		run({ image: img }) { return { image: img, width: img.width, height: img.height }; },
	},

	'core.text_encoder': {
		title: 'Text encoder',
		category: 'loaders',
		description: 'A Qwen3 GGUF. Its layers stream through VRAM one at a time while it runs.',
		inputs: { path: 'PATH(text_encoder)' },
		outputs: { encoder: 'TEXT_ENCODER' },
		run({ path }) {
			if (!llm.exists(path)) throw new Error(`no text encoder at ${path}`);
			return { encoder: { kind: 'qwen3', path } };
		},
	},

	'core.vae': {
		title: 'VAE',
		category: 'loaders',
		description: 'A Flux 2 VAE, the Flux 1 VAE (ae.safetensors) or taef1; the kind is read from the file.',
		inputs: { path: 'PATH(vae)' },
		outputs: { vae: 'VAE' },
		run({ path }) {
			const m = llm.open(path);
			const kind = vaeKind(m);
			m.close();
			return { vae: { kind, format: KIND_FORMAT[kind], path } };
		},
	},

	'core.lora': {
		title: 'LoRA',
		category: 'loaders',
		description: 'An adapter to fold into a model. Chain them: each adds to the list it is given.',
		inputs: { path: 'PATH(lora)', strength: 'FLOAT=1', lora: 'LORA?' },
		outputs: { lora: 'LORA' },
		run({ path, strength, lora }) { return { lora: [...(lora ?? []), { path, strength }] }; },
	},

	'core.vae_encode': {
		title: 'VAE encode',
		category: 'latent',
		description: 'An image into the VAE\'s latent space, cropped to width x height first (0 keeps the image\'s size, rounded down to 16).',
		inputs: { vae: 'VAE', image: 'IMAGE', width: 'INT=0', height: 'INT=0' },
		outputs: { latent: 'LATENT' },
		async run({ vae, image: img, width, height }) {
			const w = Math.floor((width || img.width) / 16) * 16;
			const h = Math.floor((height || img.height) / 16) * 16;
			const t0 = llm.now();
			const latent = await encodeImage(vae.path, image.cropTo(img, w, h));
			llm.print(`  [vae] encoded ${w}x${h} to ${latent.format} [${latent.channels}, ${latent.h}, ${latent.w}] in ${llm.since(t0)}`);
			return { latent };
		},
	},

	'core.vae_decode': {
		title: 'VAE decode',
		category: 'latent',
		inputs: { vae: 'VAE', latent: 'LATENT' },
		outputs: { image: 'IMAGE' },
		async run({ vae, latent }) {
			const t0 = llm.now();
			const factor = FORMATS[latent.format].factor;
			const width = latent.w * factor, height = latent.h * factor;
			const pixels = await decodeLatent(vae.path, latent);
			const img = image.fromTensor(pixels, width, height, 3);
			llm.print(`  [phase] vae_decode: ${llm.since(t0)}`);
			return { image: img };
		},
	},
});
