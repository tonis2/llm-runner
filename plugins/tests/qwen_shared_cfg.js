// The Qwen-Image sampler's shared CFG prefix: the empty prompt with the same
// images, encoded as a tail on the positive's caches (`encodeTail`), against
// the same prompt through its own full prefix. Their velocities must agree, and
// reserving rows must not change the positive's.
// `llm-runner tests/qwen_shared_cfg.js model=... text_model=... vision=... vae=... images=["a.png","b.png"]`
import { llm, image } from '../lib/llm.js';
import { useMatrixCores } from '../lib/ops.js';
import { encodeImage } from '../lib/latents.js';
import { nodeType } from '../lib/graph/registry.js';
import '../core/nodes.js';
import '../qwenimage/nodes.js';
import { QwenImageDiT } from '../qwenimage/dit.js';

const cfg = globalThis.__llm_config;
useMatrixCores(true);
const imgs = cfg.images.map((p) => image.load(p));
const { cond: pos } = await nodeType('qwenimage.text_encode').run({
	encoder: { kind: 'qwen3', path: cfg.text_model }, prompt: cfg.prompt ?? 'make it a watercolour painting',
	vision: cfg.vision, image1: imgs[0], image2: imgs[1], resolution: cfg.resolution ?? 384,
});
const neg = await pos.negative();
const dit = new QwenImageDiT(llm.open(cfg.model));
await dit.load();
const refs = [];
for (const img of pos.images) refs.push(await encodeImage(cfg.vae, img));
dit.prepare(16, 16);
const count = dit.channels * 256;
const x0 = llm.randomNormal(count, 3);
const velocity = async (prefix) => {
	dit.a.latent.buffer.write(x0);
	await dit.forward(prefix, 0.7);
	return new Float32Array(dit.a.velocity.buffer.readBytes().buffer);
};
const diff = (a, b) => {
	let e = 0, m = 0;
	for (let i = 0; i < a.length; i++) { e = Math.max(e, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(a[i])); }
	return `max diff ${e.toExponential(3)} of max ${m.toFixed(3)}`;
};
const tail = neg.n - neg.slots[neg.slots.length - 1].at;
llm.print(`positive ${pos.n} text rows, negative ${neg.n}, tail ${tail}`);
const plain = await dit.encodePrefix(pos, refs);
const vPlain = await velocity(plain);
plain.dispose();
const reserved = await dit.encodePrefix(pos, refs, tail);
const shared = await dit.encodeTail(reserved, neg);
const vShared = await velocity(shared);
const vReserved = await velocity(reserved);
shared.dispose();
reserved.dispose();
const own = await dit.encodePrefix(neg, refs);
const vOwn = await velocity(own);
own.dispose();
llm.print(`positive, reserved vs plain: ${diff(vPlain, vReserved)}`);
llm.print(`negative, shared vs own:     ${diff(vOwn, vShared)}`);
llm.print(`negative vs positive (scale): ${diff(vOwn, vPlain)}`);
