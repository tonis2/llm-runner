// The DINOv3 backbone out of SAM 3D Body's checkpoint (huggingface.co/facebook/
// sam-3d-body-dinov3, gated): its ViT-H+, fine-tuned on people but with the
// general features DINOv3 has for anything else. `path` is the downloaded
// folder, its model.ckpt, or a .safetensors conversion of the same state dict.

import { llm } from '/core/llm.js';
import { openTorch } from '/core/weights/torch.js';
import { useMatrixCores } from '/core/ops.js';
import { DinoV3 } from '/core/models/dinov3.js';

const PREFIX = 'backbone.encoder.';

export async function loadBackbone(path, { linear = 'q8', matrixCores = true } = {}) {
	if (!path) throw new Error('model: the sam-3d-body-dinov3 folder (or its model.ckpt)');
	let file = path;
	if (llm.exists(path) && llm.listDir(path).length) {
		const names = llm.listDir(path).map((e) => e.name);
		const pick = names.includes('model.ckpt') ? 'model.ckpt' : names.find((n) => n.endsWith('.safetensors'));
		if (!pick) throw new Error(`${path} has neither model.ckpt nor a .safetensors file`);
		file = `${path.replace(/\/+$/, '')}/${pick}`;
	}
	if (!llm.exists(file)) throw new Error(`${file} does not exist`);
	const t0 = llm.now();
	const m = file.endsWith('.safetensors') ? llm.open(file) : openTorch(file);
	if (!m.has(`${PREFIX}cls_token`)) {
		m.close();
		throw new Error(`${file} has no DINOv3 backbone (no ${PREFIX}cls_token)`);
	}
	useMatrixCores(matrixCores);
	const encoder = new DinoV3(m, { prefix: PREFIX, linear });
	await encoder.load();
	llm.print(`  backbone from ${file} in ${llm.since(t0)}`);
	return {
		encoder,
		dispose() {
			encoder.dispose();
			m.close();
		},
	};
}
