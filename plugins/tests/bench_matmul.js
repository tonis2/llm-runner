// Matmul throughput on Flux-sized shapes: shady kernels against the Slang one
// the C3 pipelines ran.
import { llm } from '../lib/llm.js';
import { kernel, pc } from '../lib/gpu.js';
const C = three.compute;
const spv = C.spirv('dependencies/llm_text.c3l/shaders/llm.spv');
const slangQ8 = C.kernel(spv, { name: 'slang_q8', entry: 'batch_matmul_q8' });
const shapes = globalThis.__llm_config.shapes ?? [[1088, 4096, 4096], [1088, 4096, 12288], [1088, 12288, 4096], [2048, 4096, 12288], [64, 4096, 12288]];
for (const [seq, inDim, out] of shapes) {
	const w = C.bytes(out * inDim / 32 * 34);
	const x = C.f32(seq * inDim);
	const y = C.f32(seq * out);
	const flops = 2 * seq * inDim * out;
	const run = (label, k, groups, reps = 10) => {
		k.dispatch([w, x, y], { workgroups: groups, push: pc('uuuu', out, inDim, seq, 0) });
		C.submit();
		const t0 = llm.now();
		for (let i = 0; i < reps; i++) k.dispatch([w, x, y], { workgroups: groups, push: pc('uuuu', out, inDim, seq, 0) });
		C.submit();
		const ms = (llm.now() - t0) / reps;
		return `${label} ${ms.toFixed(2)}ms ${(flops / ms / 1e9).toFixed(1)} TFLOPS`;
	};
	const g64 = [Math.ceil(seq / 64) * Math.ceil(out / 64), 1, 1];
	llm.print(`[${seq} x ${inDim}] -> ${out}:  ` + [
		run('slang', slangQ8, g64), run('q8', kernel('matmul_q8'), g64),
	].join(' | '));
	w.dispose(); x.dispose(); y.dispose();
}
