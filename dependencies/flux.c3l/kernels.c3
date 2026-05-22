module flux;

import vk;
import llm;
import zimage;
import std::io;

// --- DiT-specific kernels (reuse zimage DiTKernels + extra Flux kernels) ---

struct FluxDiTKernels
{
	zimage::DiTKernels dit; // All standard DiT kernels from zimage
	llm::ComputeKernel gated_residual_linear; // Linear gating (no tanh) for Flux
}

fn FluxDiTKernels? create_flux_dit_kernels(llm::DeviceContext* ctx)
{
	io::printfn("Creating Flux DiT kernels...");

	zimage::DiTKernels dit = zimage::create_dit_kernels(ctx)!!;

	char[] zi_spv = &zimage::ZIMAGE_SPV;
	vk::ShaderModule zi_shader = vk::shaderModuleCreateInfo().setCodeSize(zi_spv.len).setCode((uint*)&zi_spv[0])
		.build(ctx.device)!!;

	llm::ComputeKernel grl = llm::create_kernel(
		ctx,
		zi_shader,
		3,
		zimage::GatedResidualPC::size,
		"gated_residual_linear"
	)!!;

	zi_shader.free(ctx.device);

	return { .dit = dit, .gated_residual_linear = grl, };
}

fn void FluxDiTKernels.free(&self, vk::Device device)
{
	self.dit.patchify.free(device);
	self.dit.unpatchify.free(device);
	self.dit.adaln_modulate.free(device);
	self.dit.flow_euler_step.free(device);
	self.dit.linear_proj.free(device);
	self.dit.linear_proj_debug.free(device);
	self.dit.dit_timestep_embed.free(device);
	self.dit.broadcast_add_dim.free(device);
	self.dit.scale_buffer.free(device);
	self.dit.flash_attention.free(device);
	self.dit.scale_modulate.free(device);
	self.dit.gated_residual.free(device);
	self.dit.clip_values.free(device);
	self.dit.batch_layernorm.free(device);
	self.dit.mrope.free(device);
	self.dit.batch_rmsnorm.free(device);
	self.dit.transpose_heads.free(device);
	self.dit.transpose_channel_spatial.free(device);
	self.dit.batch_head_norm.free(device);
	self.dit.batch_matmul.free(device);
	self.dit.batch_matmul_simple.free(device);
	self.dit.batch_matmul_q8.free(device);
	self.dit.batch_matmul_q4_0.free(device);
	self.dit.batch_matmul_q4k.free(device);
	self.dit.batch_matmul_q5k.free(device);
	self.dit.batch_matmul_q6k.free(device);
	self.dit.batch_matmul_add_scaled.free(device);
	self.dit.ffn_swiglu_split.free(device);
	self.dit.batch_matmul_swiglu_q8.free(device);
	self.dit.batch_matmul_q8_coopmat.free(device);
	self.dit.shared.free(device);
	self.gated_residual_linear.free(device);
}
