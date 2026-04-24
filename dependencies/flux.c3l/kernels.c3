module flux;

import vk;
import llm;
import llm::pipelines;
import zimage;
import std::io;

// Push constant structs for Flux-specific operations (T5/CLIP)

struct SpatialAttentionPC {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
}

struct BroadcastAddPC {
    uint channels;
    uint spatial;
}

struct BatchedMatMulPC {
    uint m_dim;
    uint n_dim;
    uint k_dim;
}

// Flux compute kernels loaded from diffusion.spv (for T5/CLIP)
struct FluxKernels {
    llm::ComputeKernel batched_matmul;
    llm::ComputeKernel spatial_attention;
    llm::ComputeKernel broadcast_add;
    llm::SharedKernels shared;
}

fn FluxKernels? create_flux_kernels(llm::DeviceContext* ctx) {
    io::printfn("Creating Flux compute kernels...");
    char[] spv = &pipelines::DIFFUSION_SPV;
    vk::ShaderModule shader = vk::shaderModuleCreateInfo()
        .setCodeSize(spv.len)
        .setCode((uint*)&spv[0])
        .build(ctx.device)!!;

    FluxKernels kernels = {
        .batched_matmul   = llm::create_kernel(ctx, shader, 3, BatchedMatMulPC.sizeof, "batched_matmul")!!,
        .spatial_attention = llm::create_kernel(ctx, shader, 5, SpatialAttentionPC.sizeof, "spatial_attention")!!,
        .broadcast_add    = llm::create_kernel(ctx, shader, 2, BroadcastAddPC.sizeof, "broadcast_add")!!,
        .shared           = llm::create_shared_kernels(ctx, shader)!!,
    };

    shader.free(ctx.device);
    return kernels;
}

fn void FluxKernels.free(&self, vk::Device device) {
    self.batched_matmul.free(device);
    self.spatial_attention.free(device);
    self.broadcast_add.free(device);
    self.shared.free(device);
}

// Dispatch helpers

fn void dispatch_batched_matmul(
    vk::CommandBuffer cmd,
    FluxKernels* k,
    llm::Tensor* weight,
    llm::Tensor* input,
    llm::Tensor* output,
    BatchedMatMulPC* pc
) {
    llm::dispatch_kernel(cmd, &k.batched_matmul,
        { input.gpu_buffer.buffer, weight.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { input.size_bytes, weight.size_bytes, output.size_bytes },
        pc, llm::ceil_div(pc.m_dim, 16), llm::ceil_div(pc.n_dim, 16));
}

// --- DiT-specific kernels (reuse zimage DiTKernels + extra Flux kernels) ---

struct FluxDiTKernels {
    zimage::DiTKernels dit;                        // All standard DiT kernels from zimage
    llm::ComputeKernel gated_residual_linear;      // Linear gating (no tanh) for Flux
}

fn FluxDiTKernels? create_flux_dit_kernels(llm::DeviceContext* ctx) {
    io::printfn("Creating Flux DiT kernels...");

    // Create standard DiT kernels (patchify, unpatchify, flash_attention, mrope, batch_matmul_*, etc.)
    zimage::DiTKernels dit = zimage::create_dit_kernels(ctx)!!;

    // Load the extra gated_residual_linear kernel from zimage.spv
    char[] zi_spv = &zimage::ZIMAGE_SPV;
    vk::ShaderModule zi_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(zi_spv.len)
        .setCode((uint*)&zi_spv[0])
        .build(ctx.device)!!;

    llm::ComputeKernel grl = llm::create_kernel(
        ctx, zi_shader, 3, zimage::GatedResidualPC.sizeof, "gated_residual_linear"
    )!!;

    zi_shader.free(ctx.device);

    return {
        .dit = dit,
        .gated_residual_linear = grl,
    };
}

fn void FluxDiTKernels.free(&self, vk::Device device) {
    // Free each DiTKernel field individually (no .free(device) method on DiTKernels)
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
