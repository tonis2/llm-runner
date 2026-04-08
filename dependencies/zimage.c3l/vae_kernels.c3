module zimage;

import vk;
import llm;
import llm::pipelines;
import std::io;

// --- VAE-only push constant structs ---

struct SpatialAttentionPC {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
}

struct TransposeChannelSpatialPC {
    uint channels;
    uint spatial;
    uint direction;
}

struct VaeAttentionPC {
    uint head_dim;
    uint n_heads;
    uint seq_len;
    float scale;
    uint channels;
}

struct VaeLinearProjPC {
    uint out_dim;
    uint in_dim;
    uint seq_len;
}

// --- VAE-only Kernel Set ---
// Diffusion kernels for TAESD and Flux VAE decoding (including mid-block attention).

struct VaeKernels {
    llm::ComputeKernel conv2d;
    llm::ComputeKernel group_norm;
    llm::ComputeKernel upsample_nearest;
    llm::ComputeKernel scale_shift_clamp;
    llm::ComputeKernel relu;
    llm::ComputeKernel tanh_clamp;
    llm::ComputeKernel vae_attention;
    llm::ComputeKernel linear_proj;
    llm::ComputeKernel transpose_channel_spatial;
    llm::SharedKernels shared;  // silu, residual_add, matmul, softmax
}

fn VaeKernels? create_vae_kernels(llm::DeviceContext* ctx) {
    char[] spv = &pipelines::DIFFUSION_SPV;
    vk::ShaderModule shader = vk::shaderModuleCreateInfo()
        .setCodeSize(spv.len)
        .setCode((uint*)&spv[0])
        .build(ctx.device)!!;

    VaeKernels kernels = {
        .conv2d           = llm::create_kernel(ctx, shader, 4, pipelines::Conv2dPC.sizeof, "conv2d")!!,
        .group_norm       = llm::create_kernel(ctx, shader, 4, pipelines::GroupNormPC.sizeof, "group_norm")!!,
        .upsample_nearest = llm::create_kernel(ctx, shader, 2, pipelines::UpsamplePC.sizeof, "upsample_nearest")!!,
        .scale_shift_clamp = llm::create_kernel(ctx, shader, 2, pipelines::ScaleShiftPC.sizeof, "scale_shift_clamp")!!,
        .relu             = llm::create_kernel(ctx, shader, 1, pipelines::ReluPC.sizeof, "relu")!!,
        .tanh_clamp       = llm::create_kernel(ctx, shader, 1, pipelines::TanhClampPC.sizeof, "tanh_clamp")!!,
        .vae_attention     = llm::create_kernel(ctx, shader, 4, VaeAttentionPC.sizeof, "vae_attention")!!,
        .linear_proj       = llm::create_kernel(ctx, shader, 4, VaeLinearProjPC.sizeof, "linear_proj")!!,
        .transpose_channel_spatial = llm::create_kernel(ctx, shader, 2, TransposeChannelSpatialPC.sizeof, "transpose_channel_spatial")!!,
        .shared           = llm::create_shared_kernels(ctx, shader)!!,
    };

    shader.free(ctx.device);
    return kernels;
}

fn void VaeKernels.free(&self, vk::Device device) {
    self.conv2d.free(device);
    self.group_norm.free(device);
    self.upsample_nearest.free(device);
    self.scale_shift_clamp.free(device);
    self.relu.free(device);
    self.tanh_clamp.free(device);
    self.vae_attention.free(device);
    self.linear_proj.free(device);
    self.transpose_channel_spatial.free(device);
    self.shared.free(device);
}

// --- VAE dispatch_conv2d ---
// VAE weights are always F32, so no Q8 variant needed.

fn void dispatch_vae_conv2d(
    vk::CommandBuffer cmd,
    VaeKernels* k,
    llm::Tensor* weight,
    llm::Tensor* bias,
    llm::Tensor* input,
    llm::Tensor* output,
    pipelines::Conv2dPC* pc
) {
    uint spatial_groups = llm::ceil_div(pc.out_h * pc.out_w, 256);
    llm::dispatch_kernel(cmd, &k.conv2d,
        { weight.gpu_buffer.buffer, bias.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { weight.size_bytes, bias.size_bytes, input.size_bytes, output.size_bytes },
        pc, spatial_groups, pc.out_c);
}
