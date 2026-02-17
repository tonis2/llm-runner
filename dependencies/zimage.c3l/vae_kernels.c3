module zimage;

import vk;
import llm;
import llm::pipelines;
import std::io;

// --- VAE-only Kernel Set ---
// Lightweight subset of diffusion kernels needed for TAESD and Flux VAE decoding.

struct VaeKernels {
    llm::ComputeKernel conv2d;
    llm::ComputeKernel group_norm;
    llm::ComputeKernel upsample_nearest;
    llm::ComputeKernel scale_shift_clamp;
    llm::ComputeKernel relu;
    llm::ComputeKernel tanh_clamp;
    llm::SharedKernels shared;  // silu, residual_add
}

fn VaeKernels? create_vae_kernels(llm::DeviceContext* ctx) {
    io::printfn("Creating VAE compute kernels...");
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
