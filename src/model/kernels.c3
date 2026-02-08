module llm;

import vk;
import std::io;

// Shared compute kernels usable by both LLM and Diffusion models
struct SharedKernels {
    ComputeKernel rmsnorm;
    ComputeKernel layernorm;
    ComputeKernel silu;
    ComputeKernel gelu;
    ComputeKernel matmul;
    ComputeKernel matmul_q8;
    ComputeKernel matmul_q4_0;
    ComputeKernel matmul_q4k;
    ComputeKernel matmul_q5k;
    ComputeKernel matmul_q6k;
    ComputeKernel softmax;
    ComputeKernel residual_add;
    ComputeKernel elemwise_mul;
}

fn SharedKernels? create_shared_kernels(DeviceContext* ctx, ShaderModule shader) {
    return {
        .rmsnorm      = create_kernel(ctx, shader, 3, RMSNormPC.sizeof, "rmsnorm")!!,
        .layernorm    = create_kernel(ctx, shader, 4, LayerNormPC.sizeof, "layernorm")!!,
        .silu         = create_kernel(ctx, shader, 1, SiluPC.sizeof, "silu")!!,
        .gelu         = create_kernel(ctx, shader, 1, GeluPC.sizeof, "gelu")!!,
        .matmul       = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul")!!,
        .matmul_q8    = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q8")!!,
        .matmul_q4_0  = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q4_0")!!,
        .matmul_q4k   = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q4k")!!,
        .matmul_q5k   = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q5k")!!,
        .matmul_q6k   = create_kernel(ctx, shader, 3, MatMulPC.sizeof, "matmul_q6k")!!,
        .softmax      = create_kernel(ctx, shader, 1, SoftmaxPC.sizeof, "softmax")!!,
        .residual_add = create_kernel(ctx, shader, 2, ResidualPC.sizeof, "residual_add")!!,
        .elemwise_mul = create_kernel(ctx, shader, 2, ElemwisePC.sizeof, "elemwise_mul")!!,
    };
}

fn void SharedKernels.free(&self, Device device) {
    self.rmsnorm.free(device);
    self.layernorm.free(device);
    self.silu.free(device);
    self.gelu.free(device);
    self.matmul.free(device);
    self.matmul_q8.free(device);
    self.matmul_q4_0.free(device);
    self.matmul_q4k.free(device);
    self.matmul_q5k.free(device);
    self.matmul_q6k.free(device);
    self.softmax.free(device);
    self.residual_add.free(device);
    self.elemwise_mul.free(device);
}

fn uint ceil_div(uint a, uint b) {
    return (a + b - 1) / b;
}

fn void dispatch_matmul(
    CommandBuffer cmd,
    SharedKernels* k,
    Tensor* weight,
    Tensor* input,
    Tensor* output,
    uint out_dim,
    uint in_dim
) {
    MatMulPC pc = { .out_dim = out_dim, .in_dim = in_dim };
    ComputeKernel* kernel;
    if (weight.dtype == GGML_Q8_0) {
        kernel = &k.matmul_q8;
    } else if (weight.dtype == GGML_Q4_0) {
        kernel = &k.matmul_q4_0;
    } else if (weight.dtype == GGML_Q4_K) {
        kernel = &k.matmul_q4k;
    } else if (weight.dtype == GGML_Q5_K) {
        kernel = &k.matmul_q5k;
    } else if (weight.dtype == GGML_Q6_K) {
        kernel = &k.matmul_q6k;
    } else {
        kernel = &k.matmul;
    }
    dispatch_kernel(cmd, kernel,
        { weight.gpu_buffer.buffer, input.gpu_buffer.buffer, output.gpu_buffer.buffer },
        { weight.size_bytes, input.size_bytes, output.size_bytes },
        &pc, out_dim);
}
