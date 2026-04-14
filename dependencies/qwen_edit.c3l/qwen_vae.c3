module qwen_edit;

import vk;
import llm;
import llm::pipelines;
import zimage;
import std::io;
import std::math;
import std::core::mem;

// Qwen-Image VAE Decoder
// Architecture: Wan Video VAE with CausalConv3d, reduced to 2D for single-image decode.
// For T=1, CausalConv3d(k_t, k_h, k_w) reduces to Conv2d using weight[:,:,k_t-1,:,:].

struct VaeRMSNormPC { uint channels; uint spatial; float eps; uint _pad; }

// --- Structures ---

struct QwenVaeConv {
    llm::Tensor weight;  // [out_c, in_c, kH, kW] (extracted from 3D)
    llm::Tensor bias;
    uint in_c;
    uint out_c;
    uint kH;
    uint kW;
}

struct QwenVaeResBlock {
    llm::Tensor norm1_gamma;  // [in_c]
    QwenVaeConv conv1;        // [in_c, out_c, 3, 3]
    llm::Tensor norm2_gamma;  // [out_c]
    QwenVaeConv conv2;        // [out_c, out_c, 3, 3]
    QwenVaeConv shortcut;     // [in_c, out_c, 1, 1] (only if in_c != out_c)
    bool has_shortcut;
    uint in_c;
    uint out_c;
}

struct QwenVaeAttn {
    llm::Tensor norm_gamma;    // [channels]
    llm::Tensor q_weight;     // [channels, channels]
    llm::Tensor k_weight;     // [channels, channels]
    llm::Tensor v_weight;     // [channels, channels]
    llm::Tensor q_bias;       // [channels]
    llm::Tensor k_bias;       // [channels]
    llm::Tensor v_bias;       // [channels]
    llm::Tensor proj_weight;   // [channels, channels]
    llm::Tensor proj_bias;
    uint channels;
}

struct QwenVaeUpsampler {
    // For upsample3d (T=1): time_conv acts as 1x1 projection C→2C, we take second C channels
    QwenVaeConv time_conv;     // Only for temporal upsample blocks (0,1)
    bool has_time_conv;
    // Spatial: upsample 2x nearest + conv
    QwenVaeConv spatial_conv;  // [in_c, out_c, 3, 3]
}

struct QwenVaeDecoder {
    QwenVaeConv post_quant_conv;  // 1x1 conv [16→16]
    QwenVaeConv conv_in;          // [16→384, 3x3]

    // Mid block
    QwenVaeResBlock mid_res0;
    QwenVaeAttn mid_attn;
    QwenVaeResBlock mid_res1;

    // Up blocks: [0]=384→384+up3d, [1]=192→384+up3d, [2]=192→192+up2d, [3]=96→96
    QwenVaeResBlock[3] up0_res;   // 384→384
    QwenVaeUpsampler up0_up;
    QwenVaeResBlock[3] up1_res;   // 192→384 (first has shortcut), 384→384
    QwenVaeUpsampler up1_up;
    QwenVaeResBlock[3] up2_res;   // 192→192
    QwenVaeUpsampler up2_up;
    QwenVaeResBlock[3] up3_res;   // 96→96

    llm::Tensor norm_out_gamma;   // [96]
    QwenVaeConv conv_out;         // [96→3, 3x3]

    zimage::VaeKernels* kernels;
    llm::ComputeKernel vae_rmsnorm_kernel;
    llm::DeviceContext* ctx;
}

// --- Weight Loading ---

// Load a 3D conv weight: extract last temporal slice and transpose to [kH, kW, in_c, out_c]
fn QwenVaeConv load_conv3d_as_2d(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String weight_name, String bias_name, uint in_c, uint out_c, uint kH, uint kW) {
    llm::SafetensorsTensorInfo* winfo = sf.find_tensor(weight_name)!!;
    char* data_ptr = sf.data_base + (usz)winfo.data_start;

    uint k_t = (uint)winfo.shape[2];
    usz total_3d = (usz)out_c * in_c * k_t * kH * kW;

    // Decode BF16 → F32
    float[] src = mem::new_array(float, total_3d);
    if (winfo.dtype == "BF16") {
        for (usz j = 0; j < total_3d; j++) {
            src[j] = llm::cpu_bf16_to_f32(data_ptr + j * 2);
        }
    } else if (winfo.dtype == "F16") {
        for (usz j = 0; j < total_3d; j++) {
            src[j] = llm::cpu_f16_to_f32(data_ptr + j * 2);
        }
    } else {
        mem::copy(src.ptr, data_ptr, total_3d * 4);
    }

    // Extract last temporal slice + transpose: [out_c, in_c, k_t, kH, kW] → [kH, kW, in_c, out_c]
    usz elems_2d = (usz)out_c * in_c * kH * kW;
    float[] dst = mem::new_array(float, elems_2d);
    uint last_t = k_t - 1;

    for (uint oc = 0; oc < out_c; oc++) {
        for (uint ic = 0; ic < in_c; ic++) {
            for (uint h = 0; h < kH; h++) {
                for (uint w = 0; w < kW; w++) {
                    usz src_idx = (usz)oc * in_c * k_t * kH * kW + (usz)ic * k_t * kH * kW + (usz)last_t * kH * kW + (usz)h * kW + w;
                    usz dst_idx = (usz)h * kW * in_c * out_c + (usz)w * in_c * out_c + (usz)ic * out_c + oc;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    mem::free(src);

    vk::Memory gpu_buf = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: dst.ptr, data_size: elems_2d * 4,
    )!!;
    mem::free(dst);

    llm::Tensor weight_gpu = { .gpu_buffer = gpu_buf, .dtype = llm::GGML_F32, .size_bytes = elems_2d * 4 };
    llm::Tensor bias_gpu = llm::upload_safetensor_f32(ctx, sf, bias_name)!!;

    return {
        .weight = weight_gpu, .bias = bias_gpu,
        .in_c = in_c, .out_c = out_c, .kH = kH, .kW = kW,
    };
}

// Load a 2D conv weight (already 4D in safetensors), transpose to shader layout
fn QwenVaeConv load_conv2d(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String weight_name, String bias_name, uint in_c, uint out_c, uint kH, uint kW) {
    llm::Tensor w = llm::upload_safetensor_conv_f32(ctx, sf, weight_name)!!;
    llm::Tensor b = llm::upload_safetensor_f32(ctx, sf, bias_name)!!;
    return {
        .weight = w, .bias = b,
        .in_c = in_c, .out_c = out_c, .kH = kH, .kW = kW,
    };
}

fn llm::Tensor load_gamma(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String name) {
    return llm::upload_safetensor_f32(ctx, sf, name)!!;
}

fn QwenVaeResBlock load_resblock(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint in_c, uint out_c) {
    char[256] buf1;
    char[256] buf2;
    QwenVaeResBlock rb;
    rb.in_c = in_c;
    rb.out_c = out_c;

    rb.norm1_gamma = load_gamma(ctx, sf, format_vae_name(&buf1, prefix, "norm1.gamma"));
    rb.conv1 = load_conv3d_as_2d(ctx, sf,
        format_vae_name(&buf1, prefix, "conv1.weight"),
        format_vae_name(&buf2, prefix, "conv1.bias"),
        in_c, out_c, 3, 3);
    rb.norm2_gamma = load_gamma(ctx, sf, format_vae_name(&buf1, prefix, "norm2.gamma"));
    rb.conv2 = load_conv3d_as_2d(ctx, sf,
        format_vae_name(&buf1, prefix, "conv2.weight"),
        format_vae_name(&buf2, prefix, "conv2.bias"),
        out_c, out_c, 3, 3);

    rb.has_shortcut = in_c != out_c;
    if (rb.has_shortcut) {
        rb.shortcut = load_conv3d_as_2d(ctx, sf,
            format_vae_name(&buf1, prefix, "conv_shortcut.weight"),
            format_vae_name(&buf2, prefix, "conv_shortcut.bias"),
            in_c, out_c, 1, 1);
    }

    return rb;
}

fn String format_vae_name(char[256]* buf, String prefix, String suffix) {
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) { (*buf)[pos + i] = prefix[i]; }
    pos += prefix.len;
    for (usz i = 0; i < suffix.len; i++) { (*buf)[pos + i] = suffix[i]; }
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

fn llm::Tensor upload_f32_data(llm::DeviceContext* ctx, float* data, usz n_elems) {
    usz size = n_elems * 4;
    vk::Memory gpu = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
        properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        data: data, data_size: size,
    )!!;
    return { .gpu_buffer = gpu, .dtype = llm::GGML_F32, .size_bytes = size };
}

fn void decode_bf16_or_f32(llm::SafetensorsTensorInfo* info, char* data_ptr, float[] dst, usz count) {
    if (info.dtype == "BF16") {
        for (usz j = 0; j < count; j++) dst[j] = llm::cpu_bf16_to_f32(data_ptr + j * 2);
    } else if (info.dtype == "F16") {
        for (usz j = 0; j < count; j++) dst[j] = llm::cpu_f16_to_f32(data_ptr + j * 2);
    } else {
        mem::copy(dst.ptr, data_ptr, count * 4);
    }
}

fn QwenVaeAttn load_qwen_vae_attn(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint channels) {
    char[256] buf1;
    char[256] buf2;

    // Load norm gamma
    llm::Tensor norm_gamma = load_gamma(ctx, sf, format_vae_name(&buf1, prefix, "norm.gamma"));

    // Load combined QKV weight [3*C, C] and split into separate Q, K, V
    llm::SafetensorsTensorInfo* winfo = sf.find_tensor(format_vae_name(&buf1, prefix, "to_qkv.weight"))!!;
    char* wdata = sf.data_base + (usz)winfo.data_start;
    usz total_w = (usz)3 * channels * channels;
    float[] wsrc = mem::new_array(float, total_w);
    decode_bf16_or_f32(winfo, wdata, wsrc, total_w);

    usz slice_w = (usz)channels * channels;
    llm::Tensor q_w = upload_f32_data(ctx, wsrc.ptr, slice_w);
    llm::Tensor k_w = upload_f32_data(ctx, wsrc.ptr + slice_w, slice_w);
    llm::Tensor v_w = upload_f32_data(ctx, wsrc.ptr + 2 * slice_w, slice_w);
    mem::free(wsrc);

    // Load combined QKV bias [3*C] and split
    llm::SafetensorsTensorInfo* binfo = sf.find_tensor(format_vae_name(&buf1, prefix, "to_qkv.bias"))!!;
    char* bdata = sf.data_base + (usz)binfo.data_start;
    usz total_b = (usz)3 * channels;
    float[] bsrc = mem::new_array(float, total_b);
    decode_bf16_or_f32(binfo, bdata, bsrc, total_b);

    llm::Tensor q_b = upload_f32_data(ctx, bsrc.ptr, channels);
    llm::Tensor k_b = upload_f32_data(ctx, bsrc.ptr + channels, channels);
    llm::Tensor v_b = upload_f32_data(ctx, bsrc.ptr + 2 * channels, channels);
    mem::free(bsrc);

    // Load proj weight and bias (these are standard 2D)
    llm::Tensor proj_w = llm::upload_safetensor_f32(ctx, sf, format_vae_name(&buf1, prefix, "proj.weight"))!!;
    llm::Tensor proj_b = llm::upload_safetensor_f32(ctx, sf, format_vae_name(&buf1, prefix, "proj.bias"))!!;

    return {
        .norm_gamma = norm_gamma,
        .q_weight = q_w, .k_weight = k_w, .v_weight = v_w,
        .q_bias = q_b, .k_bias = k_b, .v_bias = v_b,
        .proj_weight = proj_w, .proj_bias = proj_b,
        .channels = channels,
    };
}

fn QwenVaeDecoder? load_qwen_vae_decoder(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, zimage::VaeKernels* kernels) {
    io::printfn("  Loading Qwen-Image VAE decoder...");
    char[256] buf;
    QwenVaeDecoder dec;
    dec.kernels = kernels;

    // Post-quant conv: 1x1x1 → effectively 1x1
    dec.post_quant_conv = load_conv3d_as_2d(ctx, sf, "post_quant_conv.weight", "post_quant_conv.bias", 16, 16, 1, 1);
    io::printfn("    post_quant_conv loaded");

    // Conv in: 3x3x3 → 3x3
    dec.conv_in = load_conv3d_as_2d(ctx, sf, "decoder.conv_in.weight", "decoder.conv_in.bias", 16, 384, 3, 3);
    io::printfn("    conv_in loaded");

    // Mid block
    dec.mid_res0 = load_resblock(ctx, sf, "decoder.mid_block.resnets.0.", 384, 384);
    dec.mid_res1 = load_resblock(ctx, sf, "decoder.mid_block.resnets.1.", 384, 384);
    io::printfn("    mid resnets loaded");

    // Mid attention (split combined QKV into separate Q/K/V)
    dec.mid_attn = load_qwen_vae_attn(ctx, sf, "decoder.mid_block.attentions.0.", 384);
    io::printfn("    mid attention loaded");

    // Up blocks
    String[3] rb_pfx0 = { "decoder.up_blocks.0.resnets.0.", "decoder.up_blocks.0.resnets.1.", "decoder.up_blocks.0.resnets.2." };
    for (uint i = 0; i < 3; i++) {
        dec.up0_res[i] = load_resblock(ctx, sf, rb_pfx0[i], 384, 384);
    }
    dec.up0_up = {
        .has_time_conv = true,
        .time_conv = load_conv3d_as_2d(ctx, sf, "decoder.up_blocks.0.upsamplers.0.time_conv.weight", "decoder.up_blocks.0.upsamplers.0.time_conv.bias", 384, 768, 1, 1),
        .spatial_conv = load_conv2d(ctx, sf, "decoder.up_blocks.0.upsamplers.0.resample.1.weight", "decoder.up_blocks.0.upsamplers.0.resample.1.bias", 384, 192, 3, 3),
    };
    io::printfn("    up_block 0 loaded");

    // Up block 1: first resblock 192→384 (has shortcut), rest 384→384
    dec.up1_res[0] = load_resblock(ctx, sf, "decoder.up_blocks.1.resnets.0.", 192, 384);
    dec.up1_res[1] = load_resblock(ctx, sf, "decoder.up_blocks.1.resnets.1.", 384, 384);
    dec.up1_res[2] = load_resblock(ctx, sf, "decoder.up_blocks.1.resnets.2.", 384, 384);
    dec.up1_up = {
        .has_time_conv = true,
        .time_conv = load_conv3d_as_2d(ctx, sf, "decoder.up_blocks.1.upsamplers.0.time_conv.weight", "decoder.up_blocks.1.upsamplers.0.time_conv.bias", 384, 768, 1, 1),
        .spatial_conv = load_conv2d(ctx, sf, "decoder.up_blocks.1.upsamplers.0.resample.1.weight", "decoder.up_blocks.1.upsamplers.0.resample.1.bias", 384, 192, 3, 3),
    };
    io::printfn("    up_block 1 loaded");

    String[3] rb_pfx2 = { "decoder.up_blocks.2.resnets.0.", "decoder.up_blocks.2.resnets.1.", "decoder.up_blocks.2.resnets.2." };
    for (uint i = 0; i < 3; i++) {
        dec.up2_res[i] = load_resblock(ctx, sf, rb_pfx2[i], 192, 192);
    }
    dec.up2_up = {
        .has_time_conv = false,
        .spatial_conv = load_conv2d(ctx, sf, "decoder.up_blocks.2.upsamplers.0.resample.1.weight", "decoder.up_blocks.2.upsamplers.0.resample.1.bias", 192, 96, 3, 3),
    };
    io::printfn("    up_block 2 loaded");

    String[3] rb_pfx3 = { "decoder.up_blocks.3.resnets.0.", "decoder.up_blocks.3.resnets.1.", "decoder.up_blocks.3.resnets.2." };
    for (uint i = 0; i < 3; i++) {
        dec.up3_res[i] = load_resblock(ctx, sf, rb_pfx3[i], 96, 96);
    }
    io::printfn("    up_block 3 loaded");

    dec.norm_out_gamma = load_gamma(ctx, sf, "decoder.norm_out.gamma");
    dec.conv_out = load_conv3d_as_2d(ctx, sf, "decoder.conv_out.weight", "decoder.conv_out.bias", 96, 3, 3, 3);
    io::printfn("    output head loaded");

    // Create VAE RMS norm kernel from vit.spv
    char[] vit_spv = &VIT_SPV;
    vk::ShaderModule vit_shader = vk::shaderModuleCreateInfo().setCodeSize(
        vit_spv.len).setCode((uint*)&vit_spv[0]).build(ctx.device)!!;
    dec.vae_rmsnorm_kernel = llm::create_kernel(ctx, vit_shader, 3, VaeRMSNormPC.sizeof, "vae_rmsnorm")!!;
    vit_shader.free(ctx.device);
    dec.ctx = ctx;

    io::printfn("  Qwen-Image VAE decoder loaded.");
    return dec;
}

fn String format_uint_suffix(char[256]* buf, uint val, String suffix) {
    usz pos = 0;
    if (val >= 10) { (*buf)[pos] = (char)('0' + val / 10); pos++; }
    (*buf)[pos] = (char)('0' + val % 10); pos++;
    for (usz i = 0; i < suffix.len; i++) { (*buf)[pos + i] = suffix[i]; }
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

// --- Forward Pass Helpers ---

fn void qvae_conv(llm::RenderGraph* g, zimage::VaeKernels* k, QwenVaeConv* conv,
    llm::Tensor* input, llm::Tensor* output, uint h, uint w) {
    uint pad = conv.kH == 1 ? 0 : 1;
    uint out_h = h;
    uint out_w = w;
    pipelines::Conv2dPC pc = {
        .in_c = conv.in_c, .out_c = conv.out_c,
        .in_h = h, .in_w = w,
        .kH = conv.kH, .kW = conv.kW,
        .stride = 1, .pad = pad,
        .groups = 1, .out_h = out_h, .out_w = out_w,
        .has_bias = 1,
    };
    uint spatial_groups = llm::ceil_div(out_h * out_w, 256);
    g.dispatch(&k.conv2d, { &conv.weight, &conv.bias, input, output }, &pc, spatial_groups, conv.out_c);
}

// VAE RMS Norm: per-spatial-position normalization across channels in [C, spatial] layout
// output[c*S+s] = input[c*S+s] / sqrt(mean(input[c,s]^2) + eps) * gamma[c]

fn void qvae_rmsnorm(llm::RenderGraph* g, llm::ComputeKernel* kernel,
    llm::Tensor* gamma, llm::Tensor* input, llm::Tensor* output,
    uint channels, uint spatial) {
    VaeRMSNormPC pc = {
        .channels = channels,
        .spatial = spatial,
        .eps = 1e-6f,
    };
    g.dispatch(kernel, { input, gamma, output }, &pc, spatial);
}

fn void qvae_resblock(llm::RenderGraph* g, zimage::VaeKernels* k, llm::ComputeKernel* rmsnorm_kernel,
    QwenVaeResBlock* rb, llm::Tensor* buf_a, llm::Tensor* buf_b, llm::Tensor* buf_c,
    uint h, uint w) {
    uint spatial = h * w;

    // norm1 → silu → conv1
    qvae_rmsnorm(g, rmsnorm_kernel, &rb.norm1_gamma, buf_a, buf_b, rb.in_c, spatial);
    llm::SiluPC silu1 = { .n = rb.in_c * spatial };
    g.dispatch(&k.shared.silu, { buf_b }, &silu1, llm::ceil_div(rb.in_c * spatial, 256));
    qvae_conv(g, k, &rb.conv1, buf_b, buf_c, h, w);

    // norm2 → silu → conv2
    qvae_rmsnorm(g, rmsnorm_kernel, &rb.norm2_gamma, buf_c, buf_b, rb.out_c, spatial);
    llm::SiluPC silu2 = { .n = rb.out_c * spatial };
    g.dispatch(&k.shared.silu, { buf_b }, &silu2, llm::ceil_div(rb.out_c * spatial, 256));
    qvae_conv(g, k, &rb.conv2, buf_b, buf_c, h, w);

    // Residual
    llm::ResidualPC res_pc = { .n = rb.out_c * spatial };
    if (rb.has_shortcut) {
        qvae_conv(g, k, &rb.shortcut, buf_a, buf_b, h, w);
        g.dispatch(&k.shared.residual_add, { buf_c, buf_b }, &res_pc, llm::ceil_div(rb.out_c * spatial, 256));
    } else {
        g.dispatch(&k.shared.residual_add, { buf_c, buf_a }, &res_pc, llm::ceil_div(rb.out_c * spatial, 256));
    }
    g.copy(buf_c, buf_a, (ulong)rb.out_c * spatial * 4);
}

fn void qvae_upsample(llm::RenderGraph* g, zimage::VaeKernels* k,
    QwenVaeUpsampler* up, llm::Tensor* buf_a, llm::Tensor* buf_b,
    uint channels, uint* h, uint* w) {
    // For temporal upsample with T=1: skip time_conv, go straight to spatial
    // The time_conv would produce 2C channels and interleave → 2 frames.
    // For T=1, we skip this and just do spatial upsample.

    // Spatial: upsample 2x nearest + conv
    pipelines::UpsamplePC up_pc = { .channels = channels, .in_h = *h, .in_w = *w };
    g.dispatch(&k.upsample_nearest, { buf_a, buf_b }, &up_pc,
        llm::ceil_div(channels * (*h) * (*w) * 4, 256));
    *h *= 2;
    *w *= 2;
    qvae_conv(g, k, &up.spatial_conv, buf_b, buf_a, *h, *w);
}

// --- Mid-block Attention ---
// RMSNorm → transpose [C,S]→[S,C] → Q/K/V projections → attention → proj_out → transpose → residual

fn void qvae_attention(llm::RenderGraph* g, zimage::VaeKernels* k, llm::ComputeKernel* rmsnorm_kernel,
    QwenVaeAttn* attn, zimage::FluxVaeActivations* acts, uint h, uint w) {
    uint channels = attn.channels;
    uint spatial = h * w;
    uint n_heads = 1;
    uint head_dim = channels;
    float scale = 1.0f / math::sqrt((float)head_dim);

    // Save residual
    g.copy(&acts.buf_a, &acts.attn_temp, (ulong)channels * spatial * 4);

    // RMSNorm on input [C, S] layout
    qvae_rmsnorm(g, rmsnorm_kernel, &attn.norm_gamma, &acts.buf_a, &acts.buf_b, channels, spatial);

    // Transpose [C, S] → [S, C]
    zimage::TransposeChannelSpatialPC tcs_fwd = { .channels = channels, .spatial = spatial, .direction = 0 };
    g.dispatch(&k.transpose_channel_spatial, { &acts.buf_b, &acts.buf_c }, &tcs_fwd,
        llm::ceil_div(channels * spatial, 256));

    // Q/K/V projections: [S, C] → [S, C] each
    zimage::VaeLinearProjPC proj_pc = { .out_dim = channels, .in_dim = channels, .seq_len = spatial };
    g.dispatch(&k.linear_proj, { &attn.q_weight, &attn.q_bias, &acts.buf_c, &acts.attn_q },
        &proj_pc, spatial * channels);
    g.dispatch(&k.linear_proj, { &attn.k_weight, &attn.k_bias, &acts.buf_c, &acts.attn_k },
        &proj_pc, spatial * channels);
    g.dispatch(&k.linear_proj, { &attn.v_weight, &attn.v_bias, &acts.buf_c, &acts.attn_v },
        &proj_pc, spatial * channels);

    // Self-attention (single-head, flash-attention style)
    zimage::VaeAttentionPC va_pc = {
        .head_dim = head_dim, .n_heads = n_heads,
        .seq_len = spatial, .scale = scale, .channels = channels,
    };
    g.dispatch(&k.vae_attention, { &acts.attn_q, &acts.attn_k, &acts.attn_v, &acts.attn_out },
        &va_pc, n_heads * spatial);

    // Output projection: [S, C] → [S, C]
    g.dispatch(&k.linear_proj, { &attn.proj_weight, &attn.proj_bias, &acts.attn_out, &acts.buf_c },
        &proj_pc, spatial * channels);

    // Transpose [S, C] → [C, S]
    zimage::TransposeChannelSpatialPC tcs_rev = { .channels = channels, .spatial = spatial, .direction = 1 };
    g.dispatch(&k.transpose_channel_spatial, { &acts.buf_c, &acts.buf_a }, &tcs_rev,
        llm::ceil_div(channels * spatial, 256));

    // Residual add
    llm::ResidualPC res_pc = { .n = channels * spatial };
    g.dispatch(&k.shared.residual_add, { &acts.buf_a, &acts.attn_temp }, &res_pc,
        llm::ceil_div(channels * spatial, 256));
}

// --- Main Forward Pass ---

fn void? QwenVaeDecoder.forward(&self, llm::DeviceContext* ctx, zimage::FluxVaeActivations* acts, uint lat_h, uint lat_w) {
    zimage::VaeKernels* k = self.kernels;
    llm::ComputeKernel* rk = &self.vae_rmsnorm_kernel;
    uint h = lat_h;
    uint w = lat_w;

    io::printfn("    VAE forward: latent [16, %d, %d]", h, w);

    // Post-quant conv: [16, h, w] → [16, h, w]
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_conv(&g, k, &self.post_quant_conv, &acts.buf_a, &acts.buf_b, h, w);
        g.copy(&acts.buf_b, &acts.buf_a, (ulong)16 * h * w * 4);
        g.flush()!!;
    }

    // Conv in: [16, h, w] → [384, h, w]
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_conv(&g, k, &self.conv_in, &acts.buf_a, &acts.buf_b, h, w);
        g.copy(&acts.buf_b, &acts.buf_a, (ulong)384 * h * w * 4);
        g.flush()!!;
    }
    io::printfn("    After conv_in: [384, %d, %d]", h, w);

    // Mid block: ResBlock → Attention → ResBlock
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_resblock(&g, k, rk, &self.mid_res0, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.flush()!!;
    }
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_attention(&g, k, rk, &self.mid_attn, acts, h, w);
        g.flush()!!;
    }
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_resblock(&g, k, rk, &self.mid_res1, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.flush()!!;
    }
    io::printfn("    After mid: [384, %d, %d]", h, w);

    // Up block 0: 3 resnets + upsample (384→384, then →192 via upsample conv)
    for (uint i = 0; i < 3; i++) {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_resblock(&g, k, rk, &self.up0_res[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.flush()!!;
    }
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_upsample(&g, k, &self.up0_up, &acts.buf_a, &acts.buf_b, 384, &h, &w);
        g.flush()!!;
    }
    io::printfn("    After up0: [192, %d, %d]", h, w);

    // Up block 1: 3 resnets (192→384, 384→384, 384→384) + upsample
    for (uint i = 0; i < 3; i++) {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_resblock(&g, k, rk, &self.up1_res[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.flush()!!;
    }
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_upsample(&g, k, &self.up1_up, &acts.buf_a, &acts.buf_b, 384, &h, &w);
        g.flush()!!;
    }
    io::printfn("    After up1: [192, %d, %d]", h, w);

    // Up block 2: 3 resnets (192→192) + upsample (spatial only)
    for (uint i = 0; i < 3; i++) {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_resblock(&g, k, rk, &self.up2_res[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.flush()!!;
    }
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_upsample(&g, k, &self.up2_up, &acts.buf_a, &acts.buf_b, 192, &h, &w);
        g.flush()!!;
    }
    io::printfn("    After up2: [96, %d, %d]", h, w);

    // Up block 3: 3 resnets (96→96), no upsample
    for (uint i = 0; i < 3; i++) {
        llm::RenderGraph g;
        g.init(ctx)!!;
        qvae_resblock(&g, k, rk, &self.up3_res[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.flush()!!;
    }
    io::printfn("    After up3: [96, %d, %d]", h, w);

    // Output: RMSNorm → SiLU → conv_out → clamp
    {
        llm::RenderGraph g;
        g.init(ctx)!!;
        uint spatial = h * w;

        qvae_rmsnorm(&g, rk, &self.norm_out_gamma, &acts.buf_a, &acts.buf_b, 96, spatial);
        llm::SiluPC silu_pc = { .n = 96 * spatial };
        g.dispatch(&k.shared.silu, { &acts.buf_b }, &silu_pc, llm::ceil_div(96 * spatial, 256));
        qvae_conv(&g, k, &self.conv_out, &acts.buf_b, &acts.buf_a, h, w);

        // Clamp to [-1, 1] then map to [0, 1]: (x + 1) / 2
        pipelines::ScaleShiftPC ss_pc = { .n = 3 * spatial, .scale = 0.5f, .shift = 0.5f };
        g.dispatch(&k.scale_shift_clamp, { &acts.buf_a, &acts.buf_a }, &ss_pc, llm::ceil_div(3 * spatial, 256));

        g.flush()!!;
    }
    io::printfn("    After output: [3, %d, %d]", h, w);
}
