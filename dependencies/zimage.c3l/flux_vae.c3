module zimage;

import vk;
import llm;
import std::io;
import std::math;
import std::core::mem;

// Flux VAE Decoder
// Standard KL-VAE with 16 latent channels
// Architecture from safetensors: conv_in(16->512) -> mid(2 resblocks + attn) ->
// up.3(512, upsample) -> up.2(512->512, upsample) -> up.1(512->256, upsample) -> up.0(256->128)
// -> norm_out(128) -> conv_out(128->3)

// --- Flux VAE Structures ---

struct FluxVaeConv {
    llm::Tensor weight;
    llm::Tensor bias;
    uint in_c;
    uint out_c;
    uint kH;
    uint kW;
}

struct FluxVaeResBlock {
    // norm1 + conv1 + norm2 + conv2 + optional nin_shortcut
    llm::Tensor norm1_weight;
    llm::Tensor norm1_bias;
    llm::Tensor conv1_weight;
    llm::Tensor conv1_bias;
    llm::Tensor norm2_weight;
    llm::Tensor norm2_bias;
    llm::Tensor conv2_weight;
    llm::Tensor conv2_bias;
    llm::Tensor nin_shortcut_weight;
    llm::Tensor nin_shortcut_bias;
    uint in_c;
    uint out_c;
    bool has_nin_shortcut;
}

struct FluxVaeAttn {
    llm::Tensor norm_weight;
    llm::Tensor norm_bias;
    llm::Tensor q_weight;
    llm::Tensor q_bias;
    llm::Tensor k_weight;
    llm::Tensor k_bias;
    llm::Tensor v_weight;
    llm::Tensor v_bias;
    llm::Tensor proj_out_weight;
    llm::Tensor proj_out_bias;
    uint channels;
}

struct FluxVaeDecoder {
    FluxVaeConv conv_in;       // [512, 16, 3, 3]

    // Mid block
    FluxVaeResBlock mid_block1;
    FluxVaeAttn mid_attn;
    FluxVaeResBlock mid_block2;

    // Up blocks (in reverse order: up.3 -> up.2 -> up.1 -> up.0)
    FluxVaeResBlock[3] up3_blocks;  // 3 resblocks at 512, + upsample
    FluxVaeConv up3_upsample;

    FluxVaeResBlock[3] up2_blocks;  // 3 resblocks at 512, + upsample
    FluxVaeConv up2_upsample;

    FluxVaeResBlock[3] up1_blocks;  // 3 resblocks 512->256, + upsample
    FluxVaeConv up1_upsample;

    FluxVaeResBlock[3] up0_blocks;  // 3 resblocks 256->128

    // Output
    llm::Tensor norm_out_weight;
    llm::Tensor norm_out_bias;
    FluxVaeConv conv_out;          // [3, 128, 3, 3]

    VaeKernels* kernels;
    llm::DeviceContext* ctx;
}

struct FluxVaeActivations {
    llm::Tensor buf_a;  // Primary buffer
    llm::Tensor buf_b;  // Secondary buffer
    llm::Tensor buf_c;  // Temp for residual
    llm::Tensor attn_q;      // [lat_spatial * 512] Q for mid-block attention
    llm::Tensor attn_k;      // [lat_spatial * 512] K for mid-block attention
    llm::Tensor attn_v;      // [lat_spatial * 512] V for mid-block attention
    llm::Tensor attn_out;    // [lat_spatial * 512] attention output
    llm::Tensor attn_temp;   // [lat_spatial * 512] residual save
}

// --- Loading Helpers ---

fn FluxVaeConv load_flux_conv(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint in_c, uint out_c) {
    char[256] buf;
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) { buf[pos] = prefix[i]; pos++; }

    char[256] wname;
    char[256] bname;
    for (usz i = 0; i < pos; i++) { wname[i] = buf[i]; bname[i] = buf[i]; }
    String w_suffix = "weight";
    String b_suffix = "bias";
    for (usz i = 0; i < w_suffix.len; i++) wname[pos + i] = w_suffix[i];
    for (usz i = 0; i < b_suffix.len; i++) bname[pos + i] = b_suffix[i];

    String w_name = (String)wname[0..pos + w_suffix.len - 1];
    String b_name = (String)bname[0..pos + b_suffix.len - 1];

    llm::Tensor w = llm::upload_safetensor_conv_f32(ctx, sf, w_name)!!;
    llm::Tensor b = llm::upload_safetensor_f32(ctx, sf, b_name)!!;

    return {
        .weight = w,
        .bias = b,
        .in_c = in_c,
        .out_c = out_c,
        .kH = 3,
        .kW = 3,
    };
}

fn FluxVaeResBlock load_flux_resblock(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint in_c, uint out_c) {
    char[256] buf;

    FluxVaeResBlock rb;
    rb.in_c = in_c;
    rb.out_c = out_c;

    // Build tensor names with prefix
    rb.norm1_weight = load_flux_vae_tensor(ctx, sf, prefix, "norm1.weight");
    rb.norm1_bias = load_flux_vae_tensor(ctx, sf, prefix, "norm1.bias");
    rb.conv1_weight = load_flux_vae_conv_weight(ctx, sf, prefix, "conv1.weight");
    rb.conv1_bias = load_flux_vae_tensor(ctx, sf, prefix, "conv1.bias");
    rb.norm2_weight = load_flux_vae_tensor(ctx, sf, prefix, "norm2.weight");
    rb.norm2_bias = load_flux_vae_tensor(ctx, sf, prefix, "norm2.bias");
    rb.conv2_weight = load_flux_vae_conv_weight(ctx, sf, prefix, "conv2.weight");
    rb.conv2_bias = load_flux_vae_tensor(ctx, sf, prefix, "conv2.bias");

    // Check for nin_shortcut (channel change)
    char[256] nin_buf;
    String nin_name = flux_vae_name(&nin_buf, prefix, "nin_shortcut.weight");
    if (try info = sf.find_tensor(nin_name)) {
        rb.nin_shortcut_weight = llm::upload_safetensor_conv_f32(ctx, sf, nin_name)!!;
        rb.nin_shortcut_bias = load_flux_vae_tensor(ctx, sf, prefix, "nin_shortcut.bias");
        rb.has_nin_shortcut = true;
    } else {
        rb.has_nin_shortcut = (in_c != out_c);
    }

    return rb;
}

fn String flux_vae_name(char[256]* buf, String prefix, String suffix) {
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) { (*buf)[pos] = prefix[i]; pos++; }
    for (usz i = 0; i < suffix.len; i++) { (*buf)[pos] = suffix[i]; pos++; }
    return (String)(*buf)[0..pos - 1];
}

fn llm::Tensor load_flux_vae_tensor(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, String suffix) {
    char[256] buf;
    String name = flux_vae_name(&buf, prefix, suffix);
    return llm::upload_safetensor_f32(ctx, sf, name)!!;
}

fn llm::Tensor load_flux_vae_conv_weight(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, String suffix) {
    char[256] buf;
    String name = flux_vae_name(&buf, prefix, suffix);
    return llm::upload_safetensor_conv_f32(ctx, sf, name)!!;
}

fn FluxVaeAttn load_flux_attn(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint channels) {
    return {
        .norm_weight = load_flux_vae_tensor(ctx, sf, prefix, "norm.weight"),
        .norm_bias = load_flux_vae_tensor(ctx, sf, prefix, "norm.bias"),
        .q_weight = load_flux_vae_tensor(ctx, sf, prefix, "q.weight"),
        .q_bias = load_flux_vae_tensor(ctx, sf, prefix, "q.bias"),
        .k_weight = load_flux_vae_tensor(ctx, sf, prefix, "k.weight"),
        .k_bias = load_flux_vae_tensor(ctx, sf, prefix, "k.bias"),
        .v_weight = load_flux_vae_tensor(ctx, sf, prefix, "v.weight"),
        .v_bias = load_flux_vae_tensor(ctx, sf, prefix, "v.bias"),
        .proj_out_weight = load_flux_vae_tensor(ctx, sf, prefix, "proj_out.weight"),
        .proj_out_bias = load_flux_vae_tensor(ctx, sf, prefix, "proj_out.bias"),
        .channels = channels,
    };
}

// --- Load Flux VAE Decoder ---

fn FluxVaeDecoder? load_flux_vae_decoder(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, VaeKernels* kernels) {
    String pfx = "decoder.";

    FluxVaeDecoder dec;
    dec.ctx = ctx;
    dec.kernels = kernels;

    dec.conv_in = load_flux_conv(ctx, sf, "decoder.conv_in.", 16, 512);
    dec.mid_block1 = load_flux_resblock(ctx, sf, "decoder.mid.block_1.", 512, 512);
    dec.mid_attn = load_flux_attn(ctx, sf, "decoder.mid.attn_1.", 512);
    dec.mid_block2 = load_flux_resblock(ctx, sf, "decoder.mid.block_2.", 512, 512);

    // up.3: 3 resblocks at 512 + upsample
    for (uint i = 0; i < 3; i++) {
        char[64] rb_prefix;
        usz plen = 0;
        String base = "decoder.up.3.block.";
        for (usz j = 0; j < base.len; j++) { rb_prefix[plen] = base[j]; plen++; }
        rb_prefix[plen] = (char)('0' + i); plen++;
        rb_prefix[plen] = '.'; plen++;
        dec.up3_blocks[i] = load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], 512, 512);
    }
    dec.up3_upsample = load_flux_conv(ctx, sf, "decoder.up.3.upsample.conv.", 512, 512);

    // up.2: 3 resblocks at 512 + upsample
    for (uint i = 0; i < 3; i++) {
        char[64] rb_prefix;
        usz plen = 0;
        String base = "decoder.up.2.block.";
        for (usz j = 0; j < base.len; j++) { rb_prefix[plen] = base[j]; plen++; }
        rb_prefix[plen] = (char)('0' + i); plen++;
        rb_prefix[plen] = '.'; plen++;
        uint in_c = (i == 0) ? 512 : 512;
        dec.up2_blocks[i] = load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], in_c, 512);
    }
    dec.up2_upsample = load_flux_conv(ctx, sf, "decoder.up.2.upsample.conv.", 512, 512);

    // up.1: 3 resblocks 512->256 + upsample
    for (uint i = 0; i < 3; i++) {
        char[64] rb_prefix;
        usz plen = 0;
        String base = "decoder.up.1.block.";
        for (usz j = 0; j < base.len; j++) { rb_prefix[plen] = base[j]; plen++; }
        rb_prefix[plen] = (char)('0' + i); plen++;
        rb_prefix[plen] = '.'; plen++;
        uint in_c = (i == 0) ? 512 : 256;
        dec.up1_blocks[i] = load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], in_c, 256);
    }
    dec.up1_upsample = load_flux_conv(ctx, sf, "decoder.up.1.upsample.conv.", 256, 256);

    // up.0: 3 resblocks 256->128
    for (uint i = 0; i < 3; i++) {
        char[64] rb_prefix;
        usz plen = 0;
        String base = "decoder.up.0.block.";
        for (usz j = 0; j < base.len; j++) { rb_prefix[plen] = base[j]; plen++; }
        rb_prefix[plen] = (char)('0' + i); plen++;
        rb_prefix[plen] = '.'; plen++;
        uint in_c = (i == 0) ? 256 : 128;
        dec.up0_blocks[i] = load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], in_c, 128);
    }

    // Output
    dec.norm_out_weight = load_flux_vae_tensor(ctx, sf, "decoder.", "norm_out.weight");
    dec.norm_out_bias = load_flux_vae_tensor(ctx, sf, "decoder.", "norm_out.bias");
    dec.conv_out = load_flux_conv(ctx, sf, "decoder.conv_out.", 128, 3);

    return dec;
}

// --- Allocate Flux VAE Activations ---

fn FluxVaeActivations? allocate_flux_vae_activations(llm::DeviceContext* ctx, uint max_size) {
    // Max buffer size is determined by the largest channels*H*W across all stages:
    //   512ch at 1/4 res: 512*(max_size/4)^2   (up.2)
    //   256ch at full res: 256*max_size^2        (after up.1 upsample)
    //   128ch at full res: 128*max_size^2        (after up.0)
    // The maximum is 256 * max_size * max_size (256ch at full resolution)
    ulong max_elems = (ulong)256 * max_size * max_size;
    ulong[4] shape = { max_elems, 0, 0, 0 };

    // Attention buffers: 512 channels at latent resolution (max_size/8)
    ulong lat_size = (ulong)max_size / 8;
    ulong attn_elems = 512 * lat_size * lat_size;
    ulong[4] attn_shape = { attn_elems, 0, 0, 0 };

    return {
        .buf_a = llm::create_f32_tensor(ctx, shape, 1)!!,
        .buf_b = llm::create_f32_tensor(ctx, shape, 1)!!,
        .buf_c = llm::create_f32_tensor(ctx, shape, 1)!!,
        .attn_q = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_k = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_v = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_temp = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
    };
}

// --- Forward Pass Block Functions (RenderGraph-based) ---

fn void flux_conv_block(
    llm::RenderGraph* g,
    VaeKernels* k,
    FluxVaeConv* conv,
    llm::Tensor* input,
    llm::Tensor* output,
    uint in_h, uint in_w,
    uint stride, uint pad
) {
    uint out_h = (in_h + 2 * pad - conv.kH) / stride + 1;
    uint out_w = (in_w + 2 * pad - conv.kW) / stride + 1;
    pipelines::Conv2dPC pc = {
        .in_c = conv.in_c,
        .out_c = conv.out_c,
        .in_h = in_h,
        .in_w = in_w,
        .kH = conv.kH,
        .kW = conv.kW,
        .stride = stride,
        .pad = pad,
        .groups = 1,
        .out_h = out_h,
        .out_w = out_w,
        .has_bias = 1,
    };
    uint spatial_groups = llm::ceil_div(out_h * out_w, 256);
    g.dispatch(&k.conv2d,
        { &conv.weight, &conv.bias, input, output },
        &pc, spatial_groups, conv.out_c);
}

fn void flux_conv2d_block(
    llm::RenderGraph* g,
    VaeKernels* k,
    llm::Tensor* weight,
    llm::Tensor* bias,
    llm::Tensor* input,
    llm::Tensor* output,
    pipelines::Conv2dPC* pc,
) {
    uint spatial_groups = llm::ceil_div(pc.out_h * pc.out_w, 256);
    g.dispatch(&k.conv2d,
        { weight, bias, input, output },
        pc, spatial_groups, pc.out_c);
}

fn void flux_groupnorm_block(
    llm::RenderGraph* g,
    VaeKernels* k,
    llm::Tensor* weight,
    llm::Tensor* bias,
    llm::Tensor* input,
    llm::Tensor* output,
    uint channels,
    uint spatial
) {
    pipelines::GroupNormPC pc = {
        .channels = channels,
        .spatial = spatial,
        .num_groups = 32,
        .eps = 1e-6f,
    };
    g.dispatch(&k.group_norm,
        { input, weight, bias, output },
        &pc, 32);
}

fn void flux_resblock_block(
    llm::RenderGraph* g,
    VaeKernels* k,
    FluxVaeResBlock* rb,
    llm::Tensor* input,     // buf_a - also receives output
    llm::Tensor* temp,      // buf_b
    llm::Tensor* skip_temp, // buf_c
    uint h, uint w
) {
    uint in_c = rb.in_c;
    uint out_c = rb.out_c;
    uint spatial = h * w;

    // GroupNorm1 -> SiLU -> Conv1
    flux_groupnorm_block(g, k, &rb.norm1_weight, &rb.norm1_bias, input, temp, in_c, spatial);

    llm::SiluPC silu_pc = { .n = in_c * spatial };
    g.dispatch(&k.shared.silu, { temp }, &silu_pc, llm::ceil_div(in_c * spatial, 256));

    pipelines::Conv2dPC pc1 = {
        .in_c = in_c, .out_c = out_c, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
    flux_conv2d_block(g, k, &rb.conv1_weight, &rb.conv1_bias, temp, skip_temp, &pc1);

    // GroupNorm2 -> SiLU -> Conv2
    flux_groupnorm_block(g, k, &rb.norm2_weight, &rb.norm2_bias, skip_temp, temp, out_c, spatial);

    llm::SiluPC silu_pc2 = { .n = out_c * spatial };
    g.dispatch(&k.shared.silu, { temp }, &silu_pc2, llm::ceil_div(out_c * spatial, 256));

    pipelines::Conv2dPC pc2 = {
        .in_c = out_c, .out_c = out_c, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
    flux_conv2d_block(g, k, &rb.conv2_weight, &rb.conv2_bias, temp, skip_temp, &pc2);

    // Skip connection
    llm::ResidualPC res_pc = { .n = out_c * spatial };
    if (rb.has_nin_shortcut && rb.nin_shortcut_weight.size_bytes > 0) {
        // 1x1 conv for channel change
        pipelines::Conv2dPC pc_nin = {
            .in_c = in_c, .out_c = out_c, .in_h = h, .in_w = w,
            .kH = 1, .kW = 1, .stride = 1, .pad = 0,
            .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
        };
        flux_conv2d_block(g, k, &rb.nin_shortcut_weight, &rb.nin_shortcut_bias, input, temp, &pc_nin);
        // Residual: skip_temp += temp
        g.dispatch(&k.shared.residual_add, { skip_temp, temp }, &res_pc, llm::ceil_div(out_c * spatial, 256));
    } else {
        // Identity skip: skip_temp += input
        g.dispatch(&k.shared.residual_add, { skip_temp, input }, &res_pc, llm::ceil_div(out_c * spatial, 256));
    }

    // Copy result back to input buffer
    g.copy(skip_temp, input, (ulong)out_c * spatial * 4);
}


fn void flux_attn_block(
    llm::RenderGraph* g,
    VaeKernels* k,
    FluxVaeAttn* attn,
    FluxVaeActivations* acts,
    uint h, uint w
) {
    uint channels = attn.channels;
    uint spatial = h * w;
    uint n_heads = 1;
    uint head_dim = channels / n_heads;
    float scale = 1.0f / math::sqrt((float)head_dim);

    // Step 1: Save residual BEFORE GroupNorm (original input)
    g.copy(&acts.buf_a, &acts.attn_temp, (ulong)channels * spatial * 4);

    // Step 2: GroupNorm on input [C, S] layout
    flux_groupnorm_block(g, k, &attn.norm_weight, &attn.norm_bias,
        &acts.buf_a, &acts.buf_b, channels, spatial);

    // Step 3: Transpose [C, S] -> [S, C]
    TransposeChannelSpatialPC tcs_fwd = {
        .channels = channels,
        .spatial = spatial,
        .direction = 0,
    };
    g.dispatch(&k.transpose_channel_spatial,
        { &acts.buf_b, &acts.buf_c },
        &tcs_fwd, llm::ceil_div(channels * spatial, 256));

    // Step 4-6: Q/K/V projections: [S, C] -> [S, C]
    VaeLinearProjPC q_pc = { .out_dim = channels, .in_dim = channels, .seq_len = spatial };
    g.dispatch(&k.linear_proj,
        { &attn.q_weight, &attn.q_bias, &acts.buf_c, &acts.attn_q },
        &q_pc, spatial * channels);
    g.dispatch(&k.linear_proj,
        { &attn.k_weight, &attn.k_bias, &acts.buf_c, &acts.attn_k },
        &q_pc, spatial * channels);
    g.dispatch(&k.linear_proj,
        { &attn.v_weight, &attn.v_bias, &acts.buf_c, &acts.attn_v },
        &q_pc, spatial * channels);

    // Step 7: Self-attention (Q/K/V in [S, C] layout, output in [S, C])
    VaeAttentionPC va_pc = {
        .head_dim = head_dim,
        .n_heads = n_heads,
        .seq_len = spatial,
        .scale = scale,
        .channels = channels,
    };
    g.dispatch(&k.vae_attention,
        { &acts.attn_q, &acts.attn_k, &acts.attn_v, &acts.attn_out },
        &va_pc, n_heads * spatial);

    // Step 8: proj_out projection: [S, C] -> [S, C]
    g.dispatch(&k.linear_proj,
        { &attn.proj_out_weight, &attn.proj_out_bias, &acts.attn_out, &acts.buf_c },
        &q_pc, spatial * channels);

    // Step 9: Transpose [S, C] -> [C, S]
    TransposeChannelSpatialPC tcs_rev = {
        .channels = channels,
        .spatial = spatial,
        .direction = 1,
    };
    g.dispatch(&k.transpose_channel_spatial,
        { &acts.buf_c, &acts.buf_a },
        &tcs_rev, llm::ceil_div(channels * spatial, 256));

    // Step 10: Residual add: buf_a += attn_temp (original input, both [C,S])
    llm::ResidualPC res_pc = { .n = channels * spatial };
    g.dispatch(&k.shared.residual_add,
        { &acts.buf_a, &acts.attn_temp },
        &res_pc, llm::ceil_div(channels * spatial, 256));
}



// --- Flux VAE Decoder Forward Pass ---
// Input: latent [16, h, w] in acts.buf_a
// Output: image [3, 8h, 8w] in acts.buf_a

fn void? FluxVaeDecoder.forward(&self, FluxVaeActivations* acts, uint lat_h, uint lat_w) {
    VaeKernels* k = self.kernels;

    uint h = lat_h;
    uint w = lat_w;

    llm::RenderGraph g;
    g.init(self.ctx)!!;

    // conv_in: [16, h, w] -> [512, h, w]
    flux_conv_block(&g, k, &self.conv_in, &acts.buf_a, &acts.buf_b, h, w, 1, 1);
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)512 * h * w * 4);

    // Mid block
    flux_resblock_block(&g, k, &self.mid_block1, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    flux_attn_block(&g, k, &self.mid_attn, acts, h, w);
    flux_resblock_block(&g, k, &self.mid_block2, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);

    g.checkpoint()!!;

    // up.3: 3 resblocks at 512 + upsample (h -> 2h)
    for (uint i = 0; i < 3; i++) {
        flux_resblock_block(&g, k, &self.up3_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.checkpoint()!!;
    }
    // Upsample 2x
    pipelines::UpsamplePC up_pc = { .channels = 512, .in_h = h, .in_w = w };
    g.dispatch(&k.upsample_nearest, { &acts.buf_a, &acts.buf_b }, &up_pc,
        llm::ceil_div(512 * h * w * 4, 256));
    h *= 2; w *= 2;
    flux_conv_block(&g, k, &self.up3_upsample, &acts.buf_b, &acts.buf_a, h, w, 1, 1);

    g.checkpoint()!!;

    // up.2: 3 resblocks at 512 + upsample
    for (uint i = 0; i < 3; i++) {
        flux_resblock_block(&g, k, &self.up2_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.checkpoint()!!;
    }
    up_pc = { .channels = 512, .in_h = h, .in_w = w };
    g.dispatch(&k.upsample_nearest, { &acts.buf_a, &acts.buf_b }, &up_pc,
        llm::ceil_div(512 * h * w * 4, 256));
    h *= 2; w *= 2;
    flux_conv_block(&g, k, &self.up2_upsample, &acts.buf_b, &acts.buf_a, h, w, 1, 1);

    g.checkpoint()!!;

    // up.1: 3 resblocks 512->256 + upsample
    for (uint i = 0; i < 3; i++) {
        flux_resblock_block(&g, k, &self.up1_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.checkpoint()!!;
    }
    up_pc = { .channels = 256, .in_h = h, .in_w = w };
    g.dispatch(&k.upsample_nearest, { &acts.buf_a, &acts.buf_b }, &up_pc,
        llm::ceil_div(256 * h * w * 4, 256));
    h *= 2; w *= 2;
    flux_conv_block(&g, k, &self.up1_upsample, &acts.buf_b, &acts.buf_a, h, w, 1, 1);

    g.checkpoint()!!;

    // up.0: 3 resblocks 256->128 (no upsample)
    for (uint i = 0; i < 3; i++) {
        flux_resblock_block(&g, k, &self.up0_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.checkpoint()!!;
    }

    // Final: GroupNorm -> SiLU -> conv_out
    flux_groupnorm_block(&g, k, &self.norm_out_weight, &self.norm_out_bias,
        &acts.buf_a, &acts.buf_b, 128, h * w);

    llm::SiluPC silu_pc = { .n = 128 * h * w };
    g.dispatch(&k.shared.silu, { &acts.buf_b }, &silu_pc, llm::ceil_div(128 * h * w, 256));

    flux_conv_block(&g, k, &self.conv_out, &acts.buf_b, &acts.buf_a, h, w, 1, 1);

    // Denormalize from [-1,1] to [0,1]: output = clamp(input * 0.5 + 0.5, 0, 1)
    pipelines::ScaleShiftPC ss_pc = { .n = 3 * h * w, .scale = 0.5f, .shift = 0.5f };
    g.dispatch(&k.scale_shift_clamp, { &acts.buf_a, &acts.buf_b }, &ss_pc,
        llm::ceil_div(3 * h * w, 256));

    // Copy result back to buf_a
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)3 * h * w * 4);

    g.flush()!!;
    // Image [3, h, w] is in buf_a
}

// --- Free ---

fn void FluxVaeConv.free(&self) {
    if (self.weight.size_bytes > 0) self.weight.free();
    if (self.bias.size_bytes > 0) self.bias.free();
}

fn void FluxVaeResBlock.free(&self) {
    if (self.norm1_weight.size_bytes > 0) self.norm1_weight.free();
    if (self.norm1_bias.size_bytes > 0) self.norm1_bias.free();
    if (self.conv1_weight.size_bytes > 0) self.conv1_weight.free();
    if (self.conv1_bias.size_bytes > 0) self.conv1_bias.free();
    if (self.norm2_weight.size_bytes > 0) self.norm2_weight.free();
    if (self.norm2_bias.size_bytes > 0) self.norm2_bias.free();
    if (self.conv2_weight.size_bytes > 0) self.conv2_weight.free();
    if (self.conv2_bias.size_bytes > 0) self.conv2_bias.free();
    if (self.nin_shortcut_weight.size_bytes > 0) self.nin_shortcut_weight.free();
    if (self.nin_shortcut_bias.size_bytes > 0) self.nin_shortcut_bias.free();
}

fn void FluxVaeAttn.free(&self) {
    if (self.norm_weight.size_bytes > 0) self.norm_weight.free();
    if (self.norm_bias.size_bytes > 0) self.norm_bias.free();
    if (self.q_weight.size_bytes > 0) self.q_weight.free();
    if (self.q_bias.size_bytes > 0) self.q_bias.free();
    if (self.k_weight.size_bytes > 0) self.k_weight.free();
    if (self.k_bias.size_bytes > 0) self.k_bias.free();
    if (self.v_weight.size_bytes > 0) self.v_weight.free();
    if (self.v_bias.size_bytes > 0) self.v_bias.free();
    if (self.proj_out_weight.size_bytes > 0) self.proj_out_weight.free();
    if (self.proj_out_bias.size_bytes > 0) self.proj_out_bias.free();
}

fn void FluxVaeDecoder.free(&self) {
    self.conv_in.free();
    self.mid_block1.free();
    self.mid_attn.free();
    self.mid_block2.free();
    for (uint i = 0; i < 3; i++) {
        self.up3_blocks[i].free();
        self.up2_blocks[i].free();
        self.up1_blocks[i].free();
        self.up0_blocks[i].free();
    }
    self.up3_upsample.free();
    self.up2_upsample.free();
    self.up1_upsample.free();
    if (self.norm_out_weight.size_bytes > 0) self.norm_out_weight.free();
    if (self.norm_out_bias.size_bytes > 0) self.norm_out_bias.free();
    self.conv_out.free();
}

fn void FluxVaeActivations.free(&self) {
    self.buf_a.free();
    self.buf_b.free();
    self.buf_c.free();
    self.attn_q.free();
    self.attn_k.free();
    self.attn_v.free();
    self.attn_out.free();
    self.attn_temp.free();
}
