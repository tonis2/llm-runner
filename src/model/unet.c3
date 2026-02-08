module llm;

import vk;
import std::io;
import std::math;
import std::core::mem;

// UNet for Stable Diffusion (SDXS distilled variant)
// Simplified structure: 1 ResBlock per level, no middle block
// Uses spatial transformers with cross-attention at 640 and 1280 channel levels

// --- UNet Weight Structures ---

// ResBlock: GroupNorm -> SiLU -> Conv3x3 -> time_emb -> GroupNorm -> SiLU -> Conv3x3
struct UnetResBlock {
    // in_layers: [0]=GroupNorm weight, [2]=Conv3x3
    Tensor in_norm_weight;   // GroupNorm weight (in_channels)
    Tensor in_norm_bias;     // GroupNorm bias
    Tensor in_conv_weight;   // Conv3x3 weight [kH, kW, in_c, out_c]
    Tensor in_conv_bias;     // Conv3x3 bias
    // emb_layers: [1]=Linear (time_emb_dim -> out_channels)
    Tensor emb_weight;       // Linear weight [1280, out_c]
    Tensor emb_bias;         // Linear bias
    // out_layers: [0]=GroupNorm weight, [3]=Conv3x3
    Tensor out_norm_weight;  // GroupNorm weight (out_channels)
    Tensor out_norm_bias;
    Tensor out_conv_weight;  // Conv3x3 weight
    Tensor out_conv_bias;
    // Skip connection (only if in_c != out_c)
    Tensor skip_weight;      // 1x1 conv weight [1, 1, in_c, out_c]
    Tensor skip_bias;
    bool has_skip;
    uint in_c;
    uint out_c;
}

// Spatial Transformer block: self-attention + cross-attention + FFN
struct SpatialTransformerWeights {
    // GroupNorm before proj_in
    Tensor norm_weight;
    Tensor norm_bias;
    // proj_in: 1x1 conv
    Tensor proj_in_weight;
    Tensor proj_in_bias;
    // Self-attention (attn1)
    Tensor attn1_q;
    Tensor attn1_k;
    Tensor attn1_v;
    Tensor attn1_out_weight;
    Tensor attn1_out_bias;
    Tensor attn1_norm_weight;
    Tensor attn1_norm_bias;
    // Cross-attention (attn2)
    Tensor attn2_q;
    Tensor attn2_k;   // context_dim -> channels
    Tensor attn2_v;   // context_dim -> channels
    Tensor attn2_out_weight;
    Tensor attn2_out_bias;
    Tensor attn2_norm_weight;
    Tensor attn2_norm_bias;
    // FFN
    Tensor ff_net0_weight;  // GEGLU: [channels, inner_dim*2]
    Tensor ff_net0_bias;
    Tensor ff_net2_weight;  // [inner_dim, channels]
    Tensor ff_net2_bias;
    Tensor ff_norm_weight;  // norm3
    Tensor ff_norm_bias;
    // proj_out: 1x1 conv
    Tensor proj_out_weight;
    Tensor proj_out_bias;
    uint channels;
}

struct UnetWeights {
    // Time embedding MLP: Linear(320, 1280) -> SiLU -> Linear(1280, 1280)
    Tensor time_emb_0_weight;
    Tensor time_emb_0_bias;
    Tensor time_emb_2_weight;
    Tensor time_emb_2_bias;

    // conv_in: input_blocks.0.0
    Tensor conv_in_weight;
    Tensor conv_in_bias;

    // Input blocks (encoder path)
    UnetResBlock input_rb1;        // input_blocks.1.0 (320->320)
    UnetResBlock input_rb4;        // input_blocks.4.0 (320->640)
    SpatialTransformerWeights input_st4;  // input_blocks.4.1
    UnetResBlock input_rb7;        // input_blocks.7.0 (640->1280)
    SpatialTransformerWeights input_st7;  // input_blocks.7.1

    // Downsample convs
    Tensor down3_weight;  // input_blocks.3.0.op (320->320, stride 2)
    Tensor down3_bias;
    Tensor down6_weight;  // input_blocks.6.0.op (640->640, stride 2)
    Tensor down6_bias;

    // Middle attentions (up_blocks.0.attentions)
    SpatialTransformerWeights mid_attn0;  // up_blocks.0.attentions.0
    SpatialTransformerWeights mid_attn1;  // up_blocks.0.attentions.1

    // Output blocks (decoder path with skip connections)
    UnetResBlock output_rb0;  // output_blocks.0.0 (2560->1280)
    UnetResBlock output_rb1;  // output_blocks.1.0 (1920->1280)
    UnetResBlock output_rb3;  // output_blocks.3.0 (1920->640)
    SpatialTransformerWeights output_st3;  // output_blocks.3.1
    UnetResBlock output_rb4;  // output_blocks.4.0 (960->640)
    SpatialTransformerWeights output_st4;  // output_blocks.4.1
    UnetResBlock output_rb6;  // output_blocks.6.0 (960->320)
    UnetResBlock output_rb7;  // output_blocks.7.0 (640->320)

    // Upsample convs
    Tensor up2_weight;  // output_blocks.2.1.conv (1280->1280)
    Tensor up2_bias;
    Tensor up5_weight;  // output_blocks.5.2.conv (640->640)
    Tensor up5_bias;

    // Output: GroupNorm -> SiLU -> Conv3x3
    Tensor out_norm_weight;  // out.0
    Tensor out_norm_bias;
    Tensor out_conv_weight;  // out.2
    Tensor out_conv_bias;
}

// --- UNet Activations ---

struct UnetActivations {
    // Working buffers for spatial features
    Tensor buf_a;       // Primary spatial buffer (max: 1280 * 16 * 16)
    Tensor buf_b;       // Secondary
    Tensor buf_c;       // Tertiary (for residual, skip)

    // Time embedding: [1280]
    Tensor time_emb;

    // Skip connection storage (saved during encoder for decoder)
    // skip0: after conv_in [320, 64, 64]
    // skip1: after input_rb1 [320, 64, 64]
    // skip3: after downsample [320, 32, 32] (before rb4)
    // skip4: after input_rb4+st4 [640, 32, 32]
    // skip6: after downsample [640, 16, 16] (before rb7)
    // skip7: after input_rb7+st7 [1280, 16, 16]
    Tensor[6] skips;

    // Attention buffers
    Tensor attn_q;
    Tensor attn_k;
    Tensor attn_v;
    Tensor attn_scores;
    Tensor attn_out;
    Tensor ffn_buf;  // For GEGLU intermediate
}

struct UnetModel {
    UnetConfig config;
    UnetWeights weights;
    UnetActivations acts;
    DiffusionKernels* kernels;
    DeviceContext* ctx;
}

// --- Tensor name helpers ---

fn String unet_name(char[256]* buf, String prefix, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) p[pos + i] = prefix[i];
    pos += prefix.len;
    for (usz i = 0; i < suffix.len; i++) p[pos + i] = suffix[i];
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

// --- Load helpers ---

fn UnetResBlock load_unet_resblock(DeviceContext* ctx, GGUFFile* gf, String base, uint in_c, uint out_c) {
    char[256] buf;
    bool has_skip = in_c != out_c;
    UnetResBlock rb = {
        .in_norm_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.0.weight")),
        .in_norm_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.0.bias")),
        .in_conv_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.2.weight")),
        .in_conv_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.2.bias")),
        .emb_weight     = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "emb_layers.1.weight")),
        .emb_bias       = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "emb_layers.1.bias")),
        .out_norm_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.0.weight")),
        .out_norm_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.0.bias")),
        .out_conv_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.3.weight")),
        .out_conv_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.3.bias")),
        .has_skip = has_skip,
        .in_c = in_c,
        .out_c = out_c,
    };

    if (has_skip) {
        rb.skip_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "skip_connection.weight"));
        rb.skip_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "skip_connection.bias"));
    }

    return rb;
}

fn SpatialTransformerWeights load_spatial_transformer(DeviceContext* ctx, GGUFFile* gf, String base, uint channels) {
    char[256] buf;
    return {
        .norm_weight     = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "norm.weight")),
        .norm_bias       = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "norm.bias")),
        .proj_in_weight  = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_in.weight")),
        .proj_in_bias    = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_in.bias")),
        .attn1_q         = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_q.weight")),
        .attn1_k         = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_k.weight")),
        .attn1_v         = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_v.weight")),
        .attn1_out_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_out.0.weight")),
        .attn1_out_bias  = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_out.0.bias")),
        .attn1_norm_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm1.weight")),
        .attn1_norm_bias = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm1.bias")),
        .attn2_q         = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_q.weight")),
        .attn2_k         = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_k.weight")),
        .attn2_v         = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_v.weight")),
        .attn2_out_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_out.0.weight")),
        .attn2_out_bias  = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_out.0.bias")),
        .attn2_norm_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm2.weight")),
        .attn2_norm_bias = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm2.bias")),
        .ff_net0_weight  = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.0.proj.weight")),
        .ff_net0_bias    = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.0.proj.bias")),
        .ff_net2_weight  = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.2.weight")),
        .ff_net2_bias    = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.2.bias")),
        .ff_norm_weight  = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm3.weight")),
        .ff_norm_bias    = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm3.bias")),
        .proj_out_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_out.weight")),
        .proj_out_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_out.bias")),
        .channels = channels,
    };
}

// --- Load UNet ---

fn UnetWeights? load_unet_weights(DeviceContext* ctx, GGUFFile* gf, UnetConfig* config) {
    io::printfn("\nLoading UNet weights...");
    String pfx = config.prefix;
    char[256] buf;

    UnetWeights w;

    // Time embedding MLP
    w.time_emb_0_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.0.weight"));
    w.time_emb_0_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.0.bias"));
    w.time_emb_2_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.2.weight"));
    w.time_emb_2_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.2.bias"));
    io::printfn("  Time embedding loaded");

    // conv_in
    w.conv_in_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.0.0.weight"));
    w.conv_in_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.0.0.bias"));

    // Input ResBlocks
    String ib1 = unet_name(&buf, pfx, "input_blocks.1.0.");
    w.input_rb1 = load_unet_resblock(ctx, gf, ib1, 320, 320);
    io::printfn("  Input block 1 loaded (320->320)");

    // Downsample 3
    w.down3_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.3.0.op.weight"));
    w.down3_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.3.0.op.bias"));

    String ib4 = unet_name(&buf, pfx, "input_blocks.4.0.");
    w.input_rb4 = load_unet_resblock(ctx, gf, ib4, 320, 640);
    String is4 = unet_name(&buf, pfx, "input_blocks.4.1.");
    w.input_st4 = load_spatial_transformer(ctx, gf, is4, 640);
    io::printfn("  Input block 4 loaded (320->640 + SpatialTransformer)");

    // Downsample 6
    w.down6_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.6.0.op.weight"));
    w.down6_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.6.0.op.bias"));

    String ib7 = unet_name(&buf, pfx, "input_blocks.7.0.");
    w.input_rb7 = load_unet_resblock(ctx, gf, ib7, 640, 1280);
    String is7 = unet_name(&buf, pfx, "input_blocks.7.1.");
    w.input_st7 = load_spatial_transformer(ctx, gf, is7, 1280);
    io::printfn("  Input block 7 loaded (640->1280 + SpatialTransformer)");

    // Middle attentions (up_blocks.0.attentions.{0,1})
    String ma0 = unet_name(&buf, pfx, "up_blocks.0.attentions.0.");
    w.mid_attn0 = load_spatial_transformer(ctx, gf, ma0, 1280);
    String ma1 = unet_name(&buf, pfx, "up_blocks.0.attentions.1.");
    w.mid_attn1 = load_spatial_transformer(ctx, gf, ma1, 1280);
    io::printfn("  Middle attentions loaded (2x SpatialTransformer 1280ch)");

    // Output blocks
    String ob0 = unet_name(&buf, pfx, "output_blocks.0.0.");
    w.output_rb0 = load_unet_resblock(ctx, gf, ob0, 2560, 1280);
    String ob1 = unet_name(&buf, pfx, "output_blocks.1.0.");
    w.output_rb1 = load_unet_resblock(ctx, gf, ob1, 1920, 1280);
    io::printfn("  Output blocks 0-1 loaded");

    w.up2_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.2.1.conv.weight"));
    w.up2_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.2.1.conv.bias"));

    String ob3 = unet_name(&buf, pfx, "output_blocks.3.0.");
    w.output_rb3 = load_unet_resblock(ctx, gf, ob3, 1920, 640);
    String os3 = unet_name(&buf, pfx, "output_blocks.3.1.");
    w.output_st3 = load_spatial_transformer(ctx, gf, os3, 640);
    String ob4 = unet_name(&buf, pfx, "output_blocks.4.0.");
    w.output_rb4 = load_unet_resblock(ctx, gf, ob4, 960, 640);
    String os4 = unet_name(&buf, pfx, "output_blocks.4.1.");
    w.output_st4 = load_spatial_transformer(ctx, gf, os4, 640);
    io::printfn("  Output blocks 3-4 loaded (with SpatialTransformers)");

    w.up5_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.5.2.conv.weight"));
    w.up5_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.5.2.conv.bias"));

    String ob6 = unet_name(&buf, pfx, "output_blocks.6.0.");
    w.output_rb6 = load_unet_resblock(ctx, gf, ob6, 960, 320);
    String ob7 = unet_name(&buf, pfx, "output_blocks.7.0.");
    w.output_rb7 = load_unet_resblock(ctx, gf, ob7, 640, 320);
    io::printfn("  Output blocks 6-7 loaded");

    // Final output
    w.out_norm_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.0.weight"));
    w.out_norm_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.0.bias"));
    w.out_conv_weight = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.2.weight"));
    w.out_conv_bias   = load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.2.bias"));
    io::printfn("  Output head loaded");

    return w;
}

// --- Allocate UNet activations ---

fn UnetActivations? allocate_unet_activations(DeviceContext* ctx, UnetConfig* config, uint latent_h, uint latent_w) {
    // Max spatial buffer: 1280 channels at 16x16, but also need 320 at 64x64
    // 320 * 64 * 64 = 1,310,720 floats
    // 1280 * 16 * 16 = 327,680 floats
    // 2560 * 16 * 16 = 655,360 floats (for concat)
    // So max is 320 * 64 * 64 = 1,310,720 but we need bigger for concat
    // Actually: 2560 * 16 * 16 = 655,360 < 320 * 64 * 64 = 1,310,720
    ulong max_spatial = (ulong)320 * latent_h * latent_w;
    // But we also need to hold 640*32*32 = 655360 and 1280*16*16 = 327680
    // And concat buffers: 2560*16*16 = 655360
    // So 320*64*64 = 1310720 is the largest single-level activation
    ulong[4] spatial_shape = { max_spatial, 0, 0, 0 };

    // Time embedding
    ulong[4] emb_shape = { 1280, 0, 0, 0 };

    // Skip connections
    Tensor[6] skips;
    // skip0: conv_in output [320, 64, 64]
    ulong[4] s0 = { (ulong)320 * latent_h * latent_w, 0, 0, 0 };
    skips[0] = create_f32_tensor(ctx, s0, 1)!!;
    // skip1: after rb1 [320, 64, 64]
    skips[1] = create_f32_tensor(ctx, s0, 1)!!;
    // skip2: after downsample [320, 32, 32] (actually this is input to rb4)
    ulong half_h = latent_h / 2;
    ulong half_w = latent_w / 2;
    ulong[4] s2 = { (ulong)320 * half_h * half_w, 0, 0, 0 };
    skips[2] = create_f32_tensor(ctx, s2, 1)!!;
    // skip3: after rb4+st4 [640, 32, 32]
    ulong[4] s3 = { (ulong)640 * half_h * half_w, 0, 0, 0 };
    skips[3] = create_f32_tensor(ctx, s3, 1)!!;
    // skip4: after downsample [640, 16, 16]
    ulong qh = half_h / 2;
    ulong qw = half_w / 2;
    ulong[4] s4 = { (ulong)640 * qh * qw, 0, 0, 0 };
    skips[4] = create_f32_tensor(ctx, s4, 1)!!;
    // skip5: after rb7+st7 [1280, 16, 16]
    ulong[4] s5 = { (ulong)1280 * qh * qw, 0, 0, 0 };
    skips[5] = create_f32_tensor(ctx, s5, 1)!!;

    // Attention buffers: max seq_len = 64*64 = 4096, max channels = 1280, n_heads = 8
    uint n_heads = config.n_heads;
    ulong max_seq = (ulong)latent_h * latent_w;  // 64*64 = 4096
    ulong max_ch = 1280;
    ulong[4] attn_shape = { max_seq * max_ch, 0, 0, 0 };
    ulong[4] scores_shape = { (ulong)n_heads * max_seq * max_seq, 0, 0, 0 };
    // GEGLU intermediate: channels * 4 * 2 (doubled for gate)
    ulong[4] ffn_shape = { max_seq * max_ch * 4, 0, 0, 0 };

    return {
        .buf_a = create_f32_tensor(ctx, spatial_shape, 1)!!,
        .buf_b = create_f32_tensor(ctx, spatial_shape, 1)!!,
        .buf_c = create_f32_tensor(ctx, spatial_shape, 1)!!,
        .time_emb = create_f32_tensor(ctx, emb_shape, 1)!!,
        .skips = skips,
        .attn_q = create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_k = create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_v = create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_scores = create_f32_tensor(ctx, scores_shape, 1)!!,
        .attn_out = create_f32_tensor(ctx, attn_shape, 1)!!,
        .ffn_buf = create_f32_tensor(ctx, ffn_shape, 1)!!,
    };
}

// --- ResBlock dispatch ---

fn void dispatch_unet_resblock(
    CommandBuffer cmd,
    DiffusionKernels* k,
    UnetResBlock* rb,
    Tensor* input,
    Tensor* output,
    Tensor* temp,
    Tensor* time_emb,
    uint h, uint w
) {
    uint in_c = rb.in_c;
    uint out_c = rb.out_c;
    uint spatial = h * w;

    // 1. GroupNorm(in) -> temp
    GroupNormPC gn_pc = { .channels = in_c, .spatial = spatial, .num_groups = 32, .eps = 1e-5f };
    dispatch_kernel(cmd, &k.group_norm,
        { input.gpu_buffer.buffer, rb.in_norm_weight.gpu_buffer.buffer,
          rb.in_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { input.size_bytes, rb.in_norm_weight.size_bytes,
          rb.in_norm_bias.size_bytes, temp.size_bytes },
        &gn_pc, ceil_div(spatial, 256), in_c / 32);
    compute_barrier(cmd);

    // 2. SiLU(temp) in-place
    SiluPC silu_pc = { .n = in_c * spatial };
    dispatch_kernel(cmd, &k.shared.silu,
        { temp.gpu_buffer.buffer },
        { temp.size_bytes },
        &silu_pc, ceil_div(in_c * spatial, 256));
    compute_barrier(cmd);

    // 3. Conv3x3(temp) -> output
    Conv2dPC conv_pc = {
        .in_c = in_c, .out_c = out_c, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &rb.in_conv_weight, &rb.in_conv_bias, temp, output, &conv_pc);
    compute_barrier(cmd);

    // 4. Time embedding projection: emb = SiLU(time_emb) then Linear -> [out_c]
    // SiLU already applied to time_emb during embedding creation
    // Linear: time_emb[1280] -> [out_c] via emb_weight
    BatchedMatMulPC emb_pc = { .m_dim = 1, .n_dim = out_c, .k_dim = 1280 };
    dispatch_batched_matmul_auto(cmd, k, &rb.emb_weight, time_emb, temp, &emb_pc);
    compute_barrier(cmd);

    // Add emb bias
    ResidualPC rbias = { .n = out_c };
    dispatch_kernel(cmd, &k.shared.residual_add,
        { temp.gpu_buffer.buffer, rb.emb_bias.gpu_buffer.buffer },
        { temp.size_bytes, rb.emb_bias.size_bytes },
        &rbias, ceil_div(out_c, 256));
    compute_barrier(cmd);

    // 5. Broadcast add: output[c, h, w] += emb[c]
    BroadcastAddPC ba_pc = { .channels = out_c, .spatial = spatial };
    dispatch_kernel(cmd, &k.broadcast_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &ba_pc, ceil_div(out_c * spatial, 256));
    compute_barrier(cmd);

    // 6. GroupNorm(output) -> temp
    gn_pc = { .channels = out_c, .spatial = spatial, .num_groups = 32, .eps = 1e-5f };
    dispatch_kernel(cmd, &k.group_norm,
        { output.gpu_buffer.buffer, rb.out_norm_weight.gpu_buffer.buffer,
          rb.out_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, rb.out_norm_weight.size_bytes,
          rb.out_norm_bias.size_bytes, temp.size_bytes },
        &gn_pc, ceil_div(spatial, 256), out_c / 32);
    compute_barrier(cmd);

    // 7. SiLU(temp) in-place
    silu_pc = { .n = out_c * spatial };
    dispatch_kernel(cmd, &k.shared.silu,
        { temp.gpu_buffer.buffer },
        { temp.size_bytes },
        &silu_pc, ceil_div(out_c * spatial, 256));
    compute_barrier(cmd);

    // 8. Conv3x3(temp) -> output
    conv_pc = {
        .in_c = out_c, .out_c = out_c, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &rb.out_conv_weight, &rb.out_conv_bias, temp, output, &conv_pc);
    compute_barrier(cmd);

    // 9. Skip connection
    if (rb.has_skip) {
        // 1x1 conv on input -> temp
        Conv2dPC skip_pc = {
            .in_c = in_c, .out_c = out_c, .in_h = h, .in_w = w,
            .kH = 1, .kW = 1, .stride = 1, .pad = 0,
            .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
        };
        dispatch_conv2d(cmd, k, &rb.skip_weight, &rb.skip_bias, input, temp, &skip_pc);
        compute_barrier(cmd);
    } else {
        // Copy input to temp for residual add
        vk::cmdCopyBuffer(cmd, input.gpu_buffer.buffer, temp.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)in_c * spatial * 4 }});
        compute_barrier(cmd);
    }

    // 10. output += temp (residual)
    ResidualPC res_pc = { .n = out_c * spatial };
    dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, ceil_div(out_c * spatial, 256));
    compute_barrier(cmd);
}

// --- Spatial Transformer dispatch ---

fn void dispatch_spatial_transformer(
    CommandBuffer cmd,
    DiffusionKernels* k,
    SpatialTransformerWeights* st,
    Tensor* input,       // [channels, h, w] — spatial features
    Tensor* output,      // same shape as input
    Tensor* temp,        // working buffer
    Tensor* context,     // [77, 768] — CLIP text conditioning
    UnetActivations* acts,
    uint h, uint w,
    uint n_heads,
    uint context_dim
) {
    uint ch = st.channels;
    uint spatial = h * w;
    uint head_dim = ch / n_heads;

    // 1. GroupNorm -> temp
    GroupNormPC gn_pc = { .channels = ch, .spatial = spatial, .num_groups = 32, .eps = 1e-6f };
    dispatch_kernel(cmd, &k.group_norm,
        { input.gpu_buffer.buffer, st.norm_weight.gpu_buffer.buffer,
          st.norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { input.size_bytes, st.norm_weight.size_bytes,
          st.norm_bias.size_bytes, temp.size_bytes },
        &gn_pc, ceil_div(spatial, 256), ch / 32);
    compute_barrier(cmd);

    // 2. proj_in: 1x1 conv temp -> output (reshape to sequence)
    Conv2dPC proj_pc = {
        .in_c = ch, .out_c = ch, .in_h = h, .in_w = w,
        .kH = 1, .kW = 1, .stride = 1, .pad = 0,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &st.proj_in_weight, &st.proj_in_bias, temp, output, &proj_pc);
    compute_barrier(cmd);
    // output is now [ch, spatial] which we treat as [spatial, ch] for attention

    // 3. Self-attention (attn1)
    // LayerNorm on output -> temp (per-position, 'spatial' positions of 'ch' dims)
    // We use layernorm per position
    for (uint pos = 0; pos < spatial; pos++) {
        LayerNormPC ln = { .dim = ch, .eps = 1e-5f };
        dispatch_kernel(cmd, &k.shared.layernorm,
            { output.gpu_buffer.buffer, st.attn1_norm_weight.gpu_buffer.buffer,
              st.attn1_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
            { output.size_bytes, st.attn1_norm_weight.size_bytes,
              st.attn1_norm_bias.size_bytes, temp.size_bytes },
            &ln, 1);
    }
    compute_barrier(cmd);

    // Q/K/V projections: batched matmul [spatial, ch] x [ch, ch]^T -> [spatial, ch]
    BatchedMatMulPC qkv_pc = { .m_dim = spatial, .n_dim = ch, .k_dim = ch };
    dispatch_batched_matmul_auto(cmd, k, &st.attn1_q, temp, &acts.attn_q, &qkv_pc);
    dispatch_batched_matmul_auto(cmd, k, &st.attn1_k, temp, &acts.attn_k, &qkv_pc);
    dispatch_batched_matmul_auto(cmd, k, &st.attn1_v, temp, &acts.attn_v, &qkv_pc);
    compute_barrier(cmd);

    // Self-attention: Q, K, V are [spatial, ch] -> reshape to [n_heads, spatial, head_dim]
    float scale = 1.0f / math::sqrt((float)head_dim);
    SpatialAttentionPC sa_pc = {
        .head_dim = head_dim,
        .n_heads = n_heads,
        .seq_len = spatial,
        .scale = scale,
    };
    dispatch_kernel(cmd, &k.spatial_attention,
        { acts.attn_q.gpu_buffer.buffer, acts.attn_k.gpu_buffer.buffer,
          acts.attn_v.gpu_buffer.buffer, acts.attn_scores.gpu_buffer.buffer,
          acts.attn_out.gpu_buffer.buffer },
        { acts.attn_q.size_bytes, acts.attn_k.size_bytes,
          acts.attn_v.size_bytes, acts.attn_scores.size_bytes,
          acts.attn_out.size_bytes },
        &sa_pc, n_heads * spatial);
    compute_barrier(cmd);

    // Output projection
    BatchedMatMulPC out_pc = { .m_dim = spatial, .n_dim = ch, .k_dim = ch };
    dispatch_batched_matmul_auto(cmd, k, &st.attn1_out_weight, &acts.attn_out, temp, &out_pc);
    compute_barrier(cmd);

    // Add bias
    BroadcastAddPC ba = { .channels = ch, .spatial = spatial };
    dispatch_kernel(cmd, &k.broadcast_add,
        { temp.gpu_buffer.buffer, st.attn1_out_bias.gpu_buffer.buffer },
        { temp.size_bytes, st.attn1_out_bias.size_bytes },
        &ba, ceil_div(ch * spatial, 256));
    compute_barrier(cmd);

    // Residual: output += attn1_output
    ResidualPC res_pc = { .n = ch * spatial };
    dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, ceil_div(ch * spatial, 256));
    compute_barrier(cmd);

    // 4. Cross-attention (attn2): Q from image, K/V from text context
    // LayerNorm on output -> temp
    for (uint pos = 0; pos < spatial; pos++) {
        LayerNormPC ln = { .dim = ch, .eps = 1e-5f };
        dispatch_kernel(cmd, &k.shared.layernorm,
            { output.gpu_buffer.buffer, st.attn2_norm_weight.gpu_buffer.buffer,
              st.attn2_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
            { output.size_bytes, st.attn2_norm_weight.size_bytes,
              st.attn2_norm_bias.size_bytes, temp.size_bytes },
            &ln, 1);
    }
    compute_barrier(cmd);

    // Q from image features: [spatial, ch] x [ch, ch]^T -> [spatial, ch]
    dispatch_batched_matmul_auto(cmd, k, &st.attn2_q, temp, &acts.attn_q, &qkv_pc);
    // K from context: [77, context_dim] x [context_dim, ch]^T -> [77, ch]
    BatchedMatMulPC kv_ctx_pc = { .m_dim = 77, .n_dim = ch, .k_dim = context_dim };
    dispatch_batched_matmul_auto(cmd, k, &st.attn2_k, context, &acts.attn_k, &kv_ctx_pc);
    dispatch_batched_matmul_auto(cmd, k, &st.attn2_v, context, &acts.attn_v, &kv_ctx_pc);
    compute_barrier(cmd);

    // Cross-attention: Q=[spatial, ch], K/V=[77, ch]
    CrossAttentionPC ca_pc = {
        .head_dim = head_dim,
        .n_heads = n_heads,
        .q_len = spatial,
        .kv_len = 77,
        .scale = scale,
    };
    dispatch_kernel(cmd, &k.cross_attention,
        { acts.attn_q.gpu_buffer.buffer, acts.attn_k.gpu_buffer.buffer,
          acts.attn_v.gpu_buffer.buffer, acts.attn_scores.gpu_buffer.buffer,
          acts.attn_out.gpu_buffer.buffer },
        { acts.attn_q.size_bytes, acts.attn_k.size_bytes,
          acts.attn_v.size_bytes, acts.attn_scores.size_bytes,
          acts.attn_out.size_bytes },
        &ca_pc, n_heads * spatial);
    compute_barrier(cmd);

    // Output projection
    dispatch_batched_matmul_auto(cmd, k, &st.attn2_out_weight, &acts.attn_out, temp, &out_pc);
    compute_barrier(cmd);
    dispatch_kernel(cmd, &k.broadcast_add,
        { temp.gpu_buffer.buffer, st.attn2_out_bias.gpu_buffer.buffer },
        { temp.size_bytes, st.attn2_out_bias.size_bytes },
        &ba, ceil_div(ch * spatial, 256));
    compute_barrier(cmd);

    // Residual
    dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, ceil_div(ch * spatial, 256));
    compute_barrier(cmd);

    // 5. FFN: LayerNorm -> GEGLU -> Linear -> residual
    // LayerNorm
    for (uint pos = 0; pos < spatial; pos++) {
        LayerNormPC ln = { .dim = ch, .eps = 1e-5f };
        dispatch_kernel(cmd, &k.shared.layernorm,
            { output.gpu_buffer.buffer, st.ff_norm_weight.gpu_buffer.buffer,
              st.ff_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
            { output.size_bytes, st.ff_norm_weight.size_bytes,
              st.ff_norm_bias.size_bytes, temp.size_bytes },
            &ln, 1);
    }
    compute_barrier(cmd);

    // GEGLU: Linear([ch] -> [inner_dim*2]), split, GELU(first_half) * second_half
    // inner_dim = ch * 4 (from ff.net.0.proj output is [ch, inner_dim*2])
    uint inner_dim = ch * 4;
    BatchedMatMulPC ff0_pc = { .m_dim = spatial, .n_dim = inner_dim * 2, .k_dim = ch };
    dispatch_batched_matmul_auto(cmd, k, &st.ff_net0_weight, temp, &acts.ffn_buf, &ff0_pc);
    compute_barrier(cmd);

    // Add bias to ffn_buf
    BroadcastAddPC ff_ba = { .channels = inner_dim * 2, .spatial = spatial };
    dispatch_kernel(cmd, &k.broadcast_add,
        { acts.ffn_buf.gpu_buffer.buffer, st.ff_net0_bias.gpu_buffer.buffer },
        { acts.ffn_buf.size_bytes, st.ff_net0_bias.size_bytes },
        &ff_ba, ceil_div(inner_dim * 2 * spatial, 256));
    compute_barrier(cmd);

    // GELU on first half, multiply with second half
    // For now: apply GELU to first inner_dim elements, then elemwise_mul with second half
    GeluPC gelu_pc = { .n = inner_dim * spatial };
    dispatch_kernel(cmd, &k.shared.gelu,
        { acts.ffn_buf.gpu_buffer.buffer },
        { acts.ffn_buf.size_bytes },
        &gelu_pc, ceil_div(inner_dim * spatial, 256));
    compute_barrier(cmd);

    // Multiply: ffn_buf[0..inner_dim*spatial-1] *= ffn_buf[inner_dim*spatial..]
    ElemwisePC ew_pc = { .n = inner_dim * spatial };
    // Use elemwise_mul treating second half as separate buffer (offset)
    // TODO: This needs an offset-aware elemwise or we copy
    // For now, we'll just use the first half (skip GEGLU gate for initial implementation)
    // This simplification will affect quality but allows testing the pipeline

    // Linear down: [inner_dim] -> [ch]
    BatchedMatMulPC ff2_pc = { .m_dim = spatial, .n_dim = ch, .k_dim = inner_dim };
    dispatch_batched_matmul_auto(cmd, k, &st.ff_net2_weight, &acts.ffn_buf, temp, &ff2_pc);
    compute_barrier(cmd);

    // Add bias
    dispatch_kernel(cmd, &k.broadcast_add,
        { temp.gpu_buffer.buffer, st.ff_net2_bias.gpu_buffer.buffer },
        { temp.size_bytes, st.ff_net2_bias.size_bytes },
        &ba, ceil_div(ch * spatial, 256));
    compute_barrier(cmd);

    // Residual
    dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, ceil_div(ch * spatial, 256));
    compute_barrier(cmd);

    // 6. proj_out: 1x1 conv output -> temp, then residual with original input
    dispatch_conv2d(cmd, k, &st.proj_out_weight, &st.proj_out_bias, output, temp, &proj_pc);
    compute_barrier(cmd);

    // Final residual: input + proj_out
    dispatch_kernel(cmd, &k.shared.residual_add,
        { input.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { input.size_bytes, temp.size_bytes },
        &res_pc, ceil_div(ch * spatial, 256));
    compute_barrier(cmd);

    // Copy result to output
    vk::cmdCopyBuffer(cmd, input.gpu_buffer.buffer, output.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)ch * spatial * 4 }});
    compute_barrier(cmd);
}

// --- UNet Forward Pass ---
// Input: noisy latent [4, 64, 64] in acts.buf_a
// Output: predicted noise [4, 64, 64] in acts.buf_a

fn void? UnetModel.forward(&self, Tensor* context, float timestep) {
    DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    DiffusionKernels* k = self.kernels;
    UnetWeights* w = &self.weights;
    UnetActivations* a = &self.acts;
    UnetConfig* cfg = &self.config;

    uint h = 64;  // latent height
    uint wd = 64;  // latent width (renamed to avoid shadowing)

    begin_compute(cmd)!!;

    // === 1. Timestep Embedding ===
    // Sinusoidal embedding -> MLP
    TimestepEmbedPC te_pc = { .dim = cfg.model_channels, .timestep = timestep, .max_period = 10000.0f };
    dispatch_kernel(cmd, &k.timestep_embed,
        { a.time_emb.gpu_buffer.buffer },
        { a.time_emb.size_bytes },
        &te_pc, ceil_div(cfg.model_channels, 256));
    compute_barrier(cmd);

    // MLP: Linear(320, 1280) -> SiLU -> Linear(1280, 1280)
    BatchedMatMulPC te_mlp0 = { .m_dim = 1, .n_dim = 1280, .k_dim = cfg.model_channels };
    dispatch_batched_matmul_auto(cmd, k, &w.time_emb_0_weight, &a.time_emb, &a.buf_c, &te_mlp0);
    compute_barrier(cmd);
    ResidualPC te_bias = { .n = 1280 };
    dispatch_kernel(cmd, &k.shared.residual_add,
        { a.buf_c.gpu_buffer.buffer, w.time_emb_0_bias.gpu_buffer.buffer },
        { a.buf_c.size_bytes, w.time_emb_0_bias.size_bytes },
        &te_bias, ceil_div(1280, 256));
    compute_barrier(cmd);

    SiluPC te_silu = { .n = 1280 };
    dispatch_kernel(cmd, &k.shared.silu,
        { a.buf_c.gpu_buffer.buffer },
        { a.buf_c.size_bytes },
        &te_silu, ceil_div(1280, 256));
    compute_barrier(cmd);

    BatchedMatMulPC te_mlp2 = { .m_dim = 1, .n_dim = 1280, .k_dim = 1280 };
    dispatch_batched_matmul_auto(cmd, k, &w.time_emb_2_weight, &a.buf_c, &a.time_emb, &te_mlp2);
    compute_barrier(cmd);
    dispatch_kernel(cmd, &k.shared.residual_add,
        { a.time_emb.gpu_buffer.buffer, w.time_emb_2_bias.gpu_buffer.buffer },
        { a.time_emb.size_bytes, w.time_emb_2_bias.size_bytes },
        &te_bias, ceil_div(1280, 256));
    compute_barrier(cmd);

    // === 2. Encoder (Input Blocks) ===

    // conv_in: [4, 64, 64] -> [320, 64, 64]
    Conv2dPC conv_in_pc = {
        .in_c = 4, .out_c = 320, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &w.conv_in_weight, &w.conv_in_bias, &a.buf_a, &a.buf_b, &conv_in_pc);
    compute_barrier(cmd);

    // Save skip0: conv_in output [320, 64, 64]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[0].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
    compute_barrier(cmd);

    // ResBlock 1: 320->320
    dispatch_unet_resblock(cmd, k, &w.input_rb1, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);
    // Result is in buf_b (resblock writes back to input)
    // Actually our resblock writes result to the input buffer via copy at the end
    // So result is in buf_b

    // Save skip1: [320, 64, 64]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[1].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
    compute_barrier(cmd);

    // Downsample 3: stride-2 conv [320, 64, 64] -> [320, 32, 32]
    Conv2dPC down3_pc = {
        .in_c = 320, .out_c = 320, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 2, .pad = 1,
        .groups = 1, .out_h = h / 2, .out_w = wd / 2, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &w.down3_weight, &w.down3_bias, &a.buf_b, &a.buf_a, &down3_pc);
    compute_barrier(cmd);
    h /= 2; wd /= 2;  // 32x32

    // Save skip2: [320, 32, 32]
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.skips[2].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
    compute_barrier(cmd);

    // ResBlock 4: 320->640 (with skip connection)
    // Copy buf_a to buf_b first (resblock takes input from first arg)
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.input_rb4, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // SpatialTransformer 4 (640ch)
    // input_rb4 result is in buf_b, copy to buf_a for spatial transformer
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
    compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.input_st4, &a.buf_a, &a.buf_b, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);
    // Result in buf_b

    // Save skip3: [640, 32, 32]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[3].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
    compute_barrier(cmd);

    // Downsample 6: stride-2 conv [640, 32, 32] -> [640, 16, 16]
    Conv2dPC down6_pc = {
        .in_c = 640, .out_c = 640, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 2, .pad = 1,
        .groups = 1, .out_h = h / 2, .out_w = wd / 2, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &w.down6_weight, &w.down6_bias, &a.buf_b, &a.buf_a, &down6_pc);
    compute_barrier(cmd);
    h /= 2; wd /= 2;  // 16x16

    // Save skip4: [640, 16, 16]
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.skips[4].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
    compute_barrier(cmd);

    // ResBlock 7: 640->1280 (with skip connection)
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.input_rb7, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // SpatialTransformer 7 (1280ch)
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)1280 * h * wd * 4 }});
    compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.input_st7, &a.buf_a, &a.buf_b, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);

    // Save skip5: [1280, 16, 16]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[5].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)1280 * h * wd * 4 }});
    compute_barrier(cmd);

    // === 3. Middle Block ===
    // Two spatial transformers at 1280ch, 16x16
    dispatch_spatial_transformer(cmd, k, &w.mid_attn0, &a.buf_b, &a.buf_a, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);
    // Result in buf_a
    dispatch_spatial_transformer(cmd, k, &w.mid_attn1, &a.buf_a, &a.buf_b, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);
    // Result in buf_b

    // === 4. Decoder (Output Blocks) ===

    // output_blocks.0: concat(buf_b[1280,16,16], skip5[1280,16,16]) -> ResBlock(2560->1280)
    ChannelConcatPC cc0 = { .channels_a = 1280, .channels_b = 1280, .spatial = h * wd };
    dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_b.gpu_buffer.buffer, a.skips[5].gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.skips[5].size_bytes, a.buf_a.size_bytes },
        &cc0, ceil_div(2560 * h * wd, 256));
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb0, &a.buf_a, &a.buf_b, &a.buf_c, &a.time_emb, h, wd);
    // Result in buf_a (resblock copies back)

    // output_blocks.1: concat(buf_a[1280,16,16], skip4[640,16,16]) -> ResBlock(1920->1280)
    ChannelConcatPC cc1 = { .channels_a = 1280, .channels_b = 640, .spatial = h * wd };
    dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_a.gpu_buffer.buffer, a.skips[4].gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.skips[4].size_bytes, a.buf_b.size_bytes },
        &cc1, ceil_div(1920 * h * wd, 256));
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb1, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // output_blocks.2: upsample 1280ch, 16x16 -> 32x32
    UpsamplePC up2 = { .channels = 1280, .in_h = h, .in_w = wd };
    dispatch_kernel(cmd, &k.upsample_nearest,
        { a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.buf_a.size_bytes },
        &up2, ceil_div(1280 * h * wd * 4, 256));
    compute_barrier(cmd);
    h *= 2; wd *= 2;  // 32x32

    // Conv after upsample
    Conv2dPC up2_conv = {
        .in_c = 1280, .out_c = 1280, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &w.up2_weight, &w.up2_bias, &a.buf_a, &a.buf_b, &up2_conv);
    compute_barrier(cmd);

    // output_blocks.3: concat(buf_b[1280,32,32], skip3[640,32,32]) -> ResBlock(1920->640) + ST
    ChannelConcatPC cc3 = { .channels_a = 1280, .channels_b = 640, .spatial = h * wd };
    dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_b.gpu_buffer.buffer, a.skips[3].gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.skips[3].size_bytes, a.buf_a.size_bytes },
        &cc3, ceil_div(1920 * h * wd, 256));
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb3, &a.buf_a, &a.buf_b, &a.buf_c, &a.time_emb, h, wd);

    // SpatialTransformer at 640ch
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
    compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.output_st3, &a.buf_b, &a.buf_a, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);

    // output_blocks.4: concat(buf_a[640,32,32], skip2[320,32,32]) -> ResBlock(960->640) + ST
    ChannelConcatPC cc4 = { .channels_a = 640, .channels_b = 320, .spatial = h * wd };
    dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_a.gpu_buffer.buffer, a.skips[2].gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.skips[2].size_bytes, a.buf_b.size_bytes },
        &cc4, ceil_div(960 * h * wd, 256));
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb4, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
    compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.output_st4, &a.buf_a, &a.buf_b, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);

    // output_blocks.5: upsample 640ch, 32x32 -> 64x64
    UpsamplePC up5 = { .channels = 640, .in_h = h, .in_w = wd };
    dispatch_kernel(cmd, &k.upsample_nearest,
        { a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.buf_a.size_bytes },
        &up5, ceil_div(640 * h * wd * 4, 256));
    compute_barrier(cmd);
    h *= 2; wd *= 2;  // 64x64

    Conv2dPC up5_conv = {
        .in_c = 640, .out_c = 640, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &w.up5_weight, &w.up5_bias, &a.buf_a, &a.buf_b, &up5_conv);
    compute_barrier(cmd);

    // output_blocks.6: concat(buf_b[640,64,64], skip1[320,64,64]) -> ResBlock(960->320)
    ChannelConcatPC cc6 = { .channels_a = 640, .channels_b = 320, .spatial = h * wd };
    dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_b.gpu_buffer.buffer, a.skips[1].gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.skips[1].size_bytes, a.buf_a.size_bytes },
        &cc6, ceil_div(960 * h * wd, 256));
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb6, &a.buf_a, &a.buf_b, &a.buf_c, &a.time_emb, h, wd);

    // output_blocks.7: concat(buf_a[320,64,64], skip0[320,64,64]) -> ResBlock(640->320)
    ChannelConcatPC cc7 = { .channels_a = 320, .channels_b = 320, .spatial = h * wd };
    dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_a.gpu_buffer.buffer, a.skips[0].gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.skips[0].size_bytes, a.buf_b.size_bytes },
        &cc7, ceil_div(640 * h * wd, 256));
    compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb7, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // === 5. Output Head ===
    // GroupNorm -> SiLU -> Conv3x3 (320 -> 4)
    uint spatial = h * wd;
    GroupNormPC out_gn = { .channels = 320, .spatial = spatial, .num_groups = 32, .eps = 1e-5f };
    dispatch_kernel(cmd, &k.group_norm,
        { a.buf_b.gpu_buffer.buffer, w.out_norm_weight.gpu_buffer.buffer,
          w.out_norm_bias.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, w.out_norm_weight.size_bytes,
          w.out_norm_bias.size_bytes, a.buf_a.size_bytes },
        &out_gn, ceil_div(spatial, 256), 320 / 32);
    compute_barrier(cmd);

    SiluPC out_silu = { .n = 320 * spatial };
    dispatch_kernel(cmd, &k.shared.silu,
        { a.buf_a.gpu_buffer.buffer },
        { a.buf_a.size_bytes },
        &out_silu, ceil_div(320 * spatial, 256));
    compute_barrier(cmd);

    Conv2dPC out_conv = {
        .in_c = 320, .out_c = 4, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
    dispatch_conv2d(cmd, k, &w.out_conv_weight, &w.out_conv_bias, &a.buf_a, &a.buf_b, &out_conv);
    compute_barrier(cmd);

    // Copy result to buf_a
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)4 * h * wd * 4 }});

    submit_and_wait(ctx)!!;
    // Result: predicted noise [4, 64, 64] in buf_a
}

// --- Load UNet Model ---

fn UnetModel? load_unet_model(DeviceContext* ctx, GGUFFile* gf, UnetConfig* config, DiffusionKernels* kernels) {
    UnetWeights weights = load_unet_weights(ctx, gf, config)!!;
    UnetActivations acts = allocate_unet_activations(ctx, config, 64, 64)!!;

    return {
        .config = *config,
        .weights = weights,
        .acts = acts,
        .kernels = kernels,
        .ctx = ctx,
    };
}

// --- Free ---

fn void UnetResBlock.free(&self) {
    self.in_norm_weight.free();
    self.in_norm_bias.free();
    self.in_conv_weight.free();
    self.in_conv_bias.free();
    self.emb_weight.free();
    self.emb_bias.free();
    self.out_norm_weight.free();
    self.out_norm_bias.free();
    self.out_conv_weight.free();
    self.out_conv_bias.free();
    if (self.has_skip) {
        self.skip_weight.free();
        self.skip_bias.free();
    }
}

fn void SpatialTransformerWeights.free(&self) {
    self.norm_weight.free();
    self.norm_bias.free();
    self.proj_in_weight.free();
    self.proj_in_bias.free();
    self.attn1_q.free(); self.attn1_k.free(); self.attn1_v.free();
    self.attn1_out_weight.free(); self.attn1_out_bias.free();
    self.attn1_norm_weight.free(); self.attn1_norm_bias.free();
    self.attn2_q.free(); self.attn2_k.free(); self.attn2_v.free();
    self.attn2_out_weight.free(); self.attn2_out_bias.free();
    self.attn2_norm_weight.free(); self.attn2_norm_bias.free();
    self.ff_net0_weight.free(); self.ff_net0_bias.free();
    self.ff_net2_weight.free(); self.ff_net2_bias.free();
    self.ff_norm_weight.free(); self.ff_norm_bias.free();
    self.proj_out_weight.free(); self.proj_out_bias.free();
}

fn void UnetWeights.free(&self) {
    self.time_emb_0_weight.free(); self.time_emb_0_bias.free();
    self.time_emb_2_weight.free(); self.time_emb_2_bias.free();
    self.conv_in_weight.free(); self.conv_in_bias.free();
    self.input_rb1.free();
    self.input_rb4.free(); self.input_st4.free();
    self.input_rb7.free(); self.input_st7.free();
    self.down3_weight.free(); self.down3_bias.free();
    self.down6_weight.free(); self.down6_bias.free();
    self.mid_attn0.free(); self.mid_attn1.free();
    self.output_rb0.free(); self.output_rb1.free();
    self.output_rb3.free(); self.output_st3.free();
    self.output_rb4.free(); self.output_st4.free();
    self.output_rb6.free(); self.output_rb7.free();
    self.up2_weight.free(); self.up2_bias.free();
    self.up5_weight.free(); self.up5_bias.free();
    self.out_norm_weight.free(); self.out_norm_bias.free();
    self.out_conv_weight.free(); self.out_conv_bias.free();
}

fn void UnetActivations.free(&self) {
    self.buf_a.free(); self.buf_b.free(); self.buf_c.free();
    self.time_emb.free();
    for (uint i = 0; i < 6; i++) self.skips[i].free();
    self.attn_q.free(); self.attn_k.free(); self.attn_v.free();
    self.attn_scores.free(); self.attn_out.free();
    self.ffn_buf.free();
}

fn void UnetModel.free(&self) {
    self.weights.free();
    self.acts.free();
}
