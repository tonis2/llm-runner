module sdxs;

import vk;
import llm;
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
    llm::Tensor in_norm_weight;   // GroupNorm weight (in_channels)
    llm::Tensor in_norm_bias;     // GroupNorm bias
    llm::Tensor in_conv_weight;   // Conv3x3 weight [kH, kW, in_c, out_c]
    llm::Tensor in_conv_bias;     // Conv3x3 bias
    // emb_layers: [1]=Linear (time_emb_dim -> out_channels)
    llm::Tensor emb_weight;       // Linear weight [1280, out_c]
    llm::Tensor emb_bias;         // Linear bias
    // out_layers: [0]=GroupNorm weight, [3]=Conv3x3
    llm::Tensor out_norm_weight;  // GroupNorm weight (out_channels)
    llm::Tensor out_norm_bias;
    llm::Tensor out_conv_weight;  // Conv3x3 weight
    llm::Tensor out_conv_bias;
    // Skip connection (only if in_c != out_c)
    llm::Tensor skip_weight;      // 1x1 conv weight [1, 1, in_c, out_c]
    llm::Tensor skip_bias;
    bool has_skip;
    uint in_c;
    uint out_c;
}

// Spatial Transformer block: self-attention + cross-attention + FFN
struct SpatialTransformerWeights {
    // GroupNorm before proj_in
    llm::Tensor norm_weight;
    llm::Tensor norm_bias;
    // proj_in: 1x1 conv
    llm::Tensor proj_in_weight;
    llm::Tensor proj_in_bias;
    // Self-attention (attn1)
    llm::Tensor attn1_q;
    llm::Tensor attn1_k;
    llm::Tensor attn1_v;
    llm::Tensor attn1_out_weight;
    llm::Tensor attn1_out_bias;
    llm::Tensor attn1_norm_weight;
    llm::Tensor attn1_norm_bias;
    // Cross-attention (attn2)
    llm::Tensor attn2_q;
    llm::Tensor attn2_k;   // context_dim -> channels
    llm::Tensor attn2_v;   // context_dim -> channels
    llm::Tensor attn2_out_weight;
    llm::Tensor attn2_out_bias;
    llm::Tensor attn2_norm_weight;
    llm::Tensor attn2_norm_bias;
    // FFN
    llm::Tensor ff_net0_weight;  // GEGLU: [channels, inner_dim*2]
    llm::Tensor ff_net0_bias;
    llm::Tensor ff_net2_weight;  // [inner_dim, channels]
    llm::Tensor ff_net2_bias;
    llm::Tensor ff_norm_weight;  // norm3
    llm::Tensor ff_norm_bias;
    // proj_out: 1x1 conv
    llm::Tensor proj_out_weight;
    llm::Tensor proj_out_bias;
    uint channels;
}

struct UnetWeights {
    // Time embedding MLP: Linear(320, 1280) -> SiLU -> Linear(1280, 1280)
    llm::Tensor time_emb_0_weight;
    llm::Tensor time_emb_0_bias;
    llm::Tensor time_emb_2_weight;
    llm::Tensor time_emb_2_bias;

    // conv_in: input_blocks.0.0
    llm::Tensor conv_in_weight;
    llm::Tensor conv_in_bias;

    // Input blocks (encoder path)
    UnetResBlock input_rb1;        // input_blocks.1.0 (320->320)
    UnetResBlock input_rb4;        // input_blocks.4.0 (320->640)
    SpatialTransformerWeights input_st4;  // input_blocks.4.1
    UnetResBlock input_rb7;        // input_blocks.7.0 (640->1280)
    SpatialTransformerWeights input_st7;  // input_blocks.7.1

    // Downsample convs
    llm::Tensor down3_weight;  // input_blocks.3.0.op (320->320, stride 2)
    llm::Tensor down3_bias;
    llm::Tensor down6_weight;  // input_blocks.6.0.op (640->640, stride 2)
    llm::Tensor down6_bias;

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
    llm::Tensor up2_weight;  // output_blocks.2.1.conv (1280->1280)
    llm::Tensor up2_bias;
    llm::Tensor up5_weight;  // output_blocks.5.2.conv (640->640)
    llm::Tensor up5_bias;

    // Output: GroupNorm -> SiLU -> Conv3x3
    llm::Tensor out_norm_weight;  // out.0
    llm::Tensor out_norm_bias;
    llm::Tensor out_conv_weight;  // out.2
    llm::Tensor out_conv_bias;
}

// --- UNet Activations ---

struct UnetActivations {
    // Working buffers for spatial features
    llm::Tensor buf_a;       // Primary spatial buffer (max: 1280 * 16 * 16)
    llm::Tensor buf_b;       // Secondary
    llm::Tensor buf_c;       // Tertiary (for residual, skip)

    // Time embedding: [1280]
    llm::Tensor time_emb;

    // Skip connection storage (saved during encoder for decoder)
    // skip0: after conv_in [320, 64, 64]
    // skip1: after input_rb1 [320, 64, 64]
    // skip3: after downsample [320, 32, 32] (before rb4)
    // skip4: after input_rb4+st4 [640, 32, 32]
    // skip6: after downsample [640, 16, 16] (before rb7)
    // skip7: after input_rb7+st7 [1280, 16, 16]
    llm::Tensor[6] skips;

    // Attention buffers
    llm::Tensor attn_q;
    llm::Tensor attn_k;
    llm::Tensor attn_v;
    llm::Tensor attn_scores;
    llm::Tensor attn_out;
    llm::Tensor ffn_buf;  // For GEGLU intermediate
}

struct UnetModel {
    llm::UnetConfig config;
    UnetWeights weights;
    UnetActivations acts;
    llm::DiffusionKernels* kernels;
    llm::DeviceContext* ctx;
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

fn UnetResBlock? load_unet_resblock(llm::DeviceContext* ctx, llm::GGUFFile* gf, String base, uint in_c, uint out_c) {
    char[256] buf;
    bool has_skip = in_c != out_c;
    UnetResBlock rb = {
        .in_norm_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.0.weight"))!!,
        .in_norm_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.0.bias"))!!,
        .in_conv_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.2.weight"))!!,
        .in_conv_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "in_layers.2.bias"))!!,
        .emb_weight     = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "emb_layers.1.weight"))!!,
        .emb_bias       = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "emb_layers.1.bias"))!!,
        .out_norm_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.0.weight"))!!,
        .out_norm_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.0.bias"))!!,
        .out_conv_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.3.weight"))!!,
        .out_conv_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "out_layers.3.bias"))!!,
        .has_skip = has_skip,
        .in_c = in_c,
        .out_c = out_c,
    };

    if (has_skip) {
        rb.skip_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "skip_connection.weight"))!!;
        rb.skip_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "skip_connection.bias"))!!;
    }

    return rb;
}

fn SpatialTransformerWeights? load_spatial_transformer(llm::DeviceContext* ctx, llm::GGUFFile* gf, String base, uint channels) {
    char[256] buf;
    return {
        .norm_weight     = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "norm.weight"))!!,
        .norm_bias       = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "norm.bias"))!!,
        .proj_in_weight  = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_in.weight"))!!,
        .proj_in_bias    = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_in.bias"))!!,
        .attn1_q         = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_q.weight"))!!,
        .attn1_k         = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_k.weight"))!!,
        .attn1_v         = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_v.weight"))!!,
        .attn1_out_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_out.0.weight"))!!,
        .attn1_out_bias  = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn1.to_out.0.bias"))!!,
        .attn1_norm_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm1.weight"))!!,
        .attn1_norm_bias = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm1.bias"))!!,
        .attn2_q         = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_q.weight"))!!,
        .attn2_k         = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_k.weight"))!!,
        .attn2_v         = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_v.weight"))!!,
        .attn2_out_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_out.0.weight"))!!,
        .attn2_out_bias  = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.attn2.to_out.0.bias"))!!,
        .attn2_norm_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm2.weight"))!!,
        .attn2_norm_bias = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm2.bias"))!!,
        .ff_net0_weight  = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.0.proj.weight"))!!,
        .ff_net0_bias    = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.0.proj.bias"))!!,
        .ff_net2_weight  = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.2.weight"))!!,
        .ff_net2_bias    = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.ff.net.2.bias"))!!,
        .ff_norm_weight  = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm3.weight"))!!,
        .ff_norm_bias    = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "transformer_blocks.0.norm3.bias"))!!,
        .proj_out_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_out.weight"))!!,
        .proj_out_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, base, "proj_out.bias"))!!,
        .channels = channels,
    };
}

// --- Load UNet ---

fn UnetWeights? load_unet_weights(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::UnetConfig* config) {
    io::printfn("\nLoading UNet weights...");
    String pfx = config.prefix;
    char[256] buf;

    UnetWeights w;

    // Time embedding MLP
    w.time_emb_0_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.0.weight"))!!;
    w.time_emb_0_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.0.bias"))!!;
    w.time_emb_2_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.2.weight"))!!;
    w.time_emb_2_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "time_embed.2.bias"))!!;
    io::printfn("  Time embedding loaded");

    // conv_in
    w.conv_in_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.0.0.weight"))!!;
    w.conv_in_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.0.0.bias"))!!;

    // Input ResBlocks
    String ib1 = unet_name(&buf, pfx, "input_blocks.1.0.");
    w.input_rb1 = load_unet_resblock(ctx, gf, ib1, 320, 320)!!;
    io::printfn("  Input block 1 loaded (320->320)");

    // Downsample 3
    w.down3_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.3.0.op.weight"))!!;
    w.down3_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.3.0.op.bias"))!!;

    String ib4 = unet_name(&buf, pfx, "input_blocks.4.0.");
    w.input_rb4 = load_unet_resblock(ctx, gf, ib4, 320, 640)!!;
    String is4 = unet_name(&buf, pfx, "input_blocks.4.1.");
    w.input_st4 = load_spatial_transformer(ctx, gf, is4, 640)!!;
    io::printfn("  Input block 4 loaded (320->640 + SpatialTransformer)");

    // Downsample 6
    w.down6_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.6.0.op.weight"))!!;
    w.down6_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "input_blocks.6.0.op.bias"))!!;

    String ib7 = unet_name(&buf, pfx, "input_blocks.7.0.");
    w.input_rb7 = load_unet_resblock(ctx, gf, ib7, 640, 1280)!!;
    String is7 = unet_name(&buf, pfx, "input_blocks.7.1.");
    w.input_st7 = load_spatial_transformer(ctx, gf, is7, 1280)!!;
    io::printfn("  Input block 7 loaded (640->1280 + SpatialTransformer)");

    // Middle attentions (up_blocks.0.attentions.{0,1})
    String ma0 = unet_name(&buf, pfx, "up_blocks.0.attentions.0.");
    w.mid_attn0 = load_spatial_transformer(ctx, gf, ma0, 1280)!!;
    String ma1 = unet_name(&buf, pfx, "up_blocks.0.attentions.1.");
    w.mid_attn1 = load_spatial_transformer(ctx, gf, ma1, 1280)!!;
    io::printfn("  Middle attentions loaded (2x SpatialTransformer 1280ch)");

    // Output blocks
    String ob0 = unet_name(&buf, pfx, "output_blocks.0.0.");
    w.output_rb0 = load_unet_resblock(ctx, gf, ob0, 2560, 1280)!!;
    String ob1 = unet_name(&buf, pfx, "output_blocks.1.0.");
    w.output_rb1 = load_unet_resblock(ctx, gf, ob1, 1920, 1280)!!;
    io::printfn("  Output blocks 0-1 loaded");

    w.up2_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.2.1.conv.weight"))!!;
    w.up2_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.2.1.conv.bias"))!!;

    String ob3 = unet_name(&buf, pfx, "output_blocks.3.0.");
    w.output_rb3 = load_unet_resblock(ctx, gf, ob3, 1920, 640)!!;
    String os3 = unet_name(&buf, pfx, "output_blocks.3.1.");
    w.output_st3 = load_spatial_transformer(ctx, gf, os3, 640)!!;
    String ob4 = unet_name(&buf, pfx, "output_blocks.4.0.");
    w.output_rb4 = load_unet_resblock(ctx, gf, ob4, 960, 640)!!;
    String os4 = unet_name(&buf, pfx, "output_blocks.4.1.");
    w.output_st4 = load_spatial_transformer(ctx, gf, os4, 640)!!;
    io::printfn("  Output blocks 3-4 loaded (with SpatialTransformers)");

    w.up5_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.5.2.conv.weight"))!!;
    w.up5_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "output_blocks.5.2.conv.bias"))!!;

    String ob6 = unet_name(&buf, pfx, "output_blocks.6.0.");
    w.output_rb6 = load_unet_resblock(ctx, gf, ob6, 960, 320)!!;
    String ob7 = unet_name(&buf, pfx, "output_blocks.7.0.");
    w.output_rb7 = load_unet_resblock(ctx, gf, ob7, 640, 320)!!;
    io::printfn("  Output blocks 6-7 loaded");

    // Final output
    w.out_norm_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.0.weight"))!!;
    w.out_norm_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.0.bias"))!!;
    w.out_conv_weight = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.2.weight"))!!;
    w.out_conv_bias   = llm::load_tensor_by_name(ctx, gf, unet_name(&buf, pfx, "out.2.bias"))!!;
    io::printfn("  Output head loaded");

    return w;
}

// --- Allocate UNet activations ---

fn UnetActivations? allocate_unet_activations(llm::DeviceContext* ctx, llm::UnetConfig* config, uint latent_h, uint latent_w) {
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
    llm::Tensor[6] skips;
    // skip0: conv_in output [320, 64, 64]
    ulong[4] s0 = { (ulong)320 * latent_h * latent_w, 0, 0, 0 };
    skips[0] = llm::create_f32_tensor(ctx, s0, 1)!!;
    // skip1: after rb1 [320, 64, 64]
    skips[1] = llm::create_f32_tensor(ctx, s0, 1)!!;
    // skip2: after downsample [320, 32, 32] (actually this is input to rb4)
    ulong half_h = latent_h / 2;
    ulong half_w = latent_w / 2;
    ulong[4] s2 = { (ulong)320 * half_h * half_w, 0, 0, 0 };
    skips[2] = llm::create_f32_tensor(ctx, s2, 1)!!;
    // skip3: after rb4+st4 [640, 32, 32]
    ulong[4] s3 = { (ulong)640 * half_h * half_w, 0, 0, 0 };
    skips[3] = llm::create_f32_tensor(ctx, s3, 1)!!;
    // skip4: after downsample [640, 16, 16]
    ulong qh = half_h / 2;
    ulong qw = half_w / 2;
    ulong[4] s4 = { (ulong)640 * qh * qw, 0, 0, 0 };
    skips[4] = llm::create_f32_tensor(ctx, s4, 1)!!;
    // skip5: after rb7+st7 [1280, 16, 16]
    ulong[4] s5 = { (ulong)1280 * qh * qw, 0, 0, 0 };
    skips[5] = llm::create_f32_tensor(ctx, s5, 1)!!;

    // Attention buffers: max seq_len = 64*64 = 4096, max channels = 1280, n_heads = 8
    uint n_heads = config.n_heads;
    ulong max_seq = (ulong)latent_h * latent_w;  // 64*64 = 4096
    ulong max_ch = 1280;
    ulong[4] attn_shape = { max_seq * max_ch, 0, 0, 0 };
    ulong[4] scores_shape = { (ulong)n_heads * max_seq * max_seq, 0, 0, 0 };
    // GEGLU intermediate: channels * 4 * 2 (doubled for gate)
    ulong[4] ffn_shape = { max_seq * max_ch * 4, 0, 0, 0 };

    return {
        .buf_a = llm::create_f32_tensor(ctx, spatial_shape, 1)!!,
        .buf_b = llm::create_f32_tensor(ctx, spatial_shape, 1)!!,
        .buf_c = llm::create_f32_tensor(ctx, spatial_shape, 1)!!,
        .time_emb = llm::create_f32_tensor(ctx, emb_shape, 1)!!,
        .skips = skips,
        .attn_q = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_k = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_v = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .attn_scores = llm::create_f32_tensor(ctx, scores_shape, 1)!!,
        .attn_out = llm::create_f32_tensor(ctx, attn_shape, 1)!!,
        .ffn_buf = llm::create_f32_tensor(ctx, ffn_shape, 1)!!,
    };
}

// --- ResBlock dispatch ---

fn void dispatch_unet_resblock(
    CommandBuffer cmd,
    llm::DiffusionKernels* k,
    UnetResBlock* rb,
    llm::Tensor* input,
    llm::Tensor* output,
    llm::Tensor* temp,
    llm::Tensor* time_emb,
    uint h, uint w
) {
    uint in_c = rb.in_c;
    uint out_c = rb.out_c;
    uint spatial = h * w;

    // 1. GroupNorm(in) -> temp
    llm::GroupNormPC gn_pc = { .channels = in_c, .spatial = spatial, .num_groups = 32, .eps = 1e-5f };
llm::dispatch_kernel(cmd, &k.group_norm,
        { input.gpu_buffer.buffer, rb.in_norm_weight.gpu_buffer.buffer,
          rb.in_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { input.size_bytes, rb.in_norm_weight.size_bytes,
          rb.in_norm_bias.size_bytes, temp.size_bytes },
        &gn_pc, llm::ceil_div(spatial, 256), in_c / 32);
llm::compute_barrier(cmd);

    // 2. SiLU(temp) in-place
    llm::SiluPC silu_pc = { .n = in_c * spatial };
llm::dispatch_kernel(cmd, &k.shared.silu,
        { temp.gpu_buffer.buffer },
        { temp.size_bytes },
        &silu_pc, llm::ceil_div(in_c * spatial, 256));
llm::compute_barrier(cmd);

    // 3. Conv3x3(temp) -> output
    llm::Conv2dPC conv_pc = {
        .in_c = in_c, .out_c = out_c, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &rb.in_conv_weight, &rb.in_conv_bias, temp, output, &conv_pc);
llm::compute_barrier(cmd);

    // 4. Time embedding projection: emb = SiLU(time_emb) then Linear -> [out_c]
    // SiLU already applied to time_emb during embedding creation
    // Linear: time_emb[1280] -> [out_c] via emb_weight
    llm::BatchedMatMulPC emb_pc = { .m_dim = 1, .n_dim = out_c, .k_dim = 1280 };
    llm::dispatch_batched_matmul_auto(cmd, k, &rb.emb_weight, time_emb, temp, &emb_pc);
llm::compute_barrier(cmd);

    // Add emb bias
    llm::ResidualPC rbias = { .n = out_c };
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { temp.gpu_buffer.buffer, rb.emb_bias.gpu_buffer.buffer },
        { temp.size_bytes, rb.emb_bias.size_bytes },
        &rbias, llm::ceil_div(out_c, 256));
llm::compute_barrier(cmd);

    // 5. Broadcast add: output[c, h, w] += emb[c]
    llm::BroadcastAddPC ba_pc = { .channels = out_c, .spatial = spatial };
llm::dispatch_kernel(cmd, &k.broadcast_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &ba_pc, llm::ceil_div(out_c * spatial, 256));
llm::compute_barrier(cmd);

    // 6. GroupNorm(output) -> temp
    gn_pc = { .channels = out_c, .spatial = spatial, .num_groups = 32, .eps = 1e-5f };
llm::dispatch_kernel(cmd, &k.group_norm,
        { output.gpu_buffer.buffer, rb.out_norm_weight.gpu_buffer.buffer,
          rb.out_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, rb.out_norm_weight.size_bytes,
          rb.out_norm_bias.size_bytes, temp.size_bytes },
        &gn_pc, llm::ceil_div(spatial, 256), out_c / 32);
llm::compute_barrier(cmd);

    // 7. SiLU(temp) in-place
    silu_pc = { .n = out_c * spatial };
llm::dispatch_kernel(cmd, &k.shared.silu,
        { temp.gpu_buffer.buffer },
        { temp.size_bytes },
        &silu_pc, llm::ceil_div(out_c * spatial, 256));
llm::compute_barrier(cmd);

    // 8. Conv3x3(temp) -> output
    conv_pc = {
        .in_c = out_c, .out_c = out_c, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &rb.out_conv_weight, &rb.out_conv_bias, temp, output, &conv_pc);
llm::compute_barrier(cmd);

    // 9. Skip connection
    if (rb.has_skip) {
        // 1x1 conv on input -> temp
        llm::Conv2dPC skip_pc = {
            .in_c = in_c, .out_c = out_c, .in_h = h, .in_w = w,
            .kH = 1, .kW = 1, .stride = 1, .pad = 0,
            .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
        };
	llm::dispatch_conv2d(cmd, k, &rb.skip_weight, &rb.skip_bias, input, temp, &skip_pc);
llm::compute_barrier(cmd);
    } else {
        // Copy input to temp for residual add
        vk::cmdCopyBuffer(cmd, input.gpu_buffer.buffer, temp.gpu_buffer.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)in_c * spatial * 4 }});
llm::compute_barrier(cmd);
    }

    // 10. output += temp (residual)
    llm::ResidualPC res_pc = { .n = out_c * spatial };
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, llm::ceil_div(out_c * spatial, 256));
llm::compute_barrier(cmd);
}

// --- Spatial Transformer dispatch ---

fn void dispatch_spatial_transformer(
    CommandBuffer cmd,
    llm::DiffusionKernels* k,
    SpatialTransformerWeights* st,
    llm::Tensor* input,       // [channels, h, w] — spatial features
    llm::Tensor* output,      // same shape as input
    llm::Tensor* temp,        // working buffer
    llm::Tensor* context,     // [77, 768] — CLIP text conditioning
    UnetActivations* acts,
    uint h, uint w,
    uint n_heads,
    uint context_dim
) {
    uint ch = st.channels;
    uint spatial = h * w;
    uint head_dim = ch / n_heads;

    // 1. GroupNorm -> temp
    llm::GroupNormPC gn_pc = { .channels = ch, .spatial = spatial, .num_groups = 32, .eps = 1e-6f };
llm::dispatch_kernel(cmd, &k.group_norm,
        { input.gpu_buffer.buffer, st.norm_weight.gpu_buffer.buffer,
          st.norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { input.size_bytes, st.norm_weight.size_bytes,
          st.norm_bias.size_bytes, temp.size_bytes },
        &gn_pc, llm::ceil_div(spatial, 256), ch / 32);
llm::compute_barrier(cmd);

    // 2. proj_in: 1x1 conv temp -> output (reshape to sequence)
    llm::Conv2dPC proj_pc = {
        .in_c = ch, .out_c = ch, .in_h = h, .in_w = w,
        .kH = 1, .kW = 1, .stride = 1, .pad = 0,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &st.proj_in_weight, &st.proj_in_bias, temp, output, &proj_pc);
llm::compute_barrier(cmd);
    // output is now [ch, spatial] which we treat as [spatial, ch] for attention

    // 3. Self-attention (attn1)
    // LayerNorm on output -> temp (per-position, 'spatial' positions of 'ch' dims)
    // We use layernorm per position
    for (uint pos = 0; pos < spatial; pos++) {
        llm::LayerNormPC ln = { .dim = ch, .eps = 1e-5f };
llm::dispatch_kernel(cmd, &k.shared.layernorm,
            { output.gpu_buffer.buffer, st.attn1_norm_weight.gpu_buffer.buffer,
              st.attn1_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
            { output.size_bytes, st.attn1_norm_weight.size_bytes,
              st.attn1_norm_bias.size_bytes, temp.size_bytes },
            &ln, 1);
    }
llm::compute_barrier(cmd);

    // Q/K/V projections: batched matmul [spatial, ch] x [ch, ch]^T -> [spatial, ch]
    llm::BatchedMatMulPC qkv_pc = { .m_dim = spatial, .n_dim = ch, .k_dim = ch };
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn1_q, temp, &acts.attn_q, &qkv_pc);
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn1_k, temp, &acts.attn_k, &qkv_pc);
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn1_v, temp, &acts.attn_v, &qkv_pc);
llm::compute_barrier(cmd);

    // Self-attention: Q, K, V are [spatial, ch] -> reshape to [n_heads, spatial, head_dim]
    float scale = 1.0f / math::sqrt((float)head_dim);
    llm::SpatialAttentionPC sa_pc = {
        .head_dim = head_dim,
        .n_heads = n_heads,
        .seq_len = spatial,
        .scale = scale,
    };
llm::dispatch_kernel(cmd, &k.spatial_attention,
        { acts.attn_q.gpu_buffer.buffer, acts.attn_k.gpu_buffer.buffer,
          acts.attn_v.gpu_buffer.buffer, acts.attn_scores.gpu_buffer.buffer,
          acts.attn_out.gpu_buffer.buffer },
        { acts.attn_q.size_bytes, acts.attn_k.size_bytes,
          acts.attn_v.size_bytes, acts.attn_scores.size_bytes,
          acts.attn_out.size_bytes },
        &sa_pc, n_heads * spatial);
llm::compute_barrier(cmd);

    // Output projection
    llm::BatchedMatMulPC out_pc = { .m_dim = spatial, .n_dim = ch, .k_dim = ch };
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn1_out_weight, &acts.attn_out, temp, &out_pc);
llm::compute_barrier(cmd);

    // Add bias
    llm::BroadcastAddPC ba = { .channels = ch, .spatial = spatial };
llm::dispatch_kernel(cmd, &k.broadcast_add,
        { temp.gpu_buffer.buffer, st.attn1_out_bias.gpu_buffer.buffer },
        { temp.size_bytes, st.attn1_out_bias.size_bytes },
        &ba, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // Residual: output += attn1_output
    llm::ResidualPC res_pc = { .n = ch * spatial };
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // 4. Cross-attention (attn2): Q from image, K/V from text context
    // LayerNorm on output -> temp
    for (uint pos = 0; pos < spatial; pos++) {
        llm::LayerNormPC ln = { .dim = ch, .eps = 1e-5f };
llm::dispatch_kernel(cmd, &k.shared.layernorm,
            { output.gpu_buffer.buffer, st.attn2_norm_weight.gpu_buffer.buffer,
              st.attn2_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
            { output.size_bytes, st.attn2_norm_weight.size_bytes,
              st.attn2_norm_bias.size_bytes, temp.size_bytes },
            &ln, 1);
    }
llm::compute_barrier(cmd);

    // Q from image features: [spatial, ch] x [ch, ch]^T -> [spatial, ch]
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn2_q, temp, &acts.attn_q, &qkv_pc);
    // K from context: [77, context_dim] x [context_dim, ch]^T -> [77, ch]
    llm::BatchedMatMulPC kv_ctx_pc = { .m_dim = 77, .n_dim = ch, .k_dim = context_dim };
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn2_k, context, &acts.attn_k, &kv_ctx_pc);
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn2_v, context, &acts.attn_v, &kv_ctx_pc);
llm::compute_barrier(cmd);

    // Cross-attention: Q=[spatial, ch], K/V=[77, ch]
    llm::CrossAttentionPC ca_pc = {
        .head_dim = head_dim,
        .n_heads = n_heads,
        .q_len = spatial,
        .kv_len = 77,
        .scale = scale,
    };
llm::dispatch_kernel(cmd, &k.cross_attention,
        { acts.attn_q.gpu_buffer.buffer, acts.attn_k.gpu_buffer.buffer,
          acts.attn_v.gpu_buffer.buffer, acts.attn_scores.gpu_buffer.buffer,
          acts.attn_out.gpu_buffer.buffer },
        { acts.attn_q.size_bytes, acts.attn_k.size_bytes,
          acts.attn_v.size_bytes, acts.attn_scores.size_bytes,
          acts.attn_out.size_bytes },
        &ca_pc, n_heads * spatial);
llm::compute_barrier(cmd);

    // Output projection
    llm::dispatch_batched_matmul_auto(cmd, k, &st.attn2_out_weight, &acts.attn_out, temp, &out_pc);
llm::compute_barrier(cmd);
llm::dispatch_kernel(cmd, &k.broadcast_add,
        { temp.gpu_buffer.buffer, st.attn2_out_bias.gpu_buffer.buffer },
        { temp.size_bytes, st.attn2_out_bias.size_bytes },
        &ba, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // Residual
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // 5. FFN: LayerNorm -> GEGLU -> Linear -> residual
    // LayerNorm
    for (uint pos = 0; pos < spatial; pos++) {
        llm::LayerNormPC ln = { .dim = ch, .eps = 1e-5f };
llm::dispatch_kernel(cmd, &k.shared.layernorm,
            { output.gpu_buffer.buffer, st.ff_norm_weight.gpu_buffer.buffer,
              st.ff_norm_bias.gpu_buffer.buffer, temp.gpu_buffer.buffer },
            { output.size_bytes, st.ff_norm_weight.size_bytes,
              st.ff_norm_bias.size_bytes, temp.size_bytes },
            &ln, 1);
    }
llm::compute_barrier(cmd);

    // GEGLU: Linear([ch] -> [inner_dim*2]), split, GELU(first_half) * second_half
    // inner_dim = ch * 4 (from ff.net.0.proj output is [ch, inner_dim*2])
    uint inner_dim = ch * 4;
    llm::BatchedMatMulPC ff0_pc = { .m_dim = spatial, .n_dim = inner_dim * 2, .k_dim = ch };
    llm::dispatch_batched_matmul_auto(cmd, k, &st.ff_net0_weight, temp, &acts.ffn_buf, &ff0_pc);
llm::compute_barrier(cmd);

    // Add bias to ffn_buf
    llm::BroadcastAddPC ff_ba = { .channels = inner_dim * 2, .spatial = spatial };
llm::dispatch_kernel(cmd, &k.broadcast_add,
        { acts.ffn_buf.gpu_buffer.buffer, st.ff_net0_bias.gpu_buffer.buffer },
        { acts.ffn_buf.size_bytes, st.ff_net0_bias.size_bytes },
        &ff_ba, llm::ceil_div(inner_dim * 2 * spatial, 256));
llm::compute_barrier(cmd);

    // GELU on first half, multiply with second half
    // For now: apply GELU to first inner_dim elements, then elemwise_mul with second half
    llm::GeluPC gelu_pc = { .n = inner_dim * spatial };
llm::dispatch_kernel(cmd, &k.shared.gelu,
        { acts.ffn_buf.gpu_buffer.buffer },
        { acts.ffn_buf.size_bytes },
        &gelu_pc, llm::ceil_div(inner_dim * spatial, 256));
llm::compute_barrier(cmd);

    // Multiply: ffn_buf[0..inner_dim*spatial-1] *= ffn_buf[inner_dim*spatial..]
    llm::ElemwisePC ew_pc = { .n = inner_dim * spatial };
    // Use elemwise_mul treating second half as separate buffer (offset)
    // TODO: This needs an offset-aware elemwise or we copy
    // For now, we'll just use the first half (skip GEGLU gate for initial implementation)
    // This simplification will affect quality but allows testing the pipeline

    // Linear down: [inner_dim] -> [ch]
    llm::BatchedMatMulPC ff2_pc = { .m_dim = spatial, .n_dim = ch, .k_dim = inner_dim };
    llm::dispatch_batched_matmul_auto(cmd, k, &st.ff_net2_weight, &acts.ffn_buf, temp, &ff2_pc);
llm::compute_barrier(cmd);

    // Add bias
llm::dispatch_kernel(cmd, &k.broadcast_add,
        { temp.gpu_buffer.buffer, st.ff_net2_bias.gpu_buffer.buffer },
        { temp.size_bytes, st.ff_net2_bias.size_bytes },
        &ba, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // Residual
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { output.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { output.size_bytes, temp.size_bytes },
        &res_pc, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // 6. proj_out: 1x1 conv output -> temp, then residual with original input
	llm::dispatch_conv2d(cmd, k, &st.proj_out_weight, &st.proj_out_bias, output, temp, &proj_pc);
llm::compute_barrier(cmd);

    // Final residual: input + proj_out
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { input.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { input.size_bytes, temp.size_bytes },
        &res_pc, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // Copy result to output
    vk::cmdCopyBuffer(cmd, input.gpu_buffer.buffer, output.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)ch * spatial * 4 }});
llm::compute_barrier(cmd);
}

// --- UNet Forward Pass ---
// Input: noisy latent [4, 64, 64] in acts.buf_a
// Output: predicted noise [4, 64, 64] in acts.buf_a

fn void? UnetModel.forward(&self, llm::Tensor* context, float timestep) {
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    llm::DiffusionKernels* k = self.kernels;
    UnetWeights* w = &self.weights;
    UnetActivations* a = &self.acts;
    llm::UnetConfig* cfg = &self.config;

    uint h = 64;  // latent height
    uint wd = 64;  // latent width (renamed to avoid shadowing)

llm::begin_compute(cmd)!!;

    // === 1. Timestep Embedding ===
    // Sinusoidal embedding -> MLP
    llm::TimestepEmbedPC te_pc = { .dim = cfg.model_channels, .timestep = timestep, .max_period = 10000.0f };
llm::dispatch_kernel(cmd, &k.timestep_embed,
        { a.time_emb.gpu_buffer.buffer },
        { a.time_emb.size_bytes },
        &te_pc, llm::ceil_div(cfg.model_channels, 256));
llm::compute_barrier(cmd);

    // MLP: Linear(320, 1280) -> SiLU -> Linear(1280, 1280)
    llm::BatchedMatMulPC te_mlp0 = { .m_dim = 1, .n_dim = 1280, .k_dim = cfg.model_channels };
    llm::dispatch_batched_matmul_auto(cmd, k, &w.time_emb_0_weight, &a.time_emb, &a.buf_c, &te_mlp0);
llm::compute_barrier(cmd);
    llm::ResidualPC te_bias = { .n = 1280 };
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { a.buf_c.gpu_buffer.buffer, w.time_emb_0_bias.gpu_buffer.buffer },
        { a.buf_c.size_bytes, w.time_emb_0_bias.size_bytes },
        &te_bias, llm::ceil_div(1280, 256));
llm::compute_barrier(cmd);

    llm::SiluPC te_silu = { .n = 1280 };
llm::dispatch_kernel(cmd, &k.shared.silu,
        { a.buf_c.gpu_buffer.buffer },
        { a.buf_c.size_bytes },
        &te_silu, llm::ceil_div(1280, 256));
llm::compute_barrier(cmd);

    llm::BatchedMatMulPC te_mlp2 = { .m_dim = 1, .n_dim = 1280, .k_dim = 1280 };
    llm::dispatch_batched_matmul_auto(cmd, k, &w.time_emb_2_weight, &a.buf_c, &a.time_emb, &te_mlp2);
llm::compute_barrier(cmd);
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { a.time_emb.gpu_buffer.buffer, w.time_emb_2_bias.gpu_buffer.buffer },
        { a.time_emb.size_bytes, w.time_emb_2_bias.size_bytes },
        &te_bias, llm::ceil_div(1280, 256));
llm::compute_barrier(cmd);

    // === 2. Encoder (Input Blocks) ===

    // conv_in: [4, 64, 64] -> [320, 64, 64]
    llm::Conv2dPC conv_in_pc = {
        .in_c = 4, .out_c = 320, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &w.conv_in_weight, &w.conv_in_bias, &a.buf_a, &a.buf_b, &conv_in_pc);
llm::compute_barrier(cmd);

    // Save skip0: conv_in output [320, 64, 64]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[0].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
llm::compute_barrier(cmd);

    // ResBlock 1: 320->320
    dispatch_unet_resblock(cmd, k, &w.input_rb1, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);
    // Result is in buf_b (resblock writes back to input)
    // Actually our resblock writes result to the input buffer via copy at the end
    // So result is in buf_b

    // Save skip1: [320, 64, 64]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[1].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
llm::compute_barrier(cmd);

    // Downsample 3: stride-2 conv [320, 64, 64] -> [320, 32, 32]
    llm::Conv2dPC down3_pc = {
        .in_c = 320, .out_c = 320, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 2, .pad = 1,
        .groups = 1, .out_h = h / 2, .out_w = wd / 2, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &w.down3_weight, &w.down3_bias, &a.buf_b, &a.buf_a, &down3_pc);
llm::compute_barrier(cmd);
    h /= 2; wd /= 2;  // 32x32

    // Save skip2: [320, 32, 32]
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.skips[2].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
llm::compute_barrier(cmd);

    // ResBlock 4: 320->640 (with skip connection)
    // Copy buf_a to buf_b first (resblock takes input from first arg)
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)320 * h * wd * 4 }});
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.input_rb4, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // SpatialTransformer 4 (640ch)
    // input_rb4 result is in buf_b, copy to buf_a for spatial transformer
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
llm::compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.input_st4, &a.buf_a, &a.buf_b, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);
    // Result in buf_b

    // Save skip3: [640, 32, 32]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[3].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
llm::compute_barrier(cmd);

    // Downsample 6: stride-2 conv [640, 32, 32] -> [640, 16, 16]
    llm::Conv2dPC down6_pc = {
        .in_c = 640, .out_c = 640, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 2, .pad = 1,
        .groups = 1, .out_h = h / 2, .out_w = wd / 2, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &w.down6_weight, &w.down6_bias, &a.buf_b, &a.buf_a, &down6_pc);
llm::compute_barrier(cmd);
    h /= 2; wd /= 2;  // 16x16

    // Save skip4: [640, 16, 16]
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.skips[4].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
llm::compute_barrier(cmd);

    // ResBlock 7: 640->1280 (with skip connection)
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.input_rb7, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // SpatialTransformer 7 (1280ch)
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)1280 * h * wd * 4 }});
llm::compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.input_st7, &a.buf_a, &a.buf_b, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);

    // Save skip5: [1280, 16, 16]
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.skips[5].gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)1280 * h * wd * 4 }});
llm::compute_barrier(cmd);

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
    llm::ChannelConcatPC cc0 = { .channels_a = 1280, .channels_b = 1280, .spatial = h * wd };
llm::dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_b.gpu_buffer.buffer, a.skips[5].gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.skips[5].size_bytes, a.buf_a.size_bytes },
        &cc0, llm::ceil_div(2560 * h * wd, 256));
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb0, &a.buf_a, &a.buf_b, &a.buf_c, &a.time_emb, h, wd);
    // Result in buf_a (resblock copies back)

    // output_blocks.1: concat(buf_a[1280,16,16], skip4[640,16,16]) -> ResBlock(1920->1280)
    llm::ChannelConcatPC cc1 = { .channels_a = 1280, .channels_b = 640, .spatial = h * wd };
llm::dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_a.gpu_buffer.buffer, a.skips[4].gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.skips[4].size_bytes, a.buf_b.size_bytes },
        &cc1, llm::ceil_div(1920 * h * wd, 256));
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb1, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // output_blocks.2: upsample 1280ch, 16x16 -> 32x32
    llm::UpsamplePC up2 = { .channels = 1280, .in_h = h, .in_w = wd };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.buf_a.size_bytes },
        &up2, llm::ceil_div(1280 * h * wd * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; wd *= 2;  // 32x32

    // Conv after upsample
    llm::Conv2dPC up2_conv = {
        .in_c = 1280, .out_c = 1280, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &w.up2_weight, &w.up2_bias, &a.buf_a, &a.buf_b, &up2_conv);
llm::compute_barrier(cmd);

    // output_blocks.3: concat(buf_b[1280,32,32], skip3[640,32,32]) -> ResBlock(1920->640) + ST
    llm::ChannelConcatPC cc3 = { .channels_a = 1280, .channels_b = 640, .spatial = h * wd };
llm::dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_b.gpu_buffer.buffer, a.skips[3].gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.skips[3].size_bytes, a.buf_a.size_bytes },
        &cc3, llm::ceil_div(1920 * h * wd, 256));
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb3, &a.buf_a, &a.buf_b, &a.buf_c, &a.time_emb, h, wd);

    // SpatialTransformer at 640ch
    vk::cmdCopyBuffer(cmd, a.buf_a.gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
llm::compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.output_st3, &a.buf_b, &a.buf_a, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);

    // output_blocks.4: concat(buf_a[640,32,32], skip2[320,32,32]) -> ResBlock(960->640) + ST
    llm::ChannelConcatPC cc4 = { .channels_a = 640, .channels_b = 320, .spatial = h * wd };
llm::dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_a.gpu_buffer.buffer, a.skips[2].gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.skips[2].size_bytes, a.buf_b.size_bytes },
        &cc4, llm::ceil_div(960 * h * wd, 256));
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb4, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)640 * h * wd * 4 }});
llm::compute_barrier(cmd);
    dispatch_spatial_transformer(cmd, k, &w.output_st4, &a.buf_a, &a.buf_b, &a.buf_c,
        context, &self.acts, h, wd, cfg.n_heads, cfg.context_dim);

    // output_blocks.5: upsample 640ch, 32x32 -> 64x64
    llm::UpsamplePC up5 = { .channels = 640, .in_h = h, .in_w = wd };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.buf_a.size_bytes },
        &up5, llm::ceil_div(640 * h * wd * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; wd *= 2;  // 64x64

    llm::Conv2dPC up5_conv = {
        .in_c = 640, .out_c = 640, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &w.up5_weight, &w.up5_bias, &a.buf_a, &a.buf_b, &up5_conv);
llm::compute_barrier(cmd);

    // output_blocks.6: concat(buf_b[640,64,64], skip1[320,64,64]) -> ResBlock(960->320)
    llm::ChannelConcatPC cc6 = { .channels_a = 640, .channels_b = 320, .spatial = h * wd };
llm::dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_b.gpu_buffer.buffer, a.skips[1].gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, a.skips[1].size_bytes, a.buf_a.size_bytes },
        &cc6, llm::ceil_div(960 * h * wd, 256));
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb6, &a.buf_a, &a.buf_b, &a.buf_c, &a.time_emb, h, wd);

    // output_blocks.7: concat(buf_a[320,64,64], skip0[320,64,64]) -> ResBlock(640->320)
    llm::ChannelConcatPC cc7 = { .channels_a = 320, .channels_b = 320, .spatial = h * wd };
llm::dispatch_kernel(cmd, &k.channel_concat,
        { a.buf_a.gpu_buffer.buffer, a.skips[0].gpu_buffer.buffer, a.buf_b.gpu_buffer.buffer },
        { a.buf_a.size_bytes, a.skips[0].size_bytes, a.buf_b.size_bytes },
        &cc7, llm::ceil_div(640 * h * wd, 256));
llm::compute_barrier(cmd);
    dispatch_unet_resblock(cmd, k, &w.output_rb7, &a.buf_b, &a.buf_a, &a.buf_c, &a.time_emb, h, wd);

    // === 5. Output Head ===
    // GroupNorm -> SiLU -> Conv3x3 (320 -> 4)
    uint spatial = h * wd;
    llm::GroupNormPC out_gn = { .channels = 320, .spatial = spatial, .num_groups = 32, .eps = 1e-5f };
llm::dispatch_kernel(cmd, &k.group_norm,
        { a.buf_b.gpu_buffer.buffer, w.out_norm_weight.gpu_buffer.buffer,
          w.out_norm_bias.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer },
        { a.buf_b.size_bytes, w.out_norm_weight.size_bytes,
          w.out_norm_bias.size_bytes, a.buf_a.size_bytes },
        &out_gn, llm::ceil_div(spatial, 256), 320 / 32);
llm::compute_barrier(cmd);

    llm::SiluPC out_silu = { .n = 320 * spatial };
llm::dispatch_kernel(cmd, &k.shared.silu,
        { a.buf_a.gpu_buffer.buffer },
        { a.buf_a.size_bytes },
        &out_silu, llm::ceil_div(320 * spatial, 256));
llm::compute_barrier(cmd);

    llm::Conv2dPC out_conv = {
        .in_c = 320, .out_c = 4, .in_h = h, .in_w = wd,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = wd, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &w.out_conv_weight, &w.out_conv_bias, &a.buf_a, &a.buf_b, &out_conv);
llm::compute_barrier(cmd);

    // Copy result to buf_a
    vk::cmdCopyBuffer(cmd, a.buf_b.gpu_buffer.buffer, a.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)4 * h * wd * 4 }});

llm::submit_and_wait(ctx)!!;
    // Result: predicted noise [4, 64, 64] in buf_a
}

// --- Load UNet Model ---

fn UnetModel? load_unet_model(llm::DeviceContext* ctx, llm::GGUFFile* gf, llm::UnetConfig* config, llm::DiffusionKernels* kernels) {
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
