module flux;

import vk;
import llm;
import llm::pipelines;
import zimage;
import std::io;
import std::math;
import std::core::mem;

// Flux VAE Encoder
// Mirrors the decoder: RGB [3, H, W] -> latent [16, H/8, W/8]
// Architecture: conv_in(3->128) -> down.0(128, downsample) -> down.1(128->256, downsample)
// -> down.2(256->512, downsample) -> down.3(512) -> mid(resblock+attn+resblock)
// -> norm_out(512) -> conv_out(512->32) -> take first 16 channels as mean

struct FluxVaeEncoder {
    zimage::FluxVaeConv conv_in;       // [128, 3, 3, 3]

    // Down blocks
    zimage::FluxVaeResBlock[2] down0_blocks;  // 128->128
    zimage::FluxVaeConv down0_downsample;     // stride-2 conv

    zimage::FluxVaeResBlock[2] down1_blocks;  // 128->256
    zimage::FluxVaeConv down1_downsample;

    zimage::FluxVaeResBlock[2] down2_blocks;  // 256->512
    zimage::FluxVaeConv down2_downsample;

    zimage::FluxVaeResBlock[2] down3_blocks;  // 512->512 (no downsample)

    // Mid block (same as decoder)
    zimage::FluxVaeResBlock mid_block1;
    zimage::FluxVaeAttn mid_attn;
    zimage::FluxVaeResBlock mid_block2;

    // Output
    llm::Tensor norm_out_weight;
    llm::Tensor norm_out_bias;
    zimage::FluxVaeConv conv_out;    // [32, 512, 3, 3]
    zimage::FluxVaeConv quant_conv;  // [32, 32, 1, 1]

    zimage::VaeKernels* kernels;
    llm::DeviceContext* ctx;
}

// Load Flux VAE Encoder from safetensors
fn FluxVaeEncoder? load_flux_vae_encoder(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, zimage::VaeKernels* kernels) {
    io::printfn("\nLoading Flux VAE encoder...");

    FluxVaeEncoder enc;
    enc.ctx = ctx;
    enc.kernels = kernels;

    enc.conv_in = zimage::load_flux_conv(ctx, sf, "encoder.conv_in.", 3, 128);

    // down.0: 2 resblocks 128->128 + downsample
    for (uint i = 0; i < 2; i++) {
        char[64] rb_prefix;
        usz plen = build_block_prefix(&rb_prefix, "encoder.down.0.block.", i);
        enc.down0_blocks[i] = zimage::load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], 128, 128);
    }
    enc.down0_downsample = load_downsample_conv(ctx, sf, "encoder.down.0.downsample.conv.", 128, 128);

    // down.1: 2 resblocks 128->256 + downsample
    for (uint i = 0; i < 2; i++) {
        char[64] rb_prefix;
        usz plen = build_block_prefix(&rb_prefix, "encoder.down.1.block.", i);
        uint in_c = (i == 0) ? 128 : 256;
        enc.down1_blocks[i] = zimage::load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], in_c, 256);
    }
    enc.down1_downsample = load_downsample_conv(ctx, sf, "encoder.down.1.downsample.conv.", 256, 256);

    // down.2: 2 resblocks 256->512 + downsample
    for (uint i = 0; i < 2; i++) {
        char[64] rb_prefix;
        usz plen = build_block_prefix(&rb_prefix, "encoder.down.2.block.", i);
        uint in_c = (i == 0) ? 256 : 512;
        enc.down2_blocks[i] = zimage::load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], in_c, 512);
    }
    enc.down2_downsample = load_downsample_conv(ctx, sf, "encoder.down.2.downsample.conv.", 512, 512);

    // down.3: 2 resblocks 512->512 (no downsample)
    for (uint i = 0; i < 2; i++) {
        char[64] rb_prefix;
        usz plen = build_block_prefix(&rb_prefix, "encoder.down.3.block.", i);
        enc.down3_blocks[i] = zimage::load_flux_resblock(ctx, sf, (String)rb_prefix[0..plen - 1], 512, 512);
    }

    // Mid block
    enc.mid_block1 = zimage::load_flux_resblock(ctx, sf, "encoder.mid.block_1.", 512, 512);
    enc.mid_attn = zimage::load_flux_attn(ctx, sf, "encoder.mid.attn_1.", 512);
    enc.mid_block2 = zimage::load_flux_resblock(ctx, sf, "encoder.mid.block_2.", 512, 512);

    // Output
    enc.norm_out_weight = zimage::load_flux_vae_tensor(ctx, sf, "encoder.", "norm_out.weight");
    enc.norm_out_bias = zimage::load_flux_vae_tensor(ctx, sf, "encoder.", "norm_out.bias");
    enc.conv_out = zimage::load_flux_conv(ctx, sf, "encoder.conv_out.", 512, 32);

    // quant_conv: 1x1 conv [32, 32]
    enc.quant_conv = load_quant_conv(ctx, sf);

    io::printfn("  VAE encoder loaded successfully");
    return enc;
}

// --- Helpers ---

fn usz build_block_prefix(char[64]* buf, String base, uint idx) {
    usz plen = 0;
    for (usz j = 0; j < base.len; j++) { (*buf)[plen] = base[j]; plen++; }
    (*buf)[plen] = (char)('0' + idx); plen++;
    (*buf)[plen] = '.'; plen++;
    return plen;
}

fn zimage::FluxVaeConv load_downsample_conv(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint in_c, uint out_c) {
    // Downsample convolutions are 3x3 with stride 2, pad 1
    // Same loading as regular conv, but stride is applied during forward pass
    return zimage::load_flux_conv(ctx, sf, prefix, in_c, out_c);
}

fn zimage::FluxVaeConv load_quant_conv(llm::DeviceContext* ctx, llm::SafetensorsFile* sf) {
    char[256] buf;
    String w_name = "quant_conv.weight";
    String b_name = "quant_conv.bias";

    llm::Tensor w = llm::upload_safetensor_conv_f32(ctx, sf, w_name)!!;
    llm::Tensor b = llm::upload_safetensor_f32(ctx, sf, b_name)!!;

    return {
        .weight = w,
        .bias = b,
        .in_c = 32,
        .out_c = 32,
        .kH = 1,
        .kW = 1,
    };
}

// --- Allocate Encoder Activations ---
// Reuses FluxVaeActivations from zimage since the buffer layout is similar

fn zimage::FluxVaeActivations? allocate_encoder_activations(llm::DeviceContext* ctx, uint image_size) {
    // Largest buffers needed:
    //   128 * image_size^2 at full resolution (down.0 input)
    //   256 * (image_size/2)^2 at half resolution
    //   512 * (image_size/4)^2 at quarter resolution
    // Max is 128 * image_size * image_size
    ulong max_elems = (ulong)128 * image_size * image_size;
    // Also need 512 * (image_size/4)^2 = 512 * image_size^2 / 16 = 32 * image_size^2
    // Actually 512 at image_size/4: 512 * (image_size/4)^2 which can be > 128*image_size^2
    // Let's compute properly:
    ulong max_a = (ulong)128 * image_size * image_size;
    ulong max_b = (ulong)512 * (image_size / 4) * (image_size / 4);
    if (max_b > max_a) max_a = max_b;
    ulong[4] shape = { max_a, 0, 0, 0 };

    // Attention at latent resolution: 512 * (image_size/8)^2
    ulong lat_size = (ulong)image_size / 8;
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

// --- VAE Encoder Forward Pass ---
// Input: image [3, H, W] in acts.buf_a (normalized to [-1, 1])
// Output: latent [16, H/8, W/8] in acts.buf_a

fn void? FluxVaeEncoder.forward(&self, zimage::FluxVaeActivations* acts, uint img_h, uint img_w) {
    zimage::VaeKernels* k = self.kernels;

    uint h = img_h;
    uint w = img_w;

    llm::RenderGraph g;
    g.init(self.ctx)!!;

    // conv_in: [3, H, W] -> [128, H, W]
    zimage::flux_conv_block(&g, k, &self.conv_in, &acts.buf_a, &acts.buf_b, h, w, 1, 1);
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)128 * h * w * 4);

    // down.0: 2 resblocks at 128 + downsample
    for (uint i = 0; i < 2; i++) {
        zimage::flux_resblock_block(&g, k, &self.down0_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }
    // Downsample: stride-2 conv [128->128, 3x3, stride=2, pad=1]
    {
        uint out_h = h / 2;
        uint out_w = w / 2;
        pipelines::Conv2dPC ds_pc = {
            .in_c = 128, .out_c = 128, .in_h = h, .in_w = w,
            .kH = 3, .kW = 3, .stride = 2, .pad = 1,
            .groups = 1, .out_h = out_h, .out_w = out_w, .has_bias = 1,
        };
        zimage::flux_conv2d_block(&g, k, &self.down0_downsample.weight, &self.down0_downsample.bias,
            &acts.buf_a, &acts.buf_b, &ds_pc);
        h = out_h; w = out_w;
        g.copy(&acts.buf_b, &acts.buf_a, (ulong)128 * h * w * 4);
    }

    g.checkpoint()!!;

    // down.1: 2 resblocks 128->256 + downsample
    for (uint i = 0; i < 2; i++) {
        zimage::flux_resblock_block(&g, k, &self.down1_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }
    {
        uint out_h = h / 2;
        uint out_w = w / 2;
        pipelines::Conv2dPC ds_pc = {
            .in_c = 256, .out_c = 256, .in_h = h, .in_w = w,
            .kH = 3, .kW = 3, .stride = 2, .pad = 1,
            .groups = 1, .out_h = out_h, .out_w = out_w, .has_bias = 1,
        };
        zimage::flux_conv2d_block(&g, k, &self.down1_downsample.weight, &self.down1_downsample.bias,
            &acts.buf_a, &acts.buf_b, &ds_pc);
        h = out_h; w = out_w;
        g.copy(&acts.buf_b, &acts.buf_a, (ulong)256 * h * w * 4);
    }

    g.checkpoint()!!;

    // down.2: 2 resblocks 256->512 + downsample
    for (uint i = 0; i < 2; i++) {
        zimage::flux_resblock_block(&g, k, &self.down2_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }
    {
        uint out_h = h / 2;
        uint out_w = w / 2;
        pipelines::Conv2dPC ds_pc = {
            .in_c = 512, .out_c = 512, .in_h = h, .in_w = w,
            .kH = 3, .kW = 3, .stride = 2, .pad = 1,
            .groups = 1, .out_h = out_h, .out_w = out_w, .has_bias = 1,
        };
        zimage::flux_conv2d_block(&g, k, &self.down2_downsample.weight, &self.down2_downsample.bias,
            &acts.buf_a, &acts.buf_b, &ds_pc);
        h = out_h; w = out_w;
        g.copy(&acts.buf_b, &acts.buf_a, (ulong)512 * h * w * 4);
    }

    g.checkpoint()!!;

    // down.3: 2 resblocks 512->512 (no downsample)
    for (uint i = 0; i < 2; i++) {
        zimage::flux_resblock_block(&g, k, &self.down3_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }

    g.checkpoint()!!;

    // Mid block: resblock + attention + resblock
    zimage::flux_resblock_block(&g, k, &self.mid_block1, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    zimage::flux_attn_block(&g, k, &self.mid_attn, acts, h, w);
    zimage::flux_resblock_block(&g, k, &self.mid_block2, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);

    g.checkpoint()!!;

    // Output: GroupNorm -> SiLU -> conv_out [512->32]
    zimage::flux_groupnorm_block(&g, k, &self.norm_out_weight, &self.norm_out_bias,
        &acts.buf_a, &acts.buf_b, 512, h * w);

    llm::SiluPC silu_pc = { .n = 512 * h * w };
    g.dispatch(&k.shared.silu, { &acts.buf_b }, &silu_pc, llm::ceil_div(512 * h * w, 256));

    zimage::flux_conv_block(&g, k, &self.conv_out, &acts.buf_b, &acts.buf_a, h, w, 1, 1);

    // quant_conv: 1x1 conv [32->32]
    {
        pipelines::Conv2dPC qc_pc = {
            .in_c = 32, .out_c = 32, .in_h = h, .in_w = w,
            .kH = 1, .kW = 1, .stride = 1, .pad = 0,
            .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
        };
        zimage::flux_conv2d_block(&g, k, &self.quant_conv.weight, &self.quant_conv.bias,
            &acts.buf_a, &acts.buf_b, &qc_pc);
    }

    // Take first 16 channels as mean (discard logvar in channels 16-31)
    // The first 16*h*w floats in buf_b are the mean, rest is logvar
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)16 * h * w * 4);

    g.flush()!!;

    io::printfn("  VAE encode: [3, %d, %d] -> [16, %d, %d]", img_h, img_w, h, w);
    // Latent [16, h, w] is in acts.buf_a
}

fn void FluxVaeEncoder.free(&self) {
    self.conv_in.free();
    for (uint i = 0; i < 2; i++) {
        self.down0_blocks[i].free();
        self.down1_blocks[i].free();
        self.down2_blocks[i].free();
        self.down3_blocks[i].free();
    }
    self.down0_downsample.free();
    self.down1_downsample.free();
    self.down2_downsample.free();
    self.mid_block1.free();
    self.mid_attn.free();
    self.mid_block2.free();
    if (self.norm_out_weight.size_bytes > 0) self.norm_out_weight.free();
    if (self.norm_out_bias.size_bytes > 0) self.norm_out_bias.free();
    self.conv_out.free();
    self.quant_conv.free();
}
