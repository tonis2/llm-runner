module zimage;

import vk;
import llm;
import std::io;
import std::math;
import std::core::mem;

// Flux VAE Encoder — mirror of FluxVaeDecoder for img2img / Kontext editing.
// Architecture (from safetensors layout, diffusers naming):
//   conv_in (3 -> 128) -> down0 (2 resnets @128 + downsample) ->
//   down1 (resnet 128->256 + resnet 256 + downsample) ->
//   down2 (resnet 256->512 + resnet 512 + downsample) ->
//   down3 (2 resnets @512, no downsample) ->
//   mid (resblock + attention + resblock) @512 ->
//   norm_out + silu + conv_out (512 -> 64)
// Downsample = stride-2 3x3 conv with diffusers' "pad bottom+right by 1" pattern.
// The existing conv2d kernel handles this naturally with pad=0, stride=2 because
// out-of-bounds reads return zero (matches the asymmetric zero-pad).

struct FluxVaeEncoder {
    FluxVaeConv conv_in;       // [128, 3, 3, 3]

    // Down blocks 0..2 each have 2 resnets + 1 downsample (stride-2 conv)
    FluxVaeResBlock[2] down0_blocks;       // 128 -> 128, 128 -> 128
    FluxVaeConv down0_downsample;          // [128, 128, 3, 3] stride 2
    FluxVaeResBlock[2] down1_blocks;       // 128 -> 256 (shortcut), 256 -> 256
    FluxVaeConv down1_downsample;          // [256, 256, 3, 3] stride 2
    FluxVaeResBlock[2] down2_blocks;       // 256 -> 512 (shortcut), 512 -> 512
    FluxVaeConv down2_downsample;          // [512, 512, 3, 3] stride 2
    FluxVaeResBlock[2] down3_blocks;       // 512 -> 512, 512 -> 512 (no downsample)

    // Mid block (same shape as decoder's mid)
    FluxVaeResBlock mid_block1;
    FluxVaeAttn mid_attn;
    FluxVaeResBlock mid_block2;

    // Output
    llm::Tensor norm_out_weight;
    llm::Tensor norm_out_bias;
    FluxVaeConv conv_out;                  // [64, 512, 3, 3]

    VaeKernels* kernels;
    llm::DeviceContext* ctx;
}

// --- Loader ---

fn FluxVaeEncoder? load_flux_vae_encoder(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, VaeKernels* kernels) {
    FluxVaeEncoder enc;
    enc.ctx = ctx;
    enc.kernels = kernels;

    if (try info = sf.find_tensor("encoder.conv_in.weight")) {
        // Confirm layout matches expectation — bail loudly if not.
        if (info.n_dims < 2 || info.shape[1] != 3) {
            io::printfn("  ERROR: encoder.conv_in expects 3 input channels, got %d", info.shape[1]);
            return llm::COMPUTE_ERROR~;
        }
    } else {
        io::printfn("  ERROR: encoder.conv_in.weight not found in safetensors");
        return llm::COMPUTE_ERROR~;
    }

    enc.conv_in = load_flux_conv(ctx, sf, "encoder.conv_in.", 3, 128);

    // down_blocks.0: 2 resnets at 128 + downsampler
    enc.down0_blocks[0] = load_flux_resblock(ctx, sf, "encoder.down_blocks.0.resnets.0.", 128, 128);
    enc.down0_blocks[1] = load_flux_resblock(ctx, sf, "encoder.down_blocks.0.resnets.1.", 128, 128);
    enc.down0_downsample = load_flux_conv(ctx, sf, "encoder.down_blocks.0.downsamplers.0.conv.", 128, 128);

    // down_blocks.1: 128 -> 256 (with shortcut), 256 -> 256, then downsampler
    enc.down1_blocks[0] = load_flux_resblock(ctx, sf, "encoder.down_blocks.1.resnets.0.", 128, 256);
    enc.down1_blocks[1] = load_flux_resblock(ctx, sf, "encoder.down_blocks.1.resnets.1.", 256, 256);
    enc.down1_downsample = load_flux_conv(ctx, sf, "encoder.down_blocks.1.downsamplers.0.conv.", 256, 256);

    // down_blocks.2: 256 -> 512 (with shortcut), 512 -> 512, then downsampler
    enc.down2_blocks[0] = load_flux_resblock(ctx, sf, "encoder.down_blocks.2.resnets.0.", 256, 512);
    enc.down2_blocks[1] = load_flux_resblock(ctx, sf, "encoder.down_blocks.2.resnets.1.", 512, 512);
    enc.down2_downsample = load_flux_conv(ctx, sf, "encoder.down_blocks.2.downsamplers.0.conv.", 512, 512);

    // down_blocks.3: 2 resnets at 512, no downsampler (final stage)
    enc.down3_blocks[0] = load_flux_resblock(ctx, sf, "encoder.down_blocks.3.resnets.0.", 512, 512);
    enc.down3_blocks[1] = load_flux_resblock(ctx, sf, "encoder.down_blocks.3.resnets.1.", 512, 512);

    // mid block (same naming convention as decoder's mid)
    enc.mid_block1 = load_flux_resblock(ctx, sf, "encoder.mid_block.resnets.0.", 512, 512);
    enc.mid_attn = load_flux_attn(ctx, sf, "encoder.mid_block.attentions.0.", 512);
    enc.mid_block2 = load_flux_resblock(ctx, sf, "encoder.mid_block.resnets.1.", 512, 512);

    // norm_out + conv_out (512 -> 64; 64 = mean[32] + logvar[32] packed)
    enc.norm_out_weight = load_flux_vae_tensor(ctx, sf, "encoder.", "conv_norm_out.weight");
    enc.norm_out_bias = load_flux_vae_tensor(ctx, sf, "encoder.", "conv_norm_out.bias");
    enc.conv_out = load_flux_conv(ctx, sf, "encoder.conv_out.", 512, 64);

    return enc;
}

// --- Activations ---
// Encoder needs to hold up to 128 channels at the full image resolution
// (post-conv_in). Decoder's allocate_flux_vae_activations sizes buffers for
// 256 * max_size * max_size, which is ample (encoder peak = 128 * H * W).

fn FluxVaeActivations? allocate_flux_vae_encoder_activations(llm::DeviceContext* ctx, uint img_size) {
    return allocate_flux_vae_activations(ctx, img_size);
}

// --- Forward Pass ---
// Input:  image [3, H, W] in [-1, 1] in acts.buf_a
// Output: encoded [64, H/8, W/8] in acts.buf_a (mean+logvar packed)

fn void? FluxVaeEncoder.forward(&self, FluxVaeActivations* acts, uint img_h, uint img_w) {
    VaeKernels* k = self.kernels;
    llm::RenderGraph g;
    g.init(self.ctx)!!;

    uint h = img_h;
    uint w = img_w;

    // conv_in: [3, H, W] -> [128, H, W]
    flux_conv_block(&g, k, &self.conv_in, &acts.buf_a, &acts.buf_b, h, w, 1, 1);
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)128 * h * w * 4);
    g.checkpoint()!!;

    // down_blocks.0: 2 resnets at 128, then downsample to H/2 x W/2
    for (uint i = 0; i < 2; i++) {
        flux_resblock_block(&g, k, &self.down0_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.checkpoint()!!;
    }
    flux_conv_downsample_block(&g, k, &self.down0_downsample, &acts.buf_a, &acts.buf_b, h, w);
    h /= 2; w /= 2;
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)128 * h * w * 4);
    g.checkpoint()!!;

    // down_blocks.1: 128 -> 256 (resnet0 with shortcut), 256, then downsample
    flux_resblock_block(&g, k, &self.down1_blocks[0], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    flux_resblock_block(&g, k, &self.down1_blocks[1], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    flux_conv_downsample_block(&g, k, &self.down1_downsample, &acts.buf_a, &acts.buf_b, h, w);
    h /= 2; w /= 2;
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)256 * h * w * 4);
    g.checkpoint()!!;

    // down_blocks.2: 256 -> 512, 512, then downsample
    flux_resblock_block(&g, k, &self.down2_blocks[0], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    flux_resblock_block(&g, k, &self.down2_blocks[1], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    flux_conv_downsample_block(&g, k, &self.down2_downsample, &acts.buf_a, &acts.buf_b, h, w);
    h /= 2; w /= 2;
    g.copy(&acts.buf_b, &acts.buf_a, (ulong)512 * h * w * 4);
    g.checkpoint()!!;

    // down_blocks.3: 2 resnets at 512, no downsample
    for (uint i = 0; i < 2; i++) {
        flux_resblock_block(&g, k, &self.down3_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
        g.checkpoint()!!;
    }

    // mid block (resblock + attention + resblock)
    flux_resblock_block(&g, k, &self.mid_block1, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    flux_attn_block(&g, k, &self.mid_attn, acts, h, w);
    flux_resblock_block(&g, k, &self.mid_block2, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    g.checkpoint()!!;

    // norm_out + silu + conv_out (512 -> 64)
    flux_groupnorm_block(&g, k, &self.norm_out_weight, &self.norm_out_bias,
        &acts.buf_a, &acts.buf_b, 512, h * w);

    llm::SiluPC silu_pc = { .n = 512 * h * w };
    g.dispatch(&k.shared.silu, { &acts.buf_b }, &silu_pc, llm::ceil_div(512 * h * w, 256));

    flux_conv_block(&g, k, &self.conv_out, &acts.buf_b, &acts.buf_a, h, w, 1, 1);

    g.flush()!!;
    // Output [64, h, w] in buf_a
}

// Stride-2 downsample conv with diffusers' asymmetric "pad bottom+right" semantics.
// Implemented via pad=0 + ceil-div output dims; the existing conv2d kernel's
// out-of-bounds reads return zero, which matches the asymmetric zero-padding.
// Requires in_h and in_w to be even.
fn void flux_conv_downsample_block(
    llm::RenderGraph* g,
    VaeKernels* k,
    FluxVaeConv* conv,
    llm::Tensor* input,
    llm::Tensor* output,
    uint in_h, uint in_w
) {
    uint out_h = in_h / 2;
    uint out_w = in_w / 2;
    pipelines::Conv2dPC pc = {
        .in_c = conv.in_c,
        .out_c = conv.out_c,
        .in_h = in_h,
        .in_w = in_w,
        .kH = conv.kH,
        .kW = conv.kW,
        .stride = 2,
        .pad = 0,
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

// --- Free ---

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
}
