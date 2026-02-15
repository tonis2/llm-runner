module sdxs;

import vk;
import llm;
import std::io;
import std::core::mem;

// VAE Decoder + Encoder for Stable Diffusion
// Uses a flat layers.N numbering scheme from the GGUF
// This is a simplified (Tiny) VAE: 64 channels throughout, no attention

// --- VAE Layer Types ---

// A single Conv2d layer: weight + optional bias
struct VaeConvLayer {
    llm::Tensor weight;
    llm::Tensor bias;
    uint in_c;
    uint out_c;
    uint kH;
    uint kW;
    bool has_bias;
}

// A ResNet block with 3 conv layers (conv.0, conv.2, conv.4)
// conv.0: GroupNorm (via weight) + Conv3x3
// conv.2: GroupNorm + Conv3x3
// conv.4: Skip connection conv (1x1 or identity)
struct VaeResBlock {
    llm::Tensor conv0_weight;
    llm::Tensor conv0_bias;
    llm::Tensor conv2_weight;
    llm::Tensor conv2_bias;
    llm::Tensor conv4_weight;
    llm::Tensor conv4_bias;
    uint channels;
}

// --- VAE Decoder ---

struct VaeDecoder {
    pipelines::VaeConfig config;

    // conv_in: 4 -> 64 channels
    VaeConvLayer conv_in;

    // ResNet blocks + upsample layers
    // Decoder structure (from tensor names):
    // layers.2,3,4: ResBlocks at 64ch (64x64 latent)
    // layers.6: upsample conv (64->128 spatial)
    // layers.7,8,9: ResBlocks at 64ch (128x128)
    // layers.11: upsample conv (128->256)
    // layers.12,13,14: ResBlocks at 64ch (256x256)
    // layers.16: upsample conv (256->512)
    // layers.17: ResBlock at 64ch (512x512)
    // layers.18: conv_out 64->3

    VaeResBlock[10] res_blocks;  // 10 res blocks total
    VaeConvLayer[3] upsample_convs;  // 3 upsample layers
    VaeConvLayer conv_out;

    pipelines::DiffusionKernels* kernels;
    llm::DeviceContext* ctx;
}

// --- VAE Encoder ---

struct VaeEncoder {
    pipelines::VaeConfig config;

    // conv_in: 3 -> 64 channels
    VaeConvLayer conv_in;

    // ResBlocks + downsample layers
    // layers.1,3,4,5: ResBlocks at 64ch (512x512)
    // layers.2: downsample conv (512->256)
    // layers.7,8,9: ResBlocks at 64ch (256x256)
    // layers.6: downsample conv (256->128)
    // layers.11,12,13: ResBlocks at 64ch (128x128)
    // layers.10: downsample conv (128->64)
    // layers.14: conv_out 64->4

    VaeResBlock[10] res_blocks;
    VaeConvLayer[3] downsample_convs;
    VaeConvLayer conv_out;

    pipelines::DiffusionKernels* kernels;
    llm::DeviceContext* ctx;
}

// --- VAE Activations (shared between encoder and decoder) ---

struct VaeActivations {
    llm::Tensor buf_a;  // Primary spatial buffer
    llm::Tensor buf_b;  // Secondary spatial buffer (ping-pong)
    llm::Tensor buf_c;  // Temporary for residual
}

// --- Tensor name helpers ---

fn String vae_tensor_name(char[128]* buf, String prefix, uint layer, String suffix) {
    char* p = &(*buf)[0];
    usz pos = 0;
    for (usz i = 0; i < prefix.len; i++) p[pos + i] = prefix[i];
    pos += prefix.len;
    String mid = "layers.";
    for (usz i = 0; i < mid.len; i++) p[pos + i] = mid[i];
    pos += mid.len;
    if (layer >= 10) { p[pos] = (char)('0' + (layer / 10) % 10); pos++; }
    p[pos] = (char)('0' + layer % 10); pos++;
    p[pos] = '.'; pos++;
    for (usz i = 0; i < suffix.len; i++) p[pos + i] = suffix[i];
    pos += suffix.len;
    return (String)(*buf)[0..pos - 1];
}

fn VaeConvLayer? load_vae_conv(llm::DeviceContext* ctx, llm::GGUFFile* gf, String prefix, uint layer, uint in_c, uint out_c, bool has_bias = true) {
    char[128] buf;
    llm::Tensor w = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "weight"))!!;
    llm::Tensor b;
    if (has_bias) {
        b = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "bias"))!!;
    }
    return {
        .weight = w,
        .bias = b,
        .in_c = in_c,
        .out_c = out_c,
        .kH = (uint)w.shape[0],
        .kW = (uint)w.shape[1],
        .has_bias = has_bias,
    };
}

fn VaeResBlock? load_vae_resblock(llm::DeviceContext* ctx, llm::GGUFFile* gf, String prefix, uint layer, uint channels) {
    char[128] buf;
    return {
        .conv0_weight = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "conv.0.weight"))!!,
        .conv0_bias = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "conv.0.bias"))!!,
        .conv2_weight = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "conv.2.weight"))!!,
        .conv2_bias = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "conv.2.bias"))!!,
        .conv4_weight = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "conv.4.weight"))!!,
        .conv4_bias = llm::load_tensor_by_name(ctx, gf, vae_tensor_name(&buf, prefix, layer, "conv.4.bias"))!!,
        .channels = channels,
    };
}

// --- Load VAE Decoder ---

fn VaeDecoder? load_vae_decoder(llm::DeviceContext* ctx, llm::GGUFFile* gf, pipelines::VaeConfig* config, pipelines::DiffusionKernels* kernels) {
    io::printfn("\nLoading VAE decoder weights...");
    String pfx = config.prefix_decoder;

    VaeConvLayer conv_in = load_vae_conv(ctx, gf, pfx, 0, 4, 64)!!;

    // ResBlocks: layers 2,3,4, 7,8,9, 12,13,14, 17
    VaeResBlock[10] res_blocks;
    uint[10] rb_layers = { 2, 3, 4, 7, 8, 9, 12, 13, 14, 17 };
    for (uint i = 0; i < 10; i++) {
        res_blocks[i] = load_vae_resblock(ctx, gf, pfx, rb_layers[i], 64)!!;
    }

    // Upsample convs: layers 6, 11, 16 (weight-only, no bias)
    VaeConvLayer[3] upsample_convs;
    uint[3] up_layers = { 6, 11, 16 };
    for (uint i = 0; i < 3; i++) {
        upsample_convs[i] = load_vae_conv(ctx, gf, pfx, up_layers[i], 64, 64, has_bias: false)!!;
    }

    VaeConvLayer conv_out = load_vae_conv(ctx, gf, pfx, 18, 64, 3)!!;

    io::printfn("  VAE decoder loaded: %d res blocks, %d upsample layers",
        (uint)10, (uint)3);

    return {
        .config = *config,
        .conv_in = conv_in,
        .res_blocks = res_blocks,
        .upsample_convs = upsample_convs,
        .conv_out = conv_out,
        .kernels = kernels,
        .ctx = ctx,
    };
}

// --- Load VAE Encoder ---

fn VaeEncoder? load_vae_encoder(llm::DeviceContext* ctx, llm::GGUFFile* gf, pipelines::VaeConfig* config, pipelines::DiffusionKernels* kernels) {
    io::printfn("\nLoading VAE encoder weights...");
    String pfx = config.prefix_encoder;

    VaeConvLayer conv_in = load_vae_conv(ctx, gf, pfx, 0, 3, 64)!!;

    // ResBlocks: layers 1,3,4,5, 7,8,9, 11,12,13
    VaeResBlock[10] res_blocks;
    uint[10] rb_layers = { 1, 3, 4, 5, 7, 8, 9, 11, 12, 13 };
    for (uint i = 0; i < 10; i++) {
        res_blocks[i] = load_vae_resblock(ctx, gf, pfx, rb_layers[i], 64)!!;
    }

    // Downsample convs: layers 2, 6, 10 (weight-only, no bias)
    VaeConvLayer[3] downsample_convs;
    uint[3] dn_layers = { 2, 6, 10 };
    for (uint i = 0; i < 3; i++) {
        downsample_convs[i] = load_vae_conv(ctx, gf, pfx, dn_layers[i], 64, 64, has_bias: false)!!;
    }

    VaeConvLayer conv_out = load_vae_conv(ctx, gf, pfx, 14, 64, 4)!!;

    io::printfn("  VAE encoder loaded: %d res blocks, %d downsample layers",
        (uint)10, (uint)3);

    return {
        .config = *config,
        .conv_in = conv_in,
        .res_blocks = res_blocks,
        .downsample_convs = downsample_convs,
        .conv_out = conv_out,
        .kernels = kernels,
        .ctx = ctx,
    };
}

// --- Allocate VAE activation buffers ---

fn VaeActivations? allocate_vae_activations(llm::DeviceContext* ctx, uint max_spatial_size, uint max_channels) {
    // Largest buffer: max_channels * max_spatial_size^2 floats
    // For decoder: 64 * 512 * 512 = 16M floats = 64MB
    ulong max_elems = (ulong)max_channels * max_spatial_size * max_spatial_size;
    ulong[4] shape = { max_elems, 0, 0, 0 };

    return {
        .buf_a = llm::create_f32_tensor(ctx, shape, 1)!!,
        .buf_b = llm::create_f32_tensor(ctx, shape, 1)!!,
        .buf_c = llm::create_f32_tensor(ctx, shape, 1)!!,
    };
}

// --- Helper: dispatch conv2d for a VaeConvLayer ---

fn void dispatch_vae_conv(
    CommandBuffer cmd,
    pipelines::DiffusionKernels* k,
    VaeConvLayer* layer,
    llm::Tensor* input,
    llm::Tensor* output,
    uint in_h, uint in_w,
    uint stride, uint pad
) {
    uint out_h = (in_h + 2 * pad - layer.kH) / stride + 1;
    uint out_w = (in_w + 2 * pad - layer.kW) / stride + 1;
    llm::Conv2dPC pc = {
        .in_c = layer.in_c,
        .out_c = layer.out_c,
        .in_h = in_h,
        .in_w = in_w,
        .kH = layer.kH,
        .kW = layer.kW,
        .stride = stride,
        .pad = pad,
        .groups = 1,
        .out_h = out_h,
        .out_w = out_w,
        .has_bias = layer.has_bias ? 1 : 0,
    };
    // When no bias, pass weight buffer as dummy (shader won't read it with has_bias=0)
    llm::Tensor* bias_tensor = layer.has_bias ? &layer.bias : &layer.weight;
	llm::dispatch_conv2d(cmd, k, &layer.weight, bias_tensor, input, output, &pc);
}

// --- Helper: dispatch ResBlock ---
// ResBlock: SiLU -> Conv3x3 -> SiLU -> Conv3x3 -> skip_conv + residual
// conv.0 = first conv, conv.2 = second conv, conv.4 = skip connection

fn void dispatch_vae_resblock(
    CommandBuffer cmd,
    pipelines::DiffusionKernels* k,
    VaeResBlock* rb,
    llm::Tensor* input,     // buf_a
    llm::Tensor* temp,      // buf_b (temporary)
    llm::Tensor* skip_temp, // buf_c (for skip connection)
    uint h, uint w
) {
    uint ch = rb.channels;
    uint spatial = h * w;

    // SiLU on input -> temp (in-place would need copy, so we use temp)
    llm::SiluPC silu_pc = { .n = ch * spatial };
    // Copy input to temp first, then apply SiLU in-place on temp
    vk::cmdCopyBuffer(cmd, input.gpu_buffer.buffer, temp.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)ch * spatial * 4 }});
llm::compute_barrier(cmd);

llm::dispatch_kernel(cmd, &k.shared.silu,
        { temp.gpu_buffer.buffer },
        { temp.size_bytes },
        &silu_pc, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // Conv3x3 (conv.0): temp -> skip_temp
    llm::Conv2dPC pc0 = {
        .in_c = ch, .out_c = ch, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &rb.conv0_weight, &rb.conv0_bias, temp, skip_temp, &pc0);
llm::compute_barrier(cmd);

    // SiLU on skip_temp
llm::dispatch_kernel(cmd, &k.shared.silu,
        { skip_temp.gpu_buffer.buffer },
        { skip_temp.size_bytes },
        &silu_pc, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // Conv3x3 (conv.2): skip_temp -> temp
	llm::dispatch_conv2d(cmd, k, &rb.conv2_weight, &rb.conv2_bias, skip_temp, temp, &pc0);
llm::compute_barrier(cmd);

    // Skip connection conv (conv.4): input -> skip_temp
    // This is typically a 3x3 conv that acts as identity or channel projection
    llm::Conv2dPC pc_skip = {
        .in_c = ch, .out_c = ch, .in_h = h, .in_w = w,
        .kH = 3, .kW = 3, .stride = 1, .pad = 1,
        .groups = 1, .out_h = h, .out_w = w, .has_bias = 1,
    };
	llm::dispatch_conv2d(cmd, k, &rb.conv4_weight, &rb.conv4_bias, input, skip_temp, &pc_skip);
llm::compute_barrier(cmd);

    // Residual add: skip_temp += temp, result in skip_temp
    llm::ResidualPC res_pc = { .n = ch * spatial };
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { skip_temp.gpu_buffer.buffer, temp.gpu_buffer.buffer },
        { skip_temp.size_bytes, temp.size_bytes },
        &res_pc, llm::ceil_div(ch * spatial, 256));
llm::compute_barrier(cmd);

    // Copy result back to input buffer for next layer
    vk::cmdCopyBuffer(cmd, skip_temp.gpu_buffer.buffer, input.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)ch * spatial * 4 }});
llm::compute_barrier(cmd);
}

// --- VAE Decoder Forward Pass ---
// Input: latent [4, 64, 64] in buf_a
// Output: image [3, 512, 512] in buf_a

fn void? VaeDecoder.forward(&self, VaeActivations* acts) {
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    pipelines::DiffusionKernels* k = self.kernels;

    uint h = 64;  // latent height
    uint w = 64;  // latent width

llm::begin_compute(cmd)!!;

    // conv_in: [4, 64, 64] -> [64, 64, 64]
    dispatch_vae_conv(cmd, k, &self.conv_in, &acts.buf_a, &acts.buf_b, h, w, 1, 1);
llm::compute_barrier(cmd);
    // Copy result to buf_a
    vk::cmdCopyBuffer(cmd, acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)64 * h * w * 4 }});
llm::compute_barrier(cmd);

    // ResBlocks 0,1,2 (layers 2,3,4) at 64x64
    for (uint i = 0; i < 3; i++) {
        dispatch_vae_resblock(cmd, k, &self.res_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }

    // Upsample 0 (layer 6): 64x64 -> 128x128
    llm::UpsamplePC up_pc = { .channels = 64, .in_h = h, .in_w = w };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { acts.buf_a.gpu_buffer.buffer, acts.buf_b.gpu_buffer.buffer },
        { acts.buf_a.size_bytes, acts.buf_b.size_bytes },
        &up_pc, llm::ceil_div(64 * h * w * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; w *= 2;  // 128x128

    // Conv after upsample (layer 6)
    dispatch_vae_conv(cmd, k, &self.upsample_convs[0], &acts.buf_b, &acts.buf_a, h, w, 1, 1);
llm::compute_barrier(cmd);

    // ResBlocks 3,4,5 (layers 7,8,9) at 128x128
    for (uint i = 3; i < 6; i++) {
        dispatch_vae_resblock(cmd, k, &self.res_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }

    // Upsample 1 (layer 11): 128x128 -> 256x256
    up_pc = { .channels = 64, .in_h = h, .in_w = w };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { acts.buf_a.gpu_buffer.buffer, acts.buf_b.gpu_buffer.buffer },
        { acts.buf_a.size_bytes, acts.buf_b.size_bytes },
        &up_pc, llm::ceil_div(64 * h * w * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; w *= 2;  // 256x256

    dispatch_vae_conv(cmd, k, &self.upsample_convs[1], &acts.buf_b, &acts.buf_a, h, w, 1, 1);
llm::compute_barrier(cmd);

    // ResBlocks 6,7,8 (layers 12,13,14) at 256x256
    for (uint i = 6; i < 9; i++) {
        dispatch_vae_resblock(cmd, k, &self.res_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }

    // Upsample 2 (layer 16): 256x256 -> 512x512
    up_pc = { .channels = 64, .in_h = h, .in_w = w };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { acts.buf_a.gpu_buffer.buffer, acts.buf_b.gpu_buffer.buffer },
        { acts.buf_a.size_bytes, acts.buf_b.size_bytes },
        &up_pc, llm::ceil_div(64 * h * w * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; w *= 2;  // 512x512

    dispatch_vae_conv(cmd, k, &self.upsample_convs[2], &acts.buf_b, &acts.buf_a, h, w, 1, 1);
llm::compute_barrier(cmd);

    // ResBlock 9 (layer 17) at 512x512
    dispatch_vae_resblock(cmd, k, &self.res_blocks[9], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);

    // conv_out: [64, 512, 512] -> [3, 512, 512]
    dispatch_vae_conv(cmd, k, &self.conv_out, &acts.buf_a, &acts.buf_b, h, w, 1, 1);
llm::compute_barrier(cmd);

    // Scale + shift to [0,1] range: output = output * 0.5 + 0.5
    llm::ScaleShiftPC ss_pc = { .n = 3 * h * w, .scale = 0.5f, .shift = 0.5f };
llm::dispatch_kernel(cmd, &k.scale_shift_clamp,
        { acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer },
        { acts.buf_b.size_bytes, acts.buf_a.size_bytes },
        &ss_pc, llm::ceil_div(3 * h * w, 256));

llm::submit_and_wait(ctx)!!;
    // Result image [3, 512, 512] is in buf_a
}

// --- VAE Encoder Forward Pass ---
// Input: image [3, H, W] in buf_a (normalized to [-1, 1])
// Output: latent [4, H/8, W/8] in buf_a

fn void? VaeEncoder.forward(&self, VaeActivations* acts, uint img_h, uint img_w) {
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    pipelines::DiffusionKernels* k = self.kernels;

    uint h = img_h;
    uint w = img_w;

llm::begin_compute(cmd)!!;

    // conv_in: [3, H, W] -> [64, H, W]
    dispatch_vae_conv(cmd, k, &self.conv_in, &acts.buf_a, &acts.buf_b, h, w, 1, 1);
llm::compute_barrier(cmd);
    vk::cmdCopyBuffer(cmd, acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)64 * h * w * 4 }});
llm::compute_barrier(cmd);

    // ResBlocks 0,1,2,3 (layers 1,3,4,5) at full res
    for (uint i = 0; i < 4; i++) {
        dispatch_vae_resblock(cmd, k, &self.res_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }

    // Downsample 0 (layer 2): stride-2 conv
    dispatch_vae_conv(cmd, k, &self.downsample_convs[0], &acts.buf_a, &acts.buf_b, h, w, 2, 1);
llm::compute_barrier(cmd);
    h /= 2; w /= 2;
    vk::cmdCopyBuffer(cmd, acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)64 * h * w * 4 }});
llm::compute_barrier(cmd);

    // ResBlocks 4,5,6 (layers 7,8,9) at half res
    for (uint i = 4; i < 7; i++) {
        dispatch_vae_resblock(cmd, k, &self.res_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }

    // Downsample 1 (layer 6): stride-2 conv
    dispatch_vae_conv(cmd, k, &self.downsample_convs[1], &acts.buf_a, &acts.buf_b, h, w, 2, 1);
llm::compute_barrier(cmd);
    h /= 2; w /= 2;
    vk::cmdCopyBuffer(cmd, acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)64 * h * w * 4 }});
llm::compute_barrier(cmd);

    // ResBlocks 7,8,9 (layers 11,12,13) at quarter res
    for (uint i = 7; i < 10; i++) {
        dispatch_vae_resblock(cmd, k, &self.res_blocks[i], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    }

    // Downsample 2 (layer 10): stride-2 conv
    dispatch_vae_conv(cmd, k, &self.downsample_convs[2], &acts.buf_a, &acts.buf_b, h, w, 2, 1);
llm::compute_barrier(cmd);
    h /= 2; w /= 2;
    vk::cmdCopyBuffer(cmd, acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)64 * h * w * 4 }});
llm::compute_barrier(cmd);

    // conv_out: [64, H/8, W/8] -> [4, H/8, W/8]
    dispatch_vae_conv(cmd, k, &self.conv_out, &acts.buf_a, &acts.buf_b, h, w, 1, 1);
llm::compute_barrier(cmd);

    // Copy to buf_a
    vk::cmdCopyBuffer(cmd, acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)4 * h * w * 4 }});

llm::submit_and_wait(ctx)!!;
    // Result latent [4, H/8, W/8] is in buf_a
}

// --- Free functions ---

fn void VaeConvLayer.free(&self) {
    self.weight.free();
    self.bias.free();
}

fn void VaeResBlock.free(&self) {
    self.conv0_weight.free();
    self.conv0_bias.free();
    self.conv2_weight.free();
    self.conv2_bias.free();
    self.conv4_weight.free();
    self.conv4_bias.free();
}

fn void VaeDecoder.free(&self) {
    self.conv_in.free();
    for (uint i = 0; i < 10; i++) self.res_blocks[i].free();
    for (uint i = 0; i < 3; i++) self.upsample_convs[i].free();
    self.conv_out.free();
}

fn void VaeEncoder.free(&self) {
    self.conv_in.free();
    for (uint i = 0; i < 10; i++) self.res_blocks[i].free();
    for (uint i = 0; i < 3; i++) self.downsample_convs[i].free();
    self.conv_out.free();
}

fn void VaeActivations.free(&self) {
    self.buf_a.free();
    self.buf_b.free();
    self.buf_c.free();
}
