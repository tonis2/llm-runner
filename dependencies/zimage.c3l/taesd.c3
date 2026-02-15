module zimage;

import vk;
import llm;
import std::io;
import std::core::mem;

// TAESD (Tiny AutoEncoder for Stable Diffusion) Decoder
// Architecture from madebyollin/taesd3 (16 latent channels for DiT/Flux)
//
// Forward pass:
//   tanh_clamp(z) -> Conv(16→64) -> ReLU ->
//   [3x TAEBlock(64) -> Upsample2x -> Conv(64→64)] x3 ->
//   TAEBlock(64) -> Conv(64→3) -> clamp[0,1]
//
// TAEBlock: conv.0 -> relu -> conv.2 -> relu -> conv.4 -> residual_add -> relu

const uint TAESD_CHANNELS = 64;
const uint TAESD_Z_CHANNELS = 16;
const uint TAESD_OUT_CHANNELS = 3;
const uint TAESD_NUM_BLOCKS = 3;  // blocks per stage

struct TAESDConv {
    llm::Tensor weight;
    llm::Tensor bias;
    uint in_c;
    uint out_c;
    bool has_bias;
}

struct TAESDBlock {
    TAESDConv conv0;  // conv.0: [n_in -> n_out, 3x3]
    TAESDConv conv2;  // conv.2: [n_out -> n_out, 3x3]
    TAESDConv conv4;  // conv.4: [n_out -> n_out, 3x3]
}

struct TAESDDecoder {
    TAESDConv conv_in;                         // index 0: Conv(16→64, bias)
    TAESDBlock[TAESD_NUM_BLOCKS] stage1;       // indices 2,3,4
    TAESDConv up1_conv;                        // index 6: Conv(64→64, no bias)
    TAESDBlock[TAESD_NUM_BLOCKS] stage2;       // indices 7,8,9
    TAESDConv up2_conv;                        // index 11: Conv(64→64, no bias)
    TAESDBlock[TAESD_NUM_BLOCKS] stage3;       // indices 12,13,14
    TAESDConv up3_conv;                        // index 16: Conv(64→64, no bias)
    TAESDBlock final_block;                    // index 17
    TAESDConv conv_out;                        // index 18: Conv(64→3, bias)

    llm::DiffusionKernels* kernels;
    llm::DeviceContext* ctx;
}

struct TAESDActivations {
    llm::Tensor buf_a;
    llm::Tensor buf_b;
    llm::Tensor buf_c;
}

// --- Loading ---

fn llm::Tensor load_taesd_tensor(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String name) {
    return llm::upload_safetensor_f32(ctx, sf, name)!!;
}

fn llm::Tensor load_taesd_conv_weight(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String name) {
    return llm::upload_safetensor_conv_f32(ctx, sf, name)!!;
}

fn TAESDConv load_taesd_conv(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint in_c, uint out_c, bool has_bias) {
    char[256] wbuf;
    char[256] bbuf;
    usz wlen = 0;
    usz blen = 0;
    for (usz i = 0; i < prefix.len; i++) { wbuf[wlen] = prefix[i]; wlen++; bbuf[blen] = prefix[i]; blen++; }
    String ws = "weight";
    String bs = "bias";
    for (usz i = 0; i < ws.len; i++) { wbuf[wlen] = ws[i]; wlen++; }
    for (usz i = 0; i < bs.len; i++) { bbuf[blen] = bs[i]; blen++; }

    TAESDConv conv;
    conv.in_c = in_c;
    conv.out_c = out_c;
    conv.has_bias = has_bias;
    conv.weight = load_taesd_conv_weight(ctx, sf, (String)wbuf[0..wlen - 1]);
    if (has_bias) {
        conv.bias = load_taesd_tensor(ctx, sf, (String)bbuf[0..blen - 1]);
    }
    return conv;
}

fn TAESDBlock load_taesd_block(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String prefix, uint in_c, uint out_c) {
    char[256] p0; char[256] p2; char[256] p4;
    usz l0 = 0; usz l2 = 0; usz l4 = 0;
    for (usz i = 0; i < prefix.len; i++) {
        p0[l0] = prefix[i]; l0++;
        p2[l2] = prefix[i]; l2++;
        p4[l4] = prefix[i]; l4++;
    }
    String s0 = "conv.0."; String s2 = "conv.2."; String s4 = "conv.4.";
    for (usz i = 0; i < s0.len; i++) { p0[l0] = s0[i]; l0++; }
    for (usz i = 0; i < s2.len; i++) { p2[l2] = s2[i]; l2++; }
    for (usz i = 0; i < s4.len; i++) { p4[l4] = s4[i]; l4++; }

    return {
        .conv0 = load_taesd_conv(ctx, sf, (String)p0[0..l0 - 1], in_c, out_c, true),
        .conv2 = load_taesd_conv(ctx, sf, (String)p2[0..l2 - 1], out_c, out_c, true),
        .conv4 = load_taesd_conv(ctx, sf, (String)p4[0..l4 - 1], out_c, out_c, true),
    };
}

fn TAESDDecoder? load_taesd_decoder(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, llm::DiffusionKernels* kernels) {
    io::printfn("\nLoading TAESD decoder...");

    TAESDDecoder dec;
    dec.ctx = ctx;
    dec.kernels = kernels;

    // Index 0: Conv2d(16→64, 3x3, pad=1, bias=true)
    dec.conv_in = load_taesd_conv(ctx, sf, "decoder.layers.0.", TAESD_Z_CHANNELS, TAESD_CHANNELS, true);
    io::printfn("  conv_in loaded (16→64)");

    // Stage 1: indices 2,3,4 - TAEBlocks(64,64)
    for (uint i = 0; i < TAESD_NUM_BLOCKS; i++) {
        char[64] prefix;
        usz plen = fmt_taesd_prefix(&prefix, i + 2);
        dec.stage1[i] = load_taesd_block(ctx, sf, (String)prefix[0..plen - 1], TAESD_CHANNELS, TAESD_CHANNELS);
    }
    // Index 6: post-upsample conv (no bias)
    dec.up1_conv = load_taesd_conv(ctx, sf, "decoder.layers.6.", TAESD_CHANNELS, TAESD_CHANNELS, false);
    io::printfn("  stage 1 loaded (3 blocks + upsample conv)");

    // Stage 2: indices 7,8,9
    for (uint i = 0; i < TAESD_NUM_BLOCKS; i++) {
        char[64] prefix;
        usz plen = fmt_taesd_prefix(&prefix, i + 7);
        dec.stage2[i] = load_taesd_block(ctx, sf, (String)prefix[0..plen - 1], TAESD_CHANNELS, TAESD_CHANNELS);
    }
    // Index 11: post-upsample conv (no bias)
    dec.up2_conv = load_taesd_conv(ctx, sf, "decoder.layers.11.", TAESD_CHANNELS, TAESD_CHANNELS, false);
    io::printfn("  stage 2 loaded (3 blocks + upsample conv)");

    // Stage 3: indices 12,13,14
    for (uint i = 0; i < TAESD_NUM_BLOCKS; i++) {
        char[64] prefix;
        usz plen = fmt_taesd_prefix(&prefix, i + 12);
        dec.stage3[i] = load_taesd_block(ctx, sf, (String)prefix[0..plen - 1], TAESD_CHANNELS, TAESD_CHANNELS);
    }
    // Index 16: post-upsample conv (no bias)
    dec.up3_conv = load_taesd_conv(ctx, sf, "decoder.layers.16.", TAESD_CHANNELS, TAESD_CHANNELS, false);
    io::printfn("  stage 3 loaded (3 blocks + upsample conv)");

    // Index 17: final TAEBlock
    {
        char[64] prefix;
        usz plen = fmt_taesd_prefix(&prefix, 17);
        dec.final_block = load_taesd_block(ctx, sf, (String)prefix[0..plen - 1], TAESD_CHANNELS, TAESD_CHANNELS);
    }

    // Index 18: Conv2d(64→3, 3x3, pad=1, bias=true)
    dec.conv_out = load_taesd_conv(ctx, sf, "decoder.layers.18.", TAESD_CHANNELS, TAESD_OUT_CHANNELS, true);
    io::printfn("  TAESD decoder loaded (19 layers).");

    return dec;
}

fn usz fmt_taesd_prefix(char[64]* buf, uint index) {
    String base = "decoder.layers.";
    usz pos = 0;
    for (usz i = 0; i < base.len; i++) { (*buf)[pos] = base[i]; pos++; }
    // Write index digits
    if (index >= 10) {
        (*buf)[pos] = (char)('0' + index / 10); pos++;
    }
    (*buf)[pos] = (char)('0' + index % 10); pos++;
    (*buf)[pos] = '.'; pos++;
    return pos;
}

// --- Activation Allocation ---

fn TAESDActivations? allocate_taesd_activations(llm::DeviceContext* ctx, uint max_img_size) {
    // Max buffer: 64ch at full output resolution (after all upsamples)
    ulong max_elems = (ulong)TAESD_CHANNELS * max_img_size * max_img_size;
    ulong[4] shape = { max_elems, 0, 0, 0 };
    return {
        .buf_a = llm::create_f32_tensor(ctx, shape, 1)!!,
        .buf_b = llm::create_f32_tensor(ctx, shape, 1)!!,
        .buf_c = llm::create_f32_tensor(ctx, shape, 1)!!,
    };
}

// --- Forward Pass Helpers ---

fn void dispatch_taesd_conv(
    CommandBuffer cmd,
    llm::DiffusionKernels* k,
    TAESDConv* conv,
    llm::Tensor* input,
    llm::Tensor* output,
    uint h, uint w
) {
    llm::Conv2dPC pc = {
        .in_c = conv.in_c,
        .out_c = conv.out_c,
        .in_h = h,
        .in_w = w,
        .kH = 3,
        .kW = 3,
        .stride = 1,
        .pad = 1,
        .groups = 1,
        .out_h = h,
        .out_w = w,
        .has_bias = conv.has_bias ? 1 : 0,
    };
	llm::dispatch_conv2d(cmd, k, &conv.weight, &conv.bias, input, output, &pc);
}

fn void dispatch_taesd_relu(CommandBuffer cmd, llm::DiffusionKernels* k, llm::Tensor* buf, uint n) {
    llm::ReluPC pc = { .n = n };
llm::dispatch_kernel(cmd, &k.relu,
        { buf.gpu_buffer.buffer },
        { buf.size_bytes },
        &pc, llm::ceil_div(n, 256));
}

// TAEBlock: conv.0 → relu → conv.2 → relu → conv.4 → residual_add → relu
// Input in `input`, uses `temp1` and `temp2` as scratch
// Output written to `temp1`
fn void dispatch_taesd_block(
    CommandBuffer cmd,
    llm::DiffusionKernels* k,
    TAESDBlock* block,
    llm::Tensor* input,   // e.g. buf_a
    llm::Tensor* temp1,   // e.g. buf_b — will contain output
    llm::Tensor* temp2,   // e.g. buf_c — scratch
    uint h, uint w
) {
    uint ch = block.conv0.out_c;
    uint spatial = h * w;
    uint n = ch * spatial;

    // conv.0: input → temp1
    dispatch_taesd_conv(cmd, k, &block.conv0, input, temp1, h, w);
llm::compute_barrier(cmd);

    // relu(temp1)
    dispatch_taesd_relu(cmd, k, temp1, n);
llm::compute_barrier(cmd);

    // conv.2: temp1 → temp2
    dispatch_taesd_conv(cmd, k, &block.conv2, temp1, temp2, h, w);
llm::compute_barrier(cmd);

    // relu(temp2)
    dispatch_taesd_relu(cmd, k, temp2, n);
llm::compute_barrier(cmd);

    // conv.4: temp2 → temp1
    dispatch_taesd_conv(cmd, k, &block.conv4, temp2, temp1, h, w);
llm::compute_barrier(cmd);

    // residual: temp1 += input
    llm::ResidualPC res_pc = { .n = n };
llm::dispatch_kernel(cmd, &k.shared.residual_add,
        { temp1.gpu_buffer.buffer, input.gpu_buffer.buffer },
        { temp1.size_bytes, input.size_bytes },
        &res_pc, llm::ceil_div(n, 256));
llm::compute_barrier(cmd);

    // relu(temp1)
    dispatch_taesd_relu(cmd, k, temp1, n);
llm::compute_barrier(cmd);
}

// --- Full Forward Pass ---
// Input: latent [16, lat_h, lat_w] in acts.buf_a
// Output: image [3, 8*lat_h, 8*lat_w] in acts.buf_a

fn void? TAESDDecoder.forward(&self, TAESDActivations* acts, uint lat_h, uint lat_w) {
    llm::DeviceContext* ctx = self.ctx;
    CommandBuffer cmd = ctx.command_buffer;
    llm::DiffusionKernels* k = self.kernels;

    uint h = lat_h;
    uint w = lat_w;

llm::begin_compute(cmd)!!;

    // tanh_clamp: z = tanh(z/3) * 3  (in-place on buf_a)
    uint latent_n = TAESD_Z_CHANNELS * h * w;
    llm::TanhClampPC tc_pc = { .n = latent_n };
llm::dispatch_kernel(cmd, &k.tanh_clamp,
        { acts.buf_a.gpu_buffer.buffer },
        { acts.buf_a.size_bytes },
        &tc_pc, llm::ceil_div(latent_n, 256));
llm::compute_barrier(cmd);

    // Conv_in: buf_a(16ch) → buf_b(64ch)
    dispatch_taesd_conv(cmd, k, &self.conv_in, &acts.buf_a, &acts.buf_b, h, w);
llm::compute_barrier(cmd);

    // ReLU(buf_b)
    dispatch_taesd_relu(cmd, k, &acts.buf_b, TAESD_CHANNELS * h * w);
llm::compute_barrier(cmd);

    // Stage 1: 3 TAEBlocks, alternating buffers
    // Block 0: buf_b → buf_a (temp=buf_c)
    dispatch_taesd_block(cmd, k, &self.stage1[0], &acts.buf_b, &acts.buf_a, &acts.buf_c, h, w);
    // Block 1: buf_a → buf_b (temp=buf_c)
    dispatch_taesd_block(cmd, k, &self.stage1[1], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    // Block 2: buf_b → buf_a (temp=buf_c)
    dispatch_taesd_block(cmd, k, &self.stage1[2], &acts.buf_b, &acts.buf_a, &acts.buf_c, h, w);
    // Result in buf_a

    // Upsample 2x: buf_a → buf_b
    llm::UpsamplePC up_pc = { .channels = TAESD_CHANNELS, .in_h = h, .in_w = w };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { acts.buf_a.gpu_buffer.buffer, acts.buf_b.gpu_buffer.buffer },
        { acts.buf_a.size_bytes, acts.buf_b.size_bytes },
        &up_pc, llm::ceil_div(TAESD_CHANNELS * h * w * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; w *= 2;

    // Post-upsample conv: buf_b → buf_a
    dispatch_taesd_conv(cmd, k, &self.up1_conv, &acts.buf_b, &acts.buf_a, h, w);
llm::compute_barrier(cmd);

llm::submit_and_wait(ctx)!!;
llm::begin_compute(cmd)!!;

    // Stage 2: 3 TAEBlocks
    dispatch_taesd_block(cmd, k, &self.stage2[0], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    dispatch_taesd_block(cmd, k, &self.stage2[1], &acts.buf_b, &acts.buf_a, &acts.buf_c, h, w);
    dispatch_taesd_block(cmd, k, &self.stage2[2], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    // Result in buf_b

    // Upsample 2x: buf_b → buf_a
    up_pc = { .channels = TAESD_CHANNELS, .in_h = h, .in_w = w };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer },
        { acts.buf_b.size_bytes, acts.buf_a.size_bytes },
        &up_pc, llm::ceil_div(TAESD_CHANNELS * h * w * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; w *= 2;

    // Post-upsample conv: buf_a → buf_b
    dispatch_taesd_conv(cmd, k, &self.up2_conv, &acts.buf_a, &acts.buf_b, h, w);
llm::compute_barrier(cmd);

llm::submit_and_wait(ctx)!!;
llm::begin_compute(cmd)!!;

    // Stage 3: 3 TAEBlocks
    dispatch_taesd_block(cmd, k, &self.stage3[0], &acts.buf_b, &acts.buf_a, &acts.buf_c, h, w);
    dispatch_taesd_block(cmd, k, &self.stage3[1], &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    dispatch_taesd_block(cmd, k, &self.stage3[2], &acts.buf_b, &acts.buf_a, &acts.buf_c, h, w);
    // Result in buf_a

    // Upsample 2x: buf_a → buf_b
    up_pc = { .channels = TAESD_CHANNELS, .in_h = h, .in_w = w };
llm::dispatch_kernel(cmd, &k.upsample_nearest,
        { acts.buf_a.gpu_buffer.buffer, acts.buf_b.gpu_buffer.buffer },
        { acts.buf_a.size_bytes, acts.buf_b.size_bytes },
        &up_pc, llm::ceil_div(TAESD_CHANNELS * h * w * 4, 256));
llm::compute_barrier(cmd);
    h *= 2; w *= 2;

    // Post-upsample conv: buf_b → buf_a
    dispatch_taesd_conv(cmd, k, &self.up3_conv, &acts.buf_b, &acts.buf_a, h, w);
llm::compute_barrier(cmd);

llm::submit_and_wait(ctx)!!;
llm::begin_compute(cmd)!!;

    // Final TAEBlock: buf_a → buf_b (temp=buf_c)
    dispatch_taesd_block(cmd, k, &self.final_block, &acts.buf_a, &acts.buf_b, &acts.buf_c, h, w);
    // Result in buf_b

    // Conv_out: buf_b(64ch) → buf_a(3ch)
    dispatch_taesd_conv(cmd, k, &self.conv_out, &acts.buf_b, &acts.buf_a, h, w);
llm::compute_barrier(cmd);

    // Clamp to [0, 1]: buf_a → buf_b → copy back to buf_a
    uint out_n = TAESD_OUT_CHANNELS * h * w;
    llm::ScaleShiftPC ss_pc = { .n = out_n, .scale = 1.0f, .shift = 0.0f };
llm::dispatch_kernel(cmd, &k.scale_shift_clamp,
        { acts.buf_a.gpu_buffer.buffer, acts.buf_b.gpu_buffer.buffer },
        { acts.buf_a.size_bytes, acts.buf_b.size_bytes },
        &ss_pc, llm::ceil_div(out_n, 256));
llm::compute_barrier(cmd);

    // Copy result to buf_a
    vk::cmdCopyBuffer(cmd, acts.buf_b.gpu_buffer.buffer, acts.buf_a.gpu_buffer.buffer, 1,
        (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = (ulong)out_n * 4 }});

llm::submit_and_wait(ctx)!!;
    // Image [3, h, w] is now in buf_a
}

// --- Free ---

fn void TAESDConv.free(&self) {
    if (self.weight.size_bytes > 0) self.weight.free();
    if (self.has_bias && self.bias.size_bytes > 0) self.bias.free();
}

fn void TAESDBlock.free(&self) {
    self.conv0.free();
    self.conv2.free();
    self.conv4.free();
}

fn void TAESDDecoder.free(&self) {
    self.conv_in.free();
    for (uint i = 0; i < TAESD_NUM_BLOCKS; i++) {
        self.stage1[i].free();
        self.stage2[i].free();
        self.stage3[i].free();
    }
    self.up1_conv.free();
    self.up2_conv.free();
    self.up3_conv.free();
    self.final_block.free();
    self.conv_out.free();
}

fn void TAESDActivations.free(&self) {
    self.buf_a.free();
    self.buf_b.free();
    self.buf_c.free();
}
