<*
 The SkinTokens mesh encoder: a point cloud in, 256 latents out.

 Architecture, read off the checkpoint rather than a paper — a
 Michelangelo/3DShape2VecSet perceiver at width 512 with 8 heads:

     input_proj      54 -> 512, applied to every point and every query
     cross_attn      256 queries attend into all 54000 points, then an MLP
     self_attn       8 residual blocks over the 256 latents alone
     ln_post
     output_proj     512 -> 896 -> RMSNorm, the handoff into the LLM

 The one asymmetry worth knowing: the encoder has no learned query parameters.
 Its queries are the projected features of 256 points picked out of the cloud by
 farthest point sampling, which is why `prepare_shape_input` hands back both a
 `features` and a `query` array — they go into the same `input_proj`.

     `MeshEncoder encoder = mesh::load_mesh_encoder(&ctx, path)!;`
     `defer encoder.free();`
     `float[] latents = encoder.encode(mem, &input)!;   // 256 x 896`
*>
module llm::mesh;

import vk;
import llm;
import std::io;
import std::core::mem;
import std::core::mem::allocator;
import std::core::string;
import std::math;

const char[*] MESH_ENCODER_SPV = $embed("../shaders/mesh_encoder.spv");
const char[*] MESH_ATTENTION_SPV = $embed("../shaders/mesh_attention.spv");

faultdef ENCODER_SHAPE_MISMATCH, ENCODER_TOO_MANY_POINTS;

const uint ENCODER_WIDTH = 512;
const uint ENCODER_HEADS = 8;
const uint ENCODER_HEAD_DIM = 64;
const uint ENCODER_BLOCKS = 8;
const uint ENCODER_MLP_DIM = 2048;
const uint ENCODER_OUT_DIM = 896;

// torch.nn.LayerNorm's default. The bridge norm is Qwen3's, which uses 1e-6.
const float LN_EPS = 1e-5f;
const float RMS_EPS = 1e-6f;

// Attention is tiled 16 queries at a time; see mesh_attention.slang.
const uint ATTN_QUERY_TILE = 16;

// --- Push constants, matching the Slang structs ---

struct MeshLayerNormPC { uint dim; uint n_rows; float eps; uint has_bias; }
struct MeshRmsNormPC { uint dim; uint n_rows; float eps; uint pad; }
struct MeshLinearPC { uint m; uint n; uint k; uint has_bias; }
struct MeshGeluPC { uint n; }
struct MeshAddPC { uint n; }

struct MeshAttentionPC {
    uint n_heads;
    uint q_len;
    uint kv_len;
    uint q_row_stride;
    uint q_head_stride;
    uint q_offset;
    uint kv_row_stride;
    uint kv_head_stride;
    uint k_offset;
    uint v_offset;
    uint out_row_stride;
    float scale;
}

struct MeshEncoderKernels {
    llm::ComputeKernel layernorm;
    llm::ComputeKernel rmsnorm;
    llm::ComputeKernel linear;
    llm::ComputeKernel gelu;
    llm::ComputeKernel residual_add;
    llm::ComputeKernel attention;
}

fn MeshEncoderKernels? create_encoder_kernels(llm::DeviceContext* ctx) {
    char[] encoder_spv = &MESH_ENCODER_SPV;
    vk::ShaderModule encoder_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(encoder_spv.len)
        .setCode((uint*)&encoder_spv[0])
        .build(ctx.device)!!;
    defer encoder_shader.free(ctx.device);

    char[] attention_spv = &MESH_ATTENTION_SPV;
    vk::ShaderModule attention_shader = vk::shaderModuleCreateInfo()
        .setCodeSize(attention_spv.len)
        .setCode((uint*)&attention_spv[0])
        .build(ctx.device)!!;
    defer attention_shader.free(ctx.device);

    return {
        .layernorm    = llm::create_kernel(ctx, encoder_shader, 4, MeshLayerNormPC::size, "mesh_layernorm")!!,
        .rmsnorm      = llm::create_kernel(ctx, encoder_shader, 3, MeshRmsNormPC::size, "mesh_rmsnorm")!!,
        .linear       = llm::create_kernel(ctx, encoder_shader, 4, MeshLinearPC::size, "mesh_linear")!!,
        .gelu         = llm::create_kernel(ctx, encoder_shader, 1, MeshGeluPC::size, "mesh_gelu")!!,
        .residual_add = llm::create_kernel(ctx, encoder_shader, 2, MeshAddPC::size, "mesh_residual_add")!!,
        .attention    = llm::create_kernel(ctx, attention_shader, 3, MeshAttentionPC::size, "mesh_attention")!!,
    };
}

fn void MeshEncoderKernels.free(&self, vk::Device device) {
    self.layernorm.free(device);
    self.rmsnorm.free(device);
    self.linear.free(device);
    self.gelu.free(device);
    self.residual_add.free(device);
    self.attention.free(device);
}

// --- Weights ---

<*
 One residual self-attention block. `qkv` is the fused projection: per head it
 emits q, k and v back to back, which is what the attention kernel's stride
 parameters are there to walk.
*>
struct SelfAttnBlock {
    llm::Tensor ln_1_weight;
    llm::Tensor ln_1_bias;
    llm::Tensor qkv_weight;
    llm::Tensor proj_weight;
    llm::Tensor proj_bias;
    llm::Tensor ln_2_weight;
    llm::Tensor ln_2_bias;
    llm::Tensor fc_weight;
    llm::Tensor fc_bias;
    llm::Tensor out_weight;
    llm::Tensor out_bias;
}

// Tolerates a half-built struct: a load that fails partway leaves the rest
// zeroed, and handing a zeroed vk::Memory to free() is not survivable.
fn void release_tensor(llm::Tensor* tensor) @local {
    if (tensor.size_bytes > 0) tensor.free();
    *tensor = {};
}

fn void SelfAttnBlock.free(&self) {
    llm::Tensor*[11] all = {
        &self.ln_1_weight, &self.ln_1_bias, &self.qkv_weight,
        &self.proj_weight, &self.proj_bias,
        &self.ln_2_weight, &self.ln_2_bias,
        &self.fc_weight, &self.fc_bias, &self.out_weight, &self.out_bias,
    };
    foreach (tensor : all) release_tensor(tensor);
}

struct MeshEncoderWeights {
    llm::Tensor input_proj_weight;
    llm::Tensor input_proj_bias;

    llm::Tensor cross_ln_1_weight;
    llm::Tensor cross_ln_1_bias;
    llm::Tensor cross_ln_2_weight;
    llm::Tensor cross_ln_2_bias;
    llm::Tensor cross_ln_3_weight;
    llm::Tensor cross_ln_3_bias;
    llm::Tensor cross_q_weight;
    llm::Tensor cross_kv_weight;
    llm::Tensor cross_proj_weight;
    llm::Tensor cross_proj_bias;
    llm::Tensor cross_fc_weight;
    llm::Tensor cross_fc_bias;
    llm::Tensor cross_out_weight;
    llm::Tensor cross_out_bias;

    SelfAttnBlock[ENCODER_BLOCKS] blocks;

    llm::Tensor ln_post_weight;
    llm::Tensor ln_post_bias;
    llm::Tensor out_proj_weight;
    llm::Tensor out_proj_bias;
    llm::Tensor out_norm_weight;
}

fn void MeshEncoderWeights.free(&self) {
    llm::Tensor*[21] all = {
        &self.input_proj_weight, &self.input_proj_bias,
        &self.cross_ln_1_weight, &self.cross_ln_1_bias,
        &self.cross_ln_2_weight, &self.cross_ln_2_bias,
        &self.cross_ln_3_weight, &self.cross_ln_3_bias,
        &self.cross_q_weight, &self.cross_kv_weight,
        &self.cross_proj_weight, &self.cross_proj_bias,
        &self.cross_fc_weight, &self.cross_fc_bias,
        &self.cross_out_weight, &self.cross_out_bias,
        &self.ln_post_weight, &self.ln_post_bias,
        &self.out_proj_weight, &self.out_proj_bias, &self.out_norm_weight,
    };
    foreach (tensor : all) release_tensor(tensor);
    foreach (&block : self.blocks) block.free();
}

// --- Activations ---
//
// Sized for one call. The two point-sized buffers dominate: at 54000 points
// that is 110 MB for the projected cloud and 221 MB for its keys and values.
// Everything else is per-latent and rounds to nothing.

struct MeshEncoderActivations {
    llm::Tensor point_features;   // [n_points, 54]
    llm::Tensor query_features;   // [n_latents, 54]
    llm::Tensor data;             // [n_points, 512], normalised in place
    llm::Tensor kv;               // [n_points, 1024]
    llm::Tensor latents;          // [n_latents, 512]
    llm::Tensor norm;             // [n_latents, 512]
    llm::Tensor query;            // [n_latents, 512]
    llm::Tensor qkv;              // [n_latents, 1536]
    llm::Tensor attn;             // [n_latents, 512]
    llm::Tensor delta;            // [n_latents, 512]
    llm::Tensor hidden;           // [n_latents, 2048]
    llm::Tensor output;           // [n_latents, 896]
    uint n_points;
    uint n_latents;
}

fn void MeshEncoderActivations.free(&self) {
    llm::Tensor*[12] all = {
        &self.point_features, &self.query_features, &self.data, &self.kv,
        &self.latents, &self.norm, &self.query, &self.qkv,
        &self.attn, &self.delta, &self.hidden, &self.output,
    };
    foreach (tensor : all) release_tensor(tensor);
}

struct MeshEncoder {
    llm::DeviceContext* ctx;
    MeshEncoderKernels kernels;
    MeshEncoderWeights weights;
    MeshEncoderActivations acts;
    bool has_acts;
}

// --- Loading ---

fn llm::Tensor? load_one(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, String name) @local {
    @pool() {
        return llm::upload_safetensor_f32(ctx, sf, string::tformat("mesh_encoder.encoder.%s", name));
    };
}

// One name at a time, each under its own pool. Formatting all eleven up front
// and loading them in a single struct literal reads better and crashes: the
// nested pools do not survive being interleaved with the literal's evaluation.
fn llm::Tensor? load_block_tensor(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, uint index, String suffix) @local {
    @pool() {
        return llm::upload_safetensor_f32(ctx, sf,
            string::tformat("mesh_encoder.encoder.self_attn.resblocks.%d.%s", index, suffix));
    };
}

fn SelfAttnBlock? load_block(llm::DeviceContext* ctx, llm::SafetensorsFile* sf, uint index) @local {
    SelfAttnBlock block;
    defer catch block.free();

    block.ln_1_weight = load_block_tensor(ctx, sf, index, "ln_1.weight")!;
    block.ln_1_bias   = load_block_tensor(ctx, sf, index, "ln_1.bias")!;
    block.qkv_weight  = load_block_tensor(ctx, sf, index, "attn.c_qkv.weight")!;
    block.proj_weight = load_block_tensor(ctx, sf, index, "attn.c_proj.weight")!;
    block.proj_bias   = load_block_tensor(ctx, sf, index, "attn.c_proj.bias")!;
    block.ln_2_weight = load_block_tensor(ctx, sf, index, "ln_2.weight")!;
    block.ln_2_bias   = load_block_tensor(ctx, sf, index, "ln_2.bias")!;
    block.fc_weight   = load_block_tensor(ctx, sf, index, "mlp.c_fc.weight")!;
    block.fc_bias     = load_block_tensor(ctx, sf, index, "mlp.c_fc.bias")!;
    block.out_weight  = load_block_tensor(ctx, sf, index, "mlp.c_proj.weight")!;
    block.out_bias    = load_block_tensor(ctx, sf, index, "mlp.c_proj.bias")!;

    return block;
}

<*
 Load the encoder from a converted SkinTokens checkpoint.

 The file is the `.mesh_encoder.safetensors` split written by
 `tools/convert_ckpt.py`; its weights are BF16 and are widened to F32 on upload.
*>
fn MeshEncoder? load_mesh_encoder(llm::DeviceContext* ctx, String path) {
    llm::SafetensorsFile sf = llm::safetensors_open(path)!;
    defer sf.close();

    MeshEncoder encoder = { .ctx = ctx };
    encoder.kernels = create_encoder_kernels(ctx)!;
    defer catch encoder.kernels.free(ctx.device);

    encoder.weights = {
        .input_proj_weight = load_one(ctx, &sf, "input_proj.weight")!,
        .input_proj_bias   = load_one(ctx, &sf, "input_proj.bias")!,

        .cross_ln_1_weight = load_one(ctx, &sf, "cross_attn.ln_1.weight")!,
        .cross_ln_1_bias   = load_one(ctx, &sf, "cross_attn.ln_1.bias")!,
        .cross_ln_2_weight = load_one(ctx, &sf, "cross_attn.ln_2.weight")!,
        .cross_ln_2_bias   = load_one(ctx, &sf, "cross_attn.ln_2.bias")!,
        .cross_ln_3_weight = load_one(ctx, &sf, "cross_attn.ln_3.weight")!,
        .cross_ln_3_bias   = load_one(ctx, &sf, "cross_attn.ln_3.bias")!,
        .cross_q_weight    = load_one(ctx, &sf, "cross_attn.attn.c_q.weight")!,
        .cross_kv_weight   = load_one(ctx, &sf, "cross_attn.attn.c_kv.weight")!,
        .cross_proj_weight = load_one(ctx, &sf, "cross_attn.attn.c_proj.weight")!,
        .cross_proj_bias   = load_one(ctx, &sf, "cross_attn.attn.c_proj.bias")!,
        .cross_fc_weight   = load_one(ctx, &sf, "cross_attn.mlp.c_fc.weight")!,
        .cross_fc_bias     = load_one(ctx, &sf, "cross_attn.mlp.c_fc.bias")!,
        .cross_out_weight  = load_one(ctx, &sf, "cross_attn.mlp.c_proj.weight")!,
        .cross_out_bias    = load_one(ctx, &sf, "cross_attn.mlp.c_proj.bias")!,

        .ln_post_weight = load_one(ctx, &sf, "ln_post.weight")!,
        .ln_post_bias   = load_one(ctx, &sf, "ln_post.bias")!,

        // The bridge into the LLM sits outside the encoder namespace.
        .out_proj_weight = llm::upload_safetensor_f32(ctx, &sf, "output_proj.0.weight")!,
        .out_proj_bias   = llm::upload_safetensor_f32(ctx, &sf, "output_proj.0.bias")!,
        .out_norm_weight = llm::upload_safetensor_f32(ctx, &sf, "output_proj.1.weight")!,
    };
    defer catch encoder.weights.free();

    for (uint i = 0; i < ENCODER_BLOCKS; i++) {
        encoder.weights.blocks[i] = load_block(ctx, &sf, i)!;
    }

    return encoder;
}

fn void MeshEncoder.free(&self) {
    if (self.has_acts) self.acts.free();
    self.weights.free();
    self.kernels.free(self.ctx.device);
    *self = {};
}

// --- Activation allocation ---

fn llm::Tensor? matrix(llm::DeviceContext* ctx, uint rows, uint cols) @local
    => llm::create_f32_tensor(ctx, { (ulong)cols, (ulong)rows, 0, 0 }, 2);

fn void? MeshEncoder.reserve(&self, uint n_points, uint n_latents) {
    if (self.has_acts && self.acts.n_points == n_points && self.acts.n_latents == n_latents) return;
    if (self.has_acts) self.acts.free();
    self.has_acts = false;

    llm::DeviceContext* ctx = self.ctx;
    usz feature_dim = encoder_input_dim();

    self.acts = {
        .point_features = matrix(ctx, n_points, (uint)feature_dim)!,
        .query_features = matrix(ctx, n_latents, (uint)feature_dim)!,
        .data           = matrix(ctx, n_points, ENCODER_WIDTH)!,
        .kv             = matrix(ctx, n_points, 2 * ENCODER_WIDTH)!,
        .latents        = matrix(ctx, n_latents, ENCODER_WIDTH)!,
        .norm           = matrix(ctx, n_latents, ENCODER_WIDTH)!,
        .query          = matrix(ctx, n_latents, ENCODER_WIDTH)!,
        .qkv            = matrix(ctx, n_latents, 3 * ENCODER_WIDTH)!,
        .attn           = matrix(ctx, n_latents, ENCODER_WIDTH)!,
        .delta          = matrix(ctx, n_latents, ENCODER_WIDTH)!,
        .hidden         = matrix(ctx, n_latents, ENCODER_MLP_DIM)!,
        .output         = matrix(ctx, n_latents, ENCODER_OUT_DIM)!,
        .n_points = n_points,
        .n_latents = n_latents,
    };
    self.has_acts = true;
}

// --- Dispatch helpers ---
//
// Each wraps one kernel so the forward pass below reads as the network rather
// than as push-constant bookkeeping.

fn void MeshEncoder.layer_norm(
    &self, llm::RenderGraph* graph,
    llm::Tensor* input, llm::Tensor* weight, llm::Tensor* bias, llm::Tensor* output,
    uint rows, uint dim)
{
    MeshLayerNormPC pc = { .dim = dim, .n_rows = rows, .eps = LN_EPS, .has_bias = 1 };
    graph.dispatch(&self.kernels.layernorm, { input, weight, bias, output }, &pc, rows);
}

fn void MeshEncoder.rms_norm(
    &self, llm::RenderGraph* graph,
    llm::Tensor* input, llm::Tensor* weight, llm::Tensor* output,
    uint rows, uint dim)
{
    MeshRmsNormPC pc = { .dim = dim, .n_rows = rows, .eps = RMS_EPS, .pad = 0 };
    graph.dispatch(&self.kernels.rmsnorm, { input, weight, output }, &pc, rows);
}

<*
 out[rows, out_dim] = in[rows, in_dim] @ weight[out_dim, in_dim]^T (+ bias).

 `bias` is still bound when unused — a push descriptor has to point somewhere —
 so callers without one pass the weight again and the shader ignores it.
*>
fn void MeshEncoder.linear(
    &self, llm::RenderGraph* graph,
    llm::Tensor* input, llm::Tensor* weight, llm::Tensor* bias, llm::Tensor* output,
    uint rows, uint out_dim, uint in_dim, bool has_bias)
{
    MeshLinearPC pc = {
        .m = rows, .n = out_dim, .k = in_dim,
        .has_bias = has_bias ? 1 : 0,
    };
    graph.dispatch(
        &self.kernels.linear,
        { input, weight, has_bias ? bias : weight, output },
        &pc,
        llm::ceil_div(out_dim, 16), llm::ceil_div(rows, 16),
    );
}

fn void MeshEncoder.gelu(&self, llm::RenderGraph* graph, llm::Tensor* data, uint count) {
    MeshGeluPC pc = { .n = count };
    graph.dispatch(&self.kernels.gelu, { data }, &pc, llm::ceil_div(count, 256));
}

fn void MeshEncoder.residual_add(&self, llm::RenderGraph* graph, llm::Tensor* acc, llm::Tensor* delta, uint count) {
    MeshAddPC pc = { .n = count };
    graph.dispatch(&self.kernels.residual_add, { acc, delta }, &pc, llm::ceil_div(count, 256));
}

fn void MeshEncoder.attention(&self, llm::RenderGraph* graph, MeshAttentionPC pc, llm::Tensor* q, llm::Tensor* kv, llm::Tensor* out) {
    graph.dispatch(
        &self.kernels.attention,
        { q, kv, out },
        &pc,
        llm::ceil_div(pc.q_len, ATTN_QUERY_TILE), ENCODER_HEADS,
    );
}

// Michelangelo scales q and k by 1/sqrt(sqrt(head_dim)) each, which is the
// usual 1/sqrt(head_dim) on the product.
fn float attention_scale() @local => 1.0f / (float)math::sqrt((double)ENCODER_HEAD_DIM);

// --- Forward pass ---

<*
 Run the encoder and read the latents back.

 @param input : `From prepare_shape_input; its features and query feed the same input_proj.`
 @return `n_latents x 896, row-major — the prefix embeddings the LLM is conditioned on.`
 @return? ENCODER_SHAPE_MISMATCH : `The input was embedded at a width this encoder does not take.`
 @return? ENCODER_TOO_MANY_POINTS : `Point or latent count does not fit a uint.`
*>
fn float[]? MeshEncoder.encode(&self, alloc::Allocator allocator, ShapeEncoderInput* input) {
    if (input.dim != encoder_input_dim()) return ENCODER_SHAPE_MISMATCH~;
    if (input.count > 0xFFFFFFFF || input.latents > 0xFFFFFFFF) return ENCODER_TOO_MANY_POINTS~;
    if (input.count == 0 || input.latents == 0) return ENCODER_SHAPE_MISMATCH~;

    uint n_points = (uint)input.count;
    uint n_latents = (uint)input.latents;
    uint width = ENCODER_WIDTH;

    self.reserve(n_points, n_latents)!;
    MeshEncoderActivations* acts = &self.acts;
    MeshEncoderWeights* w = &self.weights;

    llm::upload_to_tensor(self.ctx, &acts.point_features, input.features)!;
    llm::upload_to_tensor(self.ctx, &acts.query_features, input.query)!;

    llm::RenderGraph graph;
    graph.init(self.ctx)!;

    uint feature_dim = (uint)input.dim;
    uint latent_elements = n_latents * width;

    // The cloud and the queries go through one and the same projection; the
    // queries are just rows of the cloud that farthest point sampling picked,
    // and a per-row linear commutes with that gather.
    self.linear(&graph, &acts.point_features, &w.input_proj_weight, &w.input_proj_bias, &acts.data, n_points, width, feature_dim, true);
    self.linear(&graph, &acts.query_features, &w.input_proj_weight, &w.input_proj_bias, &acts.latents, n_latents, width, feature_dim, true);

    // Point-sized work is the expensive part of the whole encoder; give it its
    // own submission so a slow device does not trip a watchdog mid-graph.
    graph.checkpoint()!;

    // --- Cross attention: x = x + attn(ln_1(x), ln_2(data)) ---

    self.layer_norm(&graph, &acts.latents, &w.cross_ln_1_weight, &w.cross_ln_1_bias, &acts.norm, n_latents, width);
    // In place: every thread rewrites only the elements it read, and it reads
    // them for the last time in the same statement.
    self.layer_norm(&graph, &acts.data, &w.cross_ln_2_weight, &w.cross_ln_2_bias, &acts.data, n_points, width);
    self.linear(&graph, &acts.norm, &w.cross_q_weight, null, &acts.query, n_latents, width, width, false);
    self.linear(&graph, &acts.data, &w.cross_kv_weight, null, &acts.kv, n_points, 2 * width, width, false);

    graph.checkpoint()!;

    self.attention(&graph, {
        .n_heads = ENCODER_HEADS,
        .q_len = n_latents,
        .kv_len = n_points,
        .q_row_stride = width,
        .q_head_stride = ENCODER_HEAD_DIM,
        .q_offset = 0,
        // c_kv emits k then v for each head in turn.
        .kv_row_stride = 2 * width,
        .kv_head_stride = 2 * ENCODER_HEAD_DIM,
        .k_offset = 0,
        .v_offset = ENCODER_HEAD_DIM,
        .out_row_stride = width,
        .scale = attention_scale(),
    }, &acts.query, &acts.kv, &acts.attn);

    self.linear(&graph, &acts.attn, &w.cross_proj_weight, &w.cross_proj_bias, &acts.delta, n_latents, width, width, true);
    self.residual_add(&graph, &acts.latents, &acts.delta, latent_elements);

    self.feed_forward(&graph, &w.cross_ln_3_weight, &w.cross_ln_3_bias, &w.cross_fc_weight, &w.cross_fc_bias, &w.cross_out_weight, &w.cross_out_bias, n_latents);

    graph.checkpoint()!;

    // --- Self attention stack, over the 256 latents alone ---

    for (uint i = 0; i < ENCODER_BLOCKS; i++) {
        SelfAttnBlock* block = &w.blocks[i];

        self.layer_norm(&graph, &acts.latents, &block.ln_1_weight, &block.ln_1_bias, &acts.norm, n_latents, width);
        self.linear(&graph, &acts.norm, &block.qkv_weight, null, &acts.qkv, n_latents, 3 * width, width, false);

        self.attention(&graph, {
            .n_heads = ENCODER_HEADS,
            .q_len = n_latents,
            .kv_len = n_latents,
            // One fused buffer bound as both Q and KV; c_qkv emits q, k, v for
            // each head in turn, so the three only differ by their offset.
            .q_row_stride = 3 * width,
            .q_head_stride = 3 * ENCODER_HEAD_DIM,
            .q_offset = 0,
            .kv_row_stride = 3 * width,
            .kv_head_stride = 3 * ENCODER_HEAD_DIM,
            .k_offset = ENCODER_HEAD_DIM,
            .v_offset = 2 * ENCODER_HEAD_DIM,
            .out_row_stride = width,
            .scale = attention_scale(),
        }, &acts.qkv, &acts.qkv, &acts.attn);

        self.linear(&graph, &acts.attn, &block.proj_weight, &block.proj_bias, &acts.delta, n_latents, width, width, true);
        self.residual_add(&graph, &acts.latents, &acts.delta, latent_elements);

        self.feed_forward(&graph, &block.ln_2_weight, &block.ln_2_bias, &block.fc_weight, &block.fc_bias, &block.out_weight, &block.out_bias, n_latents);
    }

    // --- Out ---

    self.layer_norm(&graph, &acts.latents, &w.ln_post_weight, &w.ln_post_bias, &acts.latents, n_latents, width);
    self.linear(&graph, &acts.latents, &w.out_proj_weight, &w.out_proj_bias, &acts.output, n_latents, ENCODER_OUT_DIM, width, true);
    self.rms_norm(&graph, &acts.output, &w.out_norm_weight, &acts.output, n_latents, ENCODER_OUT_DIM);

    graph.flush()!;

    return llm::download_tensor(allocator, self.ctx, &acts.output, (usz)n_latents * ENCODER_OUT_DIM);
}

// x = x + mlp(norm(x)), shared by the cross attention block and every self
// attention block — they differ only in which weights they reach for.
fn void MeshEncoder.feed_forward(
    &self, llm::RenderGraph* graph,
    llm::Tensor* norm_weight, llm::Tensor* norm_bias,
    llm::Tensor* fc_weight, llm::Tensor* fc_bias,
    llm::Tensor* out_weight, llm::Tensor* out_bias,
    uint rows)
{
    MeshEncoderActivations* acts = &self.acts;

    self.layer_norm(graph, &acts.latents, norm_weight, norm_bias, &acts.norm, rows, ENCODER_WIDTH);
    self.linear(graph, &acts.norm, fc_weight, fc_bias, &acts.hidden, rows, ENCODER_MLP_DIM, ENCODER_WIDTH, true);
    self.gelu(graph, &acts.hidden, rows * ENCODER_MLP_DIM);
    self.linear(graph, &acts.hidden, out_weight, out_bias, &acts.delta, rows, ENCODER_WIDTH, ENCODER_MLP_DIM, true);
    self.residual_add(graph, &acts.latents, &acts.delta, rows * ENCODER_WIDTH);
}
