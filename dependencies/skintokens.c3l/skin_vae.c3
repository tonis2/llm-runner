<*
 The SkinTokens skin decoder: a bone's four codes in, a weight per vertex out.

 The transformer emits a skeleton and then, in the same sequence, four tokens
 per bone drawn from a 32768-entry codebook. Those are not weights — they are
 indices into an FSQ latent space, and turning them back into a weight field
 takes the checkpoint's conditional VAE decoder:

     FSQ.project_out    index -> 5 levels -> a 512-wide latent, 4 per bone
     cond_encoder       the point cloud -> 384 conditioning latents
     post_quant         [4 bone latents ; 384 cond latents] -> 388 x 768
     decoder self       10 blocks over those 388 tokens -> the bone's context
     decoder cross      a query point attends into that context -> one logit
     sigmoid            -> this bone's influence on this point, in [0, 1]

 The last two steps are a coordinate field: queries do not attend to each
 other, so a point's weight depends on nothing but the point and the bone. That
 is what makes it worth evaluating at the mesh's own vertices, where the
 reference evaluates it at the 54000 sampled points and then carries the result
 to the vertices with an eight-neighbour inverse-distance blend. Same field,
 read directly instead of resampled, and usually far less of it — the cost is
 vertices x bones either way.

 Everything here runs on the six kernels in `mesh_encoder.slang`, the same ones
 the shape encoder uses. The architecture is different — a Hunyuan-style DiT
 block rather than Michelangelo's — but at the level of layernorm, matmul,
 gelu, add and attention the two networks want nothing different.

     `SkinDecoder decoder = skintokens::load_skin_decoder(&ctx, path)!;`
     `defer decoder.free();`
     `float[] skin = decoder.decode(mem, &shape, codes, bones, verts, norms)!;`
*>
module skintokens;

import vk;
import llm;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;
import std::core::mem::allocator;
import std::math;
import std::math::vector;

faultdef SKIN_SHAPE_MISMATCH, SKIN_NO_BONES, SKIN_BAD_CODE;

// Read off the checkpoint's hyper-parameters. `model_width_decoder` and
// `model_width_encoder` are both 768 here, and 768/12 gives the head_dim of 64
// that mesh_attention.slang is compiled for.
const uint VAE_WIDTH = 768;
const uint VAE_HEADS = 12;
const uint VAE_FF_DIM = 4 * VAE_WIDTH;
const uint VAE_LATENT = 512;

// `model_num_layers_encoder` / `model_num_layers_decoder`. Both stacks carry
// one cross-attention block on top of these, at opposite ends: the conditioning
// encoder leads with it, the decoder finishes with it.
const uint VAE_COND_SELF_BLOCKS = 2;
const uint VAE_DECODER_SELF_BLOCKS = 10;

// `model_config_tokens_skin_cond` and `model_config_tokens_per_skin`.
const uint VAE_COND_TOKENS = 384;
const uint TOKENS_PER_SKIN = 4;

// `model_FSQ_dict_levels` is [8, 8, 8, 8, 8]: five dimensions, eight levels
// each, so 8^5 = 32768 codes — exactly the skin slice of the vocabulary.
const uint FSQ_DIM = 5;
const uint FSQ_LEVELS = 8;

// Rows the decoder's self-attention stack runs over: one bone's codes on top
// of the shared conditioning.
const uint VAE_CONTEXT_ROWS = TOKENS_PER_SKIN + VAE_COND_TOKENS;

<*
 Query points evaluated per pass.

 Only the query side of the decoder scales with the mesh, and it scales into
 the widest buffer in the network — the feed-forward's 3072 columns. Chunking
 caps that at 50 MB no matter how dense the mesh is, and costs nothing:
 queries never see each other, so a chunk boundary is not an approximation.
*>
const uint QUERY_CHUNK = 4096;

// --- Weights ---------------------------------------------------------------

<*
 A DiT block that attends to itself: norm1 -> attn1 -> norm3 -> feed-forward,
 both halves residual.

 `qkv_weight` is the block's `to_q`, `to_k` and `to_v` stacked into one
 [3*768, 768] matrix. That is not a packing convenience — it is the layout the
 weights were trained in. The reference concatenates the three projections and
 *then* splits the result into heads, so head 4's queries are the first 64
 columns of `to_k`'s output rather than anything `to_q` produced. Reading the
 three separately and slicing each into 12 heads gives a different, wrong
 assignment; stacking them and striding by 3*64 per head gives the trained one.
*>
struct DiTSelfBlock {
    llm::Tensor norm1_weight;
    llm::Tensor norm1_bias;
    llm::Tensor qkv_weight;
    llm::Tensor attn_out_weight;
    llm::Tensor attn_out_bias;
    llm::Tensor norm3_weight;
    llm::Tensor norm3_bias;
    llm::Tensor ff_in_weight;
    llm::Tensor ff_in_bias;
    llm::Tensor ff_out_weight;
    llm::Tensor ff_out_bias;
}

fn void DiTSelfBlock.free(&self) {
    llm::Tensor*[11] all = {
        &self.norm1_weight, &self.norm1_bias, &self.qkv_weight,
        &self.attn_out_weight, &self.attn_out_bias,
        &self.norm3_weight, &self.norm3_bias,
        &self.ff_in_weight, &self.ff_in_bias,
        &self.ff_out_weight, &self.ff_out_bias,
    };
    foreach (tensor : all) release_tensor(tensor);
}

<*
 A DiT block that attends into somewhere else: norm2 -> attn2 -> norm3 -> ff.

 `norm_cross` normalises the keys and values before they are projected, so it
 belongs to whatever is being attended into rather than to the queries.
 `kv_weight` stacks `to_k` and `to_v` for the same reason `qkv_weight` stacks
 three; the queries keep their own matrix because a cross block splits them
 into heads the ordinary way.
*>
struct DiTCrossBlock {
    llm::Tensor norm2_weight;
    llm::Tensor norm2_bias;
    llm::Tensor norm_cross_weight;
    llm::Tensor norm_cross_bias;
    llm::Tensor q_weight;
    llm::Tensor kv_weight;
    llm::Tensor attn_out_weight;
    llm::Tensor attn_out_bias;
    llm::Tensor norm3_weight;
    llm::Tensor norm3_bias;
    llm::Tensor ff_in_weight;
    llm::Tensor ff_in_bias;
    llm::Tensor ff_out_weight;
    llm::Tensor ff_out_bias;
}

fn void DiTCrossBlock.free(&self) {
    llm::Tensor*[14] all = {
        &self.norm2_weight, &self.norm2_bias,
        &self.norm_cross_weight, &self.norm_cross_bias,
        &self.q_weight, &self.kv_weight,
        &self.attn_out_weight, &self.attn_out_bias,
        &self.norm3_weight, &self.norm3_bias,
        &self.ff_in_weight, &self.ff_in_bias,
        &self.ff_out_weight, &self.ff_out_bias,
    };
    foreach (tensor : all) release_tensor(tensor);
}

struct SkinDecoderWeights {
    // Conditioning encoder: the point cloud down to 384 latents.
    llm::Tensor cond_proj_in_weight;
    llm::Tensor cond_proj_in_bias;
    DiTCrossBlock cond_cross;
    DiTSelfBlock[VAE_COND_SELF_BLOCKS] cond_blocks;
    llm::Tensor cond_norm_out_weight;
    llm::Tensor cond_norm_out_bias;
    llm::Tensor cond_quant_weight;
    llm::Tensor cond_quant_bias;

    // Codes back to latents, and latents up to the decoder's width.
    llm::Tensor fsq_out_weight;
    llm::Tensor fsq_out_bias;
    llm::Tensor post_quant_weight;
    llm::Tensor post_quant_bias;

    // Decoder: 10 self-attention blocks build a bone's context, one cross
    // block reads it from a query point.
    DiTSelfBlock[VAE_DECODER_SELF_BLOCKS] decoder_blocks;
    DiTCrossBlock decoder_cross;
    llm::Tensor proj_query_weight;
    llm::Tensor proj_query_bias;
    llm::Tensor out_norm_weight;
    llm::Tensor out_norm_bias;
    llm::Tensor proj_out_weight;
    llm::Tensor proj_out_bias;
}

fn void SkinDecoderWeights.free(&self) {
    llm::Tensor*[16] all = {
        &self.cond_proj_in_weight, &self.cond_proj_in_bias,
        &self.cond_norm_out_weight, &self.cond_norm_out_bias,
        &self.cond_quant_weight, &self.cond_quant_bias,
        &self.fsq_out_weight, &self.fsq_out_bias,
        &self.post_quant_weight, &self.post_quant_bias,
        &self.proj_query_weight, &self.proj_query_bias,
        &self.out_norm_weight, &self.out_norm_bias,
        &self.proj_out_weight, &self.proj_out_bias,
    };
    foreach (tensor : all) release_tensor(tensor);
    self.cond_cross.free();
    self.decoder_cross.free();
    foreach (&block : self.cond_blocks) block.free();
    foreach (&block : self.decoder_blocks) block.free();
}

struct SkinDecoder {
    llm::DeviceContext* ctx;
    MeshKernels kernels;
    SkinDecoderWeights weights;
}

// --- Loading ---------------------------------------------------------------

fn llm::Tensor? load_vae_tensor(llm::DeviceContext* ctx, llm::GGUFFile* gf, String suffix) @local {
    char[TENSOR_NAME_MAX] buffer;
    String name = (String)io::bprintf(buffer[..], "vae.model.%s", suffix)!;
    return load_named(ctx, gf, name);
}

fn llm::Tensor? load_vae_block_tensor(
    llm::DeviceContext* ctx, llm::GGUFFile* gf,
    String stack, uint index, String suffix) @local
{
    char[TENSOR_NAME_MAX] buffer;
    String name = (String)io::bprintf(buffer[..],
        "vae.model.%s.blocks.%d.%s", stack, index, suffix)!;
    return load_named(ctx, gf, name);
}

<*
 Stack two or three [768, 768] projections into one matrix.

 The parts land back to back because a torch Linear stores its weight as
 [out, in] with `in` contiguous, so stacking along `out` is a plain byte
 append. Uploading each on its own and copying it into place beats
 concatenating on the host: the loader already knows how to turn one GGUF
 tensor into one device buffer, whatever dtype it was stored in.
*>
fn llm::Tensor? fuse_projections(
    llm::DeviceContext* ctx, llm::GGUFFile* gf,
    String stack, uint index, String attn, bool with_query) @local
{
    String[3] suffixes = { "to_q.weight", "to_k.weight", "to_v.weight" };
    usz first = with_query ? 0 : 1;

    llm::Tensor[3] parts;
    usz loaded = 0;
    defer {
        for (usz i = 0; i < loaded; i++) release_tensor(&parts[i]);
    }

    ulong total_bytes = 0;
    for (usz i = first; i < 3; i++) {
        char[TENSOR_NAME_MAX] buffer;
        String name = (String)io::bprintf(buffer[..],
            "vae.model.%s.blocks.%d.%s.%s", stack, index, attn, suffixes[i])!;
        parts[loaded] = load_named(ctx, gf, name)!;
        total_bytes += (ulong)parts[loaded].size_bytes;
        loaded++;
    }

    llm::Tensor fused = llm::create_f32_tensor(ctx, { total_bytes / 4, 0, 0, 0 }, 1)!;
    defer catch fused.free();

    llm::RenderGraph graph;
    graph.init(ctx)!;
    usz offset = 0;
    for (usz i = 0; i < loaded; i++) {
        graph.copy_region(&parts[i], 0, &fused, offset, (ulong)parts[i].size_bytes);
        offset += parts[i].size_bytes;
    }
    graph.flush()!;

    return fused;
}

fn DiTSelfBlock? load_self_block(
    llm::DeviceContext* ctx, llm::GGUFFile* gf, String stack, uint index) @local
{
    DiTSelfBlock block;
    defer catch block.free();

    block.norm1_weight     = load_vae_block_tensor(ctx, gf, stack, index, "norm1.weight")!;
    block.norm1_bias       = load_vae_block_tensor(ctx, gf, stack, index, "norm1.bias")!;
    block.qkv_weight       = fuse_projections(ctx, gf, stack, index, "attn1", true)!;
    block.attn_out_weight  = load_vae_block_tensor(ctx, gf, stack, index, "attn1.to_out.0.weight")!;
    block.attn_out_bias    = load_vae_block_tensor(ctx, gf, stack, index, "attn1.to_out.0.bias")!;
    block.norm3_weight     = load_vae_block_tensor(ctx, gf, stack, index, "norm3.weight")!;
    block.norm3_bias       = load_vae_block_tensor(ctx, gf, stack, index, "norm3.bias")!;
    block.ff_in_weight     = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.0.proj.weight")!;
    block.ff_in_bias       = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.0.proj.bias")!;
    block.ff_out_weight    = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.2.weight")!;
    block.ff_out_bias      = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.2.bias")!;

    return block;
}

fn DiTCrossBlock? load_cross_block(
    llm::DeviceContext* ctx, llm::GGUFFile* gf, String stack, uint index) @local
{
    DiTCrossBlock block;
    defer catch block.free();

    block.norm2_weight      = load_vae_block_tensor(ctx, gf, stack, index, "norm2.weight")!;
    block.norm2_bias        = load_vae_block_tensor(ctx, gf, stack, index, "norm2.bias")!;
    block.norm_cross_weight = load_vae_block_tensor(ctx, gf, stack, index, "attn2.norm_cross.weight")!;
    block.norm_cross_bias   = load_vae_block_tensor(ctx, gf, stack, index, "attn2.norm_cross.bias")!;
    block.q_weight          = load_vae_block_tensor(ctx, gf, stack, index, "attn2.to_q.weight")!;
    block.kv_weight         = fuse_projections(ctx, gf, stack, index, "attn2", false)!;
    block.attn_out_weight   = load_vae_block_tensor(ctx, gf, stack, index, "attn2.to_out.0.weight")!;
    block.attn_out_bias     = load_vae_block_tensor(ctx, gf, stack, index, "attn2.to_out.0.bias")!;
    block.norm3_weight      = load_vae_block_tensor(ctx, gf, stack, index, "norm3.weight")!;
    block.norm3_bias        = load_vae_block_tensor(ctx, gf, stack, index, "norm3.bias")!;
    block.ff_in_weight      = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.0.proj.weight")!;
    block.ff_in_bias        = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.0.proj.bias")!;
    block.ff_out_weight     = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.2.weight")!;
    block.ff_out_bias       = load_vae_block_tensor(ctx, gf, stack, index, "ff.net.2.bias")!;

    return block;
}

<*
 Load the skin decoder from a converted SkinTokens checkpoint.

 The same GGUF the mesh encoder and the transformer come out of. The VAE's own
 encoder is not in the file at all — `tools/convert_ckpt.py` drops it, because
 it exists to turn ground-truth weights into tokens and inference only ever
 goes the other way.

 Kernels are built per decoder rather than borrowed from the mesh encoder. Six
 more pipeline objects over the same two SPIR-V modules costs a few
 milliseconds at load and keeps the two networks' lifetimes independent, which
 matters because the encoder's activations are worth releasing before this runs.
*>
fn SkinDecoder? load_skin_decoder(llm::DeviceContext* ctx, String path) {
    mmap::FileMmap weights = file::mmap_open(path, "rb")!;
    defer (void)weights.destroy();

    llm::GGUFFile gf = llm::gguf_parse(weights.bytes())!;
    defer gf.free();

    SkinDecoder decoder = { .ctx = ctx };
    decoder.kernels = create_mesh_kernels(ctx)!;
    defer catch decoder.kernels.free(ctx.device);
    defer catch decoder.weights.free();

    decoder.weights.cond_proj_in_weight = load_vae_tensor(ctx, &gf, "cond_encoder.proj_in.weight")!;
    decoder.weights.cond_proj_in_bias   = load_vae_tensor(ctx, &gf, "cond_encoder.proj_in.bias")!;
    decoder.weights.cond_cross          = load_cross_block(ctx, &gf, "cond_encoder", 0)!;
    for (uint i = 0; i < VAE_COND_SELF_BLOCKS; i++) {
        decoder.weights.cond_blocks[i] = load_self_block(ctx, &gf, "cond_encoder", i + 1)!;
    }
    decoder.weights.cond_norm_out_weight = load_vae_tensor(ctx, &gf, "cond_encoder.norm_out.weight")!;
    decoder.weights.cond_norm_out_bias   = load_vae_tensor(ctx, &gf, "cond_encoder.norm_out.bias")!;
    decoder.weights.cond_quant_weight    = load_vae_tensor(ctx, &gf, "cond_quant.weight")!;
    decoder.weights.cond_quant_bias      = load_vae_tensor(ctx, &gf, "cond_quant.bias")!;

    decoder.weights.fsq_out_weight    = load_vae_tensor(ctx, &gf, "FSQ.project_out.weight")!;
    decoder.weights.fsq_out_bias      = load_vae_tensor(ctx, &gf, "FSQ.project_out.bias")!;
    decoder.weights.post_quant_weight = load_vae_tensor(ctx, &gf, "post_quant.weight")!;
    decoder.weights.post_quant_bias   = load_vae_tensor(ctx, &gf, "post_quant.bias")!;

    for (uint i = 0; i < VAE_DECODER_SELF_BLOCKS; i++) {
        decoder.weights.decoder_blocks[i] = load_self_block(ctx, &gf, "decoder", i)!;
    }
    decoder.weights.decoder_cross     = load_cross_block(ctx, &gf, "decoder", VAE_DECODER_SELF_BLOCKS)!;
    decoder.weights.proj_query_weight = load_vae_tensor(ctx, &gf, "decoder.proj_query.weight")!;
    decoder.weights.proj_query_bias   = load_vae_tensor(ctx, &gf, "decoder.proj_query.bias")!;
    decoder.weights.out_norm_weight   = load_vae_tensor(ctx, &gf, "decoder.norm_out.weight")!;
    decoder.weights.out_norm_bias     = load_vae_tensor(ctx, &gf, "decoder.norm_out.bias")!;
    decoder.weights.proj_out_weight   = load_vae_tensor(ctx, &gf, "decoder.proj_out.weight")!;
    decoder.weights.proj_out_bias     = load_vae_tensor(ctx, &gf, "decoder.proj_out.bias")!;

    return decoder;
}

fn void SkinDecoder.free(&self) {
    self.weights.free();
    self.kernels.free(self.ctx.device);
    *self = {};
}

// --- Block dispatch --------------------------------------------------------

<*
 x = x + ff(norm3(x)), the tail both block kinds share.

 `scratch` is [rows, 768] and `hidden` is [rows, 3072]; both are clobbered.
*>
fn void SkinDecoder.feed_forward(
    &self, llm::RenderGraph* graph, DiTSelfBlock* ff,
    llm::Tensor* state, llm::Tensor* scratch, llm::Tensor* hidden, uint rows)
{
    self.kernels.layer_norm(graph, state, &ff.norm3_weight, &ff.norm3_bias, scratch, rows, VAE_WIDTH);
    self.kernels.linear(graph, scratch, &ff.ff_in_weight, &ff.ff_in_bias, hidden, rows, VAE_FF_DIM, VAE_WIDTH, true);
    self.kernels.gelu(graph, hidden, rows * VAE_FF_DIM);
    self.kernels.linear(graph, hidden, &ff.ff_out_weight, &ff.ff_out_bias, scratch, rows, VAE_WIDTH, VAE_FF_DIM, true);
    self.kernels.residual_add(graph, state, scratch, rows * VAE_WIDTH);
}

<*
 One self-attention block, in place on `state`.

 `qkv` is [rows, 3*768] and carries the fused projection; the attention reads
 q, k and v out of it by offset, 64 columns apart inside each head's 192.
*>
fn void SkinDecoder.self_block(
    &self, llm::RenderGraph* graph, DiTSelfBlock* block,
    llm::Tensor* state, llm::Tensor* scratch, llm::Tensor* qkv,
    llm::Tensor* attn, llm::Tensor* hidden, uint rows)
{
    self.kernels.layer_norm(graph, state, &block.norm1_weight, &block.norm1_bias, scratch, rows, VAE_WIDTH);
    self.kernels.linear(graph, scratch, &block.qkv_weight, null, qkv, rows, 3 * VAE_WIDTH, VAE_WIDTH, false);

    self.kernels.attention(graph, {
        .n_heads = VAE_HEADS,
        .q_len = rows,
        .kv_len = rows,
        .q_row_stride = 3 * VAE_WIDTH,
        .q_head_stride = 3 * ENCODER_HEAD_DIM,
        .q_offset = 0,
        .kv_row_stride = 3 * VAE_WIDTH,
        .kv_head_stride = 3 * ENCODER_HEAD_DIM,
        .k_offset = ENCODER_HEAD_DIM,
        .v_offset = 2 * ENCODER_HEAD_DIM,
        .out_row_stride = VAE_WIDTH,
        .scale = attention_scale(),
    }, qkv, qkv, attn);

    self.kernels.linear(graph, attn, &block.attn_out_weight, &block.attn_out_bias, scratch, rows, VAE_WIDTH, VAE_WIDTH, true);
    self.kernels.residual_add(graph, state, scratch, rows * VAE_WIDTH);

    self.feed_forward(graph, block, state, scratch, hidden, rows);
}

<*
 The keys and values a cross block will attend into, projected once.

 Separated from the attention itself because neither `norm_cross` nor the
 fused k/v projection depends on the queries: the decoder reuses one bone's
 keys across every query point and every chunk of them.
*>
fn void SkinDecoder.cross_keys(
    &self, llm::RenderGraph* graph, DiTCrossBlock* block,
    llm::Tensor* source, llm::Tensor* scratch, llm::Tensor* kv, uint rows)
{
    self.kernels.layer_norm(graph, source, &block.norm_cross_weight, &block.norm_cross_bias, scratch, rows, VAE_WIDTH);
    self.kernels.linear(graph, scratch, &block.kv_weight, null, kv, rows, 2 * VAE_WIDTH, VAE_WIDTH, false);
}

<*
 One cross-attention block, in place on `state`, reading keys prepared by
 `cross_keys`.

 @param kv_base : `Element offset of this bone's slab inside a shared kv buffer.`
*>
fn void SkinDecoder.cross_block(
    &self, llm::RenderGraph* graph, DiTCrossBlock* block,
    llm::Tensor* state, llm::Tensor* scratch, llm::Tensor* query,
    llm::Tensor* kv, uint kv_base, uint kv_len,
    llm::Tensor* attn, llm::Tensor* hidden, uint rows)
{
    self.kernels.layer_norm(graph, state, &block.norm2_weight, &block.norm2_bias, scratch, rows, VAE_WIDTH);
    self.kernels.linear(graph, scratch, &block.q_weight, null, query, rows, VAE_WIDTH, VAE_WIDTH, false);

    self.kernels.attention(graph, {
        .n_heads = VAE_HEADS,
        .q_len = rows,
        .kv_len = kv_len,
        // Queries split into heads the ordinary way; only k and v carry the
        // fused layout, 64 columns apart inside each head's 128.
        .q_row_stride = VAE_WIDTH,
        .q_head_stride = ENCODER_HEAD_DIM,
        .q_offset = 0,
        .kv_row_stride = 2 * VAE_WIDTH,
        .kv_head_stride = 2 * ENCODER_HEAD_DIM,
        .k_offset = kv_base,
        .v_offset = kv_base + ENCODER_HEAD_DIM,
        .out_row_stride = VAE_WIDTH,
        .scale = attention_scale(),
    }, query, kv, attn);

    self.kernels.linear(graph, attn, &block.attn_out_weight, &block.attn_out_bias, scratch, rows, VAE_WIDTH, VAE_WIDTH, true);
    self.kernels.residual_add(graph, state, scratch, rows * VAE_WIDTH);

    // The block's feed-forward half lives on the self-block struct's fields;
    // a cross block spells norm3 and ff identically.
    DiTSelfBlock tail = {
        .norm3_weight = block.norm3_weight, .norm3_bias = block.norm3_bias,
        .ff_in_weight = block.ff_in_weight, .ff_in_bias = block.ff_in_bias,
        .ff_out_weight = block.ff_out_weight, .ff_out_bias = block.ff_out_bias,
    };
    self.feed_forward(graph, &tail, state, scratch, hidden, rows);
}

// --- FSQ -------------------------------------------------------------------

<*
 Turn codebook indices into the 5-dimensional codes the FSQ projects out of.

 An index is a mixed-radix number over five digits of eight. Each digit maps to
 an evenly spaced level in [-1, 0.75]: `(digit - 4) / 4`, asymmetric because
 eight levels cannot straddle zero evenly and the reference does not try.

 @param dst : `[indices.len, 5], row-major.`
 @return? SKIN_BAD_CODE : `An index outside the 32768-entry codebook.`
 @return? SKIN_SHAPE_MISMATCH : `dst cannot hold five levels per index.`
*>
fn void? fsq_decode(float[] dst, uint[] indices) {
    if (dst.len < indices.len * FSQ_DIM) return SKIN_SHAPE_MISMATCH~;

    float half = (float)(FSQ_LEVELS / 2);
    foreach (usz i, index : indices) {
        if (index >= SKIN_TOKEN_COUNT) return SKIN_BAD_CODE~;
        uint rest = index;
        for (usz d = 0; d < FSQ_DIM; d++) {
            dst[i * FSQ_DIM + d] = ((float)(rest % FSQ_LEVELS) - half) / half;
            rest /= FSQ_LEVELS;
        }
    }
}

// --- Forward pass ----------------------------------------------------------

fn llm::Tensor? vae_matrix(llm::DeviceContext* ctx, uint rows, uint cols) @local
    => llm::create_f32_tensor(ctx, { (ulong)cols, (ulong)rows, 0, 0 }, 2);

<*
 Predict a weight per vertex per bone.

 `shape` supplies the point cloud the skeleton was generated from — the same
 cloud, in the same normalised frame, because the conditioning latents and the
 bone codes were produced against each other. `positions` and `normals` are the
 points to evaluate at, also normalised; they are ordinarily the mesh's own
 vertices.

 @param codes : `bones * 4 codebook indices, bone-major, already offset to 0.`
 @param positions : `Query points in the encoder's normalised frame.`
 @return `[positions.len, bones] row-major, each entry in [0, 1].`
 @return? SKIN_NO_BONES : `No bones, or no query points.`
 @return? SKIN_SHAPE_MISMATCH : `codes is not exactly 4 per bone, or normals do not match positions.`
 @return? SKIN_BAD_CODE : `A code outside the 32768-entry codebook.`
*>
fn float[]? SkinDecoder.decode(
    &self, alloc::Allocator allocator,
    ShapeEncoderInput* shape,
    uint[] codes,
    usz bones,
    Vec3[] positions,
    Vec3[] normals)
{
    if (bones == 0 || positions.len == 0) return SKIN_NO_BONES~;
    if (codes.len != bones * TOKENS_PER_SKIN) return SKIN_SHAPE_MISMATCH~;
    if (normals.len != positions.len) return SKIN_SHAPE_MISMATCH~;
    if (shape.cloud.positions.len == 0) return SKIN_SHAPE_MISMATCH~;

    // Two stages, two sets of buffers: the conditioning encoder's are sized on
    // the point cloud and the decoder's on the bones, and holding both alive
    // would double the peak for no reason.
    float[] cond_latents = self.encode_condition(allocator, shape)!;
    defer alloc::free(allocator, cond_latents);

    return self.decode_queries(allocator, cond_latents, codes, bones, positions, normals);
}

<*
 The point cloud down to 384 conditioning latents.

 Its own submission, and its own allocations, because this is where the memory
 goes: keys and values over a 54000-point cloud at width 768 are 332 MB, and
 nothing after this needs them.
*>
fn float[]? SkinDecoder.encode_condition(&self, alloc::Allocator allocator, ShapeEncoderInput* shape) {
    llm::DeviceContext* ctx = self.ctx;
    SkinDecoderWeights* w = &self.weights;

    usz dim = encoder_input_dim();
    uint feature_dim = (uint)dim;
    uint points = (uint)shape.cloud.positions.len;
    uint tokens = VAE_COND_TOKENS;

    // The VAE embeds the same cloud through a different embedder than the shape
    // encoder: pi-scaled frequencies with the PMPE phase term added.
    float[] features = embed_points(allocator, shape.cloud, DEFAULT_NUM_FREQS, true, true)!;
    defer alloc::free(allocator, features);

    // Conditioning queries are farthest-point picks out of the same cloud, the
    // reference's `_sample_features` with `cond_tokens` in place of the shape
    // encoder's `token_num`.
    usz[] picks = pick_queries(allocator, shape.cloud.positions, (usz)tokens, QUERY_POOL_RATIO, 0)!;
    defer alloc::free(allocator, picks);

    float[] query_features = alloc::new_array(allocator, float, (sz)((usz)tokens * dim));
    defer alloc::free(allocator, query_features);
    foreach (usz slot, index : picks) {
        for (usz c = 0; c < dim; c++) query_features[slot * dim + c] = features[index * dim + c];
    }

    llm::Tensor point_features = vae_matrix(ctx, points, feature_dim)!;
    defer point_features.free();
    llm::Tensor data = vae_matrix(ctx, points, VAE_WIDTH)!;
    defer data.free();
    llm::Tensor kv = vae_matrix(ctx, points, 2 * VAE_WIDTH)!;
    defer kv.free();

    llm::Tensor token_features = vae_matrix(ctx, tokens, feature_dim)!;
    defer token_features.free();
    llm::Tensor state = vae_matrix(ctx, tokens, VAE_WIDTH)!;
    defer state.free();
    llm::Tensor scratch = vae_matrix(ctx, tokens, VAE_WIDTH)!;
    defer scratch.free();
    llm::Tensor query = vae_matrix(ctx, tokens, VAE_WIDTH)!;
    defer query.free();
    llm::Tensor qkv = vae_matrix(ctx, tokens, 3 * VAE_WIDTH)!;
    defer qkv.free();
    llm::Tensor attn = vae_matrix(ctx, tokens, VAE_WIDTH)!;
    defer attn.free();
    llm::Tensor hidden = vae_matrix(ctx, tokens, VAE_FF_DIM)!;
    defer hidden.free();
    llm::Tensor latents = vae_matrix(ctx, tokens, VAE_LATENT)!;
    defer latents.free();

    llm::upload_to_tensor(ctx, &point_features, features)!;
    llm::upload_to_tensor(ctx, &token_features, query_features)!;

    llm::RenderGraph graph;
    graph.init(ctx)!;

    // One projection for the cloud and for the queries picked out of it, as in
    // the shape encoder: a per-row linear commutes with the gather.
    self.kernels.linear(&graph, &point_features, &w.cond_proj_in_weight, &w.cond_proj_in_bias, &data, points, VAE_WIDTH, feature_dim, true);
    self.kernels.linear(&graph, &token_features, &w.cond_proj_in_weight, &w.cond_proj_in_bias, &state, tokens, VAE_WIDTH, feature_dim, true);

    // Point-sized work gets its own submission so a slow device is not asked
    // to run the whole graph inside one watchdog interval.
    graph.checkpoint()!;

    self.cross_keys(&graph, &w.cond_cross, &data, &data, &kv, points);
    graph.checkpoint()!;

    self.cross_block(&graph, &w.cond_cross, &state, &scratch, &query, &kv, 0, points, &attn, &hidden, tokens);
    graph.checkpoint()!;

    for (uint i = 0; i < VAE_COND_SELF_BLOCKS; i++) {
        self.self_block(&graph, &w.cond_blocks[i], &state, &scratch, &qkv, &attn, &hidden, tokens);
    }

    self.kernels.layer_norm(&graph, &state, &w.cond_norm_out_weight, &w.cond_norm_out_bias, &state, tokens, VAE_WIDTH);
    self.kernels.linear(&graph, &state, &w.cond_quant_weight, &w.cond_quant_bias, &latents, tokens, VAE_LATENT, VAE_WIDTH, true);

    graph.flush()!;

    return llm::download_tensor(allocator, ctx, &latents, (usz)tokens * VAE_LATENT);
}

<*
 Bone contexts, then the query field over them.

 Contexts first and all at once: building one is 10 self-attention blocks over
 388 tokens and does not depend on the query points, so recomputing it per
 chunk would dominate everything else. Projected keys for every bone come to
 1.2 MB each, which is what makes holding them all worthwhile.
*>
fn float[]? SkinDecoder.decode_queries(
    &self, alloc::Allocator allocator,
    float[] cond_latents,
    uint[] codes,
    usz bones,
    Vec3[] positions,
    Vec3[] normals)
{
    llm::DeviceContext* ctx = self.ctx;
    SkinDecoderWeights* w = &self.weights;

    usz dim = encoder_input_dim();
    uint feature_dim = (uint)dim;
    uint n_bones = (uint)bones;
    uint rows = VAE_CONTEXT_ROWS;
    uint chunk = positions.len < (usz)QUERY_CHUNK ? (uint)positions.len : QUERY_CHUNK;

    float[] code_levels = alloc::new_array(allocator, float, (sz)(codes.len * FSQ_DIM));
    defer alloc::free(allocator, code_levels);
    fsq_decode(code_levels, codes)!;

    llm::Tensor levels = vae_matrix(ctx, (uint)codes.len, FSQ_DIM)!;
    defer levels.free();
    llm::Tensor code_latents = vae_matrix(ctx, (uint)codes.len, VAE_LATENT)!;
    defer code_latents.free();
    llm::Tensor cond = vae_matrix(ctx, VAE_COND_TOKENS, VAE_LATENT)!;
    defer cond.free();

    llm::Tensor z = vae_matrix(ctx, rows, VAE_LATENT)!;
    defer z.free();
    llm::Tensor state = vae_matrix(ctx, rows, VAE_WIDTH)!;
    defer state.free();
    llm::Tensor scratch = vae_matrix(ctx, rows, VAE_WIDTH)!;
    defer scratch.free();
    llm::Tensor qkv = vae_matrix(ctx, rows, 3 * VAE_WIDTH)!;
    defer qkv.free();
    llm::Tensor attn = vae_matrix(ctx, rows, VAE_WIDTH)!;
    defer attn.free();
    llm::Tensor hidden = vae_matrix(ctx, rows, VAE_FF_DIM)!;
    defer hidden.free();
    llm::Tensor context_kv = vae_matrix(ctx, rows, 2 * VAE_WIDTH)!;
    defer context_kv.free();

    // Every bone's projected keys and values in one buffer; the attention
    // selects a bone by adding its slab's element offset to k_offset/v_offset.
    llm::Tensor bone_kv = vae_matrix(ctx, n_bones * rows, 2 * VAE_WIDTH)!;
    defer bone_kv.free();

    llm::upload_to_tensor(ctx, &levels, code_levels)!;
    llm::upload_to_tensor(ctx, &cond, cond_latents)!;

    llm::RenderGraph graph;
    graph.init(ctx)!;

    self.kernels.linear(&graph, &levels, &w.fsq_out_weight, &w.fsq_out_bias, &code_latents, (uint)codes.len, VAE_LATENT, FSQ_DIM, true);

    ulong latent_row_bytes = (ulong)VAE_LATENT * 4;
    ulong kv_slab_bytes = (ulong)rows * 2 * VAE_WIDTH * 4;

    for (uint b = 0; b < n_bones; b++) {
        // z = [this bone's four latents ; the shared conditioning].
        graph.copy_region(&code_latents, (usz)((ulong)b * TOKENS_PER_SKIN * latent_row_bytes),
            &z, 0, (ulong)TOKENS_PER_SKIN * latent_row_bytes);
        graph.copy_region(&cond, 0,
            &z, (usz)((ulong)TOKENS_PER_SKIN * latent_row_bytes), (ulong)VAE_COND_TOKENS * latent_row_bytes);

        self.kernels.linear(&graph, &z, &w.post_quant_weight, &w.post_quant_bias, &state, rows, VAE_WIDTH, VAE_LATENT, true);
        for (uint i = 0; i < VAE_DECODER_SELF_BLOCKS; i++) {
            self.self_block(&graph, &w.decoder_blocks[i], &state, &scratch, &qkv, &attn, &hidden, rows);
        }
        self.cross_keys(&graph, &w.decoder_cross, &state, &scratch, &context_kv, rows);
        graph.copy_region(&context_kv, 0, &bone_kv, (usz)((ulong)b * kv_slab_bytes), kv_slab_bytes);

        // Ten blocks per bone is enough work per submission that batching them
        // all into one command buffer risks a device-lost on a slow GPU.
        graph.checkpoint()!;
    }

    // --- The query field ---

    llm::Tensor q_features = vae_matrix(ctx, chunk, feature_dim)!;
    defer q_features.free();
    llm::Tensor q_base = vae_matrix(ctx, chunk, VAE_WIDTH)!;
    defer q_base.free();
    llm::Tensor q_state = vae_matrix(ctx, chunk, VAE_WIDTH)!;
    defer q_state.free();
    llm::Tensor q_scratch = vae_matrix(ctx, chunk, VAE_WIDTH)!;
    defer q_scratch.free();
    llm::Tensor q_query = vae_matrix(ctx, chunk, VAE_WIDTH)!;
    defer q_query.free();
    llm::Tensor q_attn = vae_matrix(ctx, chunk, VAE_WIDTH)!;
    defer q_attn.free();
    llm::Tensor q_hidden = vae_matrix(ctx, chunk, VAE_FF_DIM)!;
    defer q_hidden.free();
    llm::Tensor q_logit = vae_matrix(ctx, chunk, 1)!;
    defer q_logit.free();
    llm::Tensor q_logits = vae_matrix(ctx, n_bones, chunk)!;
    defer q_logits.free();

    float[] chunk_features = alloc::new_array(allocator, float, (sz)((usz)chunk * dim));
    defer alloc::free(allocator, chunk_features);

    float[] skin = alloc::new_array(allocator, float, (sz)(positions.len * bones));
    defer catch alloc::free(allocator, skin);

    for (usz start = 0; start < positions.len; start += (usz)chunk) {
        usz count = positions.len - start;
        if (count > (usz)chunk) count = (usz)chunk;
        uint n = (uint)count;

        for (usz i = 0; i < count; i++) {
            embed_point(chunk_features[i * dim : dim], positions[start + i], normals[start + i],
                DEFAULT_NUM_FREQS, true, true)!;
        }
        llm::upload_to_tensor(ctx, &q_features, chunk_features[:count * dim])!;

        // proj_query is bone-independent, so it runs once per chunk and every
        // bone starts from a copy of the same projected queries.
        self.kernels.linear(&graph, &q_features, &w.proj_query_weight, &w.proj_query_bias, &q_base, n, VAE_WIDTH, feature_dim, true);

        for (uint b = 0; b < n_bones; b++) {
            graph.copy_region(&q_base, 0, &q_state, 0, (ulong)n * VAE_WIDTH * 4);

            self.cross_block(&graph, &w.decoder_cross, &q_state, &q_scratch, &q_query,
                &bone_kv, (uint)((ulong)b * rows * 2 * VAE_WIDTH), rows, &q_attn, &q_hidden, n);

            self.kernels.layer_norm(&graph, &q_state, &w.out_norm_weight, &w.out_norm_bias, &q_scratch, n, VAE_WIDTH);
            self.kernels.linear(&graph, &q_scratch, &w.proj_out_weight, &w.proj_out_bias, &q_logit, n, 1, VAE_WIDTH, true);

            // One row of q_logits per bone, so the whole chunk comes back in a
            // single readback instead of one per bone.
            graph.copy_region(&q_logit, 0, &q_logits, (usz)((ulong)b * chunk * 4), (ulong)n * 4);
        }

        // The readback runs on its own command buffer, so everything the chunk
        // recorded has to have been submitted before it is asked for.
        graph.checkpoint()!;

        float[] logits = llm::download_tensor(allocator, ctx, &q_logits, (usz)n_bones * chunk)!;
        defer alloc::free(allocator, logits);

        // The reference's sigmoid, on the host: the field is one number per
        // query per bone and it is already being copied out.
        for (usz b = 0; b < bones; b++) {
            for (usz i = 0; i < count; i++) {
                float logit = logits[b * (usz)chunk + i];
                skin[(start + i) * bones + b] = 1.0f / (1.0f + math::exp(-logit));
            }
        }
    }

    graph.flush()!;
    return skin;
}

<*
 Skin weights for a generated rig, straight from the pieces `generate_rig`
 hands back.

 The decoder answers in the frame it was conditioned in, so the mesh's vertices
 go in normalised. Their normals do not need transforming: the normalisation is
 a translate and a uniform scale, which leaves a unit normal alone.

 @param geometry : `The mesh as it was handed to generate_rig, in its own coordinates.`
 @return `[vertices, bones] row-major, each entry in [0, 1] and unnormalised across bones.`
 @return? RIG_NO_SKIN : `The sequence carries no complete skin block.`
 @return? SKIN_SHAPE_MISMATCH : `The mesh has no per-vertex normals.`
*>
fn float[]? SkinDecoder.decode_rig(
    &self, alloc::Allocator allocator,
    ShapeEncoderInput* shape,
    RigResult* rig,
    TriMesh geometry)
{
    if (!geometry.has_normals()) return SKIN_SHAPE_MISMATCH~;

    uint[] codes = rig.skin_codes(allocator)!;
    defer alloc::free(allocator, codes);

    Vec3[] points = alloc::new_array(allocator, Vec3, (sz)geometry.positions.len);
    defer alloc::free(allocator, points);
    foreach (usz i, position : geometry.positions) {
        points[i] = rig.normalization.forward(position);
    }

    return self.decode(allocator, shape, codes, rig.skeleton.joints.len, points, geometry.normals);
}
