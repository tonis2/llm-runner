<*
 Mesh in, bones out.

 The pipeline SkinTokens describes, minus the skin weights:

     prepare_shape_input   mesh -> 54000 sampled points + 512 queries
     MeshEncoder.encode    -> 512 x 896 prefix embeddings
     prefill_embeddings    those, then bos and a class token, into the LLM
     greedy decode         -> a rig token sequence
     detokenize_skeleton   -> joints and parents

 Bone positions come back in the encoder's normalised frame, so the result
 carries the `Normalization` that puts them back on the original mesh.

 Two deliberate departures from the reference, both of which cost quality
 rather than correctness. It samples from one sequence where the reference
 searches ten beams, and it applies no grammar, so a sequence can come back
 malformed — that surfaces as a fault from `detokenize_skeleton` rather than
 as a quietly wrong rig.
*>
module llm::mesh;

import vk;
import llm;
import llm_text;
import std::io;
import std::core::mem;
import std::core::mem::allocator;
import std::math::vector;
import std::math::random;

faultdef RIG_PREFIX_TOO_LONG, RIG_DECODE_STALLED, RIG_CONTEXT_EXHAUSTED;

<*
 Positions the KV cache must hold: the prefix plus everything generated after
 it. Writing past the cache does not fault — `kv_cache_store` indexes
 `h * max_seq_len * head_dim + pos * head_dim`, so an overrun lands in the next
 head's rows and then off the end of the buffer, and the damage only shows up
 later as a crash somewhere unrelated.
*>
fn bool context_fits(llm_text::LlmModel* model, usz n_prefix, uint max_new_tokens)
    => n_prefix + (usz)max_new_tokens <= (usz)model.max_seq_len;

// generate_kwargs.max_new_tokens from the checkpoint.
const uint DEFAULT_MAX_NEW_TOKENS = 2040;

struct RigOptions {
    PrepareOptions prepare;
    RigClass rig_class;
    uint max_new_tokens;
    llm::SamplingParams sampling;
    ulong seed;
}

<*
 The checkpoint's own `generate_kwargs`, minus the beam search.

 The sampling settings are not decoration. Greedy decoding on this model
 collapses immediately — it emits the coordinate bin nearest zero over and
 over, because that is the single most likely token at every step and nothing
 discourages repeating it. The repetition penalty is what breaks that loop and
 the temperature is what keeps the rest of the distribution reachable.
*>
const RigOptions DEFAULT_RIG = {
    .prepare = DEFAULT_PREPARE,
    .rig_class = NONE,
    .max_new_tokens = DEFAULT_MAX_NEW_TOKENS,
    .sampling = {
        .temperature = 1.5f,
        .top_k = 10,
        .top_p = 0.95f,
        .repetition_penalty = 2.0f,
    },
    .seed = 0,
};

<*
 A generated rig, in the mesh's own coordinates.
*>
struct RigResult {
    Skeleton skeleton;
    Normalization normalization;
    uint[] tokens;      // the whole sequence, bos first
    usz prefix_tokens;  // how many embeddings preceded it
}

fn void RigResult.free(&self, alloc::Allocator allocator) {
    self.skeleton.free(allocator);
    if (self.tokens.len > 0) alloc::free(allocator, self.tokens);
    *self = {};
}

<*
 Bone positions mapped out of the normalised frame and back onto the mesh.

 The caller owns the returned array.
*>
fn Vec3[] RigResult.joints_in_model_space(&self, alloc::Allocator allocator) {
    Vec3[] out = alloc::new_array(allocator, Vec3, (sz)self.skeleton.joints.len);
    foreach (i, joint : self.skeleton.joints) {
        out[i] = self.normalization.inverse(joint);
    }
    return out;
}

<*
 Encode a mesh, generate a skeleton for it, and decode the result.

 @param geometry : `The mesh, in its own coordinates; not modified.`
 @return? ENCODER_SHAPE_MISMATCH : `The encoder and the input disagree on width.`
 @return? RIG_PREFIX_TOO_LONG : `The prefix does not fit the prefill path.`
 @return? RIG_DECODE_STALLED : `Generation hit max_new_tokens without terminating.`
 @return? RIG_CONTEXT_EXHAUSTED : `Prefix plus max_new_tokens exceeds the model's max_seq_len.`
*>
fn RigResult? generate_rig(
    alloc::Allocator allocator,
    llm::DeviceContext* ctx,
    MeshEncoder* encoder,
    llm_text::LlmModel* model,
    TriMesh geometry,
    RigOptions options = DEFAULT_RIG)
{
    ShapeEncoderInput input = prepare_shape_input(allocator, geometry, options.prepare)!;
    defer input.free(allocator);

    float[] prefix = encoder.encode(allocator, &input)!;
    defer alloc::free(allocator, prefix);

    return generate_rig_from_prefix(allocator, ctx, model, prefix, input.normalization, options);
}

<*
 The generation half on its own, for a prefix that is already encoded.

 Split out because the encoder is the slow part and worth reusing across
 sampling attempts on the same mesh.
*>
fn RigResult? generate_rig_from_prefix(
    alloc::Allocator allocator,
    llm::DeviceContext* ctx,
    llm_text::LlmModel* model,
    float[] prefix,
    Normalization normalization,
    RigOptions options = DEFAULT_RIG)
{
    uint dim = model.config.dim;
    usz latents = prefix.len / (usz)dim;
    if (latents == 0 || prefix.len % (usz)dim != 0) return ENCODER_SHAPE_MISMATCH~;

    // The prompt the reference builds: bos, then a class token. Both are
    // ordinary tokens, so they go in as embeddings alongside the mesh prefix
    // rather than being fed one at a time afterwards.
    uint[2] start_tokens = { TOKEN_BOS, options.rig_class.token() };
    usz n_tokens = latents + start_tokens.len;
    if (!model.prefill_supported((uint)n_tokens)) return RIG_PREFIX_TOO_LONG~;
    if (!context_fits(model, n_tokens, options.max_new_tokens)) return RIG_CONTEXT_EXHAUSTED~;

    llm::Tensor hidden = llm::create_f32_tensor(ctx, { (ulong)(n_tokens * dim), 0, 0, 0 }, 1)!;
    defer hidden.free();

    // The mesh prefix fills the leading rows from the host.
    llm::upload_to_tensor(ctx, &hidden, prefix)!;

    // The start tokens' embeddings are already on the device: outside
    // encode_only mode the table is resident as F32, and `embedding_data` --
    // the CPU-side copy -- is null. So they are gathered with a buffer copy per
    // row rather than dequantised on the host.
    ulong row_bytes = (ulong)dim * 4;
    llm::RenderGraph gather;
    gather.init(ctx)!;
    foreach (usz slot, token : start_tokens) {
        gather.copy_region(
            &model.weights.token_embedding, (usz)((ulong)token * row_bytes),
            &hidden, (usz)((ulong)(latents + slot) * row_bytes),
            row_bytes);
    }
    gather.flush()!;

    model.prefill_embeddings(&hidden, (uint)n_tokens)!;

    // Decode. The prefix already produced logits for the position after the
    // last start token, so the first pick comes before any forward() call.
    usz capacity = start_tokens.len + (usz)options.max_new_tokens;
    uint[] tokens = alloc::new_array(allocator, uint, (sz)capacity);
    defer catch alloc::free(allocator, tokens);
    usz produced = 0;
    foreach (token : start_tokens) {
        tokens[produced] = token;
        produced++;
    }

    random::Sfc64Random rng;
    random::seed(&rng, options.seed);
    llm::SamplingParams sampling = options.sampling;

    // The grammar has already consumed the start tokens; from here it says
    // which tokens may follow, which is what keeps sampling inside the
    // skeleton vocabulary. The model is trained to emit skin codes straight
    // after the skeleton's eos, and nothing in the logits marks that boundary.
    RigGrammar grammar;
    foreach (token : start_tokens) grammar.accept(token);

    bool[] allowed = alloc::new_array(allocator, bool, (sz)model.config.vocab_size);
    defer alloc::free(allocator, allowed);

    bool terminated = false;
    for (uint step = 0; step < options.max_new_tokens; step++) {
        // The penalty looks only at what has been generated, not at the start
        // tokens: bos is meant to appear once and penalising it changes nothing.
        uint[] history = tokens[start_tokens.len:produced - start_tokens.len];
        grammar.allow_next(allowed);

        uint next = llm::sample_token(
            ctx, &model.acts.logits, model.config.vocab_size, &sampling, &rng, history, allowed)!;
        grammar.accept(next);
        tokens[produced] = next;
        produced++;

        if (next == TOKEN_EOS || next == TOKEN_SEQUENCE_END) {
            terminated = true;
            break;
        }
        if (produced == capacity) break;

        // Positions continue straight on from the prefix: the KV cache the
        // prefill filled and the one forward() appends to are the same.
        model.forward(next, (uint)(n_tokens + (usz)step))!;
    }

    // No explicit free here: the `defer catch` above already runs on an error
    // return, and freeing as well is a double free.
    if (!terminated) return RIG_DECODE_STALLED~;

    RigResult result = {
        .normalization = normalization,
        .tokens = tokens[:produced],
        .prefix_tokens = latents,
    };
    result.skeleton = detokenize_skeleton(allocator, result.tokens)!;
    return result;
}


