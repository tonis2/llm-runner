<*
 Mesh in, rig out.

 The pipeline SkinTokens describes:

     prepare_shape_input   mesh -> 54000 sampled points + 512 queries
     MeshEncoder.encode    -> 512 x 896 prefix embeddings
     prefill_embeddings    those, then bos and a class token, into the LLM
     constrained sampling  -> a rig token sequence, `RigGrammar` masking the
                              logits so decoding stays in the right vocabulary
     detokenize_skeleton   -> joints and parents

 One sequence carries two things. Up to the tokenizer's terminator it is a
 skeleton; after it, four codes per bone that `SkinDecoder` turns into weights.
 Nothing in the logits marks the boundary, so the grammar is what finds it —
 and `predict_skin` is what decides whether decoding crosses it at all. Stopping
 at the skeleton is a legitimate answer, and it is what the seed search in
 `examples/rig` wants.

 Bone positions come back in the encoder's normalised frame, so the result
 carries the `Normalization` that puts them back on the original mesh.

 One departure from the reference remains: this samples a single sequence where
 the reference searches ten beams. `rig_beam.c3` implements a beam search, but
 a deterministic one — the reference passes `num_beams` alongside
 `do_sample: true`, which is beam *sampling*, and the two do not produce the
 same distribution. Sampling several seeds and choosing between them with
 `RigResult.off_mesh_fraction` is the cheaper substitute, and it is what
 `examples/rig` does.
*>
module skintokens;

import vk;
import llm;
import llm_text;
import std::io;
import std::core::mem;
import std::core::mem::allocator;
import std::math::vector;
import std::math::random;
import std::os::env;
import std::time::clock;

faultdef RIG_PREFIX_TOO_LONG, RIG_DECODE_STALLED, RIG_CONTEXT_EXHAUSTED, RIG_NO_SKIN;

<*
 How many tokens may still be generated after a prefix of `n_prefix`.

 Writing past the KV cache does not fault — `kv_cache_store` indexes
 `h * max_seq_len * head_dim + pos * head_dim`, so an overrun lands in the next
 head's rows and then off the end of the buffer, and the damage only shows up
 later as a crash somewhere unrelated. Something has to bound it.

 `max_new_tokens` is a cap rather than a reservation: decoding terminates in a
 few hundred tokens in practice, so refusing to start whenever prefix plus
 budget exceeds the cache would reject every default run — 2040 against a
 512-latent prefix and a 2048-position cache. It is clamped to what the cache
 can hold instead. Zero means the prefix has filled the cache and nothing can
 follow it.
*>
fn uint decode_budget(llm_text::LlmModel* model, usz n_prefix, uint max_new_tokens)
{
    if (n_prefix >= (usz)model.max_seq_len) return 0;
    usz room = (usz)model.max_seq_len - n_prefix;
    return room < (usz)max_new_tokens ? (uint)room : max_new_tokens;
}

// generate_kwargs.max_new_tokens from the checkpoint.
const uint DEFAULT_MAX_NEW_TOKENS = 2040;

struct RigOptions {
    PrepareOptions prepare;
    RigClass rig_class;
    uint max_new_tokens;
    llm::SamplingParams sampling;
    ulong seed;
    // Keep decoding past the skeleton's terminator for the four skin codes per
    // bone the model owes. Costs one extra token per bone-quarter and buys the
    // difference between a predicted skin and a distance falloff.
    bool predict_skin;
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
    .predict_skin = true,
};

<*
 A generated rig, in the mesh's own coordinates.
*>
struct RigResult {
    Skeleton skeleton;
    Normalization normalization;
    uint[] tokens;      // the whole sequence, bos first
    usz prefix_tokens;  // how many embeddings preceded it
    usz skin_start;     // where the skin block begins in `tokens`, 0 if absent
    usz skin_count;     // how many skin tokens it holds
}

<*
 Whether the sequence carries a complete set of skin codes.

 Incomplete is a normal outcome, not an error: the skeleton comes first and a
 run that fills its token budget partway through the skin block still has a
 skeleton worth exporting.
*>
fn bool RigResult.has_skin(&self) => self.skin_count > 0
    && self.skin_count == self.skeleton.joints.len * (usz)TOKENS_PER_SKIN;

<*
 The skin block as codebook indices — four per bone, bone-major.

 The caller owns the returned array.

 @return? RIG_NO_SKIN : `The sequence has no complete skin block.`
*>
fn uint[]? RigResult.skin_codes(&self, alloc::Allocator allocator) {
    if (!self.has_skin()) return RIG_NO_SKIN~;

    uint[] codes = alloc::new_array(allocator, uint, (sz)self.skin_count);
    defer catch alloc::free(allocator, codes);

    for (usz i = 0; i < self.skin_count; i++) {
        uint token = self.tokens[self.skin_start + i];
        if (!is_skin_token(token)) return RIG_NO_SKIN~;
        codes[i] = token - SKIN_TOKEN_BASE;
    }
    return codes;
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
 The share of bones sitting outside the mesh they were generated for.

 A quality signal, not a correctness one. The model wanders when a sequence runs
 long, and the reference implementation does it too — on this cow it put 62 of
 96 joints off the mesh on a 299-token sample while its 13- to 38-joint samples
 stayed entirely clean. So it is worth sampling a few seeds and picking between
 them rather than taking the first that parses.

 Measured against the bounding box rather than the surface: a joint inside the
 box but outside the body still reads as plausible, and testing properly means
 an inside-outside test against the mesh. This catches the failure that matters,
 which is limbs flung out into space.

 @param tolerance : `Slack around the box, as a fraction of its longest side.`
*>
fn float RigResult.off_mesh_fraction(&self, Bounds bounds, float tolerance = 0.02f) {
    usz bones = self.skeleton.joints.len;
    if (bones == 0) return 1.0f;

    Vec3 extent = bounds.extent();
    float longest = math::max(extent.x, math::max(extent.y, extent.z));
    float slack = longest * tolerance;

    usz outside = 0;
    foreach (joint : self.skeleton.joints) {
        Vec3 world = self.normalization.inverse(joint);
        for (usz axis = 0; axis < 3; axis++) {
            if (world[axis] < bounds.min[axis] - slack || world[axis] > bounds.max[axis] + slack) {
                outside++;
                break;
            }
        }
    }
    return (float)outside / (float)bones;
}

<*
 The share of the mesh that no bone comes near.

 `off_mesh_fraction` asks whether the joints are inside the mesh, and a skeleton
 can satisfy that by being far too small: huddle every joint near the centre and
 not one of them is outside the box. It is blind to the opposite failure, which
 is a whole limb — or half the body — left with no bone at all. A squirrel whose
 skeleton stopped at the shoulders scored a perfect zero while 47% of its
 vertices sat more than a tenth of the mesh away from the nearest bone.

 So this asks the question of the mesh rather than of the skeleton. Distance is
 to the bone segment, not to the joints: a vertex halfway down a thigh is far
 from both hip and knee and near the bone between them, and it is the bone that
 will move it.

 Vertices rather than surface area, which biases towards the dense regions —
 the right bias, since that is where the detail a rig has to move usually is.
 Both halves of a seed's score want to be small, and they trade off against
 each other: reaching further covers more of the mesh and risks overshooting it.

 @param geometry : `The mesh as it was handed to generate_rig, in its own coordinates.`
 @param tolerance : `How near counts as covered, as a fraction of the box diagonal.`
*>
fn float RigResult.uncovered_fraction(&self, TriMesh geometry, float tolerance = 0.10f) {
    usz bones = self.skeleton.joints.len;
    if (bones == 0 || geometry.positions.len == 0) return 1.0f;

    Bounds box = geometry.bounds();
    Vec3 extent = box.extent();

    // Compared in the normalised frame the joints are already in, rather than
    // mapping every joint back for every candidate. The normalisation is a
    // translate and a uniform scale, so it multiplies every distance by the same
    // factor and the threshold can be scaled instead of the skeleton.
    float reach = extent.length() * tolerance * self.normalization.scale;
    float reach_sq = reach * reach;

    Vec3[] joints = self.skeleton.joints;
    usz far = 0;

    foreach (position : geometry.positions) {
        Vec3 point = self.normalization.forward(position);
        bool covered = false;
        foreach (usz b, joint : joints) {
            int parent = self.skeleton.parents[b];
            Vec3 head = parent >= 0 ? joints[(usz)parent] : joint;
            if (segment_distance_sq(point, head, joint) <= reach_sq) {
                covered = true;
                break;
            }
        }
        if (!covered) far++;
    }

    return (float)far / (float)geometry.positions.len;
}

<*
 Encode a mesh, generate a skeleton for it, and decode the result.

 @param geometry : `The mesh, in its own coordinates; not modified.`
 @return? ENCODER_SHAPE_MISMATCH : `The encoder and the input disagree on width.`
 @return? RIG_PREFIX_TOO_LONG : `The prefix does not fit the prefill path.`
 @return? RIG_DECODE_STALLED : `Generation hit max_new_tokens without terminating.`
 @return? RIG_CONTEXT_EXHAUSTED : `The prefix alone fills the model's KV cache.`
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
 The prefill half of a generation, run once and decoded from many times.

 What makes the reuse sound: a decode writes the KV cache only at positions
 past the prefix, so however many attempts have run, the prefix rows still
 hold exactly what the prefill left in them. The one prefill product an
 attempt does destroy is the logits — the first token of every attempt is
 sampled from them, and the attempt's own forwards then write over them — so
 they are snapshotted here and put back at the top of each decode.

 The class token is part of the prefix, so the class is fixed at prepare time;
 `decode_rig_attempt` takes its tokens from here, not from its options.
*>
struct RigPrefill {
    llm::Tensor saved_logits;
    uint n_tokens;
    uint[2] start_tokens;
}

fn void RigPrefill.free(&self) {
    self.saved_logits.free();
    *self = {};
}

<*
 Feed the encoded prefix and the start tokens through the model once.

 @return? ENCODER_SHAPE_MISMATCH : `The prefix is not a whole number of rows.`
 @return? RIG_PREFIX_TOO_LONG : `The prefix does not fit the prefill path.`
 @return? RIG_CONTEXT_EXHAUSTED : `The prefix alone fills the model's KV cache.`
*>
fn RigPrefill? prepare_rig_prefill(
    llm::DeviceContext* ctx,
    llm_text::LlmModel* model,
    float[] prefix,
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
    if (decode_budget(model, n_tokens, options.max_new_tokens) == 0) return RIG_CONTEXT_EXHAUSTED~;

    Clock prefill_clock = clock::now();
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

    ulong logits_bytes = (ulong)model.config.vocab_size * 4;
    llm::Tensor saved = llm::create_f32_tensor(ctx, { (ulong)model.config.vocab_size, 0, 0, 0 }, 1)!;
    defer catch saved.free();
    llm::RenderGraph snap;
    snap.init(ctx)!;
    snap.copy(&model.acts.logits, &saved, logits_bytes);
    snap.flush()!;

    if (try String flag = env::tget_var("RIG_PROFILE")) {
        if (flag == "1") io::printfn("  [profile] prefill %d tokens in %.2fs",
            n_tokens, prefill_clock.to_now().to_sec());
    }

    return { .saved_logits = saved, .n_tokens = (uint)n_tokens, .start_tokens = start_tokens };
}

<*
 One sampling attempt against a prefix someone already prefilled.

 Only the seed, the sampler, and `predict_skin` are read from `options`; the
 tokens ahead of the generation belong to the prefill.

 @return? RIG_DECODE_STALLED : `Generation hit max_new_tokens without terminating.`
 @return? RIG_CONTEXT_EXHAUSTED : `The prefix alone fills the model's KV cache.`
*>
fn RigResult? decode_rig_attempt(
    alloc::Allocator allocator,
    llm::DeviceContext* ctx,
    llm_text::LlmModel* model,
    RigPrefill* prefill,
    Normalization normalization,
    RigOptions options = DEFAULT_RIG)
{
    uint[2] start_tokens = prefill.start_tokens;
    usz n_tokens = prefill.n_tokens;
    uint budget = decode_budget(model, n_tokens, options.max_new_tokens);
    if (budget == 0) return RIG_CONTEXT_EXHAUSTED~;

    // Put the prefill's logits back; the previous attempt decoded over them.
    llm::RenderGraph restore;
    restore.init(ctx)!;
    restore.copy(&prefill.saved_logits, &model.acts.logits, (ulong)model.config.vocab_size * 4);
    restore.flush()!;

    // Decode. The prefix already produced logits for the position after the
    // last start token, so the first pick comes before any forward() call.
    usz capacity = start_tokens.len + (usz)budget;
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
    // after the skeleton's eos, and nothing in the logits marks that boundary —
    // so the grammar also decides whether decoding crosses it.
    RigGrammar grammar = { .tokens_per_skin = options.predict_skin ? TOKENS_PER_SKIN : 0 };
    foreach (token : start_tokens) grammar.accept(token);

    bool[] allowed = alloc::new_array(allocator, bool, (sz)model.config.vocab_size);
    defer alloc::free(allocator, allowed);

    bool terminated = false;
    usz skeleton_end = 0;

    // RIG_PROFILE=1 splits the decode wall clock into the two things it is
    // made of: the forward pass and the sampler. They are optimised
    // separately and the ratio is what says which one to touch.
    bool profile = false;
    if (try String flag = env::tget_var("RIG_PROFILE")) profile = flag == "1";
    double forward_s = 0.0;
    double sample_s = 0.0;
    double grammar_s = 0.0;
    Clock decode_clock = clock::now();

    for (uint step = 0; step < budget; step++) {
        // The penalty looks only at what has been generated, not at the start
        // tokens: bos is meant to appear once and penalising it changes nothing.
        uint[] history = tokens[start_tokens.len:produced - start_tokens.len];
        Clock op_clock = clock::now();
        grammar.allow_next(allowed);
        if (profile) grammar_s += op_clock.to_now().to_sec();

        if (profile) op_clock = clock::now();
        uint next = llm::sample_token(
            ctx, &model.acts.logits, model.config.vocab_size, &sampling, &rng, history, allowed)!;
        if (profile) sample_s += op_clock.to_now().to_sec();
        grammar.accept(next);
        tokens[produced] = next;
        produced++;

        if (next == TOKEN_EOS && !terminated) {
            terminated = true;
            skeleton_end = produced - 1;
        }
        if (grammar.finished()) break;
        if (produced == capacity) break;

        // Positions continue straight on from the prefix: the KV cache the
        // prefill filled and the one forward() appends to are the same.
        if (profile) op_clock = clock::now();
        model.forward(next, (uint)(n_tokens + (usz)step))!;
        if (profile) forward_s += op_clock.to_now().to_sec();
    }

    if (profile) {
        double total = decode_clock.to_now().to_sec();
        io::printfn(
            "  [profile] %d tokens in %.2fs (%.1f tok/s): forward %.2fs (%.1f ms/tok), sample %.2fs (%.1f ms/tok), grammar %.2fs",
            produced - start_tokens.len, total,
            (double)(produced - start_tokens.len) / total,
            forward_s, forward_s * 1000.0 / (double)(produced - start_tokens.len),
            sample_s, sample_s * 1000.0 / (double)(produced - start_tokens.len),
            grammar_s);
    }

    // No explicit free here: the `defer catch` above already runs on an error
    // return, and freeing as well is a double free.
    if (!terminated) return RIG_DECODE_STALLED~;

    RigResult result = {
        .normalization = normalization,
        .tokens = tokens[:produced],
        .prefix_tokens = n_tokens - start_tokens.len,
    };
    // A skin block counts only when the grammar saw it through to the model's
    // own terminator. Running out of budget partway leaves the skeleton intact
    // and the weights to whatever the caller falls back on.
    if (grammar.finished() && grammar.skin_emitted > 0 && grammar.skin_remaining() == 0) {
        result.skin_start = skeleton_end + 1;
        result.skin_count = (usz)grammar.skin_emitted;
    }
    result.skeleton = detokenize_skeleton(allocator, result.tokens)!;
    return result;
}

<*
 One decode attempt per seed, all in lockstep.

 A decode step's cost is its dispatch chain, not the work riding on it, so
 eight seeds decoded together cost about what one does. Every sequence starts
 from the same prefill, at the same position, and stays position-aligned for
 its whole life; a sequence whose grammar finishes simply drops out of the
 later steps. Per-sequence sampling, grammar state, and rng are untouched —
 each seed's token stream is the one the sequential loop would have produced.

 @param results : `One slot per seed, filled where the matching ok flag is set.`
 @param ok : `Whether that seed terminated in a complete skeleton.`
*>
fn void? decode_rig_attempts(
    alloc::Allocator allocator,
    llm::DeviceContext* ctx,
    llm_text::LlmModel* model,
    RigPrefill* prefill,
    Normalization normalization,
    RigOptions options,
    ulong[] seeds,
    RigResult[] results,
    bool[] ok)
{
    usz n = seeds.len;
    uint[2] start_tokens = prefill.start_tokens;
    usz n_tokens = prefill.n_tokens;
    uint budget = decode_budget(model, n_tokens, options.max_new_tokens);
    if (budget == 0) return RIG_CONTEXT_EXHAUSTED~;
    uint vocab = model.config.vocab_size;
    ulong logits_bytes = (ulong)vocab * 4;

    bool profile = false;
    if (try String flag = env::tget_var("RIG_PROFILE")) profile = flag == "1";
    Clock decode_clock = clock::now();

    // One batch, one row per seed, every row opening on the prefix's
    // attention state.
    llm_text::BatchDecode batch = model.create_batch_decode((uint)n)!;
    defer batch.free();
    model.batch_load_prefix(&batch, (uint)n_tokens)!;

    // Slotted staging the sampler reads from: forward_multi lands stream s's
    // logits at slot s. Step zero reads the prefill's logits instead — shared
    // by every sequence, and safe to share because the penalty has no history
    // to rewrite yet.
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: (sz)(logits_bytes * n),
    )!;
    defer staging.free();
    vk::Memory prefill_staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: (sz)logits_bytes,
    )!;
    defer prefill_staging.free();
    ctx.device.@single_time_command(ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, prefill.saved_logits.gpu_buffer.buffer, prefill_staging.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = logits_bytes }});
    }!!;

    // Per-sequence decode state, exactly the single attempt's, arrayed.
    usz capacity = start_tokens.len + (usz)budget;
    uint[][] seq_tokens = alloc::new_array(allocator, uint[], (sz)n);
    RigGrammar[] grammars = alloc::new_array(allocator, RigGrammar, (sz)n);
    random::Sfc64Random[] rngs = alloc::new_array(allocator, random::Sfc64Random, (sz)n);
    usz[] produced = alloc::new_array(allocator, usz, (sz)n);
    usz[] skeleton_end = alloc::new_array(allocator, usz, (sz)n);
    bool[] terminated = alloc::new_array(allocator, bool, (sz)n);
    bool[] done = alloc::new_array(allocator, bool, (sz)n);
    uint[] step_tokens = alloc::new_array(allocator, uint, (sz)n);
    bool[] allowed = alloc::new_array(allocator, bool, (sz)vocab);
    defer {
        alloc::free(allocator, grammars);
        alloc::free(allocator, rngs);
        alloc::free(allocator, produced);
        alloc::free(allocator, skeleton_end);
        alloc::free(allocator, terminated);
        alloc::free(allocator, done);
        alloc::free(allocator, step_tokens);
        alloc::free(allocator, allowed);
        alloc::free(allocator, seq_tokens);
    }
    defer catch {
        for (usz i = 0; i < n; i++) {
            if (seq_tokens[i].len > 0) alloc::free(allocator, seq_tokens[i]);
        }
    }

    for (usz i = 0; i < n; i++) {
        seq_tokens[i] = alloc::new_array(allocator, uint, (sz)capacity);
        grammars[i] = { .tokens_per_skin = options.predict_skin ? TOKENS_PER_SKIN : 0 };
        random::seed(&rngs[i], seeds[i]);
        foreach (token : start_tokens) {
            grammars[i].accept(token);
            seq_tokens[i][produced[i]] = token;
            produced[i]++;
        }
    }

    llm::SamplingParams sampling = options.sampling;
    usz steps = 0;

    for (uint step = 0; step < budget; step++) {
        // Sample each running sequence from its slot; a finished one keeps
        // repeating its last token, which its row decodes into results nobody
        // reads — with the stages batched, a shorter batch would not make the
        // step any cheaper.
        usz n_running = 0;
        for (usz i = 0; i < n; i++) {
            if (done[i]) {
                step_tokens[i] = seq_tokens[i][produced[i] - 1];
                continue;
            }
            uint[] history = seq_tokens[i][start_tokens.len:produced[i] - start_tokens.len];
            grammars[i].allow_next(allowed);
            float* data = step == 0
                ? (float*)prefill_staging.data()
                : (float*)staging.data() + i * (usz)vocab;
            uint next = llm::sample_token_data(data, vocab, &sampling, &rngs[i], history, allowed)!;
            grammars[i].accept(next);
            seq_tokens[i][produced[i]] = next;
            produced[i]++;
            step_tokens[i] = next;

            if (next == TOKEN_EOS && !terminated[i]) {
                terminated[i] = true;
                skeleton_end[i] = produced[i] - 1;
            }
            if (grammars[i].finished() || produced[i] == capacity) {
                done[i] = true;
                continue;
            }
            n_running++;
        }
        if (n_running == 0) break;

        model.forward_batch(&batch, step_tokens, (uint)(n_tokens + (usz)step), &staging)!;
        steps++;
    }

    if (profile) {
        io::printfn("  [profile] %d seeds in %d lockstep steps, %.2fs",
            n, steps, decode_clock.to_now().to_sec());
    }

    for (usz i = 0; i < n; i++) {
        ok[i] = false;
        if (terminated[i]) {
            RigResult result = {
                .normalization = normalization,
                .tokens = seq_tokens[i][:produced[i]],
                .prefix_tokens = n_tokens - start_tokens.len,
            };
            if (grammars[i].finished() && grammars[i].skin_emitted > 0 && grammars[i].skin_remaining() == 0) {
                result.skin_start = skeleton_end[i] + 1;
                result.skin_count = (usz)grammars[i].skin_emitted;
            }
            if (try skeleton = detokenize_skeleton(allocator, result.tokens)) {
                result.skeleton = skeleton;
                results[i] = result;
                ok[i] = true;
            }
        }
        // A stalled or unparseable sequence keeps nothing.
        if (!ok[i] && seq_tokens[i].len > 0) {
            alloc::free(allocator, seq_tokens[i]);
            seq_tokens[i] = {};
        }
    }
}

<*
 The generation half on its own, for a prefix that is already encoded.

 Prefill and one decode, bundled. A caller that decodes the same prefix more
 than once — a seed search, or a skin pass re-decoding the winner — should
 hold a `RigPrefill` and call `decode_rig_attempt` instead: the prefill is
 514 tokens through every layer and the decode does not disturb it, so paying
 for it per attempt buys nothing.
*>
fn RigResult? generate_rig_from_prefix(
    alloc::Allocator allocator,
    llm::DeviceContext* ctx,
    llm_text::LlmModel* model,
    float[] prefix,
    Normalization normalization,
    RigOptions options = DEFAULT_RIG)
{
    RigPrefill prefill = prepare_rig_prefill(ctx, model, prefix, options)!;
    defer prefill.free();
    return decode_rig_attempt(allocator, ctx, model, &prefill, normalization, options);
}


