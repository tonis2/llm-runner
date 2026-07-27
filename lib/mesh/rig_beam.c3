<*
 Beam search over rig token sequences.

 Sampling one sequence cannot recover from a bad early token, and a skeleton is
 strictly sequential: get one joint wrong and every bone that hangs off it is
 wrong too. That is why the checkpoint's `generate_kwargs` ask for ten beams —
 it is not a quality knob, it is the difference between a rig and a chain.

 Each beam carries its own attention state, so a step is:

     for each live beam   forward its last token on its own KV cache
     score                beam log-prob + log p(token), grammar-masked
     select               keep the best `beams`, retire any that hit eos
     reorder              give each survivor its parent's cache

 The reorder is where the cost is. A beam that keeps its own slot costs
 nothing; only a beam descending from a slot someone else claimed needs a copy,
 and then only of the positions generated so far — the prefix is identical
 across beams and is written once.
*>
module llm::mesh;

import vk;
import llm;
import llm_text;
import std::io;
import std::core::mem;
import std::core::mem::allocator;
import std::math;
import std::math::vector;

faultdef RIG_NO_BEAM_FINISHED, RIG_TOO_MANY_BEAMS;

const usz MAX_BEAMS = 16;

<*
 Beam search settings. `beams` of 1 degenerates to greedy decoding.

 `length_penalty` divides a finished beam's score by `len^penalty` before
 comparing. Longer sequences accumulate more negative log-probability, so
 without it the search always crowns the shortest rig that manages to stop —
 and the shortest legal skeleton is a single bone.
*>
struct BeamOptions {
    usz beams;
    float length_penalty;
    float temperature;
    float repetition_penalty;
    uint max_new_tokens;
    RigClass rig_class;
}

const BeamOptions DEFAULT_BEAM = {
    .beams = 8,
    .length_penalty = 1.0f,
    .temperature = 1.0f,
    .repetition_penalty = 2.0f,
    .max_new_tokens = DEFAULT_MAX_NEW_TOKENS,
    .rig_class = NONE,
};

// A live hypothesis. `tokens` points into one of two pools that swap each
// step, because a survivor reads its parent's tokens while writing its own and
// the two would otherwise be the same array.
struct Beam @local {
    uint[] tokens;
    usz len;
    double score;
    RigGrammar grammar;
    usz slot;
}

struct BeamCandidate @local {
    usz parent;
    uint token;
    double score;
}

// log(sum(exp)) over the masked, temperature-scaled logits, so scores from
// different beams are comparable. In double: a beam accumulates one of these
// per token, and a couple of hundred float additions visibly drift.
fn double log_normaliser(float[] logits, bool[] allowed, float temperature) @local {
    float highest = -1.0e30f;
    foreach (i, value : logits) {
        if (!allowed[i]) continue;
        float scaled = value / temperature;
        if (scaled > highest) highest = scaled;
    }

    double total = 0;
    foreach (i, value : logits) {
        if (!allowed[i]) continue;
        total += math::exp((double)(value / temperature - highest));
    }
    return (double)highest + math::ln(total);
}

<*
 Generate a rig by beam search over an encoded prefix.

 The model must have been loaded with `max_seq_len` covering the prefix plus
 `max_new_tokens`. Each beam gets its own KV cache —
 `n_kv_heads * max_seq_len * head_dim * 2` floats, about 235 MB at 1024
 positions for this model — so beam count is a memory decision.

 @return? ENCODER_SHAPE_MISMATCH : `The prefix is not a whole number of rows.`
 @return? RIG_TOO_MANY_BEAMS : `More beams than MAX_BEAMS, or none.`
 @return? RIG_PREFIX_TOO_LONG : `The prefix does not fit the prefill path.`
 @return? RIG_NO_BEAM_FINISHED : `No beam reached eos within max_new_tokens.`
 @return? RIG_CONTEXT_EXHAUSTED : `The prefix alone fills the model's KV cache.`
*>
fn RigResult? generate_rig_beam(
    alloc::Allocator allocator,
    llm::DeviceContext* ctx,
    llm_text::LlmModel* model,
    float[] prefix,
    Normalization normalization,
    BeamOptions options = DEFAULT_BEAM)
{
    if (options.beams == 0 || options.beams > MAX_BEAMS) return RIG_TOO_MANY_BEAMS~;

    uint dim = model.config.dim;
    usz vocab = (usz)model.config.vocab_size;
    usz latents = prefix.len / (usz)dim;
    if (latents == 0 || prefix.len % (usz)dim != 0) return ENCODER_SHAPE_MISMATCH~;

    uint[2] start_tokens = { TOKEN_BOS, options.rig_class.token() };
    usz n_prefix = latents + start_tokens.len;
    if (!model.prefill_supported((uint)n_prefix)) return RIG_PREFIX_TOO_LONG~;
    uint budget = decode_budget(model, n_prefix, options.max_new_tokens);
    if (budget == 0) return RIG_CONTEXT_EXHAUSTED~;

    // --- One prefill, shared by every beam ---

    llm::Tensor hidden = llm::create_f32_tensor(ctx, { (ulong)(n_prefix * dim), 0, 0, 0 }, 1)!;
    defer hidden.free();
    llm::upload_to_tensor(ctx, &hidden, prefix)!;

    ulong row_bytes = (ulong)dim * 4;
    llm::RenderGraph gather;
    gather.init(ctx)!;
    foreach (usz slot, token : start_tokens) {
        gather.copy_region(&model.weights.token_embedding, (usz)((ulong)token * row_bytes),
            &hidden, (usz)((ulong)(latents + slot) * row_bytes), row_bytes);
    }
    gather.flush()!;

    model.prefill_embeddings(&hidden, (uint)n_prefix)!;

    float[] prefill_logits = llm::download_tensor(allocator, ctx, &model.acts.logits, vocab)!;
    defer alloc::free(allocator, prefill_logits);

    // --- One cache per beam, each seeded with the shared prefix ---

    llm_text::KVCache original = model.kv;
    llm_text::KVCache[MAX_BEAMS] caches;
    caches[0] = original;
    usz extra_caches = 0;
    defer {
        for (usz i = 0; i < extra_caches; i++) caches[i + 1].free();
        model.kv = original;
    }
    for (usz i = 1; i < options.beams; i++) {
        caches[i] = llm_text::allocate_kv_cache(ctx, &model.config, model.max_seq_len)!;
        extra_caches++;
        model.copy_kv_range(&caches[0], &caches[i], 0, (uint)n_prefix)!;
    }

    // --- Working memory ---

    usz capacity = start_tokens.len + (usz)budget + 1;
    uint[] pool_a = alloc::new_array(allocator, uint, (sz)(capacity * options.beams));
    uint[] pool_b = alloc::new_array(allocator, uint, (sz)(capacity * options.beams));
    bool[] allowed = alloc::new_array(allocator, bool, (sz)vocab);
    float[] beam_logits = alloc::new_array(allocator, float, (sz)vocab);
    uint[] best_tokens = alloc::new_array(allocator, uint, (sz)capacity);
    defer {
        alloc::free(allocator, pool_a);
        alloc::free(allocator, pool_b);
        alloc::free(allocator, allowed);
        alloc::free(allocator, beam_logits);
    }
    defer catch alloc::free(allocator, best_tokens);

    Beam[MAX_BEAMS] beams;
    for (usz i = 0; i < options.beams; i++) {
        beams[i] = { .tokens = pool_a[i * capacity : capacity], .slot = i };
        foreach (token : start_tokens) {
            beams[i].tokens[beams[i].len] = token;
            beams[i].len++;
            beams[i].grammar.accept(token);
        }
    }

    // Every beam starts identical, so only one is live for the first step.
    // Expanding all of them would just produce the same token `beams` times
    // and the search would never widen.
    usz live = 1;
    bool use_pool_a = true;

    double best_score = -1.0e300;
    usz best_len = 0;
    usz finished = 0;

    BeamCandidate[MAX_BEAMS * 2] candidates;

    for (uint step = 0; step < budget; step++) {
        usz limit = options.beams * 2;
        usz candidate_count = 0;

        for (usz b = 0; b < live; b++) {
            if (step == 0) {
                foreach (i, value : prefill_logits) beam_logits[i] = value;
            } else {
                model.kv = caches[beams[b].slot];
                model.forward(beams[b].tokens[beams[b].len - 1], (uint)(n_prefix + (usz)step - 1))!;
                float[] fresh = llm::download_tensor(allocator, ctx, &model.acts.logits, vocab)!;
                foreach (i, value : fresh) beam_logits[i] = value;
                alloc::free(allocator, fresh);
            }

            if (options.repetition_penalty > 1.0f) {
                for (usz t = start_tokens.len; t < beams[b].len; t++) {
                    uint token = beams[b].tokens[t];
                    beam_logits[token] = beam_logits[token] < 0
                        ? beam_logits[token] * options.repetition_penalty
                        : beam_logits[token] / options.repetition_penalty;
                }
            }
            beams[b].grammar.allow_next(allowed);

            double normaliser = log_normaliser(beam_logits, allowed, options.temperature);

            for (usz t = 0; t < vocab; t++) {
                if (!allowed[t]) continue;
                double score = beams[b].score
                    + (double)(beam_logits[t] / options.temperature) - normaliser;

                if (candidate_count < limit) {
                    candidates[candidate_count] = { .parent = b, .token = (uint)t, .score = score };
                    candidate_count++;
                    continue;
                }
                usz weakest = 0;
                for (usz i = 1; i < limit; i++) {
                    if (candidates[i].score < candidates[weakest].score) weakest = i;
                }
                if (score > candidates[weakest].score) {
                    candidates[weakest] = { .parent = b, .token = (uint)t, .score = score };
                }
            }
        }

        // Best first, so selection is a walk down the list.
        for (usz i = 0; i < candidate_count; i++) {
            for (usz j = i + 1; j < candidate_count; j++) {
                if (candidates[j].score > candidates[i].score) {
                    BeamCandidate swap = candidates[i];
                    candidates[i] = candidates[j];
                    candidates[j] = swap;
                }
            }
        }

        // Survivors, in order, each still naming its parent rather than
        // referring to a candidate index — an eos in the middle of the list
        // shifts everything after it.
        BeamCandidate[MAX_BEAMS] survivors;
        usz survivor_count = 0;

        for (usz i = 0; i < candidate_count && survivor_count < options.beams; i++) {
            BeamCandidate candidate = candidates[i];

            if (candidate.token == TOKEN_EOS || candidate.token == TOKEN_SEQUENCE_END) {
                usz produced = beams[candidate.parent].len - start_tokens.len + 1;
                double normalised = candidate.score
                    / math::pow((double)produced, (double)options.length_penalty);
                if (normalised > best_score && beams[candidate.parent].len + 1 <= capacity) {
                    best_score = normalised;
                    best_len = beams[candidate.parent].len;
                    for (usz t = 0; t < best_len; t++) {
                        best_tokens[t] = beams[candidate.parent].tokens[t];
                    }
                    best_tokens[best_len] = candidate.token;
                    best_len++;
                }
                finished++;
                continue;
            }

            survivors[survivor_count] = candidate;
            survivor_count++;
        }

        if (survivor_count == 0 || finished >= options.beams) break;

        // --- Reorder the caches ---
        //
        // A survivor that is the only claimant of its parent's slot keeps it
        // for free. The rest copy, and only the generated span: every slot's
        // prefix is identical and never changes.
        bool[MAX_BEAMS] claimed;
        usz[MAX_BEAMS] assigned;
        for (usz i = 0; i < MAX_BEAMS; i++) {
            claimed[i] = false;
            assigned[i] = MAX_BEAMS;
        }

        for (usz i = 0; i < survivor_count; i++) {
            usz parent_slot = beams[survivors[i].parent].slot;
            if (!claimed[parent_slot]) {
                claimed[parent_slot] = true;
                assigned[i] = parent_slot;
            }
        }
        for (usz i = 0; i < survivor_count; i++) {
            if (assigned[i] != MAX_BEAMS) continue;
            usz free_slot = 0;
            while (free_slot < options.beams && claimed[free_slot]) free_slot++;
            // Counting says this cannot happen — pass one claims one slot per
            // distinct parent, so at least as many remain as beams still need
            // one — but reading caches[options.beams] would be a zeroed struct
            // and a segfault rather than a wrong answer.
            if (free_slot >= options.beams) return RIG_TOO_MANY_BEAMS~;
            // A slot nobody claimed holds state nobody needs, so overwriting it
            // is safe, and its source is never itself.
            model.copy_kv_range(&caches[beams[survivors[i].parent].slot], &caches[free_slot],
                (uint)n_prefix, (uint)(n_prefix + (usz)step))!;
            claimed[free_slot] = true;
            assigned[i] = free_slot;
        }

        // --- Materialise into the other pool ---
        //
        // Survivors read their parents' token arrays while writing their own;
        // sharing one pool would let a beam overwrite a parent another survivor
        // has not read yet.
        uint[] destination = use_pool_a ? pool_b : pool_a;
        Beam[MAX_BEAMS] next;

        for (usz i = 0; i < survivor_count; i++) {
            Beam* parent = &beams[survivors[i].parent];
            uint[] storage = destination[i * capacity : capacity];

            for (usz t = 0; t < parent.len; t++) storage[t] = parent.tokens[t];
            storage[parent.len] = survivors[i].token;

            next[i] = {
                .tokens = storage,
                .len = parent.len + 1,
                .score = survivors[i].score,
                .grammar = parent.grammar,
                .slot = assigned[i],
            };
            next[i].grammar.accept(survivors[i].token);
        }

        for (usz i = 0; i < survivor_count; i++) beams[i] = next[i];
        live = survivor_count;
        use_pool_a = !use_pool_a;
    }

    if (best_len == 0) return RIG_NO_BEAM_FINISHED~;

    RigResult result = {
        .normalization = normalization,
        .tokens = best_tokens[:best_len],
        .prefix_tokens = latents,
    };
    result.skeleton = detokenize_skeleton(allocator, result.tokens)!;
    return result;
}
