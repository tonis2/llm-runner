module llm;

import vk;
import std::io;
import std::core::mem;
import std::math;
import std::math::random;
import std::sort;


fn void? debug_readback(DeviceContext* ctx, Tensor* tensor, String label, uint count) {
    usz size = (usz)count * 4;
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: (sz)size,
    )!!;

    ctx.device.@single_time_command(queue: ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, tensor.gpu_buffer.buffer, staging.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = size }});
    }!!;

    float* data = (float*)staging.data();
    io::printf("  %s:", label);
    for (uint i = 0; i < count; i++) {
        io::printf(" %.6f", data[i]);
    }
    io::printfn("");
    staging.free();
}

// --- Sampling parameters ---

struct SamplingParams {
    float temperature;
    uint top_k;
    float top_p;
    // Divides the logit of anything already generated, so a model that likes
    // one token cannot spend the whole sequence on it. 1.0 (and 0.0, which is
    // what a caller that predates this field leaves behind) means off.
    float repetition_penalty;
}

struct TokenCandidate {
    float logit;
    uint id;
}

// Comparator for descending sort by logit
fn int cmp_candidates_desc(TokenCandidate a, TokenCandidate b) {
    if (a.logit > b.logit) return -1;
    if (a.logit < b.logit) return 1;
    return 0;
}

// --- Main sampling function ---

<*
 Pick the next token.

 @param history : `Tokens already generated, for the repetition penalty; empty to skip it.`
 @param allowed : `Per-token legality mask; empty to allow everything.`
*>
fn uint? sample_token(DeviceContext* ctx, Tensor* logits, uint vocab_size, SamplingParams* params, random::Sfc64Random* rng, uint[] history = {}, bool[] allowed = {}) {
    usz size = (usz)vocab_size * 4;

    // Read logits from GPU
    vk::Memory staging = vk::new_buffer(
        allocator: &ctx.allocator,
        usage: BUFFER_USAGE_TRANSFER_DST_BIT,
        properties: MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
        data: null,
        data_size: (sz)size,
    )!!;

    ctx.device.@single_time_command(queue: ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, logits.gpu_buffer.buffer, staging.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = size }});
    }!!;

    float* data = (float*)staging.data();

    // Repetition penalty, before temperature and before any greedy shortcut:
    // the point is to change which token wins, so applying it after the
    // argmax would do nothing at all. Sign matters — dividing a negative
    // logit would make the token more likely, not less.
    // Once per *distinct* token, not once per occurrence. HuggingFace gathers
    // the original logit for every position, penalises that, and scatters it
    // back, so a token repeated k times is still only divided once. Applying
    // it per occurrence compounds to penalty^k, which at 2.0 buries any token
    // the grammar has to reuse -- the branch marker above all, since one id
    // serves every fork while coordinates spread over 256.
    if (params.repetition_penalty > 1.0f) {
        foreach (usz i, token : history) {
            if (token >= vocab_size) continue;
            bool already = false;
            for (usz j = 0; j < i; j++) {
                if (history[j] == token) { already = true; break; }
            }
            if (already) continue;
            data[token] = data[token] < 0.0f
                ? data[token] * params.repetition_penalty
                : data[token] / params.repetition_penalty;
        }
    }

    // Constrained decoding: anything the caller ruled out is pushed below the
    // range any real logit occupies, so it cannot survive top-k either.
    if (allowed.len > 0) {
        for (uint i = 0; i < vocab_size; i++) {
            if (i >= allowed.len || !allowed[i]) data[i] = -1.0e30f;
        }
    }

    // Temperature 0 = greedy (argmax)
    if (params.temperature <= 0.0f) {
        float max_val = data[0];
        uint max_idx = 0;
        for (uint i = 1; i < vocab_size; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        staging.free();
        return max_idx;
    }

    // Apply temperature
    float inv_temp = 1.0f / params.temperature;
    for (uint i = 0; i < vocab_size; i++) {
        data[i] *= inv_temp;
    }

    // Build candidate array
    TokenCandidate[] candidates = mem::new_array(TokenCandidate, vocab_size);

    for (uint i = 0; i < vocab_size; i++) {
        candidates[i] = { .logit = data[i], .id = i };
    }

    // Sort descending by logit
    quicksort(candidates, cmp: &cmp_candidates_desc);

    // Apply top-k
    uint n = vocab_size;
    if (params.top_k > 0 && params.top_k < n) {
        n = params.top_k;
    }

    // Softmax over top candidates
    float max_logit = candidates[0].logit;
    float sum = 0.0f;
    for (uint i = 0; i < n; i++) {
        float val = math::exp(candidates[i].logit - max_logit);
        candidates[i].logit = val;
        sum += val;
    }
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < n; i++) {
        candidates[i].logit *= inv_sum;
    }

    // Apply top-p (nucleus sampling)
    if (params.top_p < 1.0f && params.top_p > 0.0f) {
        float cumsum = 0.0f;
        for (uint i = 0; i < n; i++) {
            cumsum += candidates[i].logit;
            if (cumsum >= params.top_p) {
                n = i + 1;
                break;
            }
        }
        // Re-normalize
        sum = 0.0f;
        for (uint i = 0; i < n; i++) sum += candidates[i].logit;
        inv_sum = 1.0f / sum;
        for (uint i = 0; i < n; i++) candidates[i].logit *= inv_sum;
    }

    // Sample from distribution
    float r = random::next_float(rng);
    float cumsum = 0.0f;
    uint result = candidates[n - 1].id;
    for (uint i = 0; i < n; i++) {
        cumsum += candidates[i].logit;
        if (r <= cumsum) {
            result = candidates[i].id;
            break;
        }
    }

    mem::free(candidates);
    staging.free();
    return result;
}

