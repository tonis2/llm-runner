module llm;

import vk;
import std::io;
import std::core::mem;
import std::math;
import std::math::random;
import std::sort;

faultdef NO_ALLOWED_TOKEN;

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

// Above this k the bounded insertion below costs more than sorting outright.
const uint TOPK_SELECT_LIMIT @local = 1024;

// An empty mask allows everything; a short one allows nothing past its end.
fn bool token_allowed(bool[] allowed, uint id) @local @inline {
    if (allowed.len == 0) return true;
    return (usz)id < allowed.len && allowed[id];
}

// --- Main sampling function ---

<*
 Pick the next token from logits still on the device.

 A readback and a call to `sample_token_data`; callers that already hold the
 logits on the host — a multi-stream decode copies every stream's logits out
 in the forward submission itself — go to that one directly.
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
    defer staging.free();

    ctx.device.@single_time_command(queue: ctx.compute_queue; CommandBuffer cmd) {
        vk::cmdCopyBuffer(cmd, logits.gpu_buffer.buffer, staging.buffer, 1,
            (BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = size }});
    }!!;

    return sample_token_data((float*)staging.data(), vocab_size, params, rng, history, allowed);
}

<*
 Pick the next token from host-visible logits.

 `data` is scratch: the repetition penalty rewrites the logits it punishes,
 so a slot must not be reused for a second sample.

 @param history : `Tokens already generated, for the repetition penalty; empty to skip it.`
 @param allowed : `Per-token legality mask; empty to allow everything.`
*>
fn uint? sample_token_data(float* data, uint vocab_size, SamplingParams* params, random::Sfc64Random* rng, uint[] history = {}, bool[] allowed = {}) {
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

    // Constrained decoding: a token the caller ruled out is not a candidate
    // at all. The obvious alternative -- write a sentinel over its logit and
    // let the sort bury it -- is what this used to do, and it is a trap: the
    // rig grammar leaves a few hundred ids legal out of 33036, so the array
    // handed to quicksort was ~33000 copies of one value, which is the input
    // that turns quicksort quadratic. 546M comparisons, 645 ms a token, on a
    // model whose entire forward pass costs 26 ms.
    uint n_allowed = vocab_size;
    if (allowed.len > 0) {
        n_allowed = 0;
        uint limit = (usz)vocab_size < allowed.len ? vocab_size : (uint)allowed.len;
        for (uint i = 0; i < limit; i++) if (allowed[i]) n_allowed++;
        if (n_allowed == 0) return NO_ALLOWED_TOKEN~;
    }

    // Temperature 0 = greedy (argmax)
    if (params.temperature <= 0.0f) {
        float max_val = 0.0f;
        uint max_idx = vocab_size;
        for (uint i = 0; i < vocab_size; i++) {
            if (!token_allowed(allowed, i)) continue;
            if (max_idx == vocab_size || data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Gather and rank, in one pass over the vocabulary. Top-k bounds how much
    // of the ranking is ever materialised: with k=10 the insertion below
    // touches ten slots and rejects everything below the running tenth-best
    // on a single compare, so nothing gets sorted that cannot be sampled.
    uint n = params.top_k > 0 && params.top_k < n_allowed ? params.top_k : n_allowed;
    TokenCandidate[] candidates = mem::new_array(TokenCandidate, (sz)n);
    defer mem::free(candidates);
    usz filled = 0;

    if (n < n_allowed && n <= TOPK_SELECT_LIMIT) {
        for (uint i = 0; i < vocab_size; i++) {
            if (!token_allowed(allowed, i)) continue;
            float logit = data[i];
            if (filled == (usz)n && logit <= candidates[filled - 1].logit) continue;
            usz at = filled < (usz)n ? filled : filled - 1;
            while (at > 0 && candidates[at - 1].logit < logit) {
                candidates[at] = candidates[at - 1];
                at--;
            }
            candidates[at] = { .logit = logit, .id = i };
            if (filled < (usz)n) filled++;
        }
    } else {
        // Unbounded (or a k so large the insertion would cost more than a
        // sort): materialise every legal candidate and sort it. Real logits
        // are distinct, so this is the case quicksort is good at.
        for (uint i = 0; i < vocab_size && filled < (usz)n; i++) {
            if (!token_allowed(allowed, i)) continue;
            candidates[filled] = { .logit = data[i], .id = i };
            filled++;
        }
        quicksort(candidates, cmp: &cmp_candidates_desc);
    }
    n = (uint)filled;

    // Temperature scales every surviving logit by the same positive factor,
    // so applying it after the ranking gives the same order and the same
    // distribution as applying it to the whole vocabulary first.
    float inv_temp = 1.0f / params.temperature;
    for (uint i = 0; i < n; i++) candidates[i].logit *= inv_temp;

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

    return result;
}

