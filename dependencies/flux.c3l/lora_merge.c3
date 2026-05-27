module flux;

import std::io;
import std::time;
import vk;
import llm;
import zimage;

// One-shot LoRA fold into Q8_0 base weights.
//
// Background: at runtime, each LoraTensor adds two extra GPU dispatches per
// forward call per site (hop-1 matvec, hop-2 add-scaled). With ~112 sites in
// a Flux 2 Klein file, that's 224 extra dispatches per step. The hop-1 kernel
// in particular is wasteful: 256-thread tree-reduction workgroups producing
// one output each.
//
// The classical fix is to bake the correction `W' = W + scale * B @ A` into
// the base weight once at load time and discard the LoRA adapter. We pay one
// pass over each affected weight and then run subsequent steps at full speed.
//
// Implementation: one Q8_0-aware merge kernel handles each site in place. For
// each (output_row, 256-elem input chunk) it dequantises the existing block,
// adds the LoRA delta, recomputes the per-block scale, and re-packs Q8_0.
// Sub-range sites (e.g. `img_q` writing into rows [0, dim) of fused
// img_qkv [3*dim, dim]) are handled by passing the row offset through the
// kernel's push constants.
//
// Non-Q8_0 base weights aren't merged here — they fall through and the LoRA
// adapter remains in play for those sites. The Flux 2 Klein Q8_0 GGUF has
// all relevant linears in Q8_0, so this fallback should never trigger.

// Dispatch the merge kernel for a single (base_weight, LoraTensor) pair.
// `dst_row_offset` is where the LoRA's b_rows land in the base weight (0
// when the LoRA targets a non-fused weight directly, or dim/2*dim/etc. when
// it targets a sub-range of a fused QKV/linear1).
fn void merge_one_site(
    llm::RenderGraph* g,
    FluxDiTKernels* dk,
    llm::Tensor* base,
    llm::LoraTensor* lora,
    uint dst_row_offset
) {
    if (!lora.present) return;
    if (base.dtype != llm::GGML_Q8_0) {
        // Fallback path not implemented — leave LoRA dispatchable at runtime
        // by keeping lora.present = true. Callers should notice and skip the
        // free.
        return;
    }

    // base shape (GGUF [col, row]): shape[0] = in_dim, shape[1] = out_dim
    uint in_dim = (uint)base.shape[0];
    // B shape after upload reversal: shape[0] = rank, shape[1] = b_rows
    uint b_rows = (uint)lora.b.shape[1];

    if ((in_dim % 256) != 0) {
        io::printfn("    LoRA merge: in_dim=%d not a multiple of 256, skipping", in_dim);
        return;
    }
    if (lora.rank > 256) {
        io::printfn("    LoRA merge: rank=%d > 256, skipping", lora.rank);
        return;
    }

    zimage::Q8LoraMergePC pc = {
        .in_dim         = in_dim,
        .b_rows         = b_rows,
        .dst_row_offset = dst_row_offset,
        .rank           = lora.rank,
        .scale          = lora.scale,
    };
    uint chunks = in_dim / 256;
    g.dispatch(&dk.dit.q8_lora_merge, { base, &lora.a, &lora.b }, &pc, chunks, b_rows);
    lora.present = false;
}

// Fold every present LoRA site in `model.lora` into the Q8_0 DiT weights
// and free the adapter. Sets `model.lora.loaded = false` so the runtime
// dispatch_lora_correction paths skip cleanly. Idempotent: no-op when no
// LoRA is loaded.
fn void? flux_merge_lora_into_weights(FluxDiTModel* model) {
    if (!model.lora.loaded) return;

    Clock t0 = clock::now();

    uint dim = flux_dim;

    llm::RenderGraph g;
    g.init(model.ctx)!!;

    // We submit and wait once at the end via g.flush(). Each merge dispatch
    // touches a disjoint slice of GPU memory so they can pipeline freely.
    for (uint l = 0; l < model.n_dual_blocks; l++) {
        FluxDualBlockWeights* bw = &model.dual_blocks[l];
        llm::FluxDualLoraLayer* lo = &model.lora.dual_layers[l];

        // Fused img_qkv [3*dim, dim] — direct + three slot sub-ranges.
        merge_one_site(&g, model.dk, &bw.img_qkv_w, &lo.img_qkv, 0);
        merge_one_site(&g, model.dk, &bw.img_qkv_w, &lo.img_q,   0);
        merge_one_site(&g, model.dk, &bw.img_qkv_w, &lo.img_k,   dim);
        merge_one_site(&g, model.dk, &bw.img_qkv_w, &lo.img_v,   2 * dim);
        merge_one_site(&g, model.dk, &bw.img_proj_w,     &lo.img_proj,     0);
        merge_one_site(&g, model.dk, &bw.img_mlp_up_w,   &lo.img_mlp_up,   0);
        merge_one_site(&g, model.dk, &bw.img_mlp_down_w, &lo.img_mlp_down, 0);

        merge_one_site(&g, model.dk, &bw.txt_qkv_w, &lo.txt_qkv, 0);
        merge_one_site(&g, model.dk, &bw.txt_qkv_w, &lo.txt_q,   0);
        merge_one_site(&g, model.dk, &bw.txt_qkv_w, &lo.txt_k,   dim);
        merge_one_site(&g, model.dk, &bw.txt_qkv_w, &lo.txt_v,   2 * dim);
        merge_one_site(&g, model.dk, &bw.txt_proj_w,     &lo.txt_proj,     0);
        merge_one_site(&g, model.dk, &bw.txt_mlp_up_w,   &lo.txt_mlp_up,   0);
        merge_one_site(&g, model.dk, &bw.txt_mlp_down_w, &lo.txt_mlp_down, 0);
    }

    for (uint l = 0; l < model.n_single_blocks; l++) {
        FluxSingleBlockWeights* sw = &model.single_blocks[l];
        llm::FluxSingleLoraLayer* lo = &model.lora.single_layers[l];

        merge_one_site(&g, model.dk, &sw.linear1_w, &lo.linear1, 0);
        merge_one_site(&g, model.dk, &sw.linear1_w, &lo.q,       0);
        merge_one_site(&g, model.dk, &sw.linear1_w, &lo.k,       dim);
        merge_one_site(&g, model.dk, &sw.linear1_w, &lo.v,       2 * dim);
        merge_one_site(&g, model.dk, &sw.linear2_w, &lo.linear2, 0);
    }

    g.flush()!!;

    // Now walk the adapter and count sites that ended up merged (present
    // cleared) vs skipped. Also free the F32 A/B buffers — they're done.
    uint total_sites_merged = 0;
    for (uint l = 0; l < model.n_dual_blocks; l++) {
        llm::FluxDualLoraLayer* lo = &model.lora.dual_layers[l];
        if (!lo.img_qkv.present      && lo.img_qkv.rank      > 0) { lo.img_qkv.a.free();      lo.img_qkv.b.free();      total_sites_merged++; }
        if (!lo.img_q.present        && lo.img_q.rank        > 0) { lo.img_q.a.free();        lo.img_q.b.free();        total_sites_merged++; }
        if (!lo.img_k.present        && lo.img_k.rank        > 0) { lo.img_k.a.free();        lo.img_k.b.free();        total_sites_merged++; }
        if (!lo.img_v.present        && lo.img_v.rank        > 0) { lo.img_v.a.free();        lo.img_v.b.free();        total_sites_merged++; }
        if (!lo.img_proj.present     && lo.img_proj.rank     > 0) { lo.img_proj.a.free();     lo.img_proj.b.free();     total_sites_merged++; }
        if (!lo.img_mlp_up.present   && lo.img_mlp_up.rank   > 0) { lo.img_mlp_up.a.free();   lo.img_mlp_up.b.free();   total_sites_merged++; }
        if (!lo.img_mlp_down.present && lo.img_mlp_down.rank > 0) { lo.img_mlp_down.a.free(); lo.img_mlp_down.b.free(); total_sites_merged++; }
        if (!lo.txt_qkv.present      && lo.txt_qkv.rank      > 0) { lo.txt_qkv.a.free();      lo.txt_qkv.b.free();      total_sites_merged++; }
        if (!lo.txt_q.present        && lo.txt_q.rank        > 0) { lo.txt_q.a.free();        lo.txt_q.b.free();        total_sites_merged++; }
        if (!lo.txt_k.present        && lo.txt_k.rank        > 0) { lo.txt_k.a.free();        lo.txt_k.b.free();        total_sites_merged++; }
        if (!lo.txt_v.present        && lo.txt_v.rank        > 0) { lo.txt_v.a.free();        lo.txt_v.b.free();        total_sites_merged++; }
        if (!lo.txt_proj.present     && lo.txt_proj.rank     > 0) { lo.txt_proj.a.free();     lo.txt_proj.b.free();     total_sites_merged++; }
        if (!lo.txt_mlp_up.present   && lo.txt_mlp_up.rank   > 0) { lo.txt_mlp_up.a.free();   lo.txt_mlp_up.b.free();   total_sites_merged++; }
        if (!lo.txt_mlp_down.present && lo.txt_mlp_down.rank > 0) { lo.txt_mlp_down.a.free(); lo.txt_mlp_down.b.free(); total_sites_merged++; }
    }
    for (uint l = 0; l < model.n_single_blocks; l++) {
        llm::FluxSingleLoraLayer* lo = &model.lora.single_layers[l];
        if (!lo.linear1.present && lo.linear1.rank > 0) { lo.linear1.a.free(); lo.linear1.b.free(); total_sites_merged++; }
        if (!lo.q.present       && lo.q.rank       > 0) { lo.q.a.free();       lo.q.b.free();       total_sites_merged++; }
        if (!lo.k.present       && lo.k.rank       > 0) { lo.k.a.free();       lo.k.b.free();       total_sites_merged++; }
        if (!lo.v.present       && lo.v.rank       > 0) { lo.v.a.free();       lo.v.b.free();       total_sites_merged++; }
        if (!lo.linear2.present && lo.linear2.rank > 0) { lo.linear2.a.free(); lo.linear2.b.free(); total_sites_merged++; }
    }
    // Free the layer arrays themselves. We mark the adapter as not loaded so
    // both forward() and free() short-circuit; this leaves any unmerged
    // sites (non-Q8_0 base) dangling, but the Flux 2 Klein GGUF has none.
    if (model.lora.n_dual > 0)   std::core::mem::free(model.lora.dual_layers);
    if (model.lora.n_single > 0) std::core::mem::free(model.lora.single_layers);
    model.lora.loaded = false;
    model.lora.dual_layers = null;
    model.lora.single_layers = null;
    model.lora.n_dual = 0;
    model.lora.n_single = 0;

    double ms = t0.to_now().to_sec() * 1000.0;
    io::printfn("  Flux LoRA merged into Q8_0 weights: %d sites, %.0f ms", total_sites_merged, ms);
}
