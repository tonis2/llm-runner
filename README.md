# llm-runner

Vulkan compute pipeline for running Z-Image and Flux diffusion in C3, with a Qwen3-8B text encoder.

## Perf — Flux Q8_0 matmul optimization log

Test config: Flux 9B Q8_0 + PULPKHOR LoRA, kontext 512×512, 4 steps, AMD RX 7800 XT (RADV NAVI32), Mesa.

| Change | avg fwd/step | Δ vs prev | Δ vs start |
|---|---:|---:|---:|
| Session baseline | 14721 ms | — | — |
| LoRA hop2 rewrite + Q8 row-scale caching | 8352 ms | −43% | −43% |
| fp16 LDS tiles (mixed-precision) | 6829 ms | −18% | −54% |
| **Packed `half2` LDS + V_DOT2-shaped inner loop** | **6599 ms** | **−3.4%** | **−55%** |

Sustained ≈ 5 TFLOPS out of ~37 TFLOPS fp32 peak (~14% of peak). Published ceiling for non-WMMA quantized GEMM under Mesa RADV/ACO is **30–45% of peak** — further gains require either WMMA (currently broken on this driver per Mesa #10847) or an ACO patch that emits `V_DOT2_F32_F16` (Mesa MR !14842, Draft).

### Optimizations tried — what didn't pay

Each measured against the packed-`half2` baseline (6599 ms). All correct (output PNG unchanged) but slower; reverted.

| Change | Result | Why it lost |
|---|---:|---|
| Double-buffered LDS (ping-pong tiles) | 7025 ms (+6%) | LDS footprint doubled to ~17 KB; halved occupancy outweighed the prefetch-overlap gain. |
| Inline per-thread scale fetch (drop scales→B barrier) | 6809 ms (+3%) | 4× extra `scale_word` global loads/thread/K-tile cost more than the saved barrier. |
| `+4` half-stride padding (llama.cpp convention) | 8199 ms (+24%) | Bumped LDS allocation into a worse bucket; the +1 stride already broadcasts in 1 cycle, +4 added overhead with no gain. |

The kernel is at a local optimum for this tile shape on this hardware. The negative results confirm: LDS pressure is binary (any change that pushes occupancy down one bucket costs more than typical kernel-structure wins).

### Remaining potential wins — to-do

- [ ] **Inline SPIR-V for `OpFDot` / V_DOT2 emission.** ACO's NIR doesn't currently pattern-match the `acc += a.x*b.x + a.y*b.y` form. Forcing the intrinsic via Slang's inline-spirv might let us hit ~2× inner-loop arithmetic throughput. Fragile, depends on RADV recognizing the form.
- [ ] **Apply the half2-LDS + V_DOT2-shaped layout to `batch_matmul_swiglu_q8`.** Used by Z-Image (not Flux — Flux uses GEGLU = `batch_matmul_q8 × 2 + silu + elementwise mul`, already optimized). Mechanical port: same transform, two B tiles instead of one.
- [ ] **Apply the same transform to `batch_matmul_q4k` / `q5k` / `q6k`.** Used by Z-Image text encoder paths; not on the Flux Q8_0 hot path.
- [ ] **Fix `VK_KHR_cooperative_matrix` (WMMA) path.** The blocker is Mesa #10847 — `groupshared float16_t` writes don't appear visible to `coopMatLoad` on RDNA3. Only path to 60–70% of fp32 peak. The shader skeleton exists (`batch_matmul_q8_coopmat`) but is gated off until the visibility bug is worked around.
- [ ] **`flash_attention` shader review.** Block-time profiling (each forward step: dual=1.5s, single=4.9s, final=3ms) shows the single-stream attention/FFN is dominant; if any attention compute is fp32-tiled like the old matmul was, the same fp16-LDS treatment likely applies.

### Already shipped

- `dispatch_lora_correction` / `batch_matmul_add_scaled` rewrite — 2D tiled dispatch with shared scratch, replaces a per-output broadcast loop.
- Q8 row-scale caching in `batch_matmul_q8` — 64 scales loaded once per K-tile into LDS, reused by all 256 threads' B dequants.
- fp16 LDS tiles in `batch_matmul_q8` — halves LDS footprint per workgroup.
- Packed `vector<float16_t,2>` LDS layout — single `ds_read_b32` returns two K-adjacent halves; inner loop shaped as `acc += a.x*b.x + a.y*b.y`.

### Earlier dead ends (pre-session)

- Removing per-block `RenderGraph.checkpoint()` calls — within noise.
- Forcing the coopmat path via `LLM_FORCE_COOPMAT=1` — produces wrong outputs and is ~42% slower than the tiled path.
- Fused QKV matmul (both 3-B-tile and single-B-tile-cycled variants) — RDNA3 occupancy halved by the larger LDS, cancelling the dispatch-overhead savings.
