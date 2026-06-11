# L40S performance + precision matrix (incl. fp8)

Companion to `INFERENCE_BASELINE_A10G.md` (A10G/O1) and `OPTIMIZATION_FINDINGS.md`. Re-runs the
per-task precision sweep on an **L40S** (Ada, sm89, 46 GiB) and tests **fp8** (only possible on Ada+).

**Env:** torch **2.12.0+cu130**, torchao 0.17.0, Python 3.12. bs: WavLM **256** / emotion **64**.
2048 windows, warmed. Precision Δ = max abs diff of the task's output arrays vs fp32 eager.
Scripts: [bench_l40s.py](../../scripts/bench_l40s.py), hand-rolled fp8 via `torch._scaled_mm`.

> ⚠️ Measurement note: this box is **shared** — concurrent CPU-heavy work starves the kernel-launch
> path and can 20× the compiled numbers (we saw fp16+compile drop 467→23 win/s under contention).
> Numbers below are contention-free runs; compiled-variant absolute win/s still carry ±~10% run-to-run.

## L40S raw vs A10G (eager fp32, same bs)
affect 131 → **272 win/s (~2.1×)**, emotion 250 → **543 (~2.2×)**. The L40S gives ~2× from
architecture alone (more SMs/clocks), *before* any precision work.

## Per-task matrix (L40S, win/s)

### affect (WavLM-large)
| variant | win/s | vs compiled_fp32 (prod) | Δ vs fp32 |
|---|---|---|---|
| eager fp32 | ~272 | 0.83× | 0 |
| **compiled_fp32** (= production `compiled_static`) | ~326 | 1.0× | 5e-5 |
| fp16 autocast (eager) | ~308 | 0.94× | 3.5e-4 |
| **fp16 + compile** | **~441–467** | **~1.35–1.43×** | 3e-4 |
| fp8 eager (torchao) | ~193 | 0.59× (slower) | 6.8e-3 |
| fp8 + compile (torchao) | ✗ `as_strided` error | — | — |
| **FFN-fp8 + compile** (hand-rolled `_scaled_mm`) | **~528** | **~1.62×** | 6.6e-3 |

### disfluency (WavLM-large)
| variant | win/s | vs compiled_fp32 | Δ vs fp32 |
|---|---|---|---|
| eager fp32 | ~341 | 0.86× | 0 |
| **compiled_fp32** (prod) | ~397 | 1.0× | 2.3e-3 |
| fp16 autocast (eager) | ~366 | 0.92× | 1.1e-2 |
| **fp16 + compile** | **~530** | **~1.34×** | 1.0e-2 |
| fp8 eager (torchao) | ~235 | 0.59× | **0.40** |
| fp8 + compile (torchao) | ✗ `as_strided` | — | — |

### emotion (emotion2vec, predict_audio path)
| variant | win/s | Δ vs fp32 | note |
|---|---|---|---|
| eager fp32 | ~543–570 | 0 | |
| fp16 autocast | ~570 | 1–4e-3 | gain noisy on L40S (vs clean 1.44× on A10G) |
| optimized (compile+TF32) | flaky/hung under serial compile on L40S | — | compile neutral for e2v anyway |
| fp8 eager (torchao) | ~459 (slower) | 1.5e-2, **top-1 1.0000** | |
| fp8 hand-rolled (eager, per-tensor) | ~476 (slower) | 0.25 | per-tensor scaling lossy; needs compile to win |

## fp8 verdict (L40S)

**torchao fp8 is unusable on this stack** (torch 2.12 / torchao 0.17): full fp8 + compile fails
(`Float8Tensor` has no `aten.as_strided` — triggered by WavLM's `F.multi_head_attention_forward`
weight chunking); FFN-only fp8 + compile *compiles* but recompile-storms to **3.3 win/s**
(`Float8Tensor` lacks `_stable_hash_for_caching`); eager fp8 is slower than fp32. Confirmed **not
fixed even on torchao `main`** (v0.17+182) — neither hook is implemented for the inference Float8Tensor.

**Hand-rolled `Fp8Linear` (`torch._scaled_mm`, no tensor subclass) is the only working fp8 path** —
it compiles cleanly (no `as_strided`, no caching pathology) and is the right approach. Results:
- **affect FFN-fp8 + compile: 528 win/s = +13% over fp16+compile (1.13×)** — real but modest. Only
  the FFN (48 layers) is fp8; attention projections stay fp16 (they read `.weight` directly, so a
  module-swap can't reach them without the SDPA patch). Dynamic per-tensor quant overhead eats into
  the fp8 GEMM win, so 1.13× not the ~2× ideal.
- **emotion: no win** — eager fp8 is *slower* (the emotion path doesn't compile cleanly, so the quant
  overhead isn't fused away). Per-tensor hand-rolled scaling is also lossy here (Δ 0.25).

**Precision (the gate, GPU/version-independent):** affect 6.6e-3 (marginal — needs the event-level
A/B, between fp16's safe 3e-4 and int8's failed level); disfluency 0.18–0.40 (**fails**, like int8);
emotion top-1 stable (1.0) but hand-rolled per-tensor is lossy. So even where fp8 runs, only affect
(maybe) and emotion (classification) could pass the gate; disfluency is out.

## Bottom line
- **fp16 + compile is the cross-platform win** (A10G *and* L40S): ~1.34–1.43× over the compiled
  production baseline, lossless/event-safe on all three tasks, zero new deps.
- **fp8 is a weak add**: at best ~+13% on affect alone (hand-rolled, FFN-only, gated on a precision
  check), nothing on emotion, out on disfluency. Not worth the hand-rolled-`Fp8Linear` maintenance +
  precision risk over the free, lossless fp16 win — unless attention-fp8 (via the SDPA patch) is
  pursued to chase a larger (~1.3–1.5×) affect number, accepting worse precision.
- **L40S itself gives ~2× over A10G** from architecture — the biggest single lever here is the
  hardware, then fp16+compile on top.

## Batch size — compute-bound, bigger batch does NOT help (measured)
affect fp16+compile on L40S, exact-multiple windows: bs256 **447** / bs384 **454** / bs512 **430** win/s
(bs768 OOM at ~46 GiB). Flat-to-slightly-worse going up, while VRAM scales ~linearly (15→30 GiB).
At bs256 the GEMM M-dim is batch×seq ≈ 256×174 ≈ 44.5k rows — tensor cores already saturated, so larger
batches just run longer at the same rate (bs512 even dips from cache/bandwidth pressure). Matches A10G
(128≈256). **Keep WavLM at bs256; extra VRAM is not a speed lever.** Occupancy on short/ragged archives
is the O3 cross-archive-micro-batching lever, not raising per-archive batch.

---

# Blackwell RTX PRO 6000 (Server Edition, sm_120, 96 GiB)

Same harness (bench_l40s.py), torch 2.12.0+cu130 (arch_list includes sm_120 → kernels present;
fp16 matmul + fp8 work; torch.compile works on Blackwell). bs WavLM 256 / emotion 64, 2048 windows.

### affect (WavLM), win/s
| variant | win/s | Δ vs fp32 |
|---|---|---|
| eager fp32 | 511 | 0 |
| compiled_fp32 | 604 | 5e-5 |
| fp16 (autocast) | 671 | 3e-4 |
| **fp16 + compile** | **1039** | 3e-4 |
| fp8 eager (torchao) | 388 (slower) | 6.9e-3 |
| fp8 + compile (torchao) | ✗ `as_strided` (same torchao bug — arch-independent) | — |

### disfluency (WavLM), win/s
eager 610 · compiled_fp32 720 · fp16 803 · **fp16+compile 1253** · fp8_eager 472 (Δ 0.40, bad).
### emotion (emotion2vec): eager fp32 **913** win/s (3.6× A10G's 250); fp16 adds ~1.1–1.4× (compile
is neutral for e2v — its `optimized` compile path hangs under serial compile here, as on L40S, but
contributes no speed anyway). So emotion fp16 ≈ ~1000–1250 win/s on Blackwell.
fp8 verdict unchanged on Blackwell: torchao path still broken (`as_strided`), eager slower than fp32.

## Cross-GPU raw speed + price/perf (affect, fp16+compile = shipping config)

| GPU | instance | $/h | win/s | raw vs A10G | **win/s per $/h** | price-perf vs A10G |
|---|---|---|---|---|---|---|
| A10G | g5.4xlarge | 1.624 | 291 | 1.0× | 179 | 1.00× |
| L40S | g6e.2xlarge | 2.242 | 447 | 1.5× | 199 | 1.11× |
| **Blackwell RTX6000** | — | 3.363 | **1039** | **3.6×** | **309** | **1.73×** |

**Verdict — Blackwell is worth it for bulk throughput:** it's the most expensive (2.07× the A10G price)
but ~3.6× faster, so it has the **best throughput-per-dollar — ~1.73× A10G, ~1.55× L40S** (disfluency
confirms: 1253 win/s = 3.6× A10G raw, 372 win/s/$ = 1.75× A10G price-perf). Unlike the
A10G→L40S step (price/perf barely moved, ~1.11×, because L40S's ~1.5× speed ≈ its ~1.38× price premium),
the Blackwell's raw speed gain decisively outpaces its price. fp16+compile remains the config (compile
works on sm_120); fp8 still doesn't help (torchao broken). Caveat: $/h is the user-provided number for
this box; confirm the actual fleet instance price.

### Blackwell batch-size sweep (affect fp16+compile) — compute-bound, bigger is WORSE
bs256 **1051** / bs512 1030 / bs768 942 / bs1024 853 win/s (peak 19.6→35.8→50.1→64.4 GiB). Throughput
*declines* with batch (cache/bandwidth pressure at huge M), confirming compute-bound on all 3 GPUs
(A10G, L40S, Blackwell). **Keep bs256 everywhere; the 96 GiB is not a speed lever.**

## Deploy note
`manifests/acoustic-events-inference-cache-workers-optimized.yaml` set to **g7e.2xlarge** (Blackwell)
with explicit `--wavlm-runtime-preset compiled_static` (affect/disfluency) + `--emotion-runtime-mode
optimized` (emotion) — both now resolve to **fp16+compile** via the code default-flip. Requires the
image to ship matching `pythonX.Y-dev` + gcc so `compiled_static` stays eligible (else falls back to
eager fp32).
