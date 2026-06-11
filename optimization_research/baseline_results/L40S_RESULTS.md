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
