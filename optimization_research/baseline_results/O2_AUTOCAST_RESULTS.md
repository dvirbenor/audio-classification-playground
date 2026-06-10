# O2 — bf16/fp16 autocast for WavLM (affect + disfluency)

Measured against the [O1 baseline](INFERENCE_BASELINE_A10G.md) on the same A10G, same harness
([compare_wavlm_runtime_knobs.py](../../scripts/compare_wavlm_runtime_knobs.py)), bs256, 2048 windows
(multiple of 256 → single static shape), real archive audio. All speedups are **vs eager fp32**.
Raw data: `baseline_results/o2_*.json`.

## Throughput

| config | affect | disfluency | vs eager fp32 |
|---|---|---|---|
| eager fp32 (baseline) | 133 win/s | 157 win/s | 1.00× |
| compiled_static (O1 prod default) | 157 | 184 | 1.17–1.18× |
| **bf16** eager | 190 | 227 | **1.43–1.44×** |
| **fp16** eager | 192 | 228 | **1.44–1.45×** |
| **bf16 + compile** | 294 | 346 | **2.21×** |
| **fp16 + compile** | 291 | 346 | **2.19–2.20×** |

- Autocast alone ≈ **1.44×** (matches the plan's predicted 1.3–1.8×, near-zero engineering).
- Autocast **stacks with compile** to **~2.2×** over eager fp32 — affect 133→291, disfluency 157→346 win/s.
  In real-time terms: affect ~73×, disfluency ~86× realtime (hop 0.25 s).

## Numerical agreement (tensor-level, vs fp32)

| config | affect max abs | disfluency fluency top-1 | disfluency type-sign |
|---|---|---|---|
| compiled (fp32) | 3e-4 | 1.0000 | — |
| **fp16** eager | **7e-4 (allclose@1e-3)** | **0.9995** | 0.9994 |
| bf16 eager | 7e-3 | 0.9976 | 0.9973 |
| fp16 + compile | 6e-3 | 0.9985 | 0.9976 |
| bf16 + compile | 7e-3 | 0.9976 | 0.9971 |

**fp16 beats bf16 here** — same speed, ~5–10× tighter drift (fp16 eager is allclose at 1e-3 for
affect; disfluency top-1 0.9995 vs bf16 0.9976). These WavLM activations are small/normalized, so
there's no fp16 overflow and its extra mantissa bits (10 vs bf16's 7) win. **This flips the plan's
"bf16 first" assumption → recommend fp16 as the default candidate** on this A10G hardware.

## VRAM / batch headroom

bs512 **OOMs even in fp16 and bf16** (and bs768 fp16 OOMs) — autocast keeps many WavLM intermediates
in fp32, so half-precision is a **pure compute win, not a memory win**. **bs256 remains the ceiling.**
→ O2 does not by itself fix GPU occupancy on short/ragged batches; **O3 (cross-archive micro-batching)
is still needed** for that. (True fp16/int8 *weights* via O4 TensorRT would also reduce memory, unlike autocast.)

## Cache-correctness ✓

`autocast_dtype` already feeds `inference_config_hash` (via `_inference_config`, runners.py:122-139,
[artifacts.py](../../audio_classification_playground/acoustic_events/inference/artifacts.py)). Flipping
the default → new config hash → new artifact dir; existing fp32 artifacts are retained. Immutability
is preserved automatically, no schema change needed.

## Recommendation

1. **Adopt fp16 autocast as the WavLM default** (affect + disfluency), stacked on `compiled_static`
   → ~2.2× fleet throughput on the two heaviest tasks, the single highest ROI/effort win in the plan.
2. **Gate on event-level A/B before flipping the production default.** Tensor top-1 agreement
   (0.9985–0.9995) is promising but not the acceptance test — compose packages fp32 vs fp16 on a
   fixed sample set and diff the *events* (count, type, start/end, score), per the plan's verification
   strategy. Disfluency boundaries are the most sensitive (top-1 0.9985 ⇒ ~0.15% window flips).
3. **Prefer fp16 over bf16** unless event-level A/B shows fp16 regressions; keep bf16 as fallback.
4. Do **not** change the default in code yet — this doc is measurement + recommendation; the
   event-level A/B harness is the next build and the gate.
