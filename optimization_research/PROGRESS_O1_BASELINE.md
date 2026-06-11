# Progress — A10G inference optimization (session dump)

**Date:** 2026-06-10 · **Branch:** feature/inference-optimization · **Status:** ✅ O1 · ✅ O2 measured + **event-A/B passed** (fp16 not yet flipped in code)

Working from `INFERENCE_OPTIMIZATION_PLAN.md`. Deliverables:
`baseline_results/INFERENCE_BASELINE_A10G.md` (O1), `baseline_results/O2_AUTOCAST_RESULTS.md` (O2),
`baseline_results/OPTIMIZATION_FINDINGS.md` (event-A/B verdict + design Q&A), `scripts/event_level_ab.py`.
This file is the running progress dump.

---

## O2 — bf16/fp16 autocast (measured, recommendation made, NOT shipped)

vs eager fp32, bs256, affect+disfluency (full table in `O2_AUTOCAST_RESULTS.md`):
- autocast alone ≈ **1.44×**; **autocast + compile ≈ 2.2×** (affect 133→291, disfluency 157→346 win/s).
- **fp16 > bf16 here**: same speed, ~5–10× tighter numerics (fp16 eager allclose@1e-3 for affect;
  disfluency top-1 0.9995 vs bf16 0.9976). **Flips the plan's "bf16 first" → recommend fp16.**
- bs512 **OOMs even in fp16/bf16** — pure compute win, no batch headroom; O3 still needed for occupancy.
- `autocast_dtype` already in `inference_config_hash` → flipping default is immutability-safe (new artifact dir).

**Gate before shipping:** event-level A/B (compose fp32 vs fp16 packages, diff events). Not built yet.
No production default changed.

---

## What's done

- [x] Confirmed env: A10G (22.06 GiB, compute 8.6), torch 2.10+cu128, Ubuntu 22.04, uv venv on system py3.10.
- [x] Pulled 5 real archives from S3 (`riverside-pro-main`) via the production resolver + manifest.
- [x] Eager fp32 per-task throughput + VRAM, bs256/128 (`profile_persistent_pipeline_vram.py`).
- [x] Eager → `compiled_static` WavLM comparison + numerical agreement (`compare_wavlm_runtime_knobs.py`).
- [x] Emotion `optimized` (compile+TF32) vs eager on the production `predict_audio` path.
- [x] Wrote baseline doc + persisted raw JSON artifacts.
- [x] Saved a project memory (`a10g-inference-baseline`).

## Headline numbers (bs: WavLM 256 / emotion 64; warmed; GPU-compute only)

| task | eager fp32 | production default | speedup | peak VRAM |
|---|---|---|---|---|
| affect (WavLM) | ~131 win/s | **157** (compiled_static) | 1.18× | ~19.5 GiB |
| disfluency (WavLM) | ~155 win/s | **184** (compiled_static) | 1.17× | ~19.5 GiB |
| emotion (e2v, predict_audio) | ~250 win/s | **243** (optimized) | 0.97× | ~4.9 GiB |

- Resident models ~3 GiB. **WavLM bs512 OOMs** → prod caps at 256.
- Inference is full-timeline (not VAD-gated). Real archives ~57 min median (~13.6k windows/task).
- Compute-only archives/hr·GPU (compiled, median archive): affect ~42, disfluency ~49, emotion ~64; all-in-one ~17.

## Key findings

1. **Compile win is modest (~1.17× WavLM, ~0× emotion)** → confirms plan thesis: real headroom is
   O2 (bf16) / O4 (TRT-fp16), not eager torch.compile. emotion2vec is at its eager ceiling on A10G.
2. **torch.compile fragility / fleet-parity risk.** Needs python3.10 dev headers for triton's
   inductor C build. WavLM `compiled_static` falls back to eager gracefully; **emotion `optimized`
   forces compile with no fallback guard.** Fleet manifests pass *no* preset flags → compile only
   engages if the production container ships headers. If not, fleet is silently eager.
3. **affect is the bottleneck** (slowest win/s + sets VRAM peak) → optimize WavLM-affect first.
4. Numerics: eager→compiled WavLM is safe (affect allclose 1e-3; disfluency top-1 agreement 1.0000).

## Decisions made (and why)

- **Baselined the compiled production-default path, not just eager.** The fleet default is
  `compiled_static`/`optimized`, so that's "current speed." Eager captured too, as the floor.
- **Used existing harnesses** per CLAUDE.md (profile + knob-compare) rather than ad-hoc timing;
  measured compiled WavLM by setting window count = multiple of 256 (single static shape ≈ compiled_static).
- **Measured emotion on `predict_audio`** (the production path) after finding the runner prefers it
  over framed `__call__`. The framed compiled path is 2.2× *slower* — a red herring, not the prod path.
- **Installed `python3.10-dev`** (was already present; headers had flickered out via EFS visibility)
  to make torch.compile build. Reversible, local dev box.
- Used `PYTORCH_ALLOC_CONF=expandable_segments:True` to dodge an affect bs256 fragmentation OOM.

## Environment changes / new files

- `baseline_results/INFERENCE_BASELINE_A10G.md` — full baseline writeup.
- `baseline_results/{baseline_profile_eager,baseline_compile_compare,real_audio_manifest}.json` — raw data.
- `PROGRESS_O1_BASELINE.md` — this file.
- Memory: `…/memory/a10g-inference-baseline.md` + `MEMORY.md`.
- Scratch (ephemeral, /tmp): seed wav, 5 real archive wavs, inductor cache `/tmp/acp_inductor_cache`.
- No source code changed. `apt-get install python3.10-dev` confirmed already-installed.

## Open questions

- Does the **production fleet image** ship python3.10 dev headers? Determines if the fleet runs
  compiled or silently eager. (No Dockerfile in repo — image built externally.)
- Real **archive duration distribution** (only 5 sampled) — needed to firm up the 600k fleet-hours figure.
- I/O-inclusive throughput: this baseline is GPU-compute only; haven't measured decode/prefetch overhead.

## Next steps (pick up here)

- **(1) Event-level A/B** ✅ DONE — fp16 passes (0 events added/dropped; affect labels 100%;
  disfluency 1 sub-type flip/63; boundary/score drift negligible). See `OPTIMIZATION_FINDINGS.md §1`.
- **(2) Flip WavLM default to fp16 autocast** — now unblocked by the A/B. New config hash handles
  immutability. *Not yet done in code.* (Optionally run the bf16 A/B first to confirm fp16 ≥ bf16.)
- **(3) SDPA attention rewrite** ✅ DONE — implemented `acoustic_events/inference/wavlm_sdpa.py`
  (repo-owned, reinstall-safe, fp32 bit-identical). Measured only **~1.05×**: WavLM is GEMM-bound
  (short ~150–174-token windows → attention ~3% of FLOPs). Keep it (free) but not a headline.
  *Not wired into model load yet.* See `OPTIMIZATION_FINDINGS.md §2`.
- **(4) O6 INT8 PTQ** ✅ TESTED → **REJECTED** (`compare_wavlm_int8.py`, see `OPTIMIZATION_FINDINGS.md §2b`):
  (a) speed blocked — int8+compile fails to trace on torch 2.10/torchao 0.17 (needs torch≥2.11), eager
  int8 is *slower* (76/88 win/s); (b) **fails the event gate** — adds/drops events, flips ~18% affect
  labels, shifts affect scores up to 1.54 / boundaries up to 4 s. Affect regression has a real int8
  accuracy cliff. Revisit only with torch≥2.11 + mixed-precision + static calibration; uncertain payoff.
- **(5) O4 TensorRT** ✅ TESTED → **REJECTED** (`benchmark_wavlm_onnx_trt.py`, `OPTIMIZATION_FINDINGS.md §2c`):
  ONNX export OK (gated-attention risk didn't bite); ONNX/TRT-fp32 bit-exact. TRT-fp32 only **1.19×** vs
  eager (structural win is small — confirms GEMM-bound); TRT-fp16 **2.44× but NaN** (fp16 overflow). Doesn't
  beat PyTorch fp16+compile (2.2×, lossless). Not worth the deploy cost. Installed onnxruntime-gpu+tensorrt
  via `uv pip` (venv only, not in uv.lock).

**All optimization levers now measured → endpoint = PyTorch fp16+compile (2.2×, lossless, event-safe).**
Remaining actions:
- **(2) Flip WavLM default to fp16** (+ optionally wire SDPA on) — the one code change left; unblocked by the A/B.
- **(a)** Worker end-to-end → I/O-inclusive archives/sec from timings JSONL (compute- vs I/O-bound).
- Verify the emotion `optimized` no-fallback compile guard against the fleet image (O1 finding #2).
