# O1 — Per-task A10G inference baseline

Baseline for the [inference optimization plan](../INFERENCE_OPTIMIZATION_PLAN.md), O1. This is
the measurement substrate every later option (O2 bf16, O4 TRT, O6 int8, …) is validated against.

## Environment

| | |
|---|---|
| GPU | NVIDIA **A10G**, 23028 MiB (~22.06 GiB usable), compute 8.6 (Ampere) |
| torch / CUDA | 2.10.0+cu128 / CUDA 12.8 |
| OS / Python | Ubuntu 22.04, CPython 3.10.12 (uv venv on system python3.10) |
| Host | this dev box (single A10G), `source env.shared.sh` for EFS model cache |

**Method.** Pure **GPU model-compute** throughput per task, warmed, at production batch sizes
(WavLM **256**, emotion **64**), measured with the existing harnesses
[profile_persistent_pipeline_vram.py](../../scripts/profile_persistent_pipeline_vram.py) and
[compare_wavlm_runtime_knobs.py](../../scripts/compare_wavlm_runtime_knobs.py), plus a small
`predict_audio` timing for emotion (its production path). Throughput is **compute-bound and
batch-independent** (130 vs 132 win/s at bs256 vs bs128), so it does **not** depend on audio
content; window count is what matters. Audio: 5 real archives pulled from S3
(`riverside-pro-main`) via the production resolver — see [real_audio_manifest.json](real_audio_manifest.json).
This baseline excludes I/O (S3 download, decode, prefetch) and orchestration overhead — it is the
GPU ceiling each plan option targets.

Raw artifacts: [baseline_profile_eager.json](baseline_profile_eager.json),
[baseline_compile_compare.json](baseline_compile_compare.json).

## Per-task throughput (windows/s, bs: WavLM 256 / emotion 64)

| task | backbone | path | **eager fp32** | **production default** | speedup | real-time factor† |
|---|---|---|---|---|---|---|
| affect | WavLM-large | framed | ~131 win/s | **157 win/s** (`compiled_static`) | **1.18×** | 33× → 39× |
| disfluency | WavLM-large | framed | ~155 win/s | **184 win/s** (`compiled_static`) | **1.17×** | 39× → 46× |
| emotion | emotion2vec+ | `predict_audio` | ~250 win/s | **243 win/s** (`optimized`) | **0.97×** | ~62× |

† audio-seconds processed per wall-clock second = win/s × hop (0.25 s). Inference runs over the
**full timeline** at every hop window (not VAD-gated — VAD is a separate CPU task), so window
count = duration / 0.25.

**Numerical agreement (eager → compiled WavLM):** affect allclose @ atol 1e-3 (max abs 3e-4);
disfluency top-1 agreement **1.0000** (max abs 6.5e-3 on type logits — sub-threshold drift, no
prediction flips). torch.compile is numerically safe for WavLM here.

## VRAM (single A10G, ~22.06 GiB usable)

| state | reserved |
|---|---|
| resident models, all 3 loaded | ~3.0 GiB |
| **all-in-one worker peak** (3 resident + affect active @bs256) | **~19.5 GiB** |
| WavLM @bs128 | ~14.0 GiB |
| WavLM **@bs512** | **OOM** — one fp32 512×3.5 s batch needs ~11 GiB on top of models |
| emotion @bs64 standalone | ~4.9 GiB |

The bs512 OOM is exactly why production caps WavLM at **256** (and `compiled_static` fixes the
static batch at 256). At bs256 the all-in-one worker sits ~2.5 GiB from the ceiling.

## Derived archive throughput (compute-only, production-default/compiled)

Real archives are full recording sessions — sampled durations **1531 / 2619 / 3395 / 3751 / 6533 s**
(median ~57 min → ~13,580 windows/task). Per-task GPU time for a median archive and the
resulting per-GPU rate (task-fleet = one task per GPU, what the manifests deploy):

| task | s / median archive | archives/hr·GPU |
|---|---|---|
| affect | ~87 s | ~42 |
| disfluency | ~74 s | ~49 |
| emotion | ~56 s | ~64 |
| **all-in-one (3 tasks/GPU)** | **~216 s** | **~17** |

Eager-only equivalent (a box where compile is unavailable): all-in-one ~243 s/archive (~15/hr).

Fleet-scale (589,648 archives; **weak** estimate — mean of only 5 sampled durations ≈ 59 min):
order **~37k GPU-hours** compiled vs **~42k** eager for all three tasks. Treat as
order-of-magnitude only; total compute scales with total audio-seconds, which needs a real
duration distribution to pin down.

## Key findings / caveats

1. **torch.compile is fragile in this environment.** It needs `python3.10` dev headers
   (`/usr/include/python3.10/Python.h`) for triton's inductor C build. They were intermittently
   invisible (transient EFS visibility); the package is installed. WavLM's `compiled_static`
   guards this with an eligibility check + runtime fallback to `fast_exact`, so it degrades
   gracefully — **but emotion's `optimized` preset (auto-default on CUDA) forces compile with no
   such guard**, so on a box where the build fails it would error rather than fall back. Worth a
   guard/parity check against the fleet image.
2. **Fleet manifests pass no runtime-preset flags** — production relies on auto-resolution
   (WavLM→`compiled_static` if eligible, emotion→`optimized`). Whether the compile speedup
   actually engages in the fleet depends on the container shipping python3.10 dev headers. If it
   doesn't, the fleet is silently running eager (the ~1.17× WavLM win is unrealized).
3. **The compile win is modest on A10G (~1.17–1.18× WavLM, ~0× emotion)** — consistent with the
   plan's thesis that the real headroom is precision/backend (O2 bf16, O4 TRT-fp16), not eager
   torch.compile. emotion2vec is already at its eager ceiling here; compile gives nothing
   (O5 is correctly Tier 2).
4. **affect is the heaviest task** (slowest win/s, sets the all-in-one VRAM peak) → optimize
   WavLM-affect first, exactly as the plan sequences it.

## Reproduce

```bash
source env.shared.sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/acp_inductor_cache PYTORCH_ALLOC_CONF=expandable_segments:True

# Eager per-task time + VRAM (all 3 tasks, bs256/128)
python scripts/profile_persistent_pipeline_vram.py --audio <archive.wav> \
  --min-windows 2048 --configs 256/256/64 256/256/64 128/128/64 --json-out baseline_profile_eager.json

# Eager vs compiled_static WavLM (bs256, window count a multiple of 256 → single static shape)
python scripts/compare_wavlm_runtime_knobs.py --audio <archive.wav> --tasks affect disfluency \
  --batch-size 256 --candidate-batch-size 256 --min-windows 256 --max-windows 2048 \
  --candidate-compile --candidate-compile-mode default --json-out baseline_compile_compare.json
```
`expandable_segments:True` avoids an allocator-fragmentation OOM for affect at the bs256 edge.
