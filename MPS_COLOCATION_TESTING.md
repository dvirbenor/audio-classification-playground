# MPS Co-Location + Static-Batch Tuning — Testing & Validation Plan

This document is the validation plan for the GPU-concurrency change that runs the
affect / disfluency / emotion task-fleets co-located on **one** Blackwell RTX PRO
6000 (96 GB) via CUDA MPS, plus a configurable WavLM static batch size. It exists
because almost all of this is verifiable on a **single GPU pod** before any fleet
rollout — the k8s fleet only adds scale, not new behavior.

## Background — why this change

On one big GPU a single small model (WavLM-large / emotion2vec) under-fills the SM
array, spatially (one model's batch doesn't cover all SMs) and temporally (tensor
cores idle during layernorm/softmax/copy phases between GEMMs). Cross-archive
micro-batching was rejected: our archives are 20 min–1 hr, so ragged-tail waste is
only ~1.6–6% and not worth a worker rewrite. The two real levers are:

1. **Cross-model overlap via CUDA MPS** — run the per-task fleets as separate
   processes sharing the GPU so one model's GEMM fills another's memory-bound gap.
2. **Static-batch tuning** — long archives have thousands of windows, so raise the
   WavLM static batch to better fill the SM array.

## What changed (files under test)

| Area | File | Change |
|---|---|---|
| Configurable batch | [wavlm_runtime.py](audio_classification_playground/acoustic_events/inference/wavlm_runtime.py) | `resolve_wavlm_runtime_settings(static_batch_size=…)` overrides the default 256 on `compiled_static` |
| Thread-through | [worker.py](audio_classification_playground/acoustic_events/orchestration/worker.py) | `wavlm_static_batch_size` through `run_worker` + `build_expected_configs`; timing record logs the **actual** batch (`wavlm_batch_size`) |
| CLI | [cli.py](audio_classification_playground/acoustic_events/orchestration/cli.py) | `--wavlm-static-batch-size` + validation |
| MPS launcher | [scripts/run_mps_colocated.sh](scripts/run_mps_colocated.sh) | per-pod entrypoint: MPS daemon + 3 co-located workers + SIGTERM forwarding + supervise |
| A/B runner | [scripts/compare_static_batch_throughput.sh](scripts/compare_static_batch_throughput.sh) | 256-vs-512 throughput comparison |
| Manifest | [manifests/acoustic-events-inference-mps-colocated-blackwell.yaml](manifests/acoustic-events-inference-mps-colocated-blackwell.yaml) | new co-located Blackwell GPU pod + leading CPU VAD fleet |

Invariants preserved (no test needed beyond the suite): locking, prefetch,
artifact layout, and `inference_config_hash` are untouched. `wavlm_static_batch_size`
is **excluded** from the semantic hash, so changing it never alters artifact identity
or outputs — only speed.

## Level 0 — unit / regression (no GPU)

```
uv run python -m pytest tests/acoustic_events/ -q
```

Expected: full suite green. The static-batch override is covered by
`tests/acoustic_events/test_wavlm_runtime.py` (override applies; rejects `fast_exact`,
granular knobs, and non-positive values). CLI wiring:

```
uv run python -m audio_classification_playground.acoustic_events.orchestration run --help | grep -A4 wavlm-static-batch-size
# conflicting combo must exit 2 with a clean error (no model load):
uv run python -m audio_classification_playground.acoustic_events.orchestration run \
  --parquet /x --output /y --affect-backbone wavlm --disfluency-backbone wavlm \
  --wavlm-runtime-preset fast_exact --wavlm-static-batch-size 512
```

## Level 1 — launcher behavior (no GPU)

The supervise/teardown logic is GPU-independent. Stub `python` on `PATH`, set
`ENABLE_MPS=0`, and run [scripts/run_mps_colocated.sh](scripts/run_mps_colocated.sh):
- clean case: all workers exit 0 → launcher exits 0;
- failure case: one worker exits non-zero → siblings get SIGTERM → launcher exits
  with that worker's code (so k8s restarts the pod).

## Level 2 — single GPU pod (the real validation, no fleet)

On a Blackwell dev pod: `source env.shared.sh` first.

1. **Static-batch sweep in isolation** — synthetic windows, no S3/MPS:
   - [scripts/profile_persistent_pipeline_vram.py](scripts/profile_persistent_pipeline_vram.py) — peak VRAM at 256 / 512 / … stays well under 96 GB across 3 co-resident clients.
   - [scripts/compare_wavlm_runtime_knobs.py](scripts/compare_wavlm_runtime_knobs.py) — per-model windows/s + numerical parity vs the FP32 reference.

2. **MPS is actually active** (not silently time-slicing): the launcher asserts the
   daemon is reachable and aborts otherwise. Confirm by hand:
   ```
   nvidia-cuda-mps-control -d
   echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control
   ```
   Watch `nvidia-smi dmon` while the co-located fleet runs — expect higher sustained
   SM/tensor utilization than any single fleet alone.

3. **Small end-to-end** — exercise the real pipeline + shared audio cache +
   VAD-gating-from-artifact on a handful of real archives:
   ```
   export PARQUET=/path/to/small_subset.parquet OUTPUT=/efs/.../scratch/smoke
   bash scripts/run_mps_colocated.sh
   ```

4. **Container/MPS prerequisites** — confirm `nvidia-cuda-mps-control` exists in the
   GPU image and the pipe dir is writable. This is exactly what the single-pod test
   catches before a fleet rollout.

## Level 3 — the 256-vs-512 throughput A/B

Goal: quantify the overall speed gain of raising the static batch under MPS.

```
source env.shared.sh
export PARQUET=/path/to/SMALL_subset.parquet     # bounded — a few dozen archives
export OUTPUT_BASE=/efs/.../scratch/batch-ab      # scratch, NOT the prod tree
bash scripts/compare_static_batch_throughput.sh   # runs bs256 then bs512
```

It runs the co-located launcher once per batch size into separate scratch dirs
(so neither run skips the other's work), times each, and prints
`orchestration timings` for both. The worker timings record the **actual**
`wavlm_batch_size`, so per-task numbers are unambiguous.

**Read the results correctly:**
- Use the per-archive `inference_sec` / `affect_sec` / `disfluency_sec` distributions
  for steady-state throughput — they **exclude** one-time compile warmup.
- Wall-clock totals **include** warmup (which differs per batch size); use them for
  end-to-end, and only after enough archives that warmup amortizes.
- Both runs do identical work (fresh dirs, VAD-gating falls back to full-timeline in
  both → fair). For a gating-representative bench, pre-populate the `vad/` artifacts
  in both dirs first (run the `vad` task-group into each).

## Level 4 — zero-drift correctness gate

Static batch size and MPS do not change the math (per-window outputs are
independent of batch grouping). Confirm empirically: run the pipeline on a fixed
sample with batch 256 vs 512, compose packages, and diff the resulting **events**
(count / type / start–end / score) with [scripts/event_level_ab.py](scripts/event_level_ab.py).
Expected: identical events.

## Acceptance criteria

- [ ] Level 0 suite green; CLI flag + validation behave.
- [ ] Level 2 confirms MPS active (concurrent kernels in `nvidia-smi dmon`) and peak
      VRAM safely under 96 GB with all 3 clients resident.
- [ ] Level 3 shows the co-located fleet's aggregate throughput at the chosen batch
      size ≥ the per-fleet baseline; pick the batch-size knee (start 256 → 512 → …).
- [ ] Level 4 shows zero event-level drift between batch sizes.

## Before fleet rollout (cluster-specific — cannot be validated on a generic pod)

Set the two `TODO(blackwell)` values in the manifest: the Blackwell node selector /
instance-type, and CPU/memory requests sized to that node (3 co-located prefetch
pipelines). Ensure the **CPU VAD fleet leads** so `vad/` artifacts are present and the
gating speedup is realized (GPU never blocks on VAD; it falls back to full-timeline).
