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

- [x] Level 0 suite green; CLI flag + validation behave.
- [x] Level 2 confirms MPS active (concurrent kernels in `nvidia-smi dmon`) and peak
      VRAM safely under 96 GB with all 3 clients resident. — **MPS active, SM peak 100%,
      aggregate peak 42.6 GiB (44.5%) at batch 512.**
- [x] Level 3 shows the co-located fleet's aggregate throughput at the chosen batch
      size ≥ the per-fleet baseline; pick the batch-size knee (start 256 → 512 → …).
      — **knee is 512 (+16% over 256); 768 adds nothing; 1024 OOMs.**
- [~] Level 4 shows zero event-level drift between batch sizes. — **PARTIAL: disfluency
      & emotion zero-drift; affect drifts marginally (batch-shape kernel effect). Pin one
      batch size fleet-wide. See results below.**

See **Validation results** below for the full measured data (2026-06-14, Blackwell RTX PRO 6000).

## Validation results (2026-06-14)

Run on a single Blackwell dev pod: **NVIDIA RTX PRO 6000 Blackwell Server Edition,
97887 MiB, driver 580.159.03, CC 12.0**, torch 2.12.0+cu130. Sample: 3 real archives
(40–66 min each) from `benchmark_audio/`, decoded-audio cache warmed, full-timeline
unless noted. Scratch under `scratch/mps-validation/` (not the prod tree).

### Level 0 — unit / regression (no GPU) ✅
- `pytest tests/acoustic_events/`: **270 passed, 1 skipped**.
- `--wavlm-static-batch-size` present in `run --help` with correct description.
- Conflicting combo (`--wavlm-runtime-preset fast_exact --wavlm-static-batch-size 512`)
  exits **2** with a clean argparse error *before any model load*.

### Level 1 — launcher supervise/teardown (no GPU, stubbed `python`, `ENABLE_MPS=0`) ✅
- Clean case: all 3 workers exit 0 → launcher exits **0**.
- Failure case: one worker exits 7 → siblings get SIGTERM (exit 143) → launcher
  propagates **7** (so k8s restarts the pod). Sibling 143-on-shutdown is correctly not
  treated as the failure code.

### Level 2 — single GPU pod ✅
- **Runtime-knob parity** (`compare_wavlm_runtime_knobs.py`, fp16-compiled vs fp32-eager,
  single static 512 shape): affect **2.27×** (1003 vs 441 win/s, allclose@1e-3),
  disfluency **1.98×** (1168 vs 590 win/s, fluency top-1 **1.0**, type-sign **1.0**,
  max logit Δ 0.033). *Caveat:* feeding an unpadded partial final batch to the compiled
  model triggers a per-shape recompile (~30 s) and a relative `TORCHINDUCTOR_CACHE_DIR`
  on EFS hits a rename race — both are bench-harness artifacts, not production paths
  (the worker pads to a single static shape and uses a local namespaced cache).
- **MPS active / concurrent kernels** (`nvidia-smi` sampling during the real fleet):
  daemon reachable; **82/86 active samples had ≥3 concurrent compute apps** (3 workers +
  MPS server = 4); **SM utilisation peaked at 100%** (a single small model under-fills
  the SM array, so 100% requires genuine overlap).
- **Aggregate VRAM (3 co-resident clients), batch 512: peak 42.6 GiB = 44.5% of 96 GB.**
- **Small e2e** (`run_mps_colocated.sh`, real pipeline): all 3 tasks × 3 archives
  `processed=3 failed=0`; **decoded-audio cache reused** (`object_hit`, no re-download);
  **VAD-gating-from-artifact live** (gated to 33–56% of windows); concurrent compile to
  the shared inductor namespace coordinated cleanly via FileLock.
- **Container/MPS prereq**: `nvidia-cuda-mps-control` present; pipe dir writable; daemon
  starts and is reachable.
- *Dev-pod note:* `run_mps_colocated.sh` launches workers as bare `python`. The dev
  `.venv` has no bare `python` on `PATH` → activate the venv (`source .venv/bin/activate`)
  before running. The k8s image is expected to have system `python`.

### Level 3 — 256-vs-512 throughput A/B ✅ (knee = 512)
Steady-state aggregate windows/s across 3 archives (warmup-excluded; affect/disfluency
take the static batch, emotion is fixed at 64):

| task | bs256 | bs512 | Δ |
|---|---|---|---|
| affect | 551 win/s | **640 win/s** | **+16%** |
| disfluency | 565 win/s | **653 win/s** | **+16%** |

Wall-clock: bs256 142 s, bs512 158 s — bs512 is *higher* only because of the one-time
512-shape `torch.compile` warmup, which over just 3 archives is not amortised (exactly
the doc's caveat). At fleet scale the one-time compile amortises and the per-window gain
dominates.

### Level 4 — event-level drift (256 vs 512) ⚠ PARTIAL
Raw model outputs are **not** bit-identical across batch sizes: emotion Δ=0 (same bs64
kernel), affect max Δ ≈ 1.5e-3, disfluency logits max Δ ≈ 2.5e-2. A determinism probe
showed affect fp16 inference is **bit-identical run-to-run at a fixed batch** (256→256 and
512→512 both Δ=0) but **256 vs 512 differ by 4.9e-4** — i.e. `torch.compile` emits a
different fp16 tiling/reduction kernel per static shape. The doc's premise that per-window
outputs are independent of batch grouping is therefore **false under fp16+compile**.

Event impact (composed via the affect/disfluency producers, full-timeline VAD):
- **emotion**: bit-identical → events identical.
- **disfluency**: 337 events, **label-agreement 1.0, exact 1.0, zero added/dropped**.
- **affect**: 738 events; **one archive 193→197 (4 added), two archives a few
  sub-tolerance boundary shifts** — the affect producer's segmentation thresholds are
  sensitive to ~5e-4 score deltas.

`wavlm_static_batch_size` is excluded from `inference_config_hash`, so two batch sizes
write to the **same artifact path** yet can yield slightly different affect events.
**Mitigation: pin one static batch size fleet-wide; never mix sizes against one output
tree.** Treat the affect drift as a known sub-threshold effect (or a hard blocker if
strict cross-batch reproducibility is required).

### Higher batch sizes — 768 and 1024 (full sweep)

| batch | affect win/s | disfluency win/s | peak VRAM (3-way) | 3-way fits? | notes |
|---|---|---|---|---|---|
| 256 | 551 | 565 | ~20 GiB | ✅ | baseline |
| **512** | **640** | **653** | **42.6 GiB (44.5%)** | ✅ | **recommended knee, +16%** |
| 768 | 631 | 657 | 90.8 GiB (95%) | ✅ (at the edge) | no throughput gain over 512 |
| 1024 | — | OOM | 89.9 GiB → OOM | ❌ | disfluency fleet fails all archives |

- **512 already saturates the SMs** (SM peak 100% at 512 and 768), so 768 buys nothing on
  throughput while pushing VRAM to a risky 95%. **512 vs 768 events are count-identical**
  (affect & disfluency drift = 0) — the small affect drift is specific to the 256 kernel
  vs the larger-batch kernels, not monotonic with batch size.
- **batch 1024 OOMs under 3-way co-location**: a single WavLM forward needs ~32 GiB
  (one process spiked to 53 GiB); the disfluency worker failed to allocate and
  `processed=0 failed=3` while affect/emotion completed.
- **Operational trap:** on OOM the worker records a per-archive failure and still exits
  **rc=0**, so k8s would *not* restart it — a too-large batch lets a fleet silently make
  zero progress. Consider a startup VRAM check or failing the worker on repeated OOM.

### Co-location efficiency — are processes slowed, or is cost ÷3?

Controlled comparison (bs512, full-timeline, same 3 archives, audio cache → pure GPU
time): each task run **alone** (1 process, no MPS) vs **3-way co-located** under MPS.

| task | solo (1 proc) | 3-way co-located | per-process speed |
|---|---|---|---|
| affect | 1035 win/s | 640 win/s | **62%** of solo |
| disfluency | 1201 win/s | 653 win/s | **54%** of solo |
| emotion | 2184 win/s | 1060 win/s | **49%** of solo |

**Aggregate:** serial inference (sum of solo) = 84 s; co-located (slowest of 3 concurrent
fleets) = 58 s → **1.44× speedup**. So it is **neither** "divide cost by 3" (would need
each process ≈100% of solo) **nor** pure time-slicing (would put each ≈33% of solo, 1.0×
aggregate). Each process runs at roughly half its solo speed and the three together finish
~1.44× faster than back-to-back. Reason: WavLM-large GEMMs at batch 512 already occupy
most of the SM array (SM peaked at 100% co-located), so MPS can only overlap the
memory-bound gaps (layernorm/softmax/copy) — a minority of runtime, hence partial overlap.
emotion2vec loses the most (49%) as the most bandwidth-bound of the three.

Net: co-location buys ~1.44× throughput per GPU vs three serial single-task passes — real,
and it stacks with the +16% from bs512 and the VAD-gating window reduction, but it is not
the 3× a genuinely SM-starved GPU would yield.

### Bottom line
Ship with **`--wavlm-static-batch-size 512`** on this card. Levels 0–3 pass; the only
caveat is the Level 4 affect drift, handled by pinning a single batch size fleet-wide.
Co-location yields ~**1.44×** aggregate throughput per GPU (not 3×) — each process runs at
~50–62% of its solo speed, the expected partial-overlap regime for SM-heavy WavLM GEMMs.

## Before fleet rollout (cluster-specific — cannot be validated on a generic pod)

Set the two `TODO(blackwell)` values in the manifest: the Blackwell node selector /
instance-type, and CPU/memory requests sized to that node (3 co-located prefetch
pipelines). Ensure the **CPU VAD fleet leads** so `vad/` artifacts are present and the
gating speedup is realized (GPU never blocks on VAD; it falls back to full-timeline).
