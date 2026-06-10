# Inference Optimization — Prioritized Options Plan

## Context

We want to make acoustic-event inference **much faster**. The system runs on g5.4xlarge (NVIDIA **A10G**, ~24 GB, Ampere — FP16 + INT8 tensor cores + TF32) and must serve **two deployment targets**:

1. **Bulk backfill (now):** a batched Kubernetes fleet processing ~600k archives. Goal: **throughput / GPU-cost** (this dominates today).
2. **On-the-fly production (next):** a **long-lived warm GPU service** that runs inference whenever a new archive is created. Models stay resident (no per-request cold start). Latency target: **don't regress** single-archive inference — no hard SLA, but realtime requests must not be starved by bulk-throughput optimizations.

These are not in conflict if we converge on a **single dynamic micro-batching inference service** (see O3): both the bulk worker and the realtime trigger feed windows into one batching queue with a small bounded max-wait, so full batches are formed across whatever work is in flight while realtime latency stays bounded. The persistent `ModelSuite` already in [models.py](../audio_classification_playground/acoustic_events/inference/models.py) is the natural basis for the warm service.

**Numerical drift tolerance: aggressive** — fp16/bf16, TensorRT fp16, and int8 PTQ are all on the table provided event-level outputs are validated via the existing A/B harnesses. Precision/backend wins (O2, O4, O5, O6) apply to **both** deployment targets, which is why they're the highest-leverage items.

This document is a **survey of options with ROI / effort / risk**, plus a recommended sequence. It is not an implementation; nothing here is built yet.

### Where the codebase already is (important — avoid re-doing this)

The eager-PyTorch path is **already heavily optimized**. Existing levers:

- **`torch.compile`** (backbone-only) for WavLM, configurable mode/dynamic — [vox_profile/wavlm_inference.py](../audio_classification_playground/vox_profile/wavlm_inference.py); for emotion2vec direct scorer — [inference/emotion2vec.py](../audio_classification_playground/acoustic_events/inference/emotion2vec.py).
- **Autocast fp16/bf16**, **TF32** (`configure_torch_matmul`), **stream-layer-sum** (avoids materializing 24 hidden states), **static-batch padding** (256) — [inference/models.py](../audio_classification_playground/acoustic_events/inference/models.py), [inference/wavlm_runtime.py](../audio_classification_playground/acoustic_events/inference/wavlm_runtime.py).
- **Direct emotion2vec scorer** bypassing FunASR's per-waveform loop; audio-fed path that strides windows on-GPU — [inference/emotion2vec.py](../audio_classification_playground/acoustic_events/inference/emotion2vec.py). Runtime presets `optimized`/`fp32-eager` — [inference/emotion_runtime.py](../audio_classification_playground/acoustic_events/inference/emotion_runtime.py).
- **Persistent models** loaded once per worker, reused across files — `ModelSuite` in [models.py](../audio_classification_playground/acoustic_events/inference/models.py).
- **Async prefetch + audio cache + task-fleets** (per-task workers with deeper lookahead) — [orchestration/worker.py](../audio_classification_playground/acoustic_events/orchestration/worker.py), [orchestration/task_groups.py](../audio_classification_playground/acoustic_events/orchestration/task_groups.py), [orchestration/prefetch.py](../audio_classification_playground/acoustic_events/orchestration/prefetch.py).
- **ONNX/TensorRT *benchmark* harnesses already exist** but are NOT wired into production — [scripts/benchmark_wavlm_onnx_trt.py](../scripts/benchmark_wavlm_onnx_trt.py), [scripts/benchmark_emotion2vec_onnx_tensorrt.py](../scripts/benchmark_emotion2vec_onnx_tensorrt.py), plus knob-comparison + VRAM-profiling scripts.

### The two biggest untapped wins

1. **No production ONNX/TensorRT runtime path.** Export + benchmarking is prototyped (incl. `PreparedWavLMWrapper` for tracing) but the persistent predictors still run eager torch. TRT-fp16/int8 on A10G is typically **2–4×** over eager for transformer encoders.
2. **Strictly one-archive-at-a-time on the GPU.** Prefetch decouples I/O, but each archive's windows are batched in isolation. Short archives → small final batches → GPU underutilization, and compile/cudagraph benefits are re-amortized per file. A **dynamic micro-batching service** (O3) that pools windows across in-flight work keeps batches always full — and is also the unifying architecture for the warm realtime service, since a newly-arrived archive simply enqueues into the same batcher.

### Key architectural seams (where changes land)

- **Clean backend seam:** persistent predictors expose a uniform callable — `AffectPredictor.__call__(windows: np.ndarray) -> dict`, `DisfluencyPredictor.__call__`, `EmotionPredictor.__call__`/`predict_audio` ([models.py](../audio_classification_playground/acoustic_events/inference/models.py)). Any alternate backend (ONNX/TRT) can slot in **behind this interface** without touching producers/composition.
- **Cache-correctness constraint:** `inference_config_hash` ([inference/artifacts.py](../audio_classification_playground/acoustic_events/inference/artifacts.py)) currently hashes task, model_id, backbone, sample_rate, window/hop, transform_policy, autocast_dtype, torch_compile, allow_tf32 — but **NOT** batch_size. Any new backend or precision **must be added to this hash** so artifacts from different runtimes don't collide. Immutability is preserved automatically: a new backend → new config hash → new artifact dir, old retained.
- **Heaviest compute:** WavLM-large (24-layer transformer) drives **affect + disfluency** (~3–5 s each per 60 s file). emotion2vec is lighter; Silero VAD runs on **CPU** off the critical path. Optimize WavLM first.

---

## Options (ranked by ROI for throughput-weighted goal)

For each: **Effort** (S/M/L), **Risk**, **Drift**, **Expected win**.

### Tier 1 — do first

**O1. Establish a per-task A10G baseline + a standing benchmark loop.** *(Effort S, Risk none, Drift none)*
Before changing anything, capture throughput (windows/s, archives/s) and VRAM per task on g5.4xlarge using the existing tools: [scripts/profile_persistent_pipeline_vram.py](../scripts/profile_persistent_pipeline_vram.py), [scripts/compare_wavlm_runtime_knobs.py](../scripts/compare_wavlm_runtime_knobs.py), and the per-archive timings JSONL the worker already writes ([orchestration/timings.py](../audio_classification_playground/acoustic_events/orchestration/timings.py)). This is the measurement substrate every later option is validated against. **Win:** none directly, but de-risks everything and tells us whether we're compute- or I/O-bound per task-fleet.

**O2. Roll out bf16/fp16 autocast as the default for WavLM affect+disfluency.** *(Effort S, Risk low, Drift small)*
The plumbing already exists (`wavlm_autocast_dtype`, `autocast_context`); it's just not the default and is flagged "experimental, needs event-level validation." Validate with [scripts/compare_wavlm_runtime_knobs.py](../scripts/compare_wavlm_runtime_knobs.py) + an event-level A/B (compose packages before/after, diff events). Add `autocast_dtype` to the config hash. **Win:** ~1.3–1.8× on the two heaviest tasks for near-zero engineering. Highest ROI/effort ratio in the whole plan — bf16 first (more numerically forgiving on Ampere), fp16 if bf16 underperforms.

**O3. Dynamic micro-batching inference service (serves both bulk + realtime).** *(Effort L, Risk med, Drift none)*
This is the **architectural keystone for the dual deployment target.** Today the GPU sees one archive's windows at a time. Introduce a batching layer that accepts *window submissions tagged by archive* from any number of producers, forms full fixed-size batches across all in-flight work, runs the resident model, and scatters results back to each owning archive. Two feeders share it:
- **Bulk worker** pushes windows from its prefetch lookahead (already 8–28 archives deep) — batches are always full, ragged tails disappear, static-batch + compile + cudagraphs pay off on *every* batch.
- **Realtime trigger** enqueues a single new archive's windows; a small **bounded max-wait** (e.g. a few ms) caps added latency so realtime never regresses — it rides along with whatever bulk work is batching, or runs alone if idle.

Touches [orchestration/worker.py](../audio_classification_playground/acoustic_events/orchestration/worker.py) and the predictor call sites; the resident `ModelSuite` and per-archive artifact writing ([artifacts.py](../audio_classification_playground/acoustic_events/inference/artifacts.py)) are unchanged. **Win:** large GPU-occupancy gain for bulk; simultaneously delivers the warm-service backbone for production. Risk: window→archive attribution, partial-completion/locking semantics, and a clean submit/await interface. **Defer the realtime trigger wiring** (event source / queue) to the production phase; build the batcher so it's feeder-agnostic from day one.

### Tier 2 — high ceiling, more work

**O4. Production TensorRT (fp16) backend for WavLM, behind the predictor seam.** *(Effort L, Risk med, Drift small)*
Promote the [scripts/benchmark_wavlm_onnx_trt.py](../scripts/benchmark_wavlm_onnx_trt.py) export path (`PreparedWavLMWrapper` → ONNX → TRT via onnxruntime-gpu TRT EP, with engine + timing cache) into an alternate predictor implementing the same `__call__` contract. Selected by a new `--wavlm-runtime-preset trt_fp16` (and a matching `inference_config_hash` entry). Requires adding `onnxruntime-gpu` to the deployable image and shipping/caching built engines on EFS (engine build is slow and shape-specific → pin static batch + window length, reuse the static-batch machinery). **Win:** typically **2–4×** over eager on A10G; stacks on top of cross-archive batching. Validate drift with the same ONNX/TRT benchmark's built-in top-1 agreement + abs-diff report, then event-level A/B.

**O5. emotion2vec ONNX/TensorRT backend.** *(Effort M, Risk low–med, Drift small)*
Same pattern via [scripts/benchmark_emotion2vec_onnx_tensorrt.py](../scripts/benchmark_emotion2vec_onnx_tensorrt.py) (extract_features → mean-pool → proj → masked softmax is a clean, compile/export-friendly graph). Lower absolute payoff than WavLM (smaller model) but the emotion fleet runs the deepest lookahead, suggesting it's a throughput target. **Win:** moderate; good once O4's harness/integration pattern is established.

**O6. INT8 PTQ for WavLM via TensorRT (calibrated).** *(Effort L, Risk high, Drift medium)*
Since aggressive drift is acceptable: build an int8 TRT engine with a calibration set drawn from real archives (entropy/minmax calibration over representative windows). A10G has INT8 tensor cores, so the ceiling is highest here. Gate strictly behind event-level A/B (does it change which events fire / their boundaries?), not just tensor allclose. **Win:** potentially another **1.3–2×** over fp16 TRT — but only pursue after O4 proves the integration and we have an event-level acceptance test. Keep fp16 as the safe fallback preset.

### Tier 3 — situational / dependent on O1 findings

**O7. Audio decode + resample acceleration.** *(Effort M, Risk low, Drift none)*
`librosa.load` is single-threaded CPU ([inference/audio.py](../audio_classification_playground/acoustic_events/inference/audio.py)). It's already hidden behind prefetch threads, so this only matters **if O1 shows I/O-bound fleets** (e.g. VAD/CPU fleet, or GPU-starved waiting on decode). Options: `torchaudio`+ffmpeg decode, `soundfile` + GPU resample, or decode-native-then-resample-on-GPU. Decide after O1.

**O8. VAD acceleration.** *(Effort S–M, Risk low, Drift small)*
Silero runs on CPU ([models.py](../audio_classification_playground/acoustic_events/inference/models.py) `VadDetector`), prefetched off the critical path. Options: Silero ONNX (`onnx=True` is a one-flag change) or GPU execution, and/or batching across the lookahead. Only worthwhile for the **VAD-only CPU fleet** or if O1 shows VAD stalling a combined worker.

**O9. SDPA/FlashAttention backend + memory-format tuning.** *(Effort S, Risk low, Drift none/small)*
Confirm WavLM attention uses the fused SDPA/FlashAttention kernel under PyTorch 2.10 (HF `attn_implementation="sdpa"`), and test `channels_last`/contiguity tweaks. Cheap experiments to run alongside O2 via the knob-comparison harness. **Win:** modest, lossless.

**O10. Fleet/batch-size + scheduling re-tuning.** *(Effort S, Risk low, Drift none)*
Revisit per-task batch sizes, prefetch workers/lookahead, and task-fleet GPU assignment ([task_groups.py](../audio_classification_playground/acoustic_events/orchestration/task_groups.py)) against O1 numbers and the new precision/backend. Pure config (manifests) — touch only when explicitly doing an orchestration-deployment pass, per CLAUDE.md.

### Explicitly out of scope
- **vLLM / CTranslate2 / faster-whisper** — these target autoregressive LLM decoding. Our models are **encoder + classifier head** (no token generation), so these frameworks don't apply. (CT2/faster-whisper would only be relevant if a Whisper *decoder* were ever used; the disfluency/affect Whisper backbone uses the encoder + a head.)
- **Distillation / smaller student models** — a model-quality project, not a runtime optimization; out of scope unless explicitly requested.

---

## Recommended sequence

1. **O1** baseline + benchmark loop (gates and validates everything).
2. **O2** bf16/fp16 default rollout — fastest win, validate event-level.
3. **O3** dynamic micro-batching service — structural win, lossless; also the warm-service backbone for production.
4. **O4** TRT-fp16 WavLM backend — biggest single model win.
5. **O5** emotion2vec ONNX/TRT — reuse O4's pattern.
6. **O6** INT8 PTQ for WavLM — highest ceiling, only after O4 + an event-level acceptance gate exist.
7. **O7–O10** as O1 findings dictate.

Stack effect (rough, throughput): O2 (~1.5×) × O3 (occupancy) × O4 (2–4×) compounds toward a multiple-× fleet throughput improvement before even reaching int8.

---

## Verification strategy (applies to every option)

Per CLAUDE.md, **use the existing harnesses in `scripts/`, not ad-hoc timing**:

- **Speed/VRAM:** [scripts/compare_wavlm_runtime_knobs.py](../scripts/compare_wavlm_runtime_knobs.py), [scripts/benchmark_wavlm_onnx_trt.py](../scripts/benchmark_wavlm_onnx_trt.py), [scripts/benchmark_emotion2vec_onnx_tensorrt.py](../scripts/benchmark_emotion2vec_onnx_tensorrt.py), [scripts/profile_persistent_pipeline_vram.py](../scripts/profile_persistent_pipeline_vram.py). Each reports speedup + abs-diff/top-1 agreement vs the FP32 reference.
- **Numerical drift (tensor-level):** built into the benchmark scripts (max/mean/p99 abs diff, top-1 agreement, flipped-prediction margin).
- **Event-level acceptance (the gate that actually matters):** run the full pipeline on a fixed sample set with old vs new backend, compose packages, and diff the resulting **events** (count, type, start/end, score). This is what determines whether "small drift" is acceptable — especially for O6 int8. Build this as a reusable comparison early (during O1) since O2/O4/O6 all depend on it.
- **Realtime latency non-regression (for O3 + warm service):** alongside throughput, measure single-archive end-to-end latency through the micro-batcher with the bounded max-wait, confirming a lone realtime archive is not delayed waiting to fill a batch. Track both p50 and tail latency, not just fleet windows/s.
- **Correctness/caching:** confirm `inference_config_hash` gains the new dimension (backend/precision) so new-runtime artifacts get fresh paths and never collide with cached FP32 ones ([inference/artifacts.py](../audio_classification_playground/acoustic_events/inference/artifacts.py)); verify cache hit/miss via worker timings.
- **Tests:** `uv run python -m pytest tests/` after any change to the inference/models seam.

On a pod, `source env.shared.sh` first so weights resolve from the EFS model cache.
