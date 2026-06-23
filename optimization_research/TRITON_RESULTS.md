# Triton Inference Server — Throughput Results & ONNX Serving Notes

**What this covers:** the measured windows/second of the **deployed Triton** path (affect /
disfluency / emotion served as ONNX models over the onnxruntime backend on one GPU), the
config that makes all three co-exist, how it compares to the in-process PyTorch/MPS path in
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md), a postmortem on getting emotion2vec to export,
and the **TensorRT engines that now supersede the ONNX path at ~2.5–3× the throughput**
([TensorRT engines](#tensorrt-engines--the-faster-backend)). Measured 2026-06-22…23 on a dedicated bench server
([manifests/triton-bench-deployment.yaml](../manifests/triton-bench-deployment.yaml)) with
[scripts/benchmark_triton_throughput.py](../scripts/benchmark_triton_throughput.py).

- **Server:** one `g7e.2xlarge` = RTX PRO 6000 Blackwell, **95.6 GiB**, Triton `25.05` (ORT backend).
- **Models:** affect (WavLM, 56 000-sample window), disfluency (WavLM, 48 000), emotion (emotion2vec, 48 000), all FP16 ONNX.
- **Harness:** CPU clients decode real archives, frame windows exactly like production, and stream
  them over gRPC; Triton's dynamic batcher merges concurrent requests. Two numbers per task —
  **client** win/s (end-to-end incl. round-trip) and **server** win/s (GPU compute only, from
  Triton's per-model stats; the number comparable to the report's win/s table).

---

## TL;DR

- **All three models serve on one GPU at `max_batch_size: 128`, ~52 GiB used (43 GiB free).** At
  256 the trio OOMs — ONNX Runtime's CUDA arena is ~5× heavier than PyTorch (see [Memory](#memory-why-batch-128)).
- **Per-task server throughput:** affect **610**, disfluency **724**, emotion **1014** win/s.
  Batch 128 == batch 256 here (compute-bound — the SMs are already saturated), so 128 costs nothing.
- **Blended (all three at once — the deployed shape): ~750 win/s per GPU** (~250 each). Co-locating
  three ONNX models on one GPU ≈ running them serially — **no packing gain**, unlike MPS's 1.49×.
- **ONNX/Triton ≈ 0.6× the in-process PyTorch path** per window (affect 610 vs the report's 1039).
  Cross-archive batching reaches the server ceiling at concurrency ≥4, but that ceiling is below PyTorch.
- **emotion went from 0 → 1014 win/s.** Its ONNX export was batch-baked (served zero inferences in
  production); fixed via a dynamo re-export (see [Emotion export postmortem](#emotion-export-postmortem)).
- **🚀 TensorRT engines are ~2.5–3× faster than this ONNX path *and* ~3× leaner — now the recommended
  backend.** Served: affect **1578**, disfluency **1787**, emotion **3127** win/s; trio fits in **17 GiB**
  (vs 52). Deployed as `triton-trt`. See [TensorRT engines](#tensorrt-engines--the-faster-backend).

---

## TensorRT engines — the faster backend

Same models, compiled to TensorRT `.plan` engines and served via Triton's `tensorrt_plan` backend
instead of onnxruntime. **This is the faster path and the one the backfill fleet now uses.** Built and
measured on the real Blackwell with TRT **10.10** (matched to the `tritonserver:25.05` image, since
engines are arch + TRT-version locked). Deploy:
[manifests/triton-trt-deployment.yaml](../manifests/triton-trt-deployment.yaml); build:
[scripts/build_trt_engines.sh](../scripts/build_trt_engines.sh).

### Throughput (served, batch 128, 3 real archives)

| task | TRT mixed-fp16 | ONNX/ORT (above) | **vs ORT** | TRT-fp32 | **vs fp32** |
|---|---|---|---|---|---|
| affect     | **1578** | 610  | **2.6×** | 418 | 3.7× |
| disfluency | **1787** | 724  | **2.5×** | 483 | 3.6× |
| emotion    | **3127** | 1014 | **3.1×** | 819 | 3.7× |

Served ≈ raw single-engine compute (Triton overhead is negligible — compute-bound). Client throughput
saturates at concurrency ≥4, same as ORT, but at a ~2.5–3× higher ceiling. With three replicas
(`triton-trt`, 3 GPUs) the fleet aggregate is ~3× these per-GPU numbers.

**No gain past batch 128** (measured, affect, single-stream): 64→1604, 128→1533, 192→1492, 256→1503
win/s — flat, and 256 is ~2% *slower* (64 is marginally best). The short-window WavLM GEMMs saturate
the SMs by batch ≤128, so larger batches add only latency. (The ~16% 128→256 gain seen on the ORT
path does **not** carry to TRT — ORT's kernels were less efficient at small batch; TRT's aren't.)

### Memory — TRT is ~3× leaner than ORT

Trio loaded **17.3 GiB** (vs ORT's 52 at batch 128), idle or under load — TRT's engine-time memory
planning avoids ORT's non-releasing BFC arena. **78 GiB free**, so batch 256 or multiple instances fit
trivially; not worth it since throughput is already compute-bound at 128.

### The precision recipe — fp16 except Softmax + Norm in fp32

Build with `--fp16` but pin **Softmax + Normalization** layers to fp32
([scripts/build_trt_engine.py](../scripts/build_trt_engine.py), by `trt.LayerType`). Why:

- Blanket `trtexec --fp16` is **numerically broken on real audio** — the attention **Softmax** overflows
  fp16, giving ~**0.42** absolute error in affect outputs (would flip every label; this is the report's
  rejected "NaN"). It's **Softmax**, not LayerNorm — pinning LayerNorm alone does nothing; pinning
  Softmax alone fixes it (3e-4). *Random* input hides this (1.5e-3) — only loud real speech exposes it.
- **Caveat:** the catastrophic error was on A10G + TRT 10.16; on **Blackwell + TRT 10.10 blanket fp16
  was also accurate** (1.5e-3). So the recipe's necessity is GPU/TRT-version-specific — but pinning
  Softmax+Norm costs **0 win/s** and is guaranteed-safe across kernel/tactic/GPU changes, so we always ship it.

### Build prerequisite — fold constants first

TRT 10.10's ONNX parser **rejects the affect fp16 model** (`convMultiInput: input tensor shape misaligns
with kernel shape`) because the fp16 converter left the fp32-kept Conv weights as a `Cast` output, not
an initializer. Fold them once — `polygraphy surgeon sanitize model.onnx --fold-constants -o folded.onnx`
(affect folds 133 nodes; disfluency/emotion fold 0). `build_trt_engines.sh` does this automatically.
Engines are **Blackwell sm_120 + TRT 10.10 locked** — rebuild if the GPU arch or Triton image changes.

### Event-safety A/B (ORT-served baseline vs TRT-served)

[scripts/event_level_ab_triton.py](../scripts/event_level_ab_triton.py), 3 archives, 10-min cap:

| task | result |
|---|---|
| affect | **0 dropped / 0 added, labels 100%** across all 3 archives ✅ |
| disfluency | 2/3 identical; **1 dropped event of ~60** on one archive (borderline; `d_score 0.004`) |
| emotion | not covered (separate producer — A/B it separately) |

Read: **fp16-vs-fp16 boundary noise** (both ORT and TRT are fp16; neither is fp32 ground truth) — a
single marginal disfluency event at a detection threshold. A near-pass, not a literal zero; worth a
broader A/B (more archives + emotion) before the final cutover.

### Deploy

Model repo on S3 at `s3://riverside-build-assets/paralinguistics-trt/` (separate from the ONNX prefix —
the `.plan`s are arch-locked). The Deployment needs `serviceAccountName: ai-dev-common-access` (IRSA for
S3) — without it Triton dies at startup with "Unable to create S3 filesystem client." Rebuild + publish:
`PUBLISH=1 bash scripts/build_trt_engines.sh`.

---

## The deployed config (the keeper)

In each [triton/<task>/config.pbtxt](../triton/affect/config.pbtxt):

```
max_batch_size: 128
dynamic_batching { preferred_batch_size: [128] ... }   # single shape → stable arena
optimization { execution_accelerators { gpu_execution_accelerator [ {
  name: "cuda"
  parameters { key: "cudnn_conv_use_max_workspace" value: "0" }   # honored
  parameters { key: "cudnn_conv_algo_search"       value: "HEURISTIC" }
  parameters { key: "arena_extend_strategy"        value: "1" }   # ignored (see below)
} ] } }
parameters { key: "enable_mem_pattern" value: { string_value: "0" } }   # the real memory lever
```

What actually moves memory, confirmed in the verbose startup log:

| knob | effect |
|---|---|
| `enable_mem_pattern: 0` | **works** — stops ORT pre-allocating a big buffer per input shape. The key lever. |
| `max_batch_size` | **works** — the only hard cap on per-model footprint. Set to 128. |
| `cudnn_conv_use_max_workspace: 0` | **works** — trims conv workspace. |
| `gpu_mem_limit`, `arena_extend_strategy` | **IGNORED** by this backend via the `cuda` accelerator params (logged as `gpu_mem_limit=<uint64 max>`, `arena_extend_strategy=kNextPowerOfTwo`). Do **not** rely on them. |

---

## Throughput

### Isolated — each task alone on the GPU

Batch 128, 3 real archives (~156 min audio). Server win/s is the steady-state GPU-compute rate;
client win/s saturates at concurrency ≥4 (one model instance is the bottleneck — past C=4 only the
queue grows, not throughput).

| task | server win/s | client @C1 | client @C4 | client @C8 | client @C16 |
|---|---|---|---|---|---|
| affect     | **610** | 417 | 597 | 595 | 598 |
| disfluency | **724** | 497 | 708 | 705 | 704 |
| emotion    | **1014** | 616 | 984 | 982 | 978 |

Batch 128 vs 256 (where 256 fits, i.e. ≤2 models): identical server win/s — the workload is
short-window and GEMM-bound, so 128 already fills the SMs.

### Blended — all three concurrent (the deployed shape)

All three tasks streamed at once, C=8 each (24 concurrent client streams):

| task | server win/s | queue |
|---|---|---|
| affect | 249 | 3.3 s |
| disfluency | 254 | 3.3 s |
| emotion | 255 | 3.2 s |
| **per-GPU aggregate** | **~758** (≈742 client) | — |

The three models time-slice one GPU (one instance each). Producing 10 000 windows of **each**
took ~41 s blended vs ~40 s if run serially — **co-location is throughput-neutral here**, neither
a gain nor a penalty. The GPU is compute-bound and fully utilized either way.

---

## Memory — why batch 128

Idle (three models loaded, no traffic): **3.2 GiB**. Under load the ONNX-Runtime CUDA arena grows
to the working set and never releases it:

| config | peak GPU mem | result |
|---|---|---|
| 2 WavLM @ 256 | ~80 GiB | ok |
| 3 models @ 256 | overflow | **emotion OOMs** on a Conv node |
| **3 models @ 128** | **52.3 GiB** | **ok, 43 GiB headroom** |

One batch-256 WavLM forward drove the arena to **~85 GiB**, versus **~15 GiB** for the same model
in PyTorch (the MPS path fit *three* models at batch 256–512 on this same card). The ~5× gap is a
serving-allocator effect — ORT's non-releasing BFC arena + per-shape `enable_mem_pattern`
pre-allocation + EXHAUSTIVE conv workspaces — not the model. `enable_mem_pattern: 0` plus a single
preferred batch shape tame most of it; `max_batch_size: 128` does the rest.

---

## ONNX/Triton vs PyTorch/MPS

| dimension | ONNX / Triton (this doc) | PyTorch / MPS ([report](OPTIMIZATION_REPORT.md)) |
|---|---|---|
| affect win/s (per GPU, one task) | ~610 | ~1039 |
| memory per WavLM @ batch 256 | ~40 GiB | ~15 GiB |
| 3 models @ batch 256 on one 96 GiB GPU | **does not fit** (→ batch 128) | fits (batch 256–512) |
| co-locating 3 on one GPU | ~neutral (≈ serial) | **1.49×** packing gain |
| cross-archive batching | yes — thin CPU clients fill the GPU | no (per-archive only) |
| operational shape | one shared server, CPU-only workers | one GPU per worker pod, MPS daemon |

**Read:** on raw per-GPU throughput the in-process PyTorch/MPS path is ~1.7× faster and packs three
models more densely. Triton's advantages are operational — a single shared GPU server fed by cheap,
stateless CPU workers via dynamic cross-archive batching — not throughput.

---

## Emotion export postmortem

emotion2vec served **zero** inferences in production: its ONNX export only ran at the batch it was
traced with (`VALIDATION_BATCH=4`) and failed every other batch with
`Reshape node_view: cannot reshape {B*H,T,T} to {B,H,T,T}`. Root causes, peeled one at a time:

1. **Attention bake** — fairseq `MultiheadAttention` reshapes through `bsz * num_heads`; the legacy
   TorchScript tracer (and `do_constant_folding=False`) bake the trace batch.
2. **Alibi cache** — `get_alibi_bias()` caches a `[heads*batch_size, T, T]` tensor in a dict and
   branches on it, baking `heads*4 = 64` and raising `ConstraintViolationError` under `torch.export`.
3. **inference_mode** — computing the reference under `torch.inference_mode()` poisoned cached
   tensors so the exporter couldn't trace them.
4. **torch.onnx bug** — the decomposition pass crashes on `alibi_scale.clamp_min(0)`.

Fix ([scripts/export_models_onnx.py](../scripts/export_models_onnx.py)): export with `dynamo=True` +
a symbolic batch `Dim`; swap a **cache-free, batch-symbolic alibi** onto the encoder; **fold the
clamped `alibi_scale` and null it** (kills the `clamp_min` op — numerically identical to the model's
own `size(0)==1` path); compute the reference under `no_grad`. A new `_validate_batch_shapes()`
(batches 1/3/7/64/256) now guards **all three** exports so a batch-fragile graph can't ship again.
TorchScript was not a way out — `trace` bakes batch identically; `script` won't compile the vendored code.

---

## Ops gotchas (each cost a debugging cycle)

- **Commit the configs.** The export's `_install_configs` regenerates `config.pbtxt` from the cloned
  repo, so an old config on the branch silently overwrites S3 on the next `export + s3 sync` — it
  re-introduced the invalid `max_workspace_size_bytes` (a TensorRT param) under the CUDA EP and broke
  model load. The corrected configs are now committed.
- **Verify the model actually uploaded.** `aws s3 sync` silently did **not** replace
  `emotion/1/model.onnx`, so we benchmarked a stale model for a cycle. Always check
  `aws s3 ls .../emotion/1/` (timestamp) after publishing. The FP16 `model.onnx` is self-contained
  (no `.data` sidecar — only `model_fp32.onnx` needs one).
- **Restart to reload.** The server localizes the S3 repo only at startup, and with one GPU per
  deployment a rolling restart deadlocks — use `kubectl delete pod -l app=…` (the Deployment
  recreates it on the freed GPU).

---

## Reproduce

```bash
# from inside the cluster (the bench server reads the same S3 model repo):
uv run python scripts/benchmark_triton_throughput.py \
    --url triton-bench.nlp-audio-understanding:8001 \
    --concurrency 1,4,8,16 --max-chunk 128 --json-out triton_trio_isolated.json
# all three at once (deployed shape):
uv run python scripts/benchmark_triton_throughput.py \
    --url ... --blended --concurrency 8 --max-chunk 128 --json-out triton_trio_blended.json
```

Related: [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) (the PyTorch/MPS path),
[MPS_OPTIMIZATION.md](MPS_OPTIMIZATION.md) (the SM-cap analog of the memory work here).
