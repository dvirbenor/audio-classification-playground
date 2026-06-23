# Triton Inference Server — Throughput Results & ONNX Serving Notes

**What this covers:** the measured windows/second of the **deployed Triton** path (affect /
disfluency / emotion served as ONNX models over the onnxruntime backend on one GPU), the
config that makes all three co-exist, how it compares to the in-process PyTorch/MPS path in
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md), and a postmortem on getting emotion2vec to
export. Measured 2026-06-22…23 on a dedicated bench server
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
