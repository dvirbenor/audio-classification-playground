# Inference optimization — consolidated findings

Companion to `INFERENCE_BASELINE_A10G.md` (O1) and `O2_AUTOCAST_RESULTS.md` (O2). Covers the
event-level A/B verdict and the design Q&A from the optimization review. A10G, torch 2.10+cu128.

---

## 1. Event-level A/B — fp16 is event-safe (the gate for shipping O2)

Harness: [scripts/event_level_ab.py](../../scripts/event_level_ab.py) — runs the real
inference→producer path at fp32 and fp16, diffs the emitted **events** (count, type, start/end,
score). 3 real archives, 600 s each, bs256. VAD coverage identical on both sides, so the diff
isolates precision. Data: [event_ab_fp16.json](event_ab_fp16.json).

| task | events (fp32 / fp16) | dropped | added | label agreement | boundary drift | score drift |
|---|---|---|---|---|---|---|
| affect | 171 / 171 | **0** | **0** | **1.000** | ≤0.978 exact; max \|Δ\| 1–2 frames (0.25–0.5 s) | max 0.186 |
| disfluency | 63 / 63 | **0** | **0** | 0.976 (1 sub-type flip / 63) | **0.000 s** | **max 0.001** |

**Verdict: fp16 passes.** No events appear or disappear on any file; affect labels are 100%
preserved; disfluency timing/scores are essentially identical. The only drift is ~2% of affect
events nudged by one frame and **one** disfluency sub-type relabel — borderline cases, far inside
this project's stated "aggressive drift tolerance". This authorizes flipping the WavLM default to
fp16 (gated only on someone sanity-reviewing those borderline cases if desired).

Notes: full-coverage VAD was used, so absolute counts aren't production counts (real VAD gates
more) — but the fp32-vs-fp16 comparison is valid because the same VAD feeds both sides. bf16 was
not run through the A/B; tensor-level it is 5–10× looser than fp16 (see O2 doc), so fp16 is the
recommended default and bf16 the fallback.

## 2. WavLM is GEMM-bound, not attention-bound (SDPA measured: only ~1.05×)

WavLM runs unfused attention (`_supports_sdpa = False` even on transformers `main`; gated
relative-position bias is the blocker — see §3). I implemented the SDPA patch
([wavlm_sdpa.py](../../audio_classification_playground/acoustic_events/inference/wavlm_sdpa.py)) and
measured it ([compare_wavlm_sdpa.py](../../scripts/compare_wavlm_sdpa.py),
[sdpa_compare.json](sdpa_compare.json)):

| config | affect (stock→SDPA) | disfluency | SDPA vs stock |
|---|---|---|---|
| fp32 | 132 → 138 (1.05×) | 157 → 163 (1.04×) | **bit-identical (0.0)** |
| fp16 | 192 → 196 (1.02×) | 228 → 234 (1.03×) | within fp16 noise (≤2.3e-2) |
| fp16+compile | 287 → 306 (1.06×) | 346 → 360 (1.04×) | within fp16 noise |

**The patch is correct (fp32 bit-identical, event-safe) but the win is only ~1.02–1.06×** — *not*
the ~1.3–1.8× originally estimated. Reason: WavLM windows are **short — ~174 tokens (affect),
~149 (disfluency)** — so per layer attention is only **~3% of matmul FLOPs**; the FFN + QKVO
projections (GEMMs) are ~97%. SDPA fuses only the attention part, capping its benefit at ~3%.
The earlier estimate wrongly assumed attention was a large cost share (true for long-sequence LLMs,
false for these audio windows). **Lesson: WavLM is GEMM-bound, not attention-bound.**

This recalibrates the remaining ladder — the lever that hits the dominant GEMMs is **precision**,
not attention fusion (incremental over fp16+compile):

| lever | mechanism | gain | effort / risk |
|---|---|---|---|
| SDPA patch | fuse attention (~3% of FLOPs) | **~1.05× (measured)**, lossless | done, M |
| **INT8 PTQ** (O6) | int8 tensor cores ~2× GEMM throughput — hits the 97% | likely the **biggest** remaining lever | L, needs event-level gate |
| TensorRT fp16 (O4) | GEMM kernel autotune + fusion + low overhead (its attention-fusion edge is small here) | unknown — *measure first* | L, integration risk |

Revised recommendation: **keep SDPA (free, lossless, compile-compatible) but it is not a headline.**
Pivot remaining effort to **INT8 (O6)** since GEMMs dominate, and *measure* TRT before committing.
Side note: fp16+compile peak VRAM is much lower (~12.9 GiB vs ~20.4 eager at bs256), because inductor
reuses buffers on the static shape.

## 3. Design Q&A

**Will the SDPA rewrite survive an env reinstall?** Only if it's a **repo-owned monkeypatch**, not
a site-packages edit. WavLM's attention lives in `transformers` (site-packages) — editing it there
is wiped by `uv sync`. The durable approach: a module under `acoustic_events` that swaps
`WavLMAttention.forward` for an SDPA version at model-load time. Version-controlled, reinstall-safe,
and doesn't touch vendored/site-packages code (consistent with CLAUDE.md). `attn_implementation="sdpa"`
won't work directly because this WavLM arch reports no SDPA support.

**Would transformers 5.x give SDPA for WavLM? No.** Verified against upstream `main` (the 5.x line):
`WavLMPreTrainedModel` still sets `_supports_sdpa = False`, `_supports_flash_attn = False`,
`_supports_flex_attn = False`, and attention is hardcoded via `F.multi_head_attention_forward` with
the gru-gated relative-position bias — which HF itself notes is "incompatible with pluggable attention
interfaces." Locally, 4.57.6 raises *"WavLMModel does not support … scaled_dot_product_attention yet"*.
So the **gated relative-position bias is the blocker** and upstream has not ported it even on main —
an upgrade won't fix it, and a 4.x→5.x major bump is independently risky here (the vendored
`vox_profile` wrappers surgically replace `backbone_model.encoder.layers`, and deps are pinned via
`uv.lock`). The fix is ours regardless: the gated bias is just an additive bias of shape
`[batch*heads, q, k]`, so it folds into `F.scaled_dot_product_attention(q, k, v, attn_mask=gated_bias)`
to reach the fused/mem-efficient kernel — a repo-owned patch (optionally via
`AttentionInterface.register(...)`), version-independent and reinstall-safe.

**ONNX vs TensorRT — either/or?** Different layers, not competitors:
- **ONNX** = interchange *format* (export target).
- **ONNX Runtime (ORT)** = *runtime* dispatching to an execution provider: **CUDA EP** (cuDNN/cuBLAS)
  or **TensorRT EP** (builds a TRT engine under the hood).
- **Native TensorRT** = build/run an engine directly (trtexec / TRT API / torch-tensorrt) — most direct.

Pipeline is `torch → ONNX → TRT engine`; ONNX is the bridge. TRT is indeed fastest on NVIDIA. Rough
ladder: `native TRT ≳ ORT+TRT-EP > ORT+CUDA-EP > eager torch`. The repo's
[benchmark_wavlm_onnx_trt.py](../../scripts/benchmark_wavlm_onnx_trt.py) uses **ORT + TRT-EP** (engine +
timing cache) — a bit slower than native TRT but far less integration work, which is why O4 picks it.
Caveat: `onnxruntime-gpu`/`tensorrt` are **not installed** in this venv, so this path is unmeasured here.

**On the L40S (48 GB):** bigger gun, not efficiency. VRAM is **not** our bottleneck — throughput is
compute-bound (identical bs128 vs bs256) and fp16 didn't even unlock bs512. The L40S helps via
*architecture* (Ada: more SMs/clocks ~1.5–2×, plus FP8), not batch size. The software levers above
are GPU-portable and the right focus now.

## 4. Plan corrections

- **fp16 > bf16** on this hardware (same speed, 5–10× tighter, event-safe) — the plan's "bf16 first"
  should become "fp16 first, bf16 fallback".
- **Half-precision (autocast) is compute-only, not memory** — does not raise the batch ceiling; O3
  still needed for occupancy. (int8/fp16 *weights* via O4/O6 would reduce memory; autocast does not.)
- **WavLM is GEMM-bound, not attention-bound** at these short windows (~150–174 tokens, attention
  ~3% of FLOPs). Correcting the earlier draft: SDPA is a ~1.05× free win, not a headline. The big
  remaining lever is **precision on the GEMMs → INT8 (O6)**, not attention fusion. TRT's "2–4×" was
  overstated for this workload (its attention-fusion edge is small here) — measure before O4.

## Status

O1 ✅ · O2 measured + event-A/B passed ✅ (fp16 default not yet flipped in code) ·
SDPA patch implemented + measured ✅ (~1.05×, lossless, not wired into model load yet) ·
**next: O6 INT8 (likely biggest remaining win), and *measure* TRT before the O4 integration.**
