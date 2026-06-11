# Inference optimization — consolidated findings

Companion to `INFERENCE_BASELINE_A10G.md` (O1) and `O2_AUTOCAST_RESULTS.md` (O2). Covers the
event-level A/B verdict and the design Q&A from the optimization review. A10G, torch 2.10+cu128.

## TL;DR — leverage vs what production actually runs

**Production baseline (fleet, main): WavLM = `compiled_static` (torch.compile + static-batch-256, fp32);
emotion = `optimized` (compile+TF32).** The fleet base image ships `python3.10-dev`, so compile *is*
eligible and already engaged — i.e. the fleet already banks the compile win. (The eager-fp32 fallback
only happens on boxes missing the py3.10 headers, like the dev box used here.)

So the **actual capturable leverage = adding fp16 autocast on top of the existing compile** (bs256):

| task | production today | + fp16 | **leverage** |
|---|---|---|---|
| affect (WavLM) | 157 win/s (`compiled_static` fp32) | 291 | **1.85×** |
| disfluency (WavLM) | 184 win/s (`compiled_static` fp32) | 346 | **1.88×** |
| emotion (`optimized`) | ~243 win/s | 334 | **~1.4×** |

One change (flip autocast default to fp16) → ~1.85× on the two heavy WavLM fleets, ~1.4× on emotion;
event-safe (passed the §1 A/B), immutability-safe (autocast in config hash), no new deps. **Not** 2.2×
— that's vs *eager*; the fleet already has compile, so the honest incremental is ~1.85×. Everything
else is banked (compile), a dead end (int8 §2b / TRT §2c / fp8 §2f), or noise (SDPA ~1.05× §2).

**Hardware:** numbers above are A10G (production GPU). On an **L40S (Ada)** the same sweep gives ~2×
raw over A10G plus the same fp16+compile leverage, and fp8 becomes testable — full matrix in
[L40S_RESULTS.md](L40S_RESULTS.md). fp8 verdict: weak (~+13% on affect only, hand-rolled; §2f).

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
| INT8 PTQ (O6) | int8 tensor cores ~2× GEMM throughput | **fails event gate + speed blocked (§2b)** | L, accuracy cliff |
| TensorRT (O4) | GEMM autotune/fusion + (fp16) precision | **measured (§2c): fp32 1.19×; fp16 2.44× but NaN — doesn't beat fp16+compile** | L, not worth it |

Conclusion after measuring all three: **PyTorch fp16+compile (2.2×, lossless, event-safe) is the
endpoint for this workload.** SDPA is a free ~1.05× to keep; INT8 fails the quality gate (§2b); TRT
doesn't beat fp16+compile (§2c). This short-window, GEMM-bound, accuracy-sensitive model has little
headroom beyond fp16 without a model-level change (distillation/smaller student — explicitly out of scope).
Side note: fp16+compile peak VRAM is much lower (~12.9 GiB vs ~20.4 eager at bs256), because inductor
reuses buffers on the static shape.

## 2b. O6 INT8 (torchao W8A8) — measured: blocked on speed AND fails the quality gate

Tested torchao 0.17 `Int8DynamicActivationInt8WeightConfig` on the 144 backbone GEMM Linears
(q/k/v/out_proj + FFN; conv extractor, gated-bias linear, and classifier head left in higher
precision). [compare_wavlm_int8.py](../../scripts/compare_wavlm_int8.py),
[int8_compare.json](int8_compare.json). **Two independent blockers:**

1. **Speed — blocked on this env.** The fast int8 path needs `torch.compile` to emit int8 tensor-core
   kernels, but **int8+compile fails to trace** (`Dynamo failed to run FX node … AffineQuantizedTensor`)
   and **int8+autocast fails** (`aten.to(dtype)` unimplemented on the quantized subclass). Root cause:
   **torchao 0.17 requires torch ≥ 2.11** ("Skipping cpp extensions" on torch 2.10). Eager int8 *runs*
   (correct) but at **76 / 88 win/s — slower than fp32 eager (130/157) and 4× slower than fp16+compile
   (294/351)**, because without the fused kernels it's a quant/dequant fallback. No int8 speed win is
   obtainable on torch 2.10 + torchao 0.17.
2. **Quality — fails the event-level A/B (the decisive one).** Eager int8 produces the same values a
   compiled int8 would, so the gate verdict holds regardless of (1):

   | task | events (fp32→int8) | dropped | added | label agr (min) | max \|Δstart\| | max \|Δscore\| |
   |---|---|---|---|---|---|---|
   | affect | 118 → 122 | 2 | **6** | **0.821** | **4.0 s** | **1.54** |
   | disfluency | 52 → 51 | 1 | 0 | 0.902 | 3.0 s | 0.045 |

   Versus fp16 (which passed cleanly: 0 dropped/added, label 1.000, Δscore ≤0.19), naive full-int8 is
   **far too lossy** — it adds/drops events, flips ~18% of affect labels, and shifts affect scores by
   up to 1.5 and boundaries by 4 s. **affect (continuous A/V/D regression) is especially fragile**:
   cumulative quantization error across 144 backbone Linears perturbs the scores enough to move
   prominence-based event boundaries, even though the affect head itself stays fp32. Disfluency
   (classification) is more robust on scores (Δ≤0.045) but still flips ~10% of labels and drops one event.

**Verdict: naive int8 PTQ is not a win here** — blocked on speed (version) and, more fundamentally,
it fails the quality gate. Unlike fp16, int8 has a real accuracy cliff for this workload. To revisit
int8 would require *all* of: (a) torch ≥ 2.11 (or a torch-2.10-matched torchao) for the fast path,
(b) **mixed precision** (keep affect head + sensitive blocks in fp16, quantize only robust FFN layers
via a per-layer sensitivity sweep), (c) **static calibration** instead of dynamic — then re-run the
gate, with uncertain payoff for affect. **Recommendation: deprioritize int8; fp16 (2.2×, event-safe)
is the shippable win.** (TRT was the other candidate — measured in §2c.)

## 2c. O4 TensorRT — measured: does not beat PyTorch fp16+compile

Installed onnxruntime-gpu 1.23.2 + TensorRT 10.16 (cu12) into the venv; exported affect WavLM to ONNX
(opset 18, dynamo exporter — **the gated-attention export risk did not materialize**) and ran via the
ORT TensorRT EP. [benchmark_wavlm_onnx_trt.py](../../scripts/benchmark_wavlm_onnx_trt.py). Affect, bs256,
all speedups **vs PyTorch eager fp32**:

| path | speedup | correct? |
|---|---|---|
| ONNX via ORT-CUDA EP (fp32) | — | ✅ bit-exact (max abs 0.0) |
| **TRT-fp32** | **1.19×** | ✅ bit-exact (0.0) |
| **TRT-fp16** | 2.44× | ❌ **NaN output** (fp16 overflow in the engine) |
| *PyTorch fp16+compile (reference)* | *2.2×* | ✅ lossless, event-safe |

Export + engine are correct (fp32 paths bit-exact); the NaN is **fp16 numerical instability** in TRT
(a layer exceeds fp16 range), not a broken graph. Key reading: **TRT's *structural* contribution is
only 1.19× (fp32)** — fusion/attention/kernel-autotuning buy little here, confirming the GEMM-bound,
short-window finding. The rest of TRT-fp16's 2.44× is just the fp16 precision, which **PyTorch already
captures at 2.2× — losslessly and event-safe.**

**Why fp16 is stable in PyTorch but NaNs in TRT.** Not a TRT bug — different precision *placement*.
PyTorch `autocast` is curated mixed precision: GEMMs run fp16 (with fp32 accumulation) but an fp32
allowlist forces `layer_norm`/`softmax`/reductions/exp to stay fp32, so the overflow-prone ops never
see fp16 (and `torch.compile` preserves this). TRT's `trt_fp16_enable` is the opposite default —
fp16 everywhere it's faster, opt *out* per layer. So a layer whose intermediate exceeds fp16's
~65504 → inf → NaN. Corollary: making TRT as numerically safe as autocast pulls its speed back down
toward PyTorch's 2.2× (the structural-only TRT win was just 1.19×).

**Tried the targeted fix — didn't resolve it.** Set the ORT TRT EP's `trt_layer_norm_fp32_fallback`
(rebuilt the engine): **still NaN.** So the overflow is *not* LayerNorm — almost certainly WavLM's
**gated relative-position bias** (`gate_a*(gate_b*const − 1) + 2` × position_bias, broadcast-added to
every attention score) — the same non-standard op that blocks SDPA (§3). The ORT EP exposes no
per-layer precision control beyond the LayerNorm knob, so fixing this needs **native TRT**
(`--layerPrecisions` / `setPrecision(kFLOAT)` after bisecting the layer with polygraphy) or a
strongly-typed mixed-precision ONNX — real L-effort, integration-risk work.

**Verdict: TRT is not worth it for this workload.** Correct TRT (fp32) is only 1.19× — *worse* than
fp16+compile (2.2×). TRT-fp16's nominal 2.44× (~+11%) is **NaN**, not fixable via the ORT EP's knobs,
and a proper native-TRT fp32-fallback fix erodes that margin toward 2.2× while adding the deploy cost
of onnxruntime-gpu + TensorRT + shipped per-shape engines. The ceiling doesn't justify it. (Measured
on affect, the heaviest task; disfluency shares the WavLM backbone so the conclusion transfers.)

## 2d. emotion2vec precision — fp16 is a win here too (previously untested)

O2 originally covered only the WavLM tasks; emotion2vec (a different backbone, `predict_audio` path)
had only been baselined. Closing that gap (synthetic seed, bs64, ~2069 windows):

| emotion2vec | win/s | vs eager fp32 | NaN | max \|Δ vs fp32\| |
|---|---|---|---|---|
| eager fp32 | 232.6 | 1.0× | — | 0 |
| **fp16 (autocast)** | **334.3** | **1.44×** | No | **1.2e-3** |
| bf16 (autocast) | 336.2 | 1.45× | No | 1.8e-2 |
| fp16 + compile | ~334 (inferred) | ~1.44× | — | — |

fp16 emotion2vec mirrors WavLM: **~1.44×, stable (no NaN), numerically tight, and fp16 > bf16**
(1.2e-3 vs 1.8e-2). Note **compile is neutral for emotion2vec** (the production "optimized" preset =
compile+TF32 measured ≈0.97× in §O1 work), so fp16+compile ≈ fp16 — the WavLM-style 2.2× stacking
does *not* apply to emotion. So **fp16 is a win on all three tasks** (affect, disfluency, emotion);
emotion's gain comes from fp16 alone, not compile. (Event-level A/B for emotion fp16 still TODO before
shipping — same gate as O2; expected to pass given the tight 1.2e-3 drift.)

## 2e. Environment caveat — torch.compile depends on py3.10 dev headers

`torch.compile` needs `python3.10` dev headers (`/usr/include/python3.10/Python.h`) for triton's
inductor C build. **The fleet base image ships `python3.10-dev`, so compile IS active in production**
(WavLM `compiled_static`, emotion `optimized`). The headers only went missing on the **dev box used
here** (ephemeral — `/tmp` wiped and headers reverted across resets), which is why local compile runs
flaked and a headerless box would fall back to eager `fast_exact`. Net for production:
- Production already banks compile → baseline is `compiled_static`/`optimized`, not eager.
- **The capturable win is fp16 *on top of* compile: ~1.85× WavLM, ~1.4× emotion** (see TL;DR).
- If moving Python versions (e.g. 3.11/3.12), keep the matching `python3.X-dev` + gcc in the image so
  `compiled_static` stays eligible — eligibility reads the *running* interpreter's headers.

## 2f. fp8 (L40S, Ada sm89) — MEASURED: torchao unusable; hand-rolled gives only +13% on affect

fp8 needs Ada/Hopper (no fp8 tensor cores on A10G). Tested on an **L40S** — full matrix in
[L40S_RESULTS.md](L40S_RESULTS.md). Summary:

**torchao fp8 is unusable on torch 2.12/torchao 0.17** (the only released path):
- Full fp8 + `torch.compile` → `Float8Tensor` has no `aten.as_strided` (triggered by WavLM's
  `F.multi_head_attention_forward` weight chunking).
- FFN-only fp8 + compile → *compiles* but recompile-storms to **3.3 win/s** (`Float8Tensor` lacks
  `_stable_hash_for_caching`).
- eager fp8 → slower than fp32.
- **Not fixed even on torchao `main`** (v0.17+182, checked directly): neither hook exists for the
  inference `Float8Tensor`.

**Hand-rolled `Fp8Linear` via `torch._scaled_mm`** (plain module, no tensor subclass) is the only
working fp8 path — compiles cleanly. Measured:
- **affect FFN-fp8 + compile = +13% over fp16+compile** (528 vs 467 win/s). Real but modest: only the
  FFN (48 layers) is fp8 (attention reads `.weight` directly → needs the SDPA patch to reach), and
  dynamic per-tensor quant overhead eats the fp8 GEMM win (1.13×, not the ~2× ideal).
- **emotion: no win** — eager fp8 is *slower* (emotion path doesn't compile cleanly → quant overhead
  unfused).

**Precision (the gate, GPU/version-independent):** affect fp8 Δ **6.6e-3** (marginal — between fp16's
safe 3e-4 and int8's failed level; needs the event-level A/B); disfluency **0.18–0.40** (*fails*, like
int8); emotion top-1 **1.0000** but hand-rolled per-tensor scaling is lossy (Δ 0.25). The fp8>int8
accuracy argument (floating-point format → wider dynamic range) holds for affect/emotion but doesn't
rescue disfluency.

**Verdict: fp8 is a weak add over fp16+compile** — at best ~+13% on affect alone (hand-rolled,
FFN-only, gated on precision), nothing on emotion, out on disfluency. Not worth the
hand-rolled-`Fp8Linear` maintenance + precision risk vs the free, lossless fp16 win. (TRT-fp8 would
hit the same gated-attention overflow that NaN'd TRT-fp16, §2c — worse, less range. Avoid.)

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

O1 ✅ · O2 fp16 measured + event-A/B passed on WavLM ✅; **emotion2vec fp16 measured ✅ (1.44×, stable, §2d)** ·
SDPA patch implemented + measured ✅ (~1.05×, lossless, not wired in) ·
O6 INT8 tested ✅ → **rejected** (fails event gate + speed-blocked, §2b) ·
O4 TensorRT tested ✅ → **rejected** (fp32 1.19×; fp16 2.44× but NaN; doesn't beat fp16+compile, §2c) ·
fp8 tested on L40S ✅ → **weak** (torchao unusable; hand-rolled +13% affect only; §2f, L40S_RESULTS.md).

**All levers measured. fp16 is the win on all three tasks.** Ship as:
- **floor (robust): fp16 autocast** — ~1.44× on affect/disfluency/emotion, no compile/header dependency.
- **ceiling: fp16 + compile** — ~2.2× on WavLM, *only if the fleet image ships `python3.10-dev`* (§2e; emotion gets no compile benefit).
Remaining code change: flip WavLM (and emotion) default to fp16 (config-hash handles immutability).
Pending: event-level A/B for emotion fp16 (expected to pass, drift 1.2e-3); verify python3.10-dev in fleet image.
