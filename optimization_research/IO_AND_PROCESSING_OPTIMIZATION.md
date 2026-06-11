# I/O and processing optimization — measured findings

Companion to `baseline_results/OPTIMIZATION_FINDINGS.md` (GPU-compute levers). This doc covers the
**per-archive I/O and decode** path: download client (boto3 vs **s5cmd**) and decoder
(librosa vs **ffmpeg**/swresample). Measured on real full archives, not synthetic.

Harness: [scripts/benchmark_io_and_decode.py](../scripts/benchmark_io_and_decode.py).
Raw data: [baseline_results/io_decode_compare.json](baseline_results/io_decode_compare.json).
Env: A10G dev box, **8 vCPU**, intra-region S3 (`riverside-pro-main`), ffmpeg 4.4.2, s5cmd v2.3.0,
librosa 0.11 + soxr 1.0, torch 2.10. 3 real archives (41–66 min, mono 44.1/48 kHz WAV, 215–348 MB,
GLACIER_IR). 3 trials each, median reported.

## TL;DR — neither lever helps this workload; the "optimized" combo is *slower*

| path | current | candidate | result |
|---|---|---|---|
| download (one ~300 MB wav) | boto3 `download_file` | **s5cmd cp** | **s5cmd loses**: 160 vs 220 MB/s (boto3 default), 246 MB/s (boto3 tuned) |
| decode (→16 kHz mono f32) | librosa (soxr_hq) | **ffmpeg** swresample | **tie**: 0.95–0.98× (ffmpeg slightly *slower* warm), and ffmpeg changes `audio_sha256` |
| end-to-end (dl + decode) | boto3 + librosa | s5cmd + ffmpeg | **0.83× — net slower** |

**Why:** per-archive I/O is already tiny — **download ~1–2 s** (~220 MB/s) + **decode ~1.2–2.0 s**
(librosa runs at **~2000× realtime** warm) = **~2.5–4 s total**, against **minutes** of GPU per
archive. The pipeline is **GPU-bound**; these levers optimize a non-bottleneck, and on single large
objects boto3's multipart already beats s5cmd's defaults. **Recommendation: adopt neither for the
inference worker.** The only dependency-free download win is bumping boto3 concurrency (~+15%, below).

**The real lever found here is not I/O at all — it's §6: VAD-gating inference.** The archives are
per-speaker stems that are ~28% speech, so ~55% of GPU compute is spent on silence; gating to speech
windows is ~2.2× (larger than fp16, and stacks), and the producers already ignore non-speech frames so
it's close to free. That's the recommended next step after shipping fp16.

### Correction to an earlier ad-hoc claim
An initial one-shot probe showed "ffmpeg 33× faster than librosa" (37.6 s vs 1.14 s). **That was a
cache-ordering artifact**, not a real result: librosa ran first and paid the **cold EFS read** of the
file; ffmpeg then read the **warm page cache**. The multi-trial benchmark below isolates it — see the
cold-read row. Lesson logged: always multi-trial and control page-cache order for decode benchmarks.

## 1. Download — boto3 vs s5cmd (per archive, median of 3)

| archive | size | boto3 default | boto3 tuned (conc=64) | s5cmd (defaults) |
|---|---|---|---|---|
| motorpresse (66m) | 348 MB | 1.58 s · 220 MB/s | **1.41 s · 247 MB/s** | 2.17 s · 160 MB/s |
| drtrish (41m) | 215 MB | 1.00 s · 215 MB/s | **0.89 s · 243 MB/s** | 1.43 s · 151 MB/s |
| vitaliy (49m) | 285 MB | 1.11 s · 256 MB/s | **1.09 s · 260 MB/s** | 1.33 s · 214 MB/s |

- **boto3 ≥ s5cmd on every archive.** boto3's `TransferConfig` multipart (default concurrency 10)
  already parallelizes a single large object; **tuning to concurrency 64 is the fastest path (~+15%)**
  and needs no new binary. s5cmd `cp` defaults (concurrency 5, 50 MB parts) are *less* aggressive for
  one big file, so it loses here — one trial even dropped to 95 MB/s.
- **Fairness caveat:** this is single-large-object, intra-region, on a fast box. s5cmd's real strength
  is **bulk parallelism across *many* objects** with low per-file overhead. That regime — the **cache
  warmer** fanning out across thousands of archives — was **not** tested here and is the one place
  s5cmd could still win (`s5cmd run` with a command file). Don't generalize this negative to the warmer.
- Network is noisy (per-trial spread 95–286 MB/s); medians used. Absolute MB/s scales with NIC/instance,
  but the *relative* ordering and the "~1–2 s, immaterial vs GPU" conclusion hold.

## 2. Decode — librosa vs ffmpeg (per archive, warm, median of 3)

| archive | librosa | ffmpeg | speedup | max\|Δ\| | rms Δ | frac \|Δ\|>1e-3 | sha match |
|---|---|---|---|---|---|---|---|
| motorpresse (66m) | 1.87 s (2108× rt) | 1.98 s (1994× rt) | 0.95× | 6.9e-2 | 3.0e-4 | 0.9% | ❌ |
| drtrish (41m) | 1.21 s (2012× rt) | 1.23 s (1977× rt) | 0.98× | 3.2e-2 | 4.8e-4 | 1.9% | ❌ |
| vitaliy (49m) | 1.51 s (1967× rt) | 1.54 s (1930× rt) | 0.98× | 8.0e-3 | 1.1e-4 | 0.2% | ❌ |

- **librosa is not a bottleneck.** With soxr (`soxr_hq`, librosa 0.11 default) it decodes+resamples a
  whole 40–66 min archive in ~1.2–2.0 s warm — **~2000× realtime**. ffmpeg/swresample is **equivalent**
  (actually a hair slower warm); the subprocess buys nothing.
  - *Caveat:* this equivalence is specific to **librosa+soxr**. An older librosa defaulting to
    `kaiser_best`/resampy (numba/python) would be far slower and ffmpeg *would* win — not the case here.
- **ffmpeg changes the cache key for zero gain.** Decoded samples differ (max\|Δ\| up to 6.9e-2;
  the sha never matches), so swapping decoders **invalidates every existing artifact + decoded-cache
  object** (the `audio_sha256` contract, CLAUDE.md) and would need the **event-level A/B gate**
  (`scripts/event_level_ab.py`) — the same gate fp16 had, and the drift here (6.9e-2) is *larger* than
  fp16's. Paying a full-corpus re-baseline + a gate run for a **0.97×** decode is strictly negative.

### 2b. soxr quality preset (soxr_hq → mq/lq) — measured, no win
Same idea as ffmpeg, but tuning the *incumbent* decoder via `librosa.load(res_type=...)` (default is
`soxr_hq`). Warm, median of 3:

| preset | decode (range) | vs hq | max\|Δ\| vs hq | sha match |
|---|---|---|---|---|
| soxr_hq (incumbent) | 1.17–1.91 s | 1.00× | — | — |
| soxr_mq | 1.18–1.88 s | **0.99–1.02×** | ≤3.9e-3 | ❌ |
| soxr_lq | 1.13–1.81 s | 1.03–1.05× | up to **3.9e-1** | ❌ |

- **mq is not faster than hq** (within noise) — at 44.1/48k→16k, libsoxr's hq/mq filter-cost difference
  is negligible. (An *old* librosa using `kaiser_best`/resampy would differ; not this setup.)
- **Decode splits ~50/50 read vs resample** (read 0.5–0.9 s; resample 0.6–1.0 s), so even a free
  resampler caps the decode win at ~0.6–1.0 s — trivial vs minutes of GPU.
- **lq** saves only ~0.05–0.1 s and is numerically way off (max\|Δ\| up to 0.39). Every preset changes
  `audio_sha256` → same re-baseline + event-A/B gate cost. **Verdict: no.**

### Cold read (the artifact, isolated)
The only cold read in the run was librosa's first touch of the 348 MB file: **22.5 s** (then 1.87 s
warm). That ~22 s is **EFS single-stream read latency for 348 MB (~15 MB/s cold), decoder-independent**
— ffmpeg pays the same when it reads cold. It does **not** apply to the worker's decode path, which
decodes a **freshly-downloaded local temp** (already in page cache → warm ~1–2 s). It *is* a real cost
on the **decoded-cache hit path**, which `np.load`s a ~200 MB `.npy` from EFS per hit — but that's an
EFS-read cost, not a decode cost, and unrelated to librosa-vs-ffmpeg.

## 3. End-to-end (download + decode, from medians)

| archive | current (boto3+librosa) | optimized (s5cmd+ffmpeg) | speedup |
|---|---|---|---|
| motorpresse | 3.5 s | 4.1 s | 0.83× |
| drtrish | 2.2 s | 2.7 s | 0.83× |
| vitaliy | 2.6 s | 2.9 s | 0.91× |

Both candidates lose, and the combined per-archive cost (~2.5–4 s) is **<2%** of the GPU time per
archive (minutes). I/O is fully hidden behind the GPU by the existing 4-deep prefetcher.

## 4. torchcodec is unusable here (and it's not an ffmpeg-version issue)
torchcodec 0.9.1 fails to load: `undefined symbol: _ZN3c1013MessageLoggerC1EPKciib` (= `c10::MessageLogger`,
a **torch** symbol) — a torch **C++ ABI mismatch**, not a missing/old ffmpeg. The loader found ffmpeg 4.4
fine (tried `core5`→none, `core4`→matched) and failed only on the torch symbol. The version matrix is
inconsistent (torch 2.10.0+cu128 / torchaudio 2.11.0+cu130 / torchcodec 0.9.1); fixing it means moving
pinned torch — risky for the whole stack, and moot since (a) the ffmpeg **CLI subprocess** is ABI-immune
and works, and (b) decode isn't a bottleneck anyway.

## 5. Recommendations

1. **Do not adopt s5cmd or ffmpeg in the inference worker.** Measured net-slower; both target a
   non-bottleneck (GPU-bound). Ship the GPU lever (fp16) — that's where the time is.
2. **Cheap, dependency-free download win (optional):** raise boto3 `TransferConfig(max_concurrency=64,
   multipart_chunksize=16MB)` in the resolver/cache download path
   ([audio_resolver.py](../audio_classification_playground/acoustic_events/orchestration/audio_resolver.py),
   [audio_cache.py](../audio_classification_playground/acoustic_events/orchestration/audio_cache.py)) —
   ~+15% on big objects. Immaterial to the worker (download already hidden), but it speeds the **cache
   warmer**, which is the genuinely I/O-bound stage.
3. **Re-test s5cmd where it can actually win:** the **warmer** copying *many* archives in one `s5cmd run`
   batch vs N boto3 threads. That's the untested regime; this single-object result doesn't settle it.
4. **Keep the GPU-bound gate.** Before any further I/O work, aggregate `prefetch_wait_sec` from the
   worker timings JSONL (`_meta/timings/*.jsonl`). If it's ≈0, I/O is hidden and none of this matters
   for throughput.

## 6. The real headroom: VAD-gated inference (~2× compute, bigger than fp16)

Unlike the I/O levers above, this one is large. The pipeline runs affect/disfluency/emotion on the
**full timeline**, but the archives are **per-speaker stems** (named per participant) — each is silent
whenever that speaker isn't talking. Measured with production-config Silero VAD
([scripts/quantify_vad_gating.py](../scripts/quantify_vad_gating.py),
[baseline_results/vad_gating_potential.json](baseline_results/vad_gating_potential.json)):

| archive | dur | speech % | affect windows kept | disf/emo kept | compute saved (affect) |
|---|---|---|---|---|---|
| motorpresse | 66 m | 24.2% | 40.6% | 38.9% | **59%** |
| drtrish | 41 m | 39.4% | 58.9% | 56.5% | 41% |
| vitaliy | 49 m | 21.5% | 34.5% | 32.8% | **66%** |
| **mean** | | **28%** | **44.6%** | ~43% | **~55%** |

Speech is only ~28% of each stem; after window dilation (a 3.5 s window with 0.25 s hop "lights up" for
any speech in its span), **~45% of windows touch speech → ~55% of GPU compute is spent on silence.**
Gating to speech windows is **~2.2× compute reduction — larger than fp16's 1.85×, and they stack**
(combined ~4× vs today). *Caveat: 3 archives, indicative not definitive (speech % ranges 21–39%); a
manifest-wide sample is needed to firm up the distribution, and a safe gate dilates a bit more (below),
trimming the realized saving somewhat.*

### Why this is closer to free than it looks: the producers already gate on speech
Filling non-speech with a sentinel barely matters, because **every consumer already ignores non-speech
frames**:
- **affect** — `global_stats(values, interior)` and per-block baseline/z-scoring run only on `interior`
  (frames inside VAD blocks); non-speech frames are never read
  ([affect/pipeline.py:137-159](../audio_classification_playground/acoustic_events/producers/affect/pipeline.py#L137-L159)).
- **disfluency** — candidates emitted only if they overlap speech (`require_vad_for_events=True`,
  `min_support_frames`).
- **emotion** — `speech_mask` gates detection; thresholds computed on `valid_mask` (speech) frames.

So VAD-gated inference + neutral fill should be **output-identical**, *provided the inference gate is a
superset of what each producer reads*.

### Sizing the change — medium, low-to-moderate risk (not a rewrite)
1. **Inference (the bulk):** thread VAD intervals into the affect/disfluency/emotion runners + persistent
   predictors (VAD already runs first in `run_all_inference`, but its intervals aren't passed to the GPU
   tasks). In each predictor batch loop, build a keep-mask over the window grid, run only kept windows,
   scatter into a full-length array, fill the rest:
   - affect A/V/D and disfluency logits → **0.0** (finite; producer ignores them anyway),
   - emotion probabilities → **uniform 1/C** (must satisfy the sum-to-1 validation). `n_frames` stays
     full-length, so all downstream frame/centre/hop alignment is unchanged.
   emotion's on-GPU `predict_audio` strided path needs the mask applied to the stride grid — slightly more
   work than affect/disfluency (which take an explicit `windows` array), same idea.
2. **Safe gate = superset (the edge-correctness rule).** A kept window is bit-identical to full-timeline
   (gating skips windows, never changes a computed window's audio), so the *only* failure mode is dropping
   a window a producer reads. Avoid it with: **keep any window that OVERLAPS a VAD interval, after bridging
   VAD gaps by ~1.0–1.5 s.** Why this is a provable superset of every consumer:
   - windows are 3.5 s (affect) / 3.0 s (disf/emo), hop 0.25 s — a frame summarizes `[i·hop, i·hop+W]`.
   - affect reads *containment* (`assign_frame_blocks`: window ⊆ merged block, gaps bridged ≤ `vad_merge_gap_sec=0.5`),
     which is **stricter than overlap** → overlap-gating covers it automatically.
   - disfluency uses overlap with `merge_gap_sec=0.5`/`min_support_sec=0.5`; emotion uses overlap with
     `support_close_gap_sec=**1.0**` (the binding constant).
   - So bridging ≥1.0 s (+margin) and gating by **overlap** (NOT containment — containment would drop
     boundary-straddling windows the producers read) is a superset of all three.
   Keeps inference loosely coupled to producers (needs VAD + one conservative bridge constant). Cost: the
   bridge+overlap dilation keeps a few more windows than the bare speech fraction, so realized saving is
   a bit under the measured 55% (the §6 `window_keep` dilated by window length but not yet by the 1 s gap).
3. **Immutability:** add a `vad_gated` flag (+ dilation params) to `inference_config` so the artifact hash
   reflects it — same pattern as `autocast_dtype`. The stored predictions differ at non-speech frames, so
   it's a new artifact lineage.
4. **Producer side:** mostly verification (affect/emotion confirmed speech-only; confirm disfluency's
   detection thresholds aren't computed over non-speech frames).
5. **Gate:** run the event-level A/B (`scripts/event_level_ab.py`) full-timeline vs gated → expect **0
   drift** (consumers are speech-gated, gate is a superset). The dilation margin is the one correctness
   lever; the A/B catches it if too tight.

**Verdict: this is the lever worth pursuing after fp16.** It targets the actual cost (GPU on silence),
the consumers already assume speech-gating, and the fill value is essentially irrelevant — making it far
closer to a "stop computing what's already discarded" optimization than a semantic change.

### IMPLEMENTED + measured (real-archive A/B, 3 archives, L40S)
Built behind `--vad-gating` (default off); see `VAD_GATING_IMPLEMENTATION_PLAN.md` for the design/status
and `baseline_results/vad_gating_ab.json` for data. Event-A/B (full vs gated, real composition path):
- **affect & emotion: bit-identical** on all 3 archives (0 dropped/added, label 1.000, 0 drift).
- **disfluency: not identical** (1 dropped event on 2/3 archives) — its region detection reads non-speech
  frames; **excluded from the default gated set** (opt-in once its producer is made speech-scoped).
- **Speed (GPU/archive):** affect+emotion gated (the safe, event-identical config) = **~1.4× mean**
  (1.28–1.50×); all-three gated would be **~2.3× mean** (1.65–2.79×) once disfluency is made gating-safe.
  Per-task speedups scale with silence (affect 1.6–2.8×, disfluency 1.7–2.9×, emotion 1.6–2.6×).

## Reproduce
```
uv run python scripts/benchmark_io_and_decode.py \
  --index benchmark_audio/index.json \
  --json-out optimization_research/baseline_results/io_decode_compare.json
# flags: --download-trials N --decode-trials N --skip-download --skip-decode
```
Per-archive, per-trial data (incl. cold trial-0) is in the JSON for report tables.
```
uv run python scripts/quantify_vad_gating.py \
  --index benchmark_audio/index.json \
  --json-out optimization_research/baseline_results/vad_gating_potential.json
```
(§6 — speech fraction + per-task window-keep / compute-saving from production-config Silero VAD.)
