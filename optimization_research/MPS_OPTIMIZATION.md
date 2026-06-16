# MPS Co-Location — Sustained-Load Benchmark Findings & Optimization

> **Result (2026-06-15):** sustained load exposed emotion2vec being SM-starved by the
> batch-512 WavLM clients (MPS gated only 1.32×, emotion inflated 3.9×). Fix = cap the
> WavLM clients' MPS SM share (40% each, emotion uncapped) + WavLM batch 512→256. **Validated:
> MPS gated 1.32×→1.49×, emotion inflation 3.9×→1.87× (tail gone), no cost to affect/disfluency.**
> Details in §6. This is the keeper config for the single-GPU production path.

Controlled A/B of **MPS-colocated vs dedicated** GPU inference on **one** Blackwell
RTX PRO 6000 (96 GB), under sustained load. This is the follow-up to the optimistic
3-archive warm validation in [MPS_COLOCATION_TESTING.md](../MPS_COLOCATION_TESTING.md),
which never ran long enough to expose contention. Harness:
[scripts/benchmark_mps_vs_dedicated.sh](../scripts/benchmark_mps_vs_dedicated.sh);
launcher under test: [scripts/run_mps_colocated.sh](../scripts/run_mps_colocated.sh).

- **Subset:** 40 archives, mean audio ≈ 63 min, up to 3.2 h (real sustained load, real tails).
- **Arms:** `dedicated` (each task alone, back-to-back, full GPU) vs `mps` (affect+disfluency+emotion co-located via CUDA MPS). Both `gated` (VAD-gated, `--require-precomputed-vad`) and `ungated` (full timeline).
- **puregpu** run = warm decoded-audio cache → pure GPU time (the clean apples-to-apples arm).
- **e2e** run = MPS only, intended cold/live-S3 — but see the caveat in §5 (it ran 100% warm).
- Logs: `puregpu …-160906-42c9g.log` (2026-06-15 14:28), `e2e …-160911-njf8j.log` (13:50),
  under `/efs/dvir/data/magic-clips-research/acoustic-understanding/scratch/mps-bench/`.

---

## 1. Headline results (puregpu, warm, apples-to-apples)

| arm / mode | wall | per-task walls (dedicated, serial) |
|---|---|---|
| dedicated · gated | **870 s** | affect 362 + disfluency 303 + emotion 205 |
| dedicated · ungated | **1319 s** | affect 552 + disfluency 481 + emotion 285 |
| mps · gated | **658 s** | (co-located) |
| mps · ungated | **1194 s** | (co-located) |

- `[gated]   dedicated serial 870 s vs mps 658 s →` **MPS speedup 1.32×**
- `[ungated] dedicated serial 1318 s vs mps 1194 s →` **MPS speedup 1.10×**
- VAD-gating speedup (ungated/gated): **dedicated 1.52×**, **mps 1.81×**

**MPS beats dedicated per-GPU in both modes**, but well short of the 1.44× from the
warm 3-archive validation — and ungated barely clears break-even at 1.10×.

e2e (MPS only): gated **709 s**, ungated **1203 s**, VAD-gating speedup **1.70×** —
within ~8% / ~1% of puregpu (but the cache was warm; §5).

---

## 2. Per-archive `inference_sec` by task

The CLI's `affect_sec/disfluency_sec/emotion_sec` columns show median 0.000 — an
aggregation artifact (each single-task record only fills its own column). Recomputed
from the raw timing JSONL grouped by `task_group`. Values = **mean / p50 / p90 / max** (s):

| run / mode | affect | disfluency | **emotion** |
|---|---|---|---|
| puregpu dedicated gated | 7.8 / 7.2 / 13.3 / 25.0 | 6.6 / 6.2 / 10.8 / 21.4 | **4.0 / 3.8 / 6.6 / 12.6** |
| puregpu dedicated ungated | 13.5 / 10.8 / 24.2 / 40.5 | 11.7 / 9.5 / 20.7 / 34.4 | **6.5 / 5.3 / 11.8 / 18.9** |
| puregpu mps gated | 13.8 / 12.9 / 24.1 / 48.4 | 13.4 / 12.3 / 22.0 / 42.8 | **15.7 / 5.7 / 42.4 / 75.7** |
| puregpu mps ungated | 25.4 / 20.9 / 46.8 / 80.0 | 24.2 / 19.4 / 43.3 / 71.8 | **29.1 / 7.7 / 74.8 / 224.6** |
| e2e mps gated | 13.9 / 13.3 / 23.9 / 49.5 | 13.2 / 12.7 / 21.1 / 39.2 | **16.3 / 6.7 / 38.5 / 79.7** |
| e2e mps ungated | 25.4 / 20.8 / 46.0 / 80.6 | 24.8 / 20.3 / 44.1 / 72.6 | **29.2 / 7.8 / 66.1 / 213.3** |

---

## 3. The dominant finding — emotion2vec is starved under co-location

Co-location inflates per-archive time for every task, but **emotion disproportionately**:

| task | dedicated→MPS inflation (gated) | notes |
|---|---|---|
| affect | 7.8 → 13.8 s = **1.8×** | WavLM, compiled_static batch 512 |
| disfluency | 6.6 → 13.4 s = **2.0×** | WavLM, compiled_static batch 512 |
| **emotion** | **4.0 → 15.7 s = 3.9×** | emotion2vec, fixed `[64, 48000]` batch |

- Emotion's per-archive sum under MPS gated (≈ 15.7 s × 40 ≈ **628 s**) ≈ the **entire
  MPS gated wall (658 s)**. Emotion is the laggard that *sets* the co-located wall.
- It is worst on long archives: emotion ungated max **224.6 s** vs dedicated **18.9 s**.
  p50 stays low (5.7 s) while mean explodes (15.7 s) → heavy right tail, i.e. emotion
  gets crushed precisely when the two WavLM streams are running their biggest batches.

**Root cause (mechanism).** WavLM `compiled_static` at batch 512 produces large GEMM
kernels that span the SM array for long, continuous stretches (that's *why* the batch
was raised — to fill SMs for a solo model). emotion2vec `optimized` uses much smaller
fixed `[64, 48000]` batches; its shorter, partly memory-bound kernels lose SM
arbitration to two batch-512 WavLM monoliths running concurrently. The static-batch
tuning that helps a *solo* WavLM becomes the thing that monopolizes SMs and starves the
co-located emotion stream under sustained load.

---

## 4. Throttle vs starvation — it is neither (it's scheduler contention)

**Not throttling.** SM-clock conditioned on under-load samples (`SM%>50`, excluding the
idle-startup floor that makes the naive global min read 180 MHz):

| arm | SM% (load) | SM-clk mean (min) MHz | ceiling | % loaded <95% clk | pwrMax | tempMax |
|---|---|---|---|---|---|---|
| puregpu dedicated gated | 97% | 2359 (2310) | 2422 | 0.0% | 551 W | 68 °C |
| puregpu dedicated ungated | 98% | 2358 (2317) | 2422 | 0.0% | 552 W | 68 °C |
| puregpu mps gated | 99% | 2356 (2280) | 2422 | 0.2% | 574 W | 72 °C |
| puregpu mps ungated | 100% | 2356 (2332) | 2422 | 0.0% | 572 W | 71 °C |
| e2e mps gated | 99% | 2358 (2302) | 2430 | 0.2% | 582 W | 73 °C |
| e2e mps ungated | 100% | 2359 (2325) | 2430 | 0.0% | 581 W | 73 °C |

Clocks hold ~98% of ceiling under sustained load, 0% sag below 95%, 73 °C max,
~580 W — no thermal/power ceiling hit.

**Not (pipeline) starvation either.** SM% under load is 99–100% in every MPS arm; the
GPU is *fuller* under MPS, not idle. e2e emotion ≈ puregpu emotion (16.3 vs 15.7 gated;
29.2 vs 29.1 ungated) — no I/O-driven gap. The GPU waited only `prefetch_wait_sec` ≈
1.6 s/archive (gated) on I/O while prefetched items aged `prefetch_ready_age_sec` ≈
145–355 s in queue waiting for the GPU → GPU-bound, items pile up *for* the GPU.

⇒ Emotion's slowdown is **intra-GPU kernel-scheduling contention between co-located MPS
clients**, not hardware throttling and not CPU/prefetch starvation.

---

## 5. Caveat — the e2e arm never exercised cold S3

The e2e run was *intended* as cold-cache + live-S3, but `object_cache_hit = 160/160`
(gated) and `120/120` (ungated), with `download_sec = 0` / `decode_sec = 0` on every
record (source objects are `DEEP_ARCHIVE`/`GLACIER_IR`, but the EFS decoded cache was
hit throughout). The `e2e/audio_cache` dir lives outside the per-arm output that
`CLEAN=1` wipes, so earlier failed e2e retries had fully populated it.

So e2e here validates the **prefetch/cache pipeline** (it fully hides the ~1.1 s cache
read behind ~16 s of GPU work) but does **not** prove robustness against a genuine cold
first-touch (live S3 GET + ffmpeg decode of cold-class objects). The e2e≈puregpu emotion
match is therefore expected, not a cold-path acquittal. **Re-run with an empty
`e2e/audio_cache`** to close this gap.

---

## 6. Fixing the emotion starvation

Goal/target: emotion sum (~628 s) currently ≈ the MPS wall (658 s). If we cut emotion's
inflation from 3.9× toward ~2×, emotion sum drops to ~320 s — below affect's 552 s sum —
so **affect becomes the wall** and MPS gated could fall from 658 s toward ~560–580 s,
lifting the speedup from 1.32× toward ~1.5×. That is the prize and the A/B success metric.

### ✅ VALIDATED (2026-06-15, A+B run: 40% caps on affect/disfluency + WavLM batch 256, 20 archives)

Applied lever A (`AFFECT_THREAD_PCT=40 DISFLUENCY_THREAD_PCT=40`, emotion uncapped) **and**
lever B (`WAVLM_STATIC_BATCH_SIZE=256`) together. The prediction held almost exactly.

| metric | baseline (no caps, b512, 40 arc) | **A+B (40/40 caps, b256, 20 arc)** |
|---|---|---|
| MPS speedup, **gated** | 1.32× | **1.49×** |
| MPS speedup, ungated | 1.10× | 1.09× (unchanged) |
| MPS VAD-gating speedup | 1.81× | 1.93× |

Per-archive `inference_sec`, MPS **gated** arm, with the dedicated (contention-free) reference
on the same archives (N changed 40→20, so compare **inflation**, not raw walls):

| task | baseline mean (max) | **A+B mean (max)** | inflation: was → now |
|---|---|---|---|
| affect | 13.8 | 13.8 (37.2) | 1.8× → 2.17× |
| disfluency | 13.4 | 12.3 (29.2) | 2.0× → 2.25× |
| **emotion** | **15.7 (75.7)** | **6.8 (14.9)** | **3.9× → 1.87×** |

- **Emotion de-starved:** mean halved (15.7→6.8 s) and the long-archive tail collapsed
  (max 75.7→14.9 s). e2e agrees (emotion gated mean 7.3 s) — holds under the live pipeline.
- **Wall now affect-bound, as predicted:** emotion sum ≈136 s (6.8×20) is no longer the
  bottleneck; affect (13.8×20 ≈ 276 s) ≈ the 290 s MPS gated wall.
- **Capping cost ≈ nil:** affect's absolute MPS time is unchanged (13.8 s) and disfluency
  improved (13.4→12.3 s) — the caps bound the WavLM SM footprint without slowing tasks that
  were already contended. (The rising inflation *ratio* is the dedicated baseline getting
  faster at b256 + a shorter-archive mix, not MPS regressing.)
- **No throttling:** under MPS load clocks held 2330–2337 / 2430 MHz, ≤73 °C, ≤606 W.
- **Ungated unchanged (1.09×):** on the full timeline each WavLM task already saturates the
  SMs, so caps just time-slice — emotion ungated still improved (29.1→13.7 s) but isn't the wall.

**Decision:** batch 256 + 40/40 caps (emotion uncapped) is the keeper for the single-GPU
production path. The fallback (dedicated GPU for emotion) is no longer needed. Next tuning
lever, if chasing >1.49×: emotion now has slack, so **loosen the caps to 45–50%** to let
affect (the new wall) run faster.

### Recommended (keep everything on one GPU) — try in this order

**(A) Cap the WavLM clients' SM share via `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (per client).**
This is the direct knob, currently deliberately *off* in
[run_mps_colocated.sh](../scripts/run_mps_colocated.sh) lines 66–68 — a decision made on
the warm 3-archive data that never showed starvation. The new sustained-load data
contradicts it. Cap the *aggressors*, leave the *victim* uncapped, e.g.:

```
affect:     CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40
disfluency: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40
emotion:    (uncapped / 100)   # grabs the guaranteed-free ~20% + all interleave gaps
```

The MPS server reads the env per client at connect, so this is a per-worker `export`
just before launch in the task loop. Asymmetric on purpose — a flat 33/33/33 hard-
partition would lose the overlap that makes MPS worth it. A/B the split (e.g. 35/35,
40/40, 45/45) — too tight a cap throttles WavLM throughput; too loose doesn't free
emotion. Watch each task's `inference_sec` *and* the arm wall.

**(B) Lower the co-located WavLM static batch (512 → 256 / 128).** Shorter WavLM kernels
yield the SMs more often → more interleave points for emotion's small kernels. The bench
used 512; the `compiled_static` default is 256. `wavlm_static_batch_size` is excluded
from the semantic hash, so this never changes artifacts — only speed/fairness. A/B
512/256/128, ideally combined with (A). Reuse
[scripts/compare_static_batch_throughput.sh](../scripts/compare_static_batch_throughput.sh)
but measure *all three* tasks' per-archive time, not just WavLM throughput.

**(C) Lower-confidence levers if (A)+(B) underdeliver:** `CUDA_MPS_CLIENT_PRIORITY`
(CUDA 11.5+) to bump emotion; or raise emotion2vec's batch so it claims more SMs per
launch (model-runtime change, and emotion is memory-bound on long audio — uncertain).

### Fallback — give emotion its own smaller dedicated GPU (e.g. A10G)

Viable and clean, but it's the fallback, not the first move:

- **Pro:** emotion fully decongested → back to ~dedicated speed; the two remaining tasks
  are *both* WavLM with near-identical kernel profiles, so they co-locate cleanly (1.8–2.0×
  mutual inflation, no odd-man-out) and MPS keeps its gain on the Blackwell.
- **Con:** 2 GPUs + 2 node pools + cross-GPU orchestration; emotion is the *fastest/cheapest*
  model, so a whole A10G running only emotion is under-utilized (you'd want N emotion
  replicas or to give the A10G more work). Breaks the one-GPU-per-pod simplicity.
- **Verdict:** only split if (A)+(B) can't push emotion below the wall-setting threshold.
  Even then, prefer "affect+disfluency co-located on the Blackwell, emotion fans out to a
  cheap-GPU pool" over scattering all three.

### Re-evaluate whether ungated MPS is worth it at all

Ungated MPS is only **1.10×** over serial — the WavLM batches already saturate the SMs,
leaving almost no gap for co-location to fill. With VAD-gating now the production path
(gating 1.5–1.8×), this matters less, but if any ungated work remains, dedicated/serial
is nearly as good there and simpler.

---

## 7. Reproduce

```
# per-task inference_sec (isolates emotion; the table above):
python - <<'PY'  # see analysis; group raw _meta/timings/*.jsonl by task_group, stat inference_sec
PY
# SM-clock-under-load throttle test: parse run/_bench_results/smi_*.log,
#   condition pclk on sm>50, report mean/min and % of loaded samples < 0.95*max.
# raw logs: .../scratch/mps-bench/{puregpu,e2e}/{job-logs,run/_bench_results}/
```

Related: [MPS_COLOCATION_TESTING.md](../MPS_COLOCATION_TESTING.md) (validation plan),
[INFERENCE_OPTIMIZATION_PLAN.md](INFERENCE_OPTIMIZATION_PLAN.md),
[IO_AND_PROCESSING_OPTIMIZATION.md](IO_AND_PROCESSING_OPTIMIZATION.md).
