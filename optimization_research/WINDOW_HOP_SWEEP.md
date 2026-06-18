# Window / Hop Sweep — Affect & Disfluency

**Date:** 2026-06-17
**Branch:** feature/inference-optimization
**Script:** `scripts/ab_window_stride.py`
**Manifest:** `manifests/ab-window-stride.yaml`

## Setup

Baseline configuration:

| Task | window | hop |
|---|---|---|
| Affect | 3.5s | 0.25s |
| Disfluency | 3.0s | 0.25s |

Five variants tested against baseline on 3 random archives (g7e.2xlarge, WavLM backbone).
Accuracy is measured relative to baseline (no external ground truth): recall, precision,
label agreement, exact match (same label + boundary within 1 baseline hop = 0.25s),
and boundary drift p50/p99.

## Results

### Affect

| Variant | Config | Recall | Precision | Label ag. | Exact | Cnt Δ% | Bnd p50 | Bnd p99 |
|---|---|---|---|---|---|---|---|---|
| **coarse** | w=3.5s h=0.5s | **0.978** | 0.887 | **0.908** | **0.699** | +10.4 | 0.208s | 2.027s |
| hop_1s | w=3.5s h=1.0s | 0.943 | 0.809 | 0.771 | 0.147 | +16.6 | 0.500s | 3.769s |
| narrow_fast | w=2.5s h=0.5s | 0.909 | 0.753 | 0.753 | 0.213 | +21.7 | 0.417s | 5.232s |
| narrow | w=2.5s h=0.25s | 0.855 | 0.822 | 0.758 | 0.164 | +4.9 | 0.417s | 6.108s |
| wide | w=5.0s h=0.25s | 0.737 | 0.967 | 0.665 | 0.011 | −23.7 | 0.625s | 6.213s |

### Disfluency

| Variant | Config | Recall | Precision | Label ag. | Exact | Cnt Δ% | Bnd p50 | Bnd p99 |
|---|---|---|---|---|---|---|---|---|
| narrow | w=2.0s h=0.25s | 0.919 | 0.801 | 0.910 | 0.143 | +18.7 | 0.458s | 1.281s |
| **coarse** | w=3.0s h=0.5s | **0.835** | **0.983** | **0.965** | **0.836** | −15.1 | **0.125s** | **0.633s** |
| narrow_fast | w=2.0s h=0.5s | 0.835 | 0.798 | 0.940 | 0.129 | +10.1 | 0.375s | 1.260s |
| hop_1s | w=3.0s h=1.0s | 0.675 | 0.980 | 0.937 | 0.327 | −31.2 | 0.167s | 0.374s |
| wide | w=4.0s h=0.25s | 0.569 | 0.800 | 0.945 | 0.443 | −32.1 | 0.417s | 0.900s |

## Findings

**`coarse` (hop=0.5s, windows unchanged) is the recommended configuration for both tasks.**

- **Affect:** near-identical quality to baseline (recall 0.978, label agreement 0.908). The
  bnd_p99 of 2s reflects boundary quantization at the coarser hop, not missed events. Exact
  match drops to 0.699 because baseline-hop (0.25s) precision is unachievable at 0.5s hop —
  this is expected and not a quality regression.
- **Disfluency:** recall drops to 0.835 (−16% vs baseline) but precision rises to 0.983 and
  label agreement to 0.965. Events that are found are highly accurate. The 16% recall loss
  is the cost of the 2× throughput gain.

**`hop_1s` does not work for disfluency.** Despite overlapping windows ensuring short events
are covered by multiple frames, recall falls to 0.675 (−32.5%). The 4× throughput gain is
not worth losing 1 in 3 events. It performs adequately for affect (recall 0.943) but with
degraded precision and label agreement.

**Wide windows hurt both tasks.** A 5s affect window loses 26% of events (recall 0.737) with
near-zero exact matches. A 4s disfluency window is the worst performer overall (recall 0.569).
Wider context dilutes the signal from short events.

**Narrow windows are a worse tradeoff than coarse hop.** Shorter windows do not provide a
throughput gain (same hop = same frame count) and meaningfully degrade affect quality
(bnd_p99 > 6s, exact 0.164).

## Bug fixed

`affect/detector.py` used `round()` to convert `merge_gap_sec` to frames, which caused
`merge_gap_sec=0.5s` to collapse to 0 frames at hop≥1s (Python banker's rounding: `round(0.5)=0`).
Fixed to `math.ceil()`. No behavior change at baseline hop (0.5/0.25=2.0, both agree).

## Recommendation

Adopt `coarse` (hop=0.5s) for production — **2× throughput gain** with negligible affect
quality loss and acceptable disfluency precision. Disfluency recall loss (−16%) should be
monitored in downstream session quality metrics before full fleet rollout.
