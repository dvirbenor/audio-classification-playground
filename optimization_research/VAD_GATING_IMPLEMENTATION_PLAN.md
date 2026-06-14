# VAD-gated inference — implementation plan

Goal: skip GPU inference on non-speech windows (≈55% of compute on these per-speaker stems; see
`IO_AND_PROCESSING_OPTIMIZATION.md` §6) **while keeping emitted events bit-identical**. Stacks with the
fp16 lever (~1.85×) for ~4× combined per-archive.

## Status: IMPLEMENTED (default off) — pending real-archive event-A/B before fleet enable

Landed on `feature/inference-optimization`:
- `inference/vad_gating.py` — `VadGating`, `bridge_intervals`, `speech_window_mask` (overlap+bridge),
  `scatter_fill`, `gate_window_arrays`, `intervals_from_array`.
- Runners (`runners.py`): affect/disfluency gate via `gate_window_arrays` (fill 0.0); emotion gates in
  `run_emotion_inference` via the **framed-windows path + raw-score scatter (fill 1.0)** — numerically
  identical to the `predict_audio` stride path on the same windows, so **no surgery into the compiled
  CUDA-graph scorer** (deviation from §3 below; lower risk). `run_all_inference` resolves intervals
  (explicit arg → vad step → on-disk vad artifact) and threads them + `VadGating` to the GPU tasks.
  `_apply_gating` tags `inference_config` only when gating is active *and* intervals are present.
- Worker/CLI: `run_worker(vad_gating_enabled, vad_gating_bridge_sec)`, reuses `pf_result.vad_intervals`,
  mirrors the tag in `build_expected_configs`; `… run … --vad-gating [--vad-gating-bridge-sec S]`.
- Tests: `test_vad_gating.py` (overlap/bridge/**superset vs affect.assign_frame_blocks over 200 random
  trials**/scatter) and `test_vad_gating_runners.py` (gated==full at kept frames, fill elsewhere,
  full-length output, fewer windows computed, distinct config hash, end-to-end intervals-from-vad-step).
  Full suite: 266 passed / 1 pre-existing skip.

### Real-archive A/B + speed — MEASURED (3 archives, L40S, eager fp32)
Harness `scripts/vad_gating_ab.py`; data `baseline_results/vad_gating_ab.json`. Runs the real ModelSuite
through `run_all_inference` full vs gated, composes events via the real composition stage with the real
Silero intervals, diffs events, and times per task.

**Event-identity (full vs gated):**
- **affect: bit-identical on all 3** (0 dropped/added, label agreement 1.000, 0 boundary/score drift).
- **emotion: bit-identical on all 3** (same).
- **disfluency: NOT identical** — 1 dropped event on 2 of 3 archives (+1 label flip, boundary drift up to
  1.75 s). Root cause confirmed: `_support_regions` builds candidate regions over the **full timeline
  before** the speech-support filter ([disfluency/pipeline.py:284](../audio_classification_playground/acoustic_events/producers/disfluency/pipeline.py#L284)),
  so filled non-speech frames shift region boundaries. → **disfluency excluded from the default gated set
  at the time of this A/B.** (Resolved 2026-06-14 by making region detection speech-scoped — see the
  follow-up in "Decision + remaining" below; re-A/B pending.)

**Speed (GPU compute/archive, speech 21–39%):**
| config | arch1 | arch2 | arch3 | mean |
|---|---|---|---|---|
| all-three gated | 2.37× | 1.65× | 2.79× | **2.27×** |
| **safe: affect+emotion gated (disfluency full)** | 1.50× | 1.28× | 1.37× | **~1.38×** |

Per-task gated speedups were large and uniform (affect 1.6–2.8×, disfluency 1.7–2.9×, emotion 1.6–2.6×),
scaling with silence fraction — confirming the lever.

### Decision + remaining
- **Shipped default = gate affect + emotion** (event-identical, ~1.4× GPU). `VadGating.tasks` defaults to
  `("affect","emotion")`; `--vad-gating` enables it; gating disfluency is an explicit opt-in
  (`--vad-gating-tasks affect disfluency emotion`).
- **Follow-up to recover disfluency's ~2.5× (→ ~2.3× overall): IMPLEMENTED (2026-06-14).** Disfluency
  region detection is now speech-scoped — `_detect` builds candidate regions over `np.where(speech_mask,
  p_disfluent, 0.0)` before `_support_regions` ([disfluency/pipeline.py:283](../audio_classification_playground/acoustic_events/producers/disfluency/pipeline.py#L283)),
  so non-speech frames can neither seed a region nor shoulder-expand a boundary into silence. Two effects:
  1. **Full-timeline output changes** (region boundaries trim to speech; pure-leak regions shrink/drop) —
     a deliberate correctness improvement (a disfluency is a speech phenomenon), so it needs a re-baseline
     + new artifact lineage, *not* a bit-identical guarantee against the old producer.
  2. **gated == full is restored** by construction: surviving region interiors only span bridged gaps
     `<= merge_gap_sec (0.5 s) <= bridge_sec (1.5 s)`, so every frame a region reads is one inference
     *computes* (never filled). Aggregation still reads the real `p_disfluent` at bridged frames.
  Unit tests updated/added in `test_disfluency_producer.py` (boundary-trims-to-speech, non-speech peak no
  longer supports, short non-speech gap still bridges); disfluency suite 23 passed.
  **Re-A/B MEASURED (2026-06-14, A10G, 3 archives, `vad_gating_ab_speechscoped.json`):** gated==full is now
  **bit-identical for disfluency on all 3** (0 dropped/added, label 1.000, 0 boundary/score drift) —
  alongside affect and emotion. Disfluency gated speedup 2.53× / 1.74× / 2.98× (scales with silence);
  GPU-total all-three-gated 2.45× / 1.69× / 2.88×, **mean 2.34×** at mean 28% speech. `event_safe: true`.
  → **disfluency added to `DEFAULT_GATED_TASKS`** (`vad_gating.py`); `--vad-gating` now gates all three GPU
  tasks. Manifest `acoustic-events-inference-cache-workers-optimized.yaml` gates the disfluency fleet too.
  **Still pending (product, not correctness):** sign-off on the full-timeline boundary change vs the *old*
  producer (speech-tight boundaries; a separate diff from this gated-vs-full A/B), and a corpus-wide
  re-baseline if the existing un-gated disfluency artifacts should be recomputed under the new producer.
- Still pending before fleet enable: a **manifest-wide speech-% sample** (3 archives = 21–39%, mean 28%),
  then enable per-fleet for affect + emotion.

## 0. The guarantee (what "compute remains identical" means precisely)

- **Events are bit-identical.** Not the stored arrays — the *outputs the producers emit*.
- Two facts make this provable, not hopeful:
  1. **A window we compute is unchanged.** Gating *skips* windows; it never alters the audio fed to a
     window we run. So every kept frame's value == full-timeline value (verify with a bit-identity test).
  2. **Producers already ignore non-speech frames** (affect `global_stats`/baseline over `interior`
     block frames only; disfluency requires speech overlap; emotion thresholds over `valid_mask`). So
     the filled non-speech frames are never read.
- Therefore: if the **gate keep-set ⊇ every frame any producer reads**, the events cannot change. The
  stored prediction arrays differ only at never-read non-speech frames → handled by a config-hash bump
  (new artifact lineage), exactly like `autocast_dtype`.

## 1. The safe gate (superset rule)

Helper (new), pure/Numpy, in `inference/audio.py` or a new `inference/vad_gating.py`:

```
def speech_window_mask(n_frames, hop_sec, window_sec, vad_intervals, *, bridge_sec) -> np.ndarray[bool]:
    # 1. bridge: merge intervals whose gap <= bridge_sec  (>= emotion support_close_gap_sec = 1.0)
    # 2. keep window i (spans [i*hop, i*hop+window_sec]) iff it OVERLAPS any bridged interval
    #    overlap, NOT containment — containment would drop boundary-straddling windows producers read
```

- `bridge_sec` default **1.5 s** (≥ max producer gap: emotion `support_close_gap_sec=1.0`, affect/disf
  `merge_gap_sec=0.5`) + margin. Single conservative constant keeps inference decoupled from producer config.
- Per-task masks differ only by `window_sec` (affect 3.5 / disf 3.0 / emo 3.0, hop 0.25); all derive from
  the **same** Silero intervals (the ones the producers consume) + the same `bridge_sec`.
- Empty intervals → all-False mask → skip GPU entirely (free win on silent stems).

**Superset proof obligation (unit test):** for random intervals, assert
`affect.assign_frame_blocks(...) >= 0` ⊆ `speech_window_mask(...)`, and analogously the disf/emo
overlap masks ⊆ the gate. This is the correctness keystone.

## 2. Thread VAD intervals + gating config into the GPU runners

Currently `run_all_inference` ([runners.py:553](../audio_classification_playground/acoustic_events/inference/runners.py#L553))
runs `vad` first but does **not** pass the intervals to affect/disfluency/emotion. Changes:

- `run_all_inference`: capture the VAD intervals (from the `vad` step's artifact, or from the injected
  precomputed `vad_detector`/`pf_result.vad_intervals` in the worker) and pass them + a `VadGating`
  settings object into each `run_*_inference`.
- `run_affect_inference` / `run_disfluency_inference` / `run_emotion_inference`: add params
  `vad_intervals: list[tuple[float,float]] | None` and `vad_gating: VadGating | None`.
- Worker ([worker.py:879](../audio_classification_playground/acoustic_events/orchestration/worker.py#L879)):
  reuse `pf_result.vad_intervals` when precomputed (avoid recompute); else they come from the in-run `vad`
  step. Pass `vad_gating` through. The `vad` task itself is never gated.

`VadGating` = `{enabled: bool, bridge_sec: float, policy: "overlap_v1"}`.

## 3. Per-task masked compute + scatter + fill

Do the gating in the **runner** (it knows audio/hop/window/intervals); keep predictors as dumb
"windows→scores" callables where possible.

**affect / disfluency** (predictors take an explicit `windows` array):
```
windows = frame_audio(...)                      # [N, win]
mask    = speech_window_mask(N, hop, window_sec, intervals, bridge_sec)  # if gating on
kept    = np.where(mask)[0]
out_sub = predictor(windows[kept])              # runs only K<=N windows (compiled_static pads K)
# scatter into full-length [N] / [N,c] arrays, fill non-kept:
arousal = np.zeros(N, f32); arousal[kept] = out_sub["arousal"]; ...   # affect fill 0.0
fluency = np.zeros((N,2), f32); fluency[kept] = ...                    # disfluency fill 0.0
```
Fill is 0.0 (finite; passes `_validate_affect/disfluency_arrays`; producers never read it).
`n_frames = N` unchanged → all downstream frame/centre/hop alignment intact.

**emotion** (production path is `DirectEmotion2vecScorer.predict_audio`, on-GPU striding):
- Add an optional `keep_mask` to `predict_audio`/`predict_windows`
  ([emotion2vec.py:257](../audio_classification_playground/acoustic_events/inference/emotion2vec.py#L257)):
  build the strided windows as today, compute only `kept` rows, scatter into the full `[N, C]` output.
- Fill non-kept **raw-score** rows with a uniform positive vector (ones) *before*
  `emotion2vec_scores_to_probabilities` so the existing fold+normalize+sum-to-1 validation passes (uniform
  → uniform distribution). Keeps the conversion/validation path untouched.
- This preserves the exact production `predict_audio` numerics for kept frames (don't switch to the framed
  path — it's a different, slower code path).

WavLM `compiled_static`: `pad_windows_to_static_batch` runs *inside* the predictor on the kept subset, so
it pads `K`→batch-multiple and trims via `valid_count` as today. Compatible; fewer batches = the saving.

## 4. Immutability / config hash

Add the gating descriptor to `inference_config` in `_inference_config`
([runners.py:970](../audio_classification_playground/acoustic_events/inference/runners.py#L970)) when
enabled, e.g. `extra["vad_gating"] = {"policy":"overlap_v1","bridge_sec":1.5}`. This flows into
`inference_config_hash` → gated artifacts get a new lineage; existing artifacts untouched until the flag
flips. Mirror in `build_expected_configs`/`expected_hashes`
([worker.py:208](../audio_classification_playground/acoustic_events/orchestration/worker.py#L208)) so
stale-detection stays consistent. Default **off**.

## 5. Verification (the acceptance gate — same discipline as fp16)

1. **Unit — superset:** `producer-read frames ⊆ gate keep-set` for random intervals (per task). Keystone.
2. **Unit — kept-frame bit-identity:** gated vs full-timeline arrays equal at `kept` indices (proves
   gating doesn't perturb computed windows).
3. **Unit — fill validity:** emotion filled rows pass sum-to-1; affect/disf filled arrays pass validators.
4. **Integration — event-level A/B** (`scripts/event_level_ab.py`): full-timeline vs gated on ≥3 real
   archives → **require 0 events added/dropped, 0 label flips, 0 boundary drift, 0 score drift.** Non-zero
   ⇒ widen `bridge_sec`. This is the ship gate.
5. **Prerequisite check:** confirm disfluency detection thresholds are **not** computed over non-speech
   frames (affect/emotion already confirmed). If they are, compute them over speech frames or fill with
   neutral "fluent" logits before that step.
6. **Producer-contract tests** (`PRODUCER_CONTRACT.md`): unchanged — events/IDs/scores identical.

## 6. Rollout

- Land behind `vad_gating.enabled=False` (no behavior change; CI green).
- Enable on a validation sample; run the §5.4 A/B; eyeball diagnostics.
- Enable **per fleet** (affect/disfluency/emotion are separate task-fleets) — disfluency/emotion are
  robust classifiers, affect (continuous A/V/D) is the sensitive one, so stage affect last.
- Quantify-first: run `quantify_vad_gating.py` over a **manifest-wide sample** to confirm the ~28% speech
  / ~55% saving holds corpus-wide before committing fleet hours.

## 7. Risks & mitigations

| risk | mitigation |
|---|---|
| `bridge_sec` too tight → a producer reads a filled frame | superset unit test + event-A/B zero-drift gate; default 1.5 s (>1.0 binding) |
| emotion fill breaks sum-to-1 validation | fill raw scores with uniform-positive *before* conversion |
| disfluency global threshold spans non-speech | §5.5 check; compute over speech frames if so |
| gated artifacts collide with full-timeline cache | config-hash bump (§4) → separate lineage |
| low-speech assumption wrong for some content | manifest-wide quantify before fleet enable; gating is per-fleet revertible |

## 8. Effort & sequencing

1. `speech_window_mask` + superset/identity unit tests — small, do first (locks the contract).
2. Thread intervals + `VadGating` through `run_all_inference`/runners + worker — medium plumbing.
3. affect + disfluency masked scatter/fill — small (explicit windows array).
4. emotion `predict_audio` keep_mask — medium (on-GPU stride scatter).
5. config-hash flag + expected-config mirror — small.
6. event-A/B run + per-fleet rollout — validation.

Net: **medium effort, low-to-moderate risk.** The producers already assuming speech-gating + the superset
gate + the event-A/B backstop make this a "stop computing what's discarded" change, not a semantic rewrite.
Expected payoff ~2× GPU (on top of fp16), the largest remaining lever short of new hardware or a model change.
