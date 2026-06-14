# Model output schemas (inference artifacts)

How the inference stage persists model predictions, what the arrays mean, and at
what resolution they are indexed. This is the contract every downstream stage
(`composition/`, `producers/`, `event_packages/`) reads against — it does **not**
depend on whether a run was VAD-gated.

Source of truth: [artifacts.py](artifacts.py) (save/load), [runners.py](runners.py)
(what each task writes), [vad_gating.py](vad_gating.py) (gated fills).

## 1. On-disk layout

Each `(recording, audio, task, config)` produces one artifact **directory** with
two files:

```
<task>/
  predictions.npz   # the numerical arrays
  manifest.json     # metadata: task, audio hash, config, config hash, model, timing
```

Directory placement differs by caller (the *files inside* are identical):

- **Standalone inference API:** `<recording_id>/<audio_sha256>/<task>/<config_hash>/`
  — config hash in the path, so multiple configs coexist.
- **Orchestration fleet:** flat `<output>/<session_id>/<archive_id>/<task>/`
  — **no** config hash in the path; one artifact per `(archive, task)` slot,
  selected by `--completion-policy`.

## 2. `predictions.npz` — a dict map of full-length dense arrays

It is **not** a list-of-lists and **not** sparse. It is a standard NumPy `.npz`
(`np.savez_compressed`, [artifacts.py:154](artifacts.py#L154)) — a container of
**named** arrays. Loading returns `{name: ndarray}`
([load_prediction_artifact](artifacts.py#L106)).

Every array is a dense, full-timeline array whose **first axis is the frame
index** `i` (see §3 for the time grid). Trailing axes are the per-frame output.

| task | array keys | shape | dtype | meaning |
|---|---|---|---|---|
| `affect` | `arousal`, `valence`, `dominance` | `[N]` each | f32 | continuous affect dimensions, one scalar per frame |
| `disfluency` | `fluency_logits` | `[N, 2]` | f32 | raw logits (fluent / disfluent) |
| | `disfluency_type_logits` | `[N, 5]` | f32 | raw logits over the 5 disfluency types |
| `emotion` | `probabilities` | `[N, 8]` | f32 | normalized probabilities (rows sum to 1); column order = manifest `labels` |
| `vad` | `intervals_sec` | `[n, 2]` | f32 | **not framed** — `[start_sec, end_sec]` speech segments (see §4) |

Stored values are the raw model space: affect = continuous scalars, disfluency =
**logits** (producers apply sigmoid/softmax), emotion = already-normalized
**probabilities**.

## 3. Frame resolution — the time grid

**The first axis is not audio samples.** 16 kHz is the sample rate the models
*consume* (`SAMPLE_RATE = 16_000`, [artifacts.py:16](artifacts.py#L16)); it is not
the prediction resolution. Predictions are emitted on a coarse sliding-window grid:

- **Stride between rows = `DEFAULT_HOP_SEC = 0.25 s` → 4 Hz, identical for all
  three GPU tasks** ([runners.py:75](runners.py#L75)). One row = one model inference;
  consecutive rows step 0.25 s (= 4000 audio samples).
- **Window length (context each row integrates) differs per task:**

  | task | hop (resolution) | window (context) | frame rate |
  |---|---|---|---|
  | affect | 0.25 s | `AFFECT_WINDOW_SEC = 3.5 s` | 4 Hz |
  | disfluency | 0.25 s | `DISFLUENCY_WINDOW_SEC = 3.0 s` | 4 Hz |
  | emotion | 0.25 s | `EMOTION_WINDOW_SEC = 3.0 s` | 4 Hz |

  ([runners.py:72-75](runners.py#L72-L75))

### Index ↔ time

Row `i` summarizes the audio span `[i·hop, i·hop + window]` (manifest
`timing.window_semantics = "frame summarizes [i*hop, i*hop + window]"`,
[runners.py:1278](runners.py#L1278)):

```
start_sec(i)  = i · 0.25
center_sec(i) = i · 0.25 + window_sec / 2
```

(`_frame_centers`, [producers/disfluency/pipeline.py:606](../producers/disfluency/pipeline.py#L606)).

### Two consequences

1. **`N` is not identical across tasks for the same audio.** `N ≈ floor((duration −
   window) / 0.25) + 1`, so a longer window → fewer rows. Example from a 40-min
   stem: affect `N = 9739` vs disfluency/emotion `N = 9741` — affect's 0.5 s-larger
   window is exactly `0.5 / 0.25 = 2` fewer hops.
2. **Do not align tasks by row index.** At the same `i`, affect and disfluency have
   different center times (offset by `(3.5 − 3.0)/2 = 0.25 s`), because the center
   depends on `window_sec`. Align by **time**: each task carries its own `window_sec`
   in `timing`, and producers convert index→seconds themselves. Composition already
   does this.

## 4. VAD is the exception (interval-based, not framed)

The `vad` artifact stores continuous Silero speech segments, **not** a 0.25 s grid
([runners.py:629](runners.py#L629)):

- `intervals_sec`: `[n, 2]` float32, each row `[start_sec, end_sec]`.
- `timing`: `window_sec = 0.0`, `hop_sec = 0.0`, `window_semantics =
  "sparse_intervals_sec"`, `n_frames = number of intervals`.

The gate and the producers project these intervals onto each task's own 0.25 s
frame grid (using that task's `window_sec`), which is why one VAD pass serves all
three GPU tasks.

## 5. `manifest.json` schema

Built in [artifacts.py:255](artifacts.py#L255); `schema = "acoustic_predictions.v1"`,
written last so a half-written artifact is never `status: "complete"`.

| field | meaning |
|---|---|
| `schema` | `"acoustic_predictions.v1"` |
| `status` | `"complete"` (set only after the npz is written + validated) |
| `task` | `affect` / `disfluency` / `emotion` / `vad` |
| `recording_id` | recording identity (fleet: `session_id`/`archive_id`) |
| `audio` | `{path, sha256, sample_rate, duration_sec, hash_semantics: "decoded_mono_16khz_float32"[, source_key]}` |
| `inference_config` | the config that produced these arrays (includes the VAD-gating descriptor when gated — see §6) |
| `inference_config_hash` | 16-hex hash of `inference_config` → artifact lineage |
| `model` | `{family, id, …}` |
| `timing` | `{sample_rate, window_sec, hop_sec, n_frames, window_semantics}` |
| `runtime` | device / batch-size / dtype knobs |
| `arrays` | per-array `{shape, dtype}` (mirrors the npz; lets a reader introspect without loading) |
| `labels` | emotion only: ordered class names for the `probabilities` columns |
| `created_at` | UTC ISO timestamp |

`audio.sha256` is computed from **decoded mono 16 kHz float32 samples**
([decoded_audio_sha256](artifacts.py#L51)) — independent of model, config, or gating.

## 6. VAD-gated vs un-gated — identical schema

VAD gating skips GPU compute on non-speech windows, then scatters the computed rows
back into a **full-length** array and fills the rest ([gate_window_arrays](vad_gating.py#L160),
[scatter_fill](vad_gating.py#L142)). So a gated artifact is **byte-schema-identical**
to an un-gated one — same keys, same `N`, same dtypes. Only two things change:

1. **Values at non-speech rows are filled** (never computed): `0.0` for affect and
   disfluency; a uniform row for emotion (raw score filled `1.0` before softmax, so
   the row stays a valid sum-to-1 distribution).
2. **`inference_config_hash` differs**, because `inference_config.extra.vad_gating =
   {policy: "overlap_v1", bridge_sec: 1.5}` is recorded → gated artifacts get a
   distinct lineage.

Verified on real artifacts (≈41 % non-speech stem): gated vs full had the **same
shape** on every array; only the non-speech rows differed (e.g. affect `arousal`:
4005/9739 rows). The differing-row count tracks the silence fraction exactly.

### Why downstream reads both uniformly

- **One reader, no branching.** `load_prediction_artifact` returns the same
  `{name: full_array}` for gated and un-gated; producers/composition consume them
  identically. Nothing downstream inspects the gating flag — the producers' speech
  mask (derived from the `vad` artifact) is what ignores non-speech frames, which is
  *why* gated output equals full output at the event level.
- **Mixed trees are fine.** Composition only requires the task artifacts to share the
  same `audio.sha256` (gating does not change it), **not** the same
  `inference_config_hash`. So an archive with gated-affect + un-gated-disfluency
  composes cleanly, and because gated ≡ full at the event level, the emitted events
  are the same either way.

## See also

- [README.md](README.md) — running the inference stage.
- [../PRODUCER_CONTRACT.md](../PRODUCER_CONTRACT.md) — how producers turn these arrays into events.
- [../../../optimization_research/VAD_GATING_IMPLEMENTATION_PLAN.md](../../../optimization_research/VAD_GATING_IMPLEMENTATION_PLAN.md) — the gating design, fills, and A/B evidence.
