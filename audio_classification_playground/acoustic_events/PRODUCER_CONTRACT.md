# Acoustic Event Producer Contract

This document is the implementation contract for every acoustic event
producer that wants to feed the review app. It is intentionally broader than
affect: the same session can contain affect deviations, categorical emotions,
disfluencies, non-verbal vocalizations, emphasis, or future model outputs.

The canonical package is now:

```text
audio_classification_playground/acoustic_events/
  schema.py
  review/
  producers/
    affect/
```

`audio_classification_playground.affective_events` remains only as a
compatibility facade for old notebooks and imports. New code should not add
modules or producers under `affective_events`.

## Design Rule

Producer logic is task-specific. Review storage, labels, schema validation,
track rendering, and inheritance are generic.

A producer owns:

- model loading and inference;
- task-specific event extraction;
- task-specific configuration;
- task-specific diagnostics inside `ProducerRun.outputs` or `Event.extra`.

The generic system owns:

- `ProducerRun`, `Event`, `RegularGridTrack`, `MarkerTrack`;
- session save/load;
- label inheritance;
- review API and UI;
- waveform rendering and audio playback.

## Producer Layout

Add each new producer under:

```text
audio_classification_playground/acoustic_events/producers/<task_or_family>/
  __init__.py
  config.py
  pipeline.py
  types.py          # optional, only for producer-local inputs
  detector.py       # optional
  preprocessing.py  # optional
```

Examples:

```text
producers/affect/
producers/emotion/
producers/disfluency/
producers/vocalization/
producers/emphasis/
```

Use `schema.py` for review-facing objects. Do not create parallel event
dataclasses unless they are private intermediate candidates that never leave
the producer.

## Producer Output

Every producer should expose a function that returns:

```python
producer_run: ProducerRun
tracks: list[RegularGridTrack | MarkerTrack]
events: list[Event]
```

The caller composes one or more producers into a session:

```python
from audio_classification_playground.acoustic_events.review.storage import save_session

session_path = save_session(
    events=affect_events + emotion_events,
    tracks=affect_tracks + emotion_tracks,
    producer_runs=[affect_run, emotion_run],
    vad=vad,
    audio_path=audio_path,
    session_dir=session_dir,
)
```

Sessions are immutable. Rerunning one producer means creating a new combined
session from retained producer outputs plus the newly produced outputs.
The caller is responsible for composition in v1.

## ProducerRun

`ProducerRun` describes one extractor/model run.

```python
ProducerRun(
    producer_id="disfluency.v1",
    task="disfluency",
    source_model="wavlm-disfluency-2026-05",
    config={...},
    config_hash="...",
    outputs={...},
)
```

Rules:

- `producer_id` is stable and unique within a session.
- `task` is the broad review lens: `affect`, `emotion`, `disfluency`, etc.
- `source_model` identifies the model or extractor family.
- `config` should contain enough information to reproduce the run.
- `config_hash` is deterministic for the producer config.
- `outputs` stores producer-scoped artifacts that are not events or tracks.

Example: affect VAD-derived analysis blocks live in
`ProducerRun.outputs["blocks"]`, not in the top-level session.

## Prediction Tracks

Tracks are the model evidence shown next to the waveform. Events connect to
their evidence through `source_track_ids`.

### RegularGridTrack

Use `RegularGridTrack` when output is sampled on a regular hop grid.

```python
RegularGridTrack(
    track_id="emotion.anger",
    producer_id="emotion.v1",
    task="emotion",
    name="anger",
    value_type="probability",
    renderer="probability",
    values=anger_probability,  # shape: frames
    hop_sec=0.25,
    window_sec=1.0,
)
```

Supported grid renderers:

- `line`: continuous values such as arousal, valence, dominance.
- `probability`: one probability curve.
- `multi_probability`: 2-D `frames x channels` array.

`multi_probability` must provide channel names:

```python
RegularGridTrack(
    track_id="disfluency.type",
    producer_id="disfluency.v1",
    task="disfluency",
    name="disfluency type",
    value_type="probability",
    renderer="multi_probability",
    values=type_probs,  # shape: frames x classes
    hop_sec=0.25,
    window_sec=1.0,
    channels=("filled_pause", "repetition", "restart"),
)
```

Disfluency can be grid-based or marker-based. The schema does not force a
choice. A framewise model can emit regular-grid tracks; a sparse detector can
emit marker tracks and events with no regular source track.

### MarkerTrack

Use `MarkerTrack` when evidence is sparse and not on a hop grid.

```python
MarkerTrack(
    track_id="vocalization.markers",
    producer_id="vocalization.v1",
    task="vocalization",
    name="non-verbal vocalizations",
    renderer="marker",
    items=(
        MarkerItem(
            start_sec=42.1,
            end_sec=42.8,
            label="laugh",
            score=0.91,
            payload={"raw_class": "laughter"},
        ),
    ),
)
```

Minimum marker item contract:

```python
{
    "start_sec": float,
    "end_sec": float | None,
    "label": str,
    "score": float | None,
    "payload": dict,
}
```

Keep marker payloads small. If marker outputs become large, add a storage
extension rather than hiding large blobs in JSON.

## Event

All reviewable spans are `Event` objects.

```python
Event(
    event_id="emotion.v1.categorical.000001",
    producer_id="emotion.v1",
    task="emotion",
    event_type="categorical",
    label="anger",
    start_sec=12.0,
    end_sec=15.0,
    duration_sec=3.0,
    source_track_ids=("emotion.anger",),
    score=0.87,
    score_name="probability",
    direction=None,
    parent_id=None,
    children=(),
    evidence={"top_classes": {"anger": 0.87, "neutral": 0.08}},
    extra={"threshold": 0.7},
)
```

Rules:

- `event_id` format is `{producer_id}.{event_type}.{NNNNNN}`.
- Event IDs are unique within a session.
- `producer_id` must match one `ProducerRun`.
- `task` should match the producer task.
- `source_track_ids` should reference emitted tracks when tracks explain the
  event.
- `source_track_ids=()` is valid for marker-only or externally supplied spans.
- `duration_sec` should equal `end_sec - start_sec`.

Allowed `score_name` values are defined in `schema.SCORE_NAMES`:

- `peak_z`
- `probability`
- `confidence`
- `prominence_z`
- `logit`
- `margin`

Add a new score name to `SCORE_NAMES` before using it.

## Evidence vs Extra

Use `evidence` for reviewer-facing explanation: values that help a human
decide whether the event is real.

Use `extra` for diagnostics, thresholds, internal frame indices, intermediate
candidate state, or reproducibility metadata.

Do not duplicate the top-level score in `evidence`. If the score is `peak_z`,
then `score=...` and `score_name="peak_z"` are the canonical score fields.
Related explanatory values like `signed_z`, `baseline_at_peak`, and `delta`
can live in `evidence`.

## Event Creation Checklist

For each producer:

1. Choose a stable `producer_id`, for example `emotion.v1`.
2. Choose a broad `task`, for example `emotion`.
3. Build one `ProducerRun`.
4. Emit zero or more tracks.
5. Extract events from the producer output.
6. Assign monotonic event IDs with `{producer_id}.{event_type}.{NNNNNN}`.
7. Fill common event fields first: time span, label, score, score name,
   direction if meaningful, source tracks.
8. Put reviewer-facing explanation in `evidence`.
9. Put diagnostics in `extra`.
10. Save through `review.storage.save_session`.

The review app does not create detection events. It only displays, filters,
and labels upstream outputs.

## Affect Producer Mapping

The affect producer lives at:

```python
audio_classification_playground.acoustic_events.producers.affect
```

Its public convenience API is also re-exported from:

```python
audio_classification_playground.acoustic_events
```

Affect leaf event:

- `producer_id="affect.default"` unless explicitly overridden.
- `task="affect"`.
- `event_type="deviation"`.
- `label="{axis}_deviation"`.
- `source_track_ids=("affect.{axis}",)`.
- `score=peak_z`.
- `score_name="peak_z"`.
- `direction="+"` or `"-"`.
- `evidence` contains `signed_z`, `peak_time_sec`, `baseline_at_peak`,
  `scale_at_peak`, `delta`, and `block_ids`.
- `extra` contains frame indices, seed bounds, shoulder z values, and merge
  diagnostics.

Affect joint event:

- `task="affect"`.
- `event_type="joint"`.
- `label="joint"`.
- `source_track_ids` is the union of child source tracks.
- `children` is the structural truth.
- the display signature, such as `A+ V- D0`, is derived from children and
  evidence, not stored as the canonical label.

## Review App Contract

The review app consumes sessions written by:

```python
audio_classification_playground.acoustic_events.review.storage.save_session
```

Session JSON contains:

- `event_schema="acoustic_events.v1"`;
- `session_id`;
- `producer_runs`;
- `tracks_meta`;
- `tracks_data_path`;
- `events`;
- `labels`;
- audio metadata;
- top-level `vad_intervals`;
- no top-level `blocks`.

The API exposes `/api/tracks`. `/api/signals` is retired.

Default UI behavior:

- if an event is selected, show waveform plus tracks with `track.task ==
  event.task`;
- highlight `event.source_track_ids`;
- if no event is selected, show tracks for the selected task;
- if no task is selected, show the first task that has tracks;
- render `evidence` in the event header;
- keep `extra` hidden unless a debug/details affordance is opened.

## Label Inheritance

Regular events match by:

- `task`;
- `label`;
- `source_track_ids`;
- temporal overlap.

`source_track_ids=()` is a valid match key for marker-only events.

Affect joint events additionally require the child `(label, direction)` set to
match before overlap inheritance applies. This prevents a label from moving
from an `A+ V-` joint to an `A+ D-` joint just because the time span overlaps.

## Tests Required For Each Producer

Each new producer should add tests for:

- unique event IDs with `{producer_id}.{event_type}.{NNNNNN}`;
- valid `score_name`;
- source tracks exist, unless intentionally empty;
- regular-grid track shape, hop, window, renderer, and channels;
- marker item minimum fields;
- tracks-only session if the producer supports exploration without events;
- `save_session` round trip;
- label inheritance after a rerun;
- any producer-specific edge cases that could silently change review meaning.

## Why The Affect Implementation Moved

The first generic-schema migration kept the affect implementation under
`affective_events.v2` so detection behavior and review schema could be changed
without also changing import paths. That made the behavioral diff easier to
audit.

Now the canonical namespace exists, the implementation has moved to
`acoustic_events.producers.affect`. The old `affective_events` and
`affective_events.v2` modules are compatibility wrappers only.
