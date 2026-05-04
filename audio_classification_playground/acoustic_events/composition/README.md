# Acoustic Event Review Packages

This package composes inference artifacts into producer events and review-ready
tracks:

```text
audio -> inference artifacts -> producer events -> review_package.v1
```

Composition is deliberately explicit. You pass exactly one artifact per task:
affect, disfluency, emotion, and VAD. The composer validates that all artifacts
come from the same decoded audio hash, runs one producer output per task, and
writes a deterministic review package.

## Compose A Package

After running inference, compose the four artifacts:

```bash
uv run python -m audio_classification_playground.acoustic_events.composition compose \
  --affect-artifact artifacts/<recording_id>/<audio_sha256>/affect/<hash>/ \
  --disfluency-artifact artifacts/<recording_id>/<audio_sha256>/disfluency/<hash>/ \
  --emotion-artifact artifacts/<recording_id>/<audio_sha256>/emotion/<hash>/ \
  --vad-artifact artifacts/<recording_id>/<audio_sha256>/vad/<hash>/ \
  --out review_packages/
```

The output path is:

```text
review_packages/<recording_id>/<package_id>/
  package.json
  labels.json
  tracks/*.npz
```

`package.json` is immutable producer evidence. `labels.json` is the only mutable
file and is updated by the review app.

## Optional Producer Configs

Use JSON files to override producer defaults:

```bash
uv run python -m audio_classification_playground.acoustic_events.composition compose \
  --affect-artifact ... \
  --disfluency-artifact ... \
  --emotion-artifact ... \
  --vad-artifact ... \
  --out review_packages/ \
  --disfluency-config configs/disfluency-strict.json
```

The resolved post-merge producer config is stored in `producer_runs[].config`.
Changing a producer config changes the package fingerprint and package id.

## Launch Review

```bash
uv run python -m audio_classification_playground.acoustic_events.review \
  --package review_packages/<recording_id>/<package_id>/
```

The review app now reads `review_package.v1` directories. Legacy session JSON
can still be read by the deprecated storage helpers, but it is not the review
server entrypoint.

## Python API

```python
from audio_classification_playground.acoustic_events.composition import (
    compose_review_package,
)

package_path = compose_review_package(
    affect_artifact="artifacts/clip/<audio_sha256>/affect/<hash>",
    disfluency_artifact="artifacts/clip/<audio_sha256>/disfluency/<hash>",
    emotion_artifact="artifacts/clip/<audio_sha256>/emotion/<hash>",
    vad_artifact="artifacts/clip/<audio_sha256>/vad/<hash>",
    out_dir="review_packages",
    task_configs={"disfluency": "configs/disfluency-strict.json"},
)
```

Lower-level helpers are available when you want to inspect producer output before
writing a package:

```python
from audio_classification_playground.acoustic_events.composition import (
    compose_affect_from_artifacts,
    compose_disfluency_from_artifacts,
    compose_emotion_from_artifacts,
)
```

The full review-package composer passes VAD into the categorical emotion
producer. Calling `compose_emotion_from_artifacts()` directly without a
`vad_artifact` still produces tracks and a valid producer run, but no default
emotion events.

## Determinism

`package_id` is the first 24 characters of a content fingerprint. The
fingerprint covers the audio hash, inference artifact identities, producer
configs, emitted events, and track metadata. It excludes local filesystem paths
so the same inputs compose to the same package id in different workspaces.

`package.json` is written as sorted, pretty JSON for stable diffs and readable
debugging.
