# Acoustic Event Inference Artifacts

This package turns the notebook inference workflows into reusable prediction
artifacts:

```text
audio -> inference artifacts -> later producer inputs
```

It does not create review sessions and does not run producers as a user-facing
workflow. It only stores the model outputs that the producers will later
consume.

## Output Layout

Artifacts are stored as:

```text
<out>/<recording_id>/<audio_sha256>/<task>/<inference_config_hash>/
  manifest.json
  predictions.npz
```

`audio_sha256` is computed from decoded mono 16 kHz float32 samples, so cache
lookup decodes the audio first. `inference_config_hash` is separate from
producer `config_hash`.

## Run One Model

Dimensional affect, with explicit backbone:

```bash
uv run python -m audio_classification_playground.acoustic_events.inference run affect \
  --audio /path/to/audio.mp3 \
  --backbone wavlm \
  --out artifacts/
```

Disfluency, with explicit backbone:

```bash
uv run python -m audio_classification_playground.acoustic_events.inference run disfluency \
  --audio /path/to/audio.mp3 \
  --backbone whisper \
  --out artifacts/
```

Categorical emotion:

```bash
uv run python -m audio_classification_playground.acoustic_events.inference run emotion \
  --audio /path/to/audio.mp3 \
  --out artifacts/
```

Shared VAD:

```bash
uv run python -m audio_classification_playground.acoustic_events.inference run vad \
  --audio /path/to/audio.mp3 \
  --out artifacts/
```

Add `--reuse-cache` to reuse a complete matching artifact instead of rerunning
the model. Add `--recording-id my_recording` to control the organizational
directory name.

## Run All Inference

```bash
uv run python -m audio_classification_playground.acoustic_events.inference run-all \
  --audio /path/to/audio.mp3 \
  --affect-backbone wavlm \
  --disfluency-backbone whisper \
  --out artifacts/ \
  --reuse-cache
```

`run-all` runs VAD, affect, disfluency, and emotion sequentially. It fails fast
if any task fails. Completed per-task artifacts remain valid.

## List Cached Artifacts

```bash
uv run python -m audio_classification_playground.acoustic_events.inference list-cached \
  --audio /path/to/audio.mp3 \
  --out artifacts/
```

You can narrow the result:

```bash
uv run python -m audio_classification_playground.acoustic_events.inference list-cached \
  --audio /path/to/audio.mp3 \
  --out artifacts/ \
  --task emotion
```

## Python API

```python
from audio_classification_playground.acoustic_events.inference import run_all_inference

result = run_all_inference(
    "/path/to/audio.mp3",
    out_dir="artifacts",
    affect_backbone="wavlm",
    disfluency_backbone="whisper",
    reuse_cache=True,
)

print(result.artifacts["affect"].path)
print(result.reused)
```

Single-task example:

```python
from audio_classification_playground.acoustic_events.inference import run_emotion_inference

emotion = run_emotion_inference(
    "/path/to/audio.mp3",
    out_dir="artifacts",
    reuse_cache=True,
)
print(emotion.artifact.path)
```

## Later Producer Inputs

Adapters convert artifacts into the existing producer input shapes:

```python
from audio_classification_playground.acoustic_events.inference import (
    artifact_to_affect_signals,
    artifact_to_disfluency_logits,
    artifact_to_emotion_probabilities,
    artifact_to_vad,
)

vad = artifact_to_vad(result.artifacts["vad"])
signals = artifact_to_affect_signals(result.artifacts["affect"])
fluency_logits, type_logits, hop, window, duration = artifact_to_disfluency_logits(
    result.artifacts["disfluency"]
)
probs, labels, hop, window, duration = artifact_to_emotion_probabilities(
    result.artifacts["emotion"]
)
```

The adapters are validation/building-block helpers. They do not save review
sessions.

## Logging

The CLI and default Python runner progress use the
`audio_classification_playground.acoustic_events.inference` logger, configured
to write INFO logs to stdout. Pass a custom `progress` callable to any runner if
you want to route progress elsewhere.
