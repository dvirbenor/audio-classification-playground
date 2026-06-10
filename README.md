# Audio Classification Playground

An end-to-end audio understanding platform that processes speech recordings to extract and review **acoustic events** — dimensional affect (arousal/valence/dominance), categorical emotion, and speech disfluency — at scale. It spans single-file inference, Kubernetes-native batch processing of hundreds of thousands of archives, content-addressed artifact caching, and an interactive web UI for human review.

---

## Features

- **Multi-task acoustic inference** — Voice Activity Detection (Silero), dimensional **affect** (WavLM → arousal/valence/dominance), **categorical emotion** (Emotion2vec), and **disfluency** detection (WavLM or Whisper backbones) from a single CLI.
- **Content-addressed caching** — artifacts are keyed by `recording_id / audio_sha256 / task / config_hash`, so re-running with the same audio and config reuses prior work automatically.
- **Distributed batch orchestration** — Kubernetes worker fleet designed for ~600k archives, with atomic EFS lock files (no races), async audio prefetch/decode threads, GPU-resident models, graceful `SIGTERM` handling, and stale-lock reclaim.
- **Explicit composition pipeline** — inference → artifacts → review packages, each a discrete, inspectable step.
- **Unified event/track schema** — every producer emits to the canonical `acoustic_events.v1` schema (`Event`, `ProducerRun`, `RegularGridTrack`, `MarkerTrack`).
- **Immutable provenance** — producer evidence (`package.json`) is content-signed; only `labels.json` (reviewer annotations) is mutable.
- **Interactive review UI** — FastAPI app with waveform + track visualization, range-served audio, and label editing/inheritance.
- **Atomic event packages & session store** — compact JSONL event extraction and date-partitioned Parquet aggregation for downstream analytics and transcript decoration.
- **Inference optimization** — runtime-mode selection (optimized vs. fp32-eager), WavLM compilation presets, plus benchmarking scripts (ONNX/TensorRT) for emotion2vec and WavLM.

---

## Architecture

The system is organized as a pipeline of independent stages under `audio_classification_playground/acoustic_events/`. Each stage reads the previous stage's on-disk artifacts and writes its own, which makes every step independently runnable, cacheable, and debuggable.

```
Raw audio
   │
   ▼
┌──────────────┐   VAD intervals, affect signals, disfluency logits,
│  inference   │   emotion probabilities  →  content-addressed artifacts
└──────────────┘   <out>/<recording_id>/<audio_sha256>/<task>/<config_hash>/
   │
   ▼
┌──────────────┐   runs each producer over the 4 artifacts; assembles a
│ composition  │   deterministic review package (content hash = package id)
└──────────────┘   package.json (immutable) + labels.json (mutable) + tracks/
   │
   ├──────────────────────────────┐
   ▼                              ▼
┌──────────────┐         ┌──────────────────┐
│   review     │         │  event_packages  │  compact events.jsonl per archive
│ (FastAPI UI) │         └──────────────────┘
└──────────────┘                  │
                                  ▼
                          ┌──────────────────┐
                          │  session_store   │  date-partitioned Parquet,
                          └──────────────────┘  session-level aggregation
```

**Module responsibilities**

| Module | Responsibility |
|--------|----------------|
| `acoustic_events/schema.py` | Canonical `Event`, `ProducerRun`, `RegularGridTrack`, `MarkerTrack` schema shared across the pipeline. |
| `acoustic_events/inference/` | Audio loading & normalization (mono 16 kHz), per-file model runners, artifact persistence (NPZ + manifest), runtime-mode selection. |
| `acoustic_events/producers/` | Task-specific event extractors: `affect/` (dimensional deviation + joint events), `disfluency/` (filled pauses, repetitions, restarts), `emotion/` (VAD-aware categorical events). |
| `acoustic_events/orchestration/` | Distributed K8s worker: manifest loading, atomic locking, async prefetch, decoded-audio LRU cache, task groups, timing/error/progress reporting. |
| `acoustic_events/composition/` | Combines four completed artifacts into an immutable, fingerprinted review package. |
| `acoustic_events/event_packages/` | Worker that extracts atomic events to compact JSONL with a completion index. |
| `acoustic_events/session_store/` | Aggregates per-archive events up to session level, writing date-partitioned Parquet. |
| `acoustic_events/review/` | FastAPI server + static frontend serving packages, waveforms, tracks, and mutable labels. |

**Supporting / exploratory modules**

- `affective_events/` — legacy compatibility facade that re-exports the new `acoustic_events` implementations (the original affect pipeline; `v2/` was the staging ground before migration).
- `panns/`, `beats/` — self-contained audio-tagging / music-understanding models used for exploration.
- `vox_profile/` — WavLM-based speaker-attribute models (emotion, fluency).
- `synthetic/` — synthetic audio generation for tests and validation.
- `scripts/` — one-off profiling and ONNX/TensorRT benchmarking utilities.
- `notebooks/` — Jupyter exploratory work (model development, signal exploration).
- `manifests/` — Kubernetes job YAML templates for the worker fleet.

**Key design decisions**

- **Explicit, staged composition** over a monolithic pipeline — each stage is a CLI command with on-disk inputs/outputs.
- **Content-addressed artifacts** — reuse is automatic and provenance is verifiable.
- **Immutable producer evidence** — only reviewer labels mutate, keeping model output auditable.
- **Schema unification** — one event/track schema replaces per-task schemas, so the review UI and downstream consumers are model-agnostic.

---

## Prerequisites

**Runtime**

- Python **3.10–3.13** (`>=3.10,<3.14`)
- For GPU inference: CUDA 11.8+ capable GPU. CPU-only works but is significantly slower.

**Hardware**

- Single file (CPU): 8 GB+ RAM.
- GPU batch inference: 12–24 GB VRAM depending on task/batch size (e.g. on A100: affect ~6 GB @ batch 256, disfluency ~4 GB @ batch 384, emotion ~8 GB @ batch 64 optimized; VAD runs on CPU).
- ~50 GB free disk for model weights downloaded from Hugging Face Hub on first use (WavLM-Large, Whisper-Large, Emotion2vec, Silero VAD, etc.).

**External services (for distributed orchestration only)**

- An **S3 bucket** as the audio source (single-file runs can read local paths).
- **EFS / shared NFS** for worker locks, error logs, timing metadata, and artifacts.
- A **Kubernetes cluster** — not required for single-file or small-batch runs.

Models download automatically on first use; no manual model setup is required.

---

## Installation & Setup

The project uses a [Hatchling](https://hatch.pypa.io/) build backend and a standard `pyproject.toml`. [`uv`](https://docs.astral.sh/uv/) is recommended.

```bash
# Clone
git clone <repo-url>
cd audio-classification-playground

# Option A — uv (recommended)
uv sync --all-groups          # installs runtime + dev (pytest) dependencies

# Option B — pip + venv
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

The first inference run downloads the required model weights from Hugging Face Hub. Ensure network access and sufficient disk space (~50 GB for the full model set).

---

## Usage

All commands run as Python modules under `audio_classification_playground.acoustic_events.<module>`. Prefix with `uv run` if using `uv` (shown), or activate your venv and drop the prefix.

### 1. Single-file inference

```bash
# Run all tasks (VAD + affect + disfluency + emotion), reusing cached artifacts
uv run python -m audio_classification_playground.acoustic_events.inference run-all \
  --audio /path/to/audio.mp3 \
  --affect-backbone wavlm \
  --disfluency-backbone whisper \
  --out artifacts/ \
  --reuse-cache

# Or a single task
uv run python -m audio_classification_playground.acoustic_events.inference run affect \
  --audio /path/to/audio.mp3 --backbone wavlm --out artifacts/
```

Artifacts land in `artifacts/<recording_id>/<audio_sha256>/<task>/<config_hash>/`.

### 2. Compose a review package

```bash
uv run python -m audio_classification_playground.acoustic_events.composition compose \
  --affect-artifact     artifacts/clip/<hash>/affect/<hash>/ \
  --disfluency-artifact artifacts/clip/<hash>/disfluency/<hash>/ \
  --emotion-artifact    artifacts/clip/<hash>/emotion/<hash>/ \
  --vad-artifact        artifacts/clip/<hash>/vad/<hash>/ \
  --out review-packages/
```

### 3. Launch the review UI

```bash
uv run python -m audio_classification_playground.acoustic_events.review \
  --package review-packages/<recording_id>/<package_id>/
# Opens http://localhost:8000
```

### 4. Distributed batch processing

```bash
# (Optional) warm a shared decoded-audio cache, one per fleet
uv run python -m audio_classification_playground.acoustic_events.orchestration warm-cache \
  --parquet /efs/manifests/all_archives.parquet \
  --output  /efs/inference-output \
  --audio-cache-dir /efs/audio-cache \
  --max-cache-bytes 1099511627776

# All-in-one worker
uv run python -m audio_classification_playground.acoustic_events.orchestration run \
  --parquet /efs/manifests/all_archives.parquet \
  --output  /efs/inference-output \
  --affect-backbone wavlm --disfluency-backbone wavlm --device cuda

# Task-specialized worker (one fleet per task group)
uv run python -m audio_classification_playground.acoustic_events.orchestration run \
  --parquet /efs/manifests/all_archives.parquet \
  --output  /efs/inference-output \
  --task-group affect --affect-batch-size 256 --device cuda
```

### 5. Event packaging & session aggregation

```bash
# Extract atomic events to JSONL (watch mode polls for new completions)
uv run python -m audio_classification_playground.acoustic_events.event_packages run \
  --parquet /efs/manifests/all_archives.parquet \
  --inference-output /efs/inference \
  --events-output    /efs/events \
  --watch --poll-interval-sec 300

# Aggregate to session-level Parquet
uv run python -m audio_classification_playground.acoustic_events.session_store populate \
  --manifest /efs/manifests/all_archives.parquet \
  --events-output /efs/events \
  --store-output  /efs/session-events/
```

### Python API

```python
from audio_classification_playground.acoustic_events.inference import (
    run_all_inference, run_affect_inference, ModelSuite, AudioData,
)
from audio_classification_playground.acoustic_events.composition import (
    compose_review_package,
)
from audio_classification_playground.acoustic_events import (
    Event, ProducerRun, RegularGridTrack, MarkerTrack, extract_events,
)
```

---

## Commands & Scripts

Module CLIs (`python -m audio_classification_playground.acoustic_events.<module> <command>`):

| Module | Command(s) | Description |
|--------|-----------|-------------|
| `inference` | `run {affect,disfluency,emotion,vad}` | Run a single inference task on one file. |
| `inference` | `run-all` | Run VAD + all tasks sequentially. |
| `inference` | `list-cached` | List cached artifacts. |
| `orchestration` | `run` | Worker loop: claim → prefetch → infer → write. |
| `orchestration` | `warm-cache` | Prefill the shared decoded-audio LRU cache. |
| `orchestration` | `progress` / `status` | Report completion status / live fleet heartbeat. |
| `orchestration` | `errors` / `timings` | Summarize audio/inference errors / per-archive timing distribution. |
| `orchestration` | `reclaim-stale` | Unlock hung worker locks. |
| `composition` | `compose` | Build a review package from four artifacts. |
| `event_packages` | `run` / `watch` / `eventify` | Extract atomic events (fleet, polling, or single archive). |
| `event_packages` | `progress` / `compact-index` / `reconcile-index` | Track and maintain the completion index. |
| `session_store` | `populate` | Aggregate archives into session-level Parquet. |
| `review` | *(no subcommand)* | Launch the FastAPI review UI. |

Common development commands:

| Task | Command |
|------|---------|
| Install (runtime + dev) | `uv sync --all-groups` |
| Run tests | `uv run pytest tests/` |
| Run a single test | `uv run pytest tests/acoustic_events/test_disfluency_producer.py` |
| Tests with coverage | `uv run pytest --cov=audio_classification_playground tests/` |
| Build wheel | `uv run hatch build` |
| Launch review UI | `uv run python -m audio_classification_playground.acoustic_events.review --package <dir>` |

> Per-module deep-dive docs live in each module's own `README.md` (notably `acoustic_events/inference/`, `orchestration/`, and `composition/`). Design principles are documented in `affective_events/METHODOLOGY.md`.
