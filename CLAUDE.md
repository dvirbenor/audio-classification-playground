# CLAUDE.md

### Project Architecture & Big Picture

This is a research/production codebase for **acoustic event detection on speech** — extracting discrete, reviewable events (affective deviations, categorical emotion, disfluencies, non-verbal vocalizations) from audio. It runs both as a local research playground (notebooks) and as a batched Kubernetes inference pipeline over ~600k audio archives. The source audio is pulled from S3; the inference artifacts, decoded-audio cache, and work-coordination metadata live on shared EFS storage (see **Data Storage & Retrieval** below).

The canonical package is `audio_classification_playground.acoustic_events`. Everything flows through a deliberate, stage-by-stage pipeline where each stage has its own subpackage, a `__main__.py` CLI, and a README:

```
audio → inference artifacts → producer events → {review packages | atomic event packages} → session store
```

- **`inference/`** — Turns raw audio into reusable model prediction artifacts (`predictions.npz` + `manifest.json`). Stores model outputs only; does not run producers. Cache key is `audio_sha256` computed from *decoded mono 16 kHz float32 samples*, plus a separate `inference_config_hash`. Backbones include WavLM, Whisper, emotion2vec.
- **`producers/<task>/`** (`affect`, `emotion`, `disfluency`) — Task-specific logic only: model loading, event extraction, config, diagnostics. Each producer returns `(ProducerRun, list[tracks], list[Event])`. The affect producer is the reference implementation and is re-exported from the package root.
- **`composition/`** — Composes exactly one artifact per task (affect/disfluency/emotion/VAD) into an immutable `review_package.v1`. Validates all artifacts share the same decoded-audio hash.
- **`event_packages/`** — Builds compact per-archive atomic event collections (`events.jsonl` + `package.json`) for downstream transcript decoration. No labels, no track arrays.
- **`review/`** — FastAPI + static JS app (`review/static/`) that displays, filters, and labels upstream events. **It never creates events.** Consumes sessions; exposes `/api/tracks` (`/api/signals` is retired).
- **`session_store/`** — Aggregates per-archive event packages into date-partitioned session-level parquet. Writes a session row only once all archives of that session complete. Idempotent.
- **`orchestration/`** — Drives batched inference across many GPU/CPU pods: worker loop, manifest-driven work claiming, file locking, prefetch, audio cache + cache warmer, heartbeat, timings. Supports an all-in-one worker mode and a per-task "task-fleet" mode.

**The generic schema vs. producer split is the core architectural rule** — read `acoustic_events/PRODUCER_CONTRACT.md` before touching events, tracks, or producers. The schema layer (`Event`, `ProducerRun`, `RegularGridTrack`, `MarkerTrack`, `save_session`, label inheritance, review API/UI) is generic and task-agnostic. Producers own everything task-specific. Sessions/packages are **immutable**; `labels.json` is the only mutable file.

`audio_classification_playground.affective_events` is a **compatibility facade only** (for old notebooks). Do not add modules or producers there — new work goes under `acoustic_events`.

Vendored model code lives in `panns/`, `beats/`, `vox_profile/` and `synthetic/` (synthetic test-sample generation). These are largely standalone.

### Data Storage & Retrieval

The "audio archives" the pipeline runs over are **not** files on EFS — that's a derived/output store. Source audio lives in S3, and the unit of work is logical:

- **Work index (manifest):** A parquet file (passed via `--parquet`; the k8s specs default to `/efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet`) lists every archive. It's read in `orchestration/manifest.py` with columns `session_id, archive_id, file_parent_dir, date`, deduplicated by `(session_id, archive_id)`.
- **What an "archive" is:** One `(session_id, archive_id)` pair (`orchestration/manifest.py` `ArchiveEntity`) — a logical recording, *not* a tar/zip. Its audio is whatever lives under the S3 prefix `file_parent_dir`.
- **Source audio (retrieval):** Pulled from S3 bucket `riverside-pro-main` (`orchestration/audio_resolver.py`). The resolver lists `s3://riverside-pro-main/<file_parent_dir>/` and picks the best `.wav` (excluding `enhanced`/`cfr` stems), falling back to `.mp3`, then downloads and decodes to mono 16 kHz float32. Objects in cold storage classes raise a resolution error rather than restoring.
- **Decoded-audio cache (EFS):** Optional shared cache of decoded f32 samples, enabled with `--audio-cache-dir`/`--max-cache-bytes` and pre-populated by the `warm-cache` command (`orchestration/audio_cache.py`, `cache_warmer.py`). This is the only sense in which "audio" lives on EFS.
- **Outputs (EFS):** Orchestration writes artifacts under `<output_base>/<session_id>/<archive_id>/<task>/` (worker overrides the inference API's default path fn). Note this differs from the standalone inference API layout `<recording_id>/<audio_sha256>/<task>/<config_hash>/` documented under Coding Standards — in the fleet, `recording_id`↔`session_id`/`archive_id` and the path is flattened. Work-coordination metadata (locks, error JSONs, timings, progress) lives under `<output_base>/_meta/`.

### Development Commands

This project uses **`uv`** (not pip/poetry). A frozen `uv.lock` is committed. There is no Makefile, no configured linter/formatter, and no type checker — do not assume `ruff`/`black`/`mypy` are available.

- **Format:** *(none configured — match surrounding style)*
- **Lint:** *(none configured)*
- **Type-check:** *(none configured)*
- **Test (all):** `uv run python -m pytest tests/`
- **Test (single file):** `uv run python -m pytest tests/acoustic_events/test_fusion.py`
- **Test (single case):** `uv run python -m pytest tests/acoustic_events/test_fusion.py::FusionTest::test_joint_construction_and_parent_ids`
- **Build/Run:** Each pipeline stage is a module CLI invoked as `uv run python -m audio_classification_playground.acoustic_events.<stage> <command> [flags]` (e.g. `inference run affect ...`, `composition compose ...`, `event_packages run ...`, `session_store populate ...`, `review`). See each subpackage's `README.md` for exact flags.
- **Sync deps:** `uv sync --frozen`

When running on a pod, `source env.shared.sh` first so all frameworks resolve weights from the shared EFS model cache (`HF_HOME`, `MODELSCOPE_CACHE`, `TORCH_HOME`).

### Coding Standards & Preferences

- **Python 3.10–3.13.** Code must run on 3.10 (the `.venv` interpreter is 3.10); avoid syntax newer than that.
- Tests are written with the stdlib **`unittest`** framework (`TestCase` classes), even though they run under pytest. Match this style — do not introduce pytest-fixture-based tests in existing test modules.
- Events, tracks, and producer runs use the dataclasses in `acoustic_events/schema.py`. **Do not create parallel event dataclasses** for review-facing objects (private intermediate candidates inside a producer are fine).
- Follow the `PRODUCER_CONTRACT.md` invariants exactly when emitting events: `event_id` format `{producer_id}.{event_type}.{NNNNNN}`, IDs unique per session, `duration_sec == end_sec - start_sec`, `score_name` must already exist in `schema.SCORE_NAMES` (add it there first if needed). Use `evidence` for reviewer-facing explanation, `extra` for diagnostics; never duplicate the top-level score in `evidence`.
- Sessions and packages are **immutable**. Re-running one producer means composing a new package from retained outputs plus new outputs — never mutate an existing session in place. `labels.json` is the sole mutable artifact.
- Artifact/package directory layouts and hash keys are contracts other stages depend on. Preserve the `<recording_id>/<audio_sha256>/<task>/<config_hash>/` structure and keep `audio_sha256` derived from decoded 16 kHz mono float32 samples.
- When the methodology doc and an algorithm description disagree, `affective_events/METHODOLOGY.md` is the canonical reference for affect detection.

### Tools & Testing Setup

- **Test runner:** `pytest` (dev dependency) discovering stdlib `unittest` tests under `tests/`, mirroring the package layout (`tests/acoustic_events/`).
- **Web layer:** `fastapi` + `uvicorn` for the review server; the frontend is plain static HTML/JS/CSS in `review/static/` (no build step, no framework).
- **A/B and benchmark harnesses** live in `scripts/` (e.g. ONNX/TensorRT benchmarks, VRAM profiling, runtime-knob comparisons). Use these patterns when validating inference-performance changes rather than ad-hoc timing.
- Each new producer must add the tests enumerated in `PRODUCER_CONTRACT.md` (unique event IDs, valid score names, track shapes, `save_session` round-trip, label inheritance after rerun).

### Important Usage Notes

- **Do not add code under `affective_events/`** — it is a frozen compatibility facade.
- **Do not edit vendored model packages** (`panns/`, `beats/`, `vox_profile/`) for stylistic reasons; they track upstream implementations.
- **Notebooks (`notebooks/`) are exploratory** and may contain stale/work-copy variants (`*-workcopy`, `*-continuation`). Treat them as scratch, not as source of truth; the contract docs and `acoustic_events` code win.
- **`manifests/`** are Kubernetes job specs for the cloud inference fleet (ClearML/EFS/ECR). Editing them affects real batch jobs — change only when explicitly working on orchestration deployment.
- **Never commit credentials.** Manifests reference SSH keys and cloud hosts via mounted secrets and env vars; keep it that way.
- Do not mix task-fleet orchestration workers with all-in-one workers on the same output tree (documented in `orchestration/README.md`).
- `README.md` at repo root is intentionally empty — per-subpackage READMEs are the real documentation.
