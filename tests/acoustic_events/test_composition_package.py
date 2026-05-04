import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from audio_classification_playground.acoustic_events.composition import (
    compose_affect_from_artifacts,
    compose_disfluency_from_artifacts,
    compose_emotion_from_artifacts,
    compose_review_package,
    load_review_package,
)
from audio_classification_playground.acoustic_events.composition.package import (
    build_package_payload,
    write_review_package,
)
from audio_classification_playground.acoustic_events.inference import run_all_inference
from audio_classification_playground.acoustic_events.review.server import make_app
from audio_classification_playground.acoustic_events.schema import MarkerItem, MarkerTrack


EMOTION2VEC_LABELS = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]


class CompositionPackageTest(unittest.TestCase):
    def test_compose_package_is_idempotent_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _make_artifacts(root)
            out_dir = root / "packages"

            first = compose_review_package(
                affect_artifact=artifacts["affect"].path,
                disfluency_artifact=artifacts["disfluency"].path,
                emotion_artifact=artifacts["emotion"].path,
                vad_artifact=artifacts["vad"].path,
                out_dir=out_dir,
            )
            package_json = first / "package.json"
            first_bytes = package_json.read_bytes()
            first_mtime = package_json.stat().st_mtime_ns
            time.sleep(0.001)
            second = compose_review_package(
                affect_artifact=artifacts["affect"].path,
                disfluency_artifact=artifacts["disfluency"].path,
                emotion_artifact=artifacts["emotion"].path,
                vad_artifact=artifacts["vad"].path,
                out_dir=out_dir,
            )

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, package_json.read_bytes())
            self.assertEqual(first_mtime, package_json.stat().st_mtime_ns)

            pkg = load_review_package(first)
            track_rel = next(
                meta["data_path"]
                for meta in pkg.tracks_meta.values()
                if "data_path" in meta
            )
            track_path = first / track_rel
            track_path.unlink()
            third = compose_review_package(
                affect_artifact=artifacts["affect"].path,
                disfluency_artifact=artifacts["disfluency"].path,
                emotion_artifact=artifacts["emotion"].path,
                vad_artifact=artifacts["vad"].path,
                out_dir=out_dir,
            )
            self.assertEqual(first, third)
            self.assertTrue(track_path.is_file())
            self.assertEqual(first_bytes, package_json.read_bytes())

            self.assertEqual(pkg.package["schema"], "review_package.v1")
            self.assertEqual(pkg.package_id, pkg.package_fingerprint[:24])
            self.assertEqual(pkg.labels, {})
            self.assertIn("vad_intervals", pkg.package)
            self.assertTrue(pkg.events)

            by_task = {run["task"]: run for run in pkg.package["producer_runs"]}
            self.assertTrue(by_task["emotion"]["outputs"]["composition"]["vad_applied"])
            for task in ("affect", "disfluency", "emotion"):
                prov = by_task[task]["outputs"]["inference_artifacts"][task]
                source = artifacts[task].manifest
                self.assertEqual(prov["audio_sha256"], source["audio"]["sha256"])
                self.assertEqual(prov["inference_config_hash"], source["inference_config_hash"])
                self.assertEqual(prov["model"], source["model"])
            self.assertIn("vad", by_task["emotion"]["outputs"]["inference_artifacts"])

    def test_config_override_changes_package_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _make_artifacts(root)
            baseline = compose_review_package(
                affect_artifact=artifacts["affect"].path,
                disfluency_artifact=artifacts["disfluency"].path,
                emotion_artifact=artifacts["emotion"].path,
                vad_artifact=artifacts["vad"].path,
                out_dir=root / "packages",
            )
            cfg_path = root / "disfluency_config.json"
            cfg_path.write_text(json.dumps({"seed_threshold": 0.85}), encoding="utf-8")
            changed = compose_review_package(
                affect_artifact=artifacts["affect"].path,
                disfluency_artifact=artifacts["disfluency"].path,
                emotion_artifact=artifacts["emotion"].path,
                vad_artifact=artifacts["vad"].path,
                out_dir=root / "packages",
                task_configs={"disfluency": cfg_path},
            )

            self.assertNotEqual(baseline, changed)
            run = next(
                r for r in load_review_package(changed).package["producer_runs"]
                if r["task"] == "disfluency"
            )
            self.assertEqual(run["config"]["seed_threshold"], 0.85)

    def test_rejects_mixed_audio_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _make_artifacts(root / "one")
            second = _make_artifacts(root / "two", audio_name="other.wav", frequency=3.0)

            with self.assertRaisesRegex(ValueError, "mixed audio_sha256"):
                compose_review_package(
                    affect_artifact=first["affect"].path,
                    disfluency_artifact=second["disfluency"].path,
                    emotion_artifact=first["emotion"].path,
                    vad_artifact=first["vad"].path,
                    out_dir=root / "packages",
                )

    def test_task_composers_are_deterministic_in_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))

            a1 = compose_affect_from_artifacts(
                affect_artifact=artifacts["affect"],
                vad_artifact=artifacts["vad"],
            )
            a2 = compose_affect_from_artifacts(
                affect_artifact=artifacts["affect"],
                vad_artifact=artifacts["vad"],
            )
            d1 = compose_disfluency_from_artifacts(disfluency_artifact=artifacts["disfluency"])
            d2 = compose_disfluency_from_artifacts(disfluency_artifact=artifacts["disfluency"])
            e1 = compose_emotion_from_artifacts(emotion_artifact=artifacts["emotion"])
            e2 = compose_emotion_from_artifacts(emotion_artifact=artifacts["emotion"])

            self.assertEqual([e.as_dict() for e in a1[2]], [e.as_dict() for e in a2[2]])
            self.assertEqual([e.as_dict() for e in d1[2]], [e.as_dict() for e in d2[2]])
            self.assertEqual([e.as_dict() for e in e1[2]], [e.as_dict() for e in e2[2]])

    def test_review_api_reads_package_tracks_and_updates_only_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _make_artifacts(root)
            package_path = compose_review_package(
                affect_artifact=artifacts["affect"].path,
                disfluency_artifact=artifacts["disfluency"].path,
                emotion_artifact=artifacts["emotion"].path,
                vad_artifact=artifacts["vad"].path,
                out_dir=root / "packages",
            )
            package_json = package_path / "package.json"
            labels_json = package_path / "labels.json"
            before_package = package_json.read_bytes()
            before_labels = labels_json.read_bytes()

            client = TestClient(make_app(package_path))
            session = client.get("/api/session").json()
            tracks = client.get("/api/tracks").json()
            self.assertEqual(session["package_id"], package_path.name)
            self.assertIn("disfluency.fluency", tracks["tracks"])
            waveform = client.get("/api/waveform")
            self.assertEqual(waveform.status_code, 200)
            self.assertFalse((package_path / "waveform.peaks.json").exists())
            self.assertTrue(
                (
                    package_path.parent
                    / ".review_cache"
                    / f"{package_path.name}.waveform.peaks.json"
                ).is_file()
            )

            event_id = session["events"][0]["event_id"]
            response = client.post(f"/api/label/{event_id}", json={"verdict": "tp", "tags": ["ok"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(package_json.read_bytes(), before_package)
            self.assertNotEqual(labels_json.read_bytes(), before_labels)
            self.assertEqual(json.loads(labels_json.read_text())[event_id]["verdict"], "tp")

            response = client.delete(f"/api/label/{event_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(labels_json.read_text()), {})

    def test_marker_tracks_are_inline_without_data_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "clip.wav"
            _write_audio(audio)
            track = MarkerTrack(
                track_id="vocalization.markers",
                producer_id="vocalization.default",
                task="vocalization",
                name="vocalization markers",
                renderer="marker",
                items=(MarkerItem(0.1, 0.2, "laugh", 0.8),),
            )
            package = build_package_payload(
                recording_id="clip",
                audio={
                    "path": str(audio.resolve()),
                    "sha256": "hash",
                    "sample_rate": 16000,
                    "duration_sec": 6.0,
                    "hash_semantics": "decoded_mono_16khz_float32",
                },
                vad_intervals=[],
                inference_artifacts={},
                producer_runs=[],
                events=[],
                tracks_meta={
                    track.track_id: {
                        "kind": "marker",
                        "track_id": track.track_id,
                        "producer_id": track.producer_id,
                        "task": track.task,
                        "name": track.name,
                        "renderer": track.renderer,
                        "items": [item.as_dict() for item in track.items],
                        "meta": {},
                    }
                },
            )
            package_path = write_review_package(out_dir=root / "packages", package_payload=package, tracks=[track])

            meta = load_review_package(package_path).tracks_meta["vocalization.markers"]
            self.assertNotIn("data_path", meta)
            tracks = TestClient(make_app(package_path)).get("/api/tracks").json()
            self.assertEqual(tracks["tracks"]["vocalization.markers"][0]["label"], "laugh")

    def test_legacy_session_json_is_not_review_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy.json"
            legacy.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                make_app(legacy)


def _make_artifacts(root: Path, *, audio_name="clip.wav", frequency: float = 0.0):
    root.mkdir(parents=True, exist_ok=True)
    audio = root / audio_name
    _write_audio(audio, frequency=frequency)
    result = run_all_inference(
        audio,
        out_dir=root / "artifacts",
        affect_backbone="wavlm",
        disfluency_backbone="wavlm",
        predictors={
            "affect": _fake_affect,
            "disfluency": _fake_disfluency,
            "emotion": _fake_emotion,
        },
        vad_detector=_fake_vad,
        progress=_quiet,
    )
    return result.artifacts


def _write_audio(path: Path, *, frequency: float = 0.0) -> Path:
    sr = 16000
    t = np.arange(sr * 6, dtype=np.float32) / sr
    samples = np.zeros_like(t) if frequency == 0.0 else 0.1 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, samples.astype(np.float32), sr)
    return path


def _fake_vad(samples, sample_rate):
    return [(0.0, len(samples) / sample_rate)]


def _fake_affect(windows):
    n = len(windows)
    values = np.zeros(n, dtype=np.float32)
    if n >= 5:
        values[n // 2] = 0.9
    return {
        "arousal": values,
        "valence": np.zeros(n, dtype=np.float32),
        "dominance": np.zeros(n, dtype=np.float32),
    }


def _fake_disfluency(windows):
    n = len(windows)
    fluency = np.zeros((n, 2), dtype=np.float32)
    types = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        p = 0.9 if 2 <= i <= min(5, n - 1) else 0.1
        logit = math.log(p / (1.0 - p))
        fluency[i] = [0.0, logit]
    if n:
        block_logit = math.log(0.9 / 0.1)
        types[2:min(6, n), 0] = block_logit
    return {
        "fluency_logits": fluency,
        "disfluency_type_logits": types,
    }


def _fake_emotion(windows):
    n = len(windows)
    scores = np.tile(
        np.array([[0.05, 0.02, 0.02, 0.1, 0.6, 0.1, 0.03, 0.03, 0.05]], dtype=np.float32),
        (n, 1),
    )
    return scores, EMOTION2VEC_LABELS


def _quiet(message):
    pass


if __name__ == "__main__":
    unittest.main()
