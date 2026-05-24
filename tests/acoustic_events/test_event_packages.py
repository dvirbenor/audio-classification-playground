import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_classification_playground.acoustic_events.event_packages import (
    EventPackageConfigs,
    append_completion_row,
    compact_completion_index,
    eventify_archive,
    load_event_package,
)
from audio_classification_playground.acoustic_events.event_packages.cli import _run_one_pass
from audio_classification_playground.acoustic_events.event_packages.package import (
    completion_rows_from_index_and_shards,
)
from audio_classification_playground.acoustic_events.inference import run_all_inference
from audio_classification_playground.acoustic_events.orchestration.manifest import ArchiveEntity


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


class EventPackageTest(unittest.TestCase):
    def test_eventify_archive_writes_compact_atomic_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inference_archive = _make_flat_artifacts(root)
            events_archive = root / "events" / "s1" / "a1"

            result = eventify_archive(
                inference_archive_dir=inference_archive,
                output_archive_dir=events_archive,
                session_id="s1",
                archive_id="a1",
                date="2026-05-24",
            )

            self.assertEqual(result.status, "packaged")
            pkg = load_event_package(events_archive)
            self.assertEqual(pkg.package["schema"], "event_package.v1")
            self.assertEqual(pkg.package["status"], "complete")
            self.assertEqual(pkg.package["session_id"], "s1")
            rows = [
                json.loads(line)
                for line in (events_archive / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows)
            self.assertNotIn("session_id", rows[0])
            self.assertNotIn("archive_id", rows[0])
            self.assertNotIn("parent_id", rows[0])
            self.assertNotIn("children", rows[0])

            affect = next(row for row in rows if row["task"] == "affect")
            self.assertEqual(affect["event_type"], "deviation")
            self.assertEqual(affect["label"], "arousal+")
            self.assertEqual(affect["labels"], ["arousal+"])
            self.assertEqual(affect["axis"], "arousal")
            self.assertEqual(affect["direction"], "+")
            self.assertEqual(affect["metadata"]["producer_label"], "arousal_deviation")
            self.assertEqual(affect["score_name"], "peak_z")

            disfluency = next(row for row in rows if row["task"] == "disfluency")
            self.assertEqual(disfluency["label"], "block")
            self.assertIn("block", disfluency["labels"])
            self.assertIn("interjection", disfluency["labels"])
            self.assertEqual(disfluency["score_name"], "probability")

    def test_eventify_archive_skips_complete_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inference_archive = _make_flat_artifacts(root)
            events_archive = root / "events" / "s1" / "a1"

            first = eventify_archive(
                inference_archive_dir=inference_archive,
                output_archive_dir=events_archive,
                session_id="s1",
                archive_id="a1",
            )
            package_json = events_archive / "package.json"
            before = package_json.read_bytes()
            before_mtime = package_json.stat().st_mtime_ns
            time.sleep(0.001)
            second = eventify_archive(
                inference_archive_dir=inference_archive,
                output_archive_dir=events_archive,
                session_id="s1",
                archive_id="a1",
            )

            self.assertEqual(first.status, "packaged")
            self.assertEqual(second.status, "skipped_complete")
            self.assertEqual(package_json.read_bytes(), before)
            self.assertEqual(package_json.stat().st_mtime_ns, before_mtime)

    def test_eventify_archive_reports_not_ready_without_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inference_archive = root / "inference" / "s1" / "a1"
            (inference_archive / "vad").mkdir(parents=True)

            result = eventify_archive(
                inference_archive_dir=inference_archive,
                output_archive_dir=root / "events" / "s1" / "a1",
                session_id="s1",
                archive_id="a1",
            )

            self.assertEqual(result.status, "not_ready")
            self.assertFalse((root / "events" / "s1" / "a1" / "package.json").exists())

    def test_completion_index_compacts_only_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inference_archive = _make_flat_artifacts(root)
            events_archive = root / "events" / "s1" / "a1"
            result = eventify_archive(
                inference_archive_dir=inference_archive,
                output_archive_dir=events_archive,
                session_id="s1",
                archive_id="a1",
                date="2026-05-24",
            )
            append_completion_row(
                events_base=root / "events",
                worker_id="worker1",
                package_dir=events_archive,
                package_payload=result.package_payload,
            )
            shard = root / "events" / "_meta" / "completed_shards" / "date=2026-05-24" / "worker1.jsonl"
            with shard.open("ab") as f:
                f.write(b'{"session_id":"bad"')

            compact_completion_index(root / "events")
            rows = completion_rows_from_index_and_shards(root / "events")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["session_id"], "s1")
            self.assertEqual(rows[0]["archive_id"], "a1")

    def test_run_worker_uses_completion_index_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_flat_artifacts(root)
            events_output = root / "events"
            entities = [
                ArchiveEntity(
                    session_id="s1",
                    archive_id="a1",
                    file_parent_dir="",
                    date="2026-05-24",
                )
            ]
            common = dict(
                entities=entities,
                inference_output=root / "inference",
                events_output=events_output,
                configs=EventPackageConfigs(),
                num_shards=1,
                shard_index=0,
                force=False,
                validate_inputs=False,
                max_attempts=3,
                retry_failed=False,
                reclaim_stale_minutes=60.0,
            )

            first = _run_one_pass(worker_id="worker1", **common)
            compact_completion_index(events_output)
            second = _run_one_pass(worker_id="worker2", **common)
            rows = completion_rows_from_index_and_shards(events_output)

            self.assertEqual(first["packaged"], 1)
            self.assertEqual(second["skipped_complete"], 1)
            self.assertEqual(second["processed"], 0)
            self.assertEqual(len(rows), 1)


def _make_flat_artifacts(root: Path) -> Path:
    audio = _write_audio(root / "clip.wav")
    inference_archive = root / "inference" / "s1" / "a1"

    def artifact_path(task: str) -> Path:
        return inference_archive / task

    run_all_inference(
        audio,
        out_dir=root / "inference",
        affect_backbone="wavlm",
        disfluency_backbone="wavlm",
        predictors={
            "affect": _fake_affect,
            "disfluency": _fake_disfluency,
            "emotion": _fake_emotion,
        },
        vad_detector=_fake_vad,
        artifact_path_fn=artifact_path,
        progress=lambda message: None,
    )
    return inference_archive


def _write_audio(path: Path) -> Path:
    sr = 16000
    samples = np.zeros(sr * 10, dtype=np.float32)
    sf.write(path, samples, sr)
    return path


def _fake_vad(samples, sample_rate):
    return [(0.0, len(samples) / sample_rate)]


def _fake_affect(windows):
    n = len(windows)
    values = np.zeros(n, dtype=np.float32)
    if n >= 12:
        center = n // 2
        values[max(0, center - 4): min(n, center + 5)] = 0.9
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
        strong = math.log(0.9 / 0.1)
        types[2:min(6, n), 0] = strong  # Block
        types[2:min(6, n), 4] = strong  # Interjection
    return {
        "fluency_logits": fluency,
        "disfluency_type_logits": types,
    }


def _fake_emotion(windows):
    n = len(windows)
    scores = np.tile(
        np.array([[0.05, 0.02, 0.02, 0.9, 0.05, 0.02, 0.02, 0.02, 0.0]], dtype=np.float32),
        (n, 1),
    )
    return scores, EMOTION2VEC_LABELS


if __name__ == "__main__":
    unittest.main()
