"""Tests for the session_store module."""
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from audio_classification_playground.acoustic_events.session_store import (
    PARQUET_SCHEMA,
    SessionEventsRecord,
    build_session_record,
    session_fingerprint,
    thin_event,
)
from audio_classification_playground.acoustic_events.session_store.populate import (
    _atomic_write_parquet,
    _load_and_build_row,
    _process_date_partition,
    ReadySession,
    populate_session_store,
)
from audio_classification_playground.acoustic_events.event_packages.package import (
    write_event_package,
    build_event_package_payload,
    append_completion_row,
)
from audio_classification_playground.acoustic_events.orchestration.manifest import ArchiveEntity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _raw_emotion_event(start: float = 6.375, end: float = 7.375) -> dict:
    return {
        "event_id": "emotion.categorical.v1.categorical.000000",
        "producer_id": "emotion.categorical.v1",
        "task": "emotion",
        "event_type": "categorical",
        "label": "happiness",
        "labels": ["happiness"],
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "source_track_ids": ["emotion.categorical.probabilities"],
        "score": 0.999,
        "score_name": "probability",
        "direction": None,
    }


def _raw_affect_event(start: float = 22.5, end: float = 25.25) -> dict:
    return {
        "event_id": "affect.default.deviation.000000",
        "producer_id": "affect.default",
        "task": "affect",
        "event_type": "deviation",
        "label": "arousal+",
        "labels": ["arousal+"],
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "source_track_ids": ["affect.arousal"],
        "score": 2.998,
        "score_name": "peak_z",
        "direction": "+",
        "axis": "arousal",
        "metadata": {"producer_label": "arousal_deviation"},
    }


def _raw_disfluency_event(start: float = 24.625, end: float = 27.375) -> dict:
    return {
        "event_id": "disfluency.default.instance.000000",
        "producer_id": "disfluency.default",
        "task": "disfluency",
        "event_type": "instance",
        "label": "word_repetition",
        "labels": ["word_repetition", "interjection"],
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "source_track_ids": ["disfluency.fluency", "disfluency.type"],
        "score": 0.966,
        "score_name": "probability",
        "direction": None,
    }


def _make_event_package(
    base: Path,
    session_id: str,
    archive_id: str,
    events: list[dict],
    date: str = "2025-02-19",
) -> Path:
    """Write a complete event package to disk and return its directory."""
    package_dir = base / session_id / archive_id
    payload = build_event_package_payload(
        session_id=session_id,
        archive_id=archive_id,
        date=date,
        audio={"sha256": "abc123", "duration_sec": 100.0, "sample_rate": 16000},
        source_artifacts={"affect": {"task": "affect", "audio_sha256": "abc123"}},
        producer_runs=[],
        event_policy={"atomic_only": True},
        producer_configs={},
        event_config_fingerprint_value="cfg_fp_test",
        input_fingerprint_value="input_fp_test",
        events=events,
    )
    write_event_package(
        package_dir=package_dir,
        package_payload=payload,
        events=events,
    )
    return package_dir


# ---------------------------------------------------------------------------
# thin_event tests
# ---------------------------------------------------------------------------


class TestThinEvent(unittest.TestCase):
    def test_emotion_event_keeps_required_fields_only(self):
        raw = _raw_emotion_event()
        thinned = thin_event(raw)
        assert thinned.start_sec == 6.375
        assert thinned.end_sec == 7.375
        assert thinned.task == "emotion"
        assert thinned.label == "happiness"
        assert thinned.score == 0.999
        assert thinned.score_name == "probability"
        assert thinned.direction is None
        assert thinned.axis is None
        assert thinned.labels is None

    def test_affect_event_keeps_direction_and_axis(self):
        raw = _raw_affect_event()
        thinned = thin_event(raw)
        assert thinned.direction == "+"
        assert thinned.axis == "arousal"
        assert thinned.labels is None  # single-element list stripped

    def test_disfluency_event_keeps_multi_labels(self):
        raw = _raw_disfluency_event()
        thinned = thin_event(raw)
        assert thinned.labels == ["word_repetition", "interjection"]
        assert thinned.direction is None

    def test_empty_string_direction_treated_as_absent(self):
        raw = _raw_affect_event()
        raw["direction"] = ""
        thinned = thin_event(raw)
        assert thinned.direction is None

    def test_empty_string_axis_treated_as_absent(self):
        raw = _raw_affect_event()
        raw["axis"] = ""
        thinned = thin_event(raw)
        assert thinned.axis is None

    def test_serialization_excludes_none(self):
        raw = _raw_emotion_event()
        thinned = thin_event(raw)
        dumped = thinned.model_dump(exclude_none=True)
        assert "direction" not in dumped
        assert "axis" not in dumped
        assert "labels" not in dumped


# ---------------------------------------------------------------------------
# session_fingerprint tests
# ---------------------------------------------------------------------------


class TestSessionFingerprint(unittest.TestCase):
    def test_deterministic(self):
        pairs = [("archive_b", "fp_b"), ("archive_a", "fp_a")]
        fp1 = session_fingerprint(pairs)
        fp2 = session_fingerprint(pairs)
        assert fp1 == fp2

    def test_order_independent(self):
        fp1 = session_fingerprint([("a", "fp1"), ("b", "fp2")])
        fp2 = session_fingerprint([("b", "fp2"), ("a", "fp1")])
        assert fp1 == fp2

    def test_different_fingerprints_differ(self):
        fp1 = session_fingerprint([("a", "fp1")])
        fp2 = session_fingerprint([("a", "fp2")])
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# build_session_record tests
# ---------------------------------------------------------------------------


class TestBuildSessionRecord(unittest.TestCase):
    def test_builds_correct_structure(self):
        events_a = [_raw_emotion_event(), _raw_affect_event()]
        events_b = [_raw_disfluency_event()]
        record = build_session_record(
            session_id="sess-1",
            date="2025-02-19",
            archive_events=[("archive_b", events_b), ("archive_a", events_a)],
        )
        assert record.session_id == "sess-1"
        assert record.date == "2025-02-19"
        assert len(record.event_items) == 2
        # Sorted by archive_id
        assert record.event_items[0].archive_id == "archive_a"
        assert record.event_items[1].archive_id == "archive_b"
        assert len(record.event_items[0].events_data) == 2
        assert len(record.event_items[1].events_data) == 1

    def test_events_sorted_by_start_sec_then_label(self):
        e1 = _raw_emotion_event(start=10.0, end=11.0)
        e2 = _raw_affect_event(start=5.0, end=7.0)
        record = build_session_record(
            session_id="s1",
            date="2025-01-01",
            archive_events=[("a1", [e1, e2])],
        )
        events = record.event_items[0].events_data
        assert events[0].start_sec == 5.0
        assert events[1].start_sec == 10.0

    def test_model_dump_roundtrip(self):
        events = [_raw_emotion_event(), _raw_affect_event(), _raw_disfluency_event()]
        record = build_session_record(
            session_id="s1",
            date="2025-02-19",
            archive_events=[("a1", events)],
        )
        dumped = record.model_dump(exclude_none=True)
        restored = SessionEventsRecord.model_validate(dumped)
        assert restored == record


# ---------------------------------------------------------------------------
# Populate integration tests
# ---------------------------------------------------------------------------


class TestPopulateIntegration(unittest.TestCase):
    def _setup_scenario(self, tmp: Path):
        """Create a two-archive session with event packages and completion shards."""
        events_output = tmp / "events"
        store_output = tmp / "store"
        manifest_path = tmp / "manifest.parquet"

        events_a = [_raw_emotion_event(), _raw_affect_event()]
        events_b = [_raw_disfluency_event()]

        pkg_a = _make_event_package(events_output, "sess-1", "arch-a", events_a)
        pkg_b = _make_event_package(events_output, "sess-1", "arch-b", events_b)

        from audio_classification_playground.acoustic_events.event_packages.package import (
            load_event_package,
        )
        for pkg_dir in (pkg_a, pkg_b):
            pkg = load_event_package(pkg_dir)
            append_completion_row(
                events_base=events_output,
                worker_id="test-worker",
                package_dir=pkg_dir,
                package_payload=pkg.package,
            )

        import pyarrow as pa
        entities = [
            {"session_id": "sess-1", "archive_id": "arch-a", "file_parent_dir": "/tmp", "date": "2025-02-19"},
            {"session_id": "sess-1", "archive_id": "arch-b", "file_parent_dir": "/tmp", "date": "2025-02-19"},
        ]
        table = pa.Table.from_pylist(entities)
        pq.write_table(table, manifest_path)

        return manifest_path, events_output, store_output

    def test_complete_session_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, events_output, store_output = self._setup_scenario(Path(tmp))
            summary = populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
            )
            assert summary["sessions_written"] == 1
            assert summary["sessions_incomplete"] == 0

            parquet_path = store_output / "2025-02-19.parquet"
            assert parquet_path.is_file()
            table = pq.read_table(parquet_path)
            assert table.num_rows == 1
            assert table.column("session_id")[0].as_py() == "sess-1"

            data = json.loads(table.column("data")[0].as_py())
            assert data["session_id"] == "sess-1"
            assert len(data["event_items"]) == 2

    def test_noop_rerun_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, events_output, store_output = self._setup_scenario(Path(tmp))
            populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
            )
            summary = populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
            )
            assert summary["sessions_unchanged"] == 1
            assert summary["sessions_written"] == 0
            assert summary["sessions_updated"] == 0
            assert summary["dates_skipped_no_delta"] == 1

    def test_force_rewrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, events_output, store_output = self._setup_scenario(Path(tmp))
            populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
            )
            summary = populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
                force=True,
            )
            assert summary["sessions_updated"] == 1
            assert summary["sessions_unchanged"] == 0

    def test_incomplete_session_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            events_output = tmp / "events"
            store_output = tmp / "store"
            manifest_path = tmp / "manifest.parquet"

            # Only create package for one archive
            _make_event_package(events_output, "sess-1", "arch-a", [_raw_emotion_event()])
            from audio_classification_playground.acoustic_events.event_packages.package import (
                load_event_package,
            )
            pkg = load_event_package(events_output / "sess-1" / "arch-a")
            append_completion_row(
                events_base=events_output,
                worker_id="test-worker",
                package_dir=events_output / "sess-1" / "arch-a",
                package_payload=pkg.package,
            )

            # Manifest says session has TWO archives
            import pyarrow as pa
            entities = [
                {"session_id": "sess-1", "archive_id": "arch-a", "file_parent_dir": "/tmp", "date": "2025-02-19"},
                {"session_id": "sess-1", "archive_id": "arch-b", "file_parent_dir": "/tmp", "date": "2025-02-19"},
            ]
            table = pa.Table.from_pylist(entities)
            pq.write_table(table, manifest_path)

            summary = populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
            )
            assert summary["sessions_incomplete"] == 1
            assert summary["sessions_written"] == 0
            assert not (store_output / "2025-02-19.parquet").exists()

    def test_load_failed_gracefully_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            events_output = tmp / "events"
            store_output = tmp / "store"
            manifest_path = tmp / "manifest.parquet"

            # Create packages for both archives
            _make_event_package(events_output, "sess-1", "arch-a", [_raw_emotion_event()])
            _make_event_package(events_output, "sess-1", "arch-b", [_raw_disfluency_event()])

            from audio_classification_playground.acoustic_events.event_packages.package import (
                load_event_package,
            )
            for aid in ("arch-a", "arch-b"):
                pkg = load_event_package(events_output / "sess-1" / aid)
                append_completion_row(
                    events_base=events_output,
                    worker_id="test-worker",
                    package_dir=events_output / "sess-1" / aid,
                    package_payload=pkg.package,
                )

            # Now corrupt arch-b's package
            (events_output / "sess-1" / "arch-b" / "package.json").write_text("CORRUPT")

            import pyarrow as pa
            entities = [
                {"session_id": "sess-1", "archive_id": "arch-a", "file_parent_dir": "/tmp", "date": "2025-02-19"},
                {"session_id": "sess-1", "archive_id": "arch-b", "file_parent_dir": "/tmp", "date": "2025-02-19"},
            ]
            table = pa.Table.from_pylist(entities)
            pq.write_table(table, manifest_path)

            summary = populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
            )
            assert summary["sessions_load_failed"] == 1
            assert summary["sessions_written"] == 0

    def test_date_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            events_output = tmp / "events"
            store_output = tmp / "store"
            manifest_path = tmp / "manifest.parquet"

            # Session on two dates
            _make_event_package(events_output, "sess-1", "arch-a", [_raw_emotion_event()], date="2025-02-19")
            _make_event_package(events_output, "sess-1", "arch-b", [_raw_disfluency_event()], date="2025-02-20")

            from audio_classification_playground.acoustic_events.event_packages.package import (
                load_event_package,
            )
            for aid, date in [("arch-a", "2025-02-19"), ("arch-b", "2025-02-20")]:
                pkg = load_event_package(events_output / "sess-1" / aid)
                append_completion_row(
                    events_base=events_output,
                    worker_id="test-worker",
                    package_dir=events_output / "sess-1" / aid,
                    package_payload=pkg.package,
                )

            import pyarrow as pa
            entities = [
                {"session_id": "sess-1", "archive_id": "arch-a", "file_parent_dir": "/tmp", "date": "2025-02-19"},
                {"session_id": "sess-1", "archive_id": "arch-b", "file_parent_dir": "/tmp", "date": "2025-02-20"},
            ]
            table = pa.Table.from_pylist(entities)
            pq.write_table(table, manifest_path)

            # Only process one date
            summary = populate_session_store(
                manifest_path=manifest_path,
                events_output=events_output,
                store_output=store_output,
                dates=["2025-02-19"],
            )
            assert summary["sessions_written"] == 1
            assert (store_output / "2025-02-19.parquet").is_file()
            assert not (store_output / "2025-02-20.parquet").exists()


if __name__ == "__main__":
    unittest.main()
