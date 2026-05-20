"""Tests for orchestration progress scanning, grouped error summaries, and --fast CLI.

All filesystem tests use ``tempfile.TemporaryDirectory`` with deterministic
layouts.  Concurrent-removal tolerance is verified via monkeypatched
``os.scandir`` that raises ``FileNotFoundError`` mid-iteration.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_classification_playground.acoustic_events.orchestration.errors import (
    AUDIO_ERRORS_DIR,
    INFERENCE_ERRORS_DIR,
    ErrorGroup,
    summarize_errors_grouped,
)
from audio_classification_playground.acoustic_events.orchestration.manifest import (
    ArchiveEntity,
)
from audio_classification_playground.acoustic_events.orchestration.progress import (
    TASKS,
    QuickSummary,
    _walk_completed_tasks,
    _walk_completed_tasks_scandir,
    quick_disk_summary,
    scan_progress,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task_artifacts(base: Path, sid: str, aid: str, tasks: list[str]) -> None:
    """Create manifest.json + predictions.npz for each task."""
    for task in tasks:
        task_dir = base / sid / aid / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "manifest.json").write_text("{}")
        (task_dir / "predictions.npz").write_text("data")


def _make_audio_error(base: Path, sid: str, aid: str, error_type: str,
                      is_permanent: bool, detail: str = "some detail",
                      filename: str | None = None) -> None:
    errors_dir = base / AUDIO_ERRORS_DIR
    errors_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"{sid}__{aid}__{error_type}.json"
    payload = {
        "session_id": sid,
        "archive_id": aid,
        "error_type": error_type,
        "detail": detail,
        "is_permanent": is_permanent,
    }
    (errors_dir / fname).write_text(json.dumps(payload))


def _make_inference_error(base: Path, sid: str, aid: str, error_type: str,
                          detail: str = "boom",
                          filename: str | None = None) -> None:
    errors_dir = base / INFERENCE_ERRORS_DIR
    errors_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"{sid}__{aid}__{error_type}.json"
    payload = {
        "session_id": sid,
        "archive_id": aid,
        "error_type": error_type,
        "detail": detail,
    }
    (errors_dir / fname).write_text(json.dumps(payload))


def _make_lock(base: Path, sid: str, aid: str) -> None:
    locks_dir = base / "_meta" / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    (locks_dir / f"{sid}__{aid}.lock").write_text("worker=test\n")


def _entity(sid: str, aid: str) -> ArchiveEntity:
    return ArchiveEntity(session_id=sid, archive_id=aid, file_parent_dir="")


# ===========================================================================
# Step 2: Walk-based progress tests
# ===========================================================================


class TestWalkCompletedTasks(unittest.TestCase):

    def test_missing_output_base_returns_empty(self):
        result = _walk_completed_tasks(Path("/nonexistent/path"))
        self.assertEqual(result, {})

    def test_empty_output_base(self):
        with tempfile.TemporaryDirectory() as d:
            result = _walk_completed_tasks(Path(d))
            self.assertEqual(result, {})

    def test_fully_complete_archive(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            result = _walk_completed_tasks(base)
            self.assertEqual(result, {("s1", "a1"): set(TASKS)})

    def test_partially_complete_archive(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", ["vad", "affect"])
            result = _walk_completed_tasks(base)
            self.assertEqual(result, {("s1", "a1"): {"vad", "affect"}})

    def test_no_tasks_complete_excluded(self):
        """Archive dir exists but has no valid task dirs."""
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "s1" / "a1").mkdir(parents=True)
            result = _walk_completed_tasks(base)
            self.assertEqual(result, {})

    def test_incomplete_task_missing_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            task_dir = base / "s1" / "a1" / "vad"
            task_dir.mkdir(parents=True)
            (task_dir / "manifest.json").write_text("{}")
            result = _walk_completed_tasks(base)
            self.assertEqual(result, {})

    def test_meta_directory_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            # _meta should be ignored even if it looks like a session
            meta_dir = base / "_meta" / "fake_archive" / "vad"
            meta_dir.mkdir(parents=True)
            (meta_dir / "manifest.json").write_text("{}")
            (meta_dir / "predictions.npz").write_text("data")
            result = _walk_completed_tasks(base)
            self.assertNotIn(("_meta", "fake_archive"), result)
            self.assertEqual(len(result), 1)

    def test_underscore_prefixed_session_not_skipped(self):
        """Session IDs starting with _ (but not == _meta) are valid."""
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "_special_session", "a1", list(TASKS))
            result = _walk_completed_tasks(base)
            self.assertIn(("_special_session", "a1"), result)

    def test_multiple_archives_across_sessions(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            _make_task_artifacts(base, "s1", "a2", ["vad"])
            _make_task_artifacts(base, "s2", "a3", list(TASKS))
            result = _walk_completed_tasks(base)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[("s1", "a1")], set(TASKS))
            self.assertEqual(result[("s1", "a2")], {"vad"})
            self.assertEqual(result[("s2", "a3")], set(TASKS))

    def test_files_in_session_dir_ignored(self):
        """Regular files at the session level should not crash the walker."""
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            (base / "s1" / "stray_file.txt").write_text("noise")
            result = _walk_completed_tasks(base)
            self.assertEqual(len(result), 1)


class TestWalkConcurrentRemoval(unittest.TestCase):
    """Verify the walker tolerates directories vanishing mid-iteration."""

    def test_archive_dir_removed_during_scan(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            _make_task_artifacts(base, "s1", "a2", list(TASKS))

            real_scandir = os.scandir
            call_count = {"archives": 0}

            class _FailingScandirCtx:
                """Wraps a real scandir iterator, raising on the second archive."""

                def __init__(self, it):
                    self._it = it

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    self._it.close()

                def __iter__(self):
                    for entry in self._it:
                        if entry.is_dir(follow_symlinks=False):
                            call_count["archives"] += 1
                            if call_count["archives"] == 2:
                                raise FileNotFoundError("vanished")
                        yield entry

            def patched_scandir(path):
                ctx = real_scandir(path)
                p = str(path)
                # Only intercept the session-level scandir (contains archive dirs)
                if p.endswith("s1"):
                    return _FailingScandirCtx(ctx)
                return ctx

            with patch("audio_classification_playground.acoustic_events."
                        "orchestration.progress.os.scandir", side_effect=patched_scandir):
                result = _walk_completed_tasks_scandir(base)

            # One archive should still be captured despite the error
            self.assertEqual(len(result), 1)


class TestScanProgress(unittest.TestCase):

    def test_full_progress_with_all_states(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)

            _make_task_artifacts(base, "s1", "complete", list(TASKS))
            _make_task_artifacts(base, "s1", "partial", ["vad", "affect"])
            _make_lock(base, "s1", "locked_one")
            # s1/error_one has a permanent audio error (no artifacts)

            entities = [
                _entity("s1", "complete"),
                _entity("s1", "partial"),
                _entity("s1", "locked_one"),
                _entity("s1", "error_one"),
                _entity("s1", "untouched"),
            ]
            perm_errors = {("s1", "error_one")}
            inf_errors = {("s1", "partial"): 2}

            summary = scan_progress(
                base, entities,
                permanent_audio_errors=perm_errors,
                inference_error_counts=inf_errors,
            )

            self.assertEqual(summary.total_entities, 5)
            self.assertEqual(summary.complete, 1)
            self.assertEqual(summary.partial, 1)
            self.assertEqual(summary.permanent_audio_errors, 1)
            self.assertEqual(summary.inference_errors_by_archive, 1)
            self.assertEqual(summary.locked, 1)
            self.assertEqual(summary.remaining, 3)  # 5 - 1 complete - 1 perm error
            self.assertEqual(summary.task_counts["vad"], 2)
            self.assertEqual(summary.task_counts["affect"], 2)
            self.assertEqual(summary.task_counts["disfluency"], 1)
            self.assertEqual(summary.task_counts["emotion"], 1)

    def test_permanent_errors_skipped_before_completion_check(self):
        """An archive with perm error should NOT be counted as complete."""
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            entities = [_entity("s1", "a1")]
            perm_errors = {("s1", "a1")}

            summary = scan_progress(
                base, entities, permanent_audio_errors=perm_errors,
            )
            self.assertEqual(summary.complete, 0)
            self.assertEqual(summary.permanent_audio_errors, 1)

    def test_empty_entities(self):
        with tempfile.TemporaryDirectory() as d:
            summary = scan_progress(Path(d), [])
            self.assertEqual(summary.total_entities, 0)
            self.assertEqual(summary.complete, 0)

    def test_missing_output_base(self):
        entities = [_entity("s1", "a1")]
        summary = scan_progress(Path("/nonexistent"), entities)
        self.assertEqual(summary.complete, 0)
        self.assertEqual(summary.remaining, 1)


# ===========================================================================
# Step 4: Grouped error summary tests
# ===========================================================================


class TestSummarizeErrorsGrouped(unittest.TestCase):

    def test_missing_dir_returns_empty(self):
        result = summarize_errors_grouped(Path("/nonexistent"))
        self.assertEqual(result, [])

    def test_single_error_type(self):
        with tempfile.TemporaryDirectory() as d:
            errors_dir = Path(d)
            for i in range(3):
                payload = {
                    "session_id": f"s{i}",
                    "archive_id": "a1",
                    "error_type": "no_matching_file",
                    "detail": f"detail {i}",
                    "is_permanent": True,
                }
                (errors_dir / f"err_{i}.json").write_text(json.dumps(payload))

            groups = summarize_errors_grouped(errors_dir)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].error_type, "no_matching_file")
            self.assertEqual(groups[0].record_count, 3)
            self.assertEqual(len(groups[0].unique_archives), 3)
            self.assertTrue(groups[0].is_permanent)

    def test_duplicate_archives_deduped(self):
        with tempfile.TemporaryDirectory() as d:
            errors_dir = Path(d)
            for i in range(5):
                payload = {
                    "session_id": "s1",
                    "archive_id": "a1",
                    "error_type": "RuntimeError",
                    "detail": f"attempt {i}",
                }
                (errors_dir / f"err_{i}.json").write_text(json.dumps(payload))

            groups = summarize_errors_grouped(errors_dir)
            self.assertEqual(groups[0].record_count, 5)
            self.assertEqual(len(groups[0].unique_archives), 1)

    def test_sorted_by_count_desc_then_type_asc(self):
        with tempfile.TemporaryDirectory() as d:
            errors_dir = Path(d)
            # 3 records of type B
            for i in range(3):
                (errors_dir / f"b_{i}.json").write_text(json.dumps({
                    "session_id": f"s{i}", "archive_id": "a1",
                    "error_type": "BBBError", "detail": "",
                }))
            # 3 records of type A (same count, should come first alphabetically)
            for i in range(3):
                (errors_dir / f"a_{i}.json").write_text(json.dumps({
                    "session_id": f"s{i}", "archive_id": "a1",
                    "error_type": "AAAError", "detail": "",
                }))
            # 1 record of type C
            (errors_dir / "c_0.json").write_text(json.dumps({
                "session_id": "s0", "archive_id": "a1",
                "error_type": "CCCError", "detail": "",
            }))

            groups = summarize_errors_grouped(errors_dir)
            types = [g.error_type for g in groups]
            self.assertEqual(types, ["AAAError", "BBBError", "CCCError"])

    def test_malformed_json_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            errors_dir = Path(d)
            (errors_dir / "good.json").write_text(json.dumps({
                "session_id": "s1", "archive_id": "a1",
                "error_type": "RuntimeError", "detail": "ok",
            }))
            (errors_dir / "bad.json").write_text("not json at all {{{")
            (errors_dir / "missing_key.json").write_text(json.dumps({
                "error_type": "RuntimeError",
            }))
            (errors_dir / "not_json.txt").write_text("ignored")

            groups = summarize_errors_grouped(errors_dir)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].record_count, 1)

    def test_multiline_detail_collapsed(self):
        with tempfile.TemporaryDirectory() as d:
            errors_dir = Path(d)
            (errors_dir / "err.json").write_text(json.dumps({
                "session_id": "s1", "archive_id": "a1",
                "error_type": "RuntimeError",
                "detail": "line one\nline two\nline three",
            }))
            groups = summarize_errors_grouped(errors_dir)
            self.assertNotIn("\n", groups[0].example_detail)


# ===========================================================================
# Step 6: Quick disk summary / --fast tests
# ===========================================================================


class TestQuickDiskSummary(unittest.TestCase):

    def test_missing_output_returns_zeros(self):
        s = quick_disk_summary(Path("/nonexistent"))
        self.assertEqual(s.complete, 0)
        self.assertEqual(s.partial, 0)
        self.assertEqual(s.lock_count, 0)
        self.assertEqual(s.audio_error_records, 0)
        self.assertEqual(s.inference_error_records, 0)

    def test_complete_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            _make_task_artifacts(base, "s1", "a2", ["vad"])

            s = quick_disk_summary(base)
            self.assertEqual(s.complete, 1)
            self.assertEqual(s.partial, 1)
            self.assertEqual(s.task_counts["vad"], 2)
            self.assertEqual(s.task_counts["affect"], 1)

    def test_locks_counted(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_lock(base, "s1", "a1")
            _make_lock(base, "s1", "a2")

            s = quick_disk_summary(base)
            self.assertEqual(s.lock_count, 2)

    def test_audio_errors_counted(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_audio_error(base, "s1", "a1", "no_matching_file",
                              is_permanent=True, filename="e1.json")
            _make_audio_error(base, "s1", "a2", "download_failed",
                              is_permanent=False, filename="e2.json")
            _make_audio_error(base, "s1", "a1", "no_matching_file",
                              is_permanent=True, filename="e3.json")

            s = quick_disk_summary(base)
            self.assertEqual(s.audio_error_records, 3)
            self.assertEqual(s.permanent_audio_error_archives, 1)

    def test_inference_errors_counted(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_inference_error(base, "s1", "a1", "RuntimeError", filename="e1.json")
            _make_inference_error(base, "s1", "a1", "RuntimeError", filename="e2.json")
            _make_inference_error(base, "s2", "a2", "OOMError", filename="e3.json")

            s = quick_disk_summary(base)
            self.assertEqual(s.inference_error_records, 3)
            self.assertEqual(s.inference_error_archives, 2)


class TestCLIProgressFast(unittest.TestCase):
    """Test argparse validation for --fast / --parquet interaction."""

    def test_fast_without_parquet_succeeds(self):
        from audio_classification_playground.acoustic_events.orchestration.cli import main

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            _make_task_artifacts(base, "s1", "a1", list(TASKS))
            main(["progress", "--output", str(base), "--fast"])

    def test_no_fast_no_parquet_exits(self):
        from audio_classification_playground.acoustic_events.orchestration.cli import main

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                main(["progress", "--output", d])


# ===========================================================================
# Glacier transient classification and s3_key in error payloads
# ===========================================================================


class TestAppendAudioErrorS3Key(unittest.TestCase):
    """Verify s3_key is included in the JSON payload written by append_audio_error."""

    def test_s3_key_written_to_json(self):
        from audio_classification_playground.acoustic_events.orchestration.audio_resolver import (
            AudioResolutionError,
        )
        from audio_classification_playground.acoustic_events.orchestration.errors import (
            append_audio_error,
        )

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            err = AudioResolutionError(
                session_id="s1",
                archive_id="a1",
                file_parent_dir="pfx",
                error_type="glacier_storage_class",
                detail="s3://bucket/key is in GLACIER",
                s3_key="accounts/studio/takes/file.wav",
            )
            path = append_audio_error(base, err)
            data = json.loads(path.read_text())

            self.assertEqual(data["s3_key"], "accounts/studio/takes/file.wav")
            self.assertFalse(data["is_permanent"])

    def test_s3_key_defaults_to_empty_for_no_matching_file(self):
        from audio_classification_playground.acoustic_events.orchestration.audio_resolver import (
            AudioResolutionError,
        )
        from audio_classification_playground.acoustic_events.orchestration.errors import (
            append_audio_error,
        )

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            err = AudioResolutionError(
                session_id="s1",
                archive_id="a1",
                file_parent_dir="pfx",
                error_type="no_matching_file",
                detail="No WAV/MP3 found",
            )
            path = append_audio_error(base, err)
            data = json.loads(path.read_text())

            self.assertEqual(data["s3_key"], "")
            self.assertTrue(data["is_permanent"])


class TestGlacierTransientInGroupedSummary(unittest.TestCase):
    """Verify glacier errors appear as transient in grouped summaries."""

    def test_glacier_is_transient_in_summary(self):
        with tempfile.TemporaryDirectory() as d:
            errors_dir = Path(d)
            payload = {
                "session_id": "s1",
                "archive_id": "a1",
                "error_type": "glacier_storage_class",
                "detail": "s3://bucket/key is in GLACIER",
                "s3_key": "key",
                "is_permanent": False,
            }
            (errors_dir / "err.json").write_text(json.dumps(payload))

            groups = summarize_errors_grouped(errors_dir)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].error_type, "glacier_storage_class")
            self.assertFalse(groups[0].is_permanent)
            self.assertEqual(groups[0].record_count, 1)

    def test_permanent_audio_error_types_excludes_glacier(self):
        from audio_classification_playground.acoustic_events.orchestration.errors import (
            PERMANENT_AUDIO_ERROR_TYPES,
        )

        self.assertNotIn("glacier_storage_class", PERMANENT_AUDIO_ERROR_TYPES)
        self.assertIn("no_matching_file", PERMANENT_AUDIO_ERROR_TYPES)


if __name__ == "__main__":
    unittest.main()
