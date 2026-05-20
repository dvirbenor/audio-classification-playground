import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from audio_classification_playground.acoustic_events.orchestration.timings import (
    DEFAULT_TIMING_FIELDS,
    FieldStats,
    derive_vad_mode,
    format_timing_csv,
    format_timing_summary,
    load_timing_records,
    summarize_timings,
    summarize_timings_by_worker,
)


def _sample_record(**overrides):
    base = {
        "worker_id": "pod-a_1234",
        "session_id": "s1",
        "archive_id": "a1",
        "ts": "2026-05-15T01:00:00Z",
        "s3_key": "accounts/studio/takes/a1.wav",
        "audio_source_extension": ".wav",
        "audio_object_size_bytes": 123,
        "audio_storage_class": "STANDARD",
        "audio_duration_sec": 10.0,
        "prefetch_scheduler_wait_sec": 0.08,
        "prefetch_get_wait_sec": 0.02,
        "prefetch_wait_sec": 0.1,
        "decode_queue_wait_sec": 0.03,
        "download_decode_sec": 0.5,
        "vad_queue_wait_sec": 0.04,
        "vad_precompute_sec": 0.3,
        "prefetch_submit_to_ready_sec": 0.9,
        "prefetch_ready_age_sec": 0.2,
        "precomputed_vad": True,
        "vad_reused": False,
        "affect_reused": False,
        "disfluency_reused": False,
        "emotion_reused": False,
        "vad_sec": 0.01,
        "affect_sec": 1.0,
        "disfluency_sec": 2.0,
        "emotion_sec": 0.5,
        "inference_sec": 3.51,
        "total_sec": 4.11,
    }
    base.update(overrides)
    return base


class TestLoadTimingRecords(unittest.TestCase):
    def test_skips_corrupt_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            timings_dir = Path(tmpdir) / "_meta" / "timings"
            timings_dir.mkdir(parents=True)
            jsonl = timings_dir / "worker1.jsonl"
            good = _sample_record()
            jsonl.write_text(
                json.dumps(good) + "\n"
                "NOT VALID JSON\n"
                + json.dumps(_sample_record(archive_id="a2")) + "\n",
                encoding="utf-8",
            )
            records = load_timing_records(tmpdir)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["archive_id"], "a1")
            self.assertEqual(records[1]["archive_id"], "a2")

    def test_returns_empty_when_no_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = load_timing_records(tmpdir)
            self.assertEqual(records, [])

    def test_reads_multiple_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            timings_dir = Path(tmpdir) / "_meta" / "timings"
            timings_dir.mkdir(parents=True)
            (timings_dir / "w1.jsonl").write_text(
                json.dumps(_sample_record(worker_id="w1")) + "\n", encoding="utf-8",
            )
            (timings_dir / "w2.jsonl").write_text(
                json.dumps(_sample_record(worker_id="w2")) + "\n", encoding="utf-8",
            )
            records = load_timing_records(tmpdir)
            self.assertEqual(len(records), 2)
            worker_ids = {r["worker_id"] for r in records}
            self.assertEqual(worker_ids, {"w1", "w2"})


class TestDeriveVadMode(unittest.TestCase):
    def test_prefetched(self):
        self.assertEqual(
            derive_vad_mode({"precomputed_vad": True, "vad_reused": False}),
            "prefetched",
        )

    def test_cached(self):
        self.assertEqual(
            derive_vad_mode({"precomputed_vad": False, "vad_reused": True}),
            "cached",
        )

    def test_inline(self):
        self.assertEqual(
            derive_vad_mode({"precomputed_vad": False, "vad_reused": False}),
            "inline",
        )


class TestSummarizeTimings(unittest.TestCase):
    def test_percentiles_with_known_values(self):
        records = [_sample_record(affect_sec=float(i)) for i in range(1, 101)]
        summary = summarize_timings(records, fields=("affect_sec",))
        stats = summary["affect_sec"]
        self.assertEqual(stats.count, 100)
        self.assertAlmostEqual(stats.mean, 50.5)
        self.assertAlmostEqual(stats.min, 1.0)
        self.assertAlmostEqual(stats.max, 100.0)
        self.assertAlmostEqual(stats.p50, 50.0)
        self.assertAlmostEqual(stats.p90, 90.0)
        self.assertAlmostEqual(stats.p99, 99.0)

    def test_single_record(self):
        records = [_sample_record(affect_sec=5.0)]
        summary = summarize_timings(records, fields=("affect_sec",))
        stats = summary["affect_sec"]
        self.assertEqual(stats.count, 1)
        self.assertAlmostEqual(stats.mean, 5.0)
        self.assertAlmostEqual(stats.std, 0.0)
        self.assertAlmostEqual(stats.min, 5.0)
        self.assertAlmostEqual(stats.max, 5.0)
        self.assertAlmostEqual(stats.p50, 5.0)

    def test_empty_records(self):
        summary = summarize_timings([], fields=("affect_sec",))
        stats = summary["affect_sec"]
        self.assertEqual(stats.count, 0)

    def test_bool_fields_excluded(self):
        records = [_sample_record()]
        summary = summarize_timings(records)
        self.assertNotIn("precomputed_vad", summary)

    def test_mixed_old_and_new_records_do_not_fake_missing_new_fields(self):
        legacy = _sample_record()
        for field in (
            "prefetch_scheduler_wait_sec",
            "prefetch_get_wait_sec",
            "decode_queue_wait_sec",
            "vad_queue_wait_sec",
            "prefetch_submit_to_ready_sec",
            "prefetch_ready_age_sec",
        ):
            legacy.pop(field)
        modern = _sample_record(prefetch_scheduler_wait_sec=2.5)
        summary = summarize_timings(
            [legacy, modern],
            fields=("prefetch_scheduler_wait_sec", "prefetch_wait_sec"),
        )
        self.assertEqual(summary["prefetch_scheduler_wait_sec"].count, 1)
        self.assertAlmostEqual(summary["prefetch_scheduler_wait_sec"].mean, 2.5)
        self.assertEqual(summary["prefetch_wait_sec"].count, 2)

    def test_default_fields_all_present(self):
        records = [_sample_record()]
        summary = summarize_timings(records)
        self.assertEqual(set(summary.keys()), set(DEFAULT_TIMING_FIELDS))


class TestSummarizeByWorker(unittest.TestCase):
    def test_groups_correctly(self):
        records = [
            _sample_record(worker_id="w1", affect_sec=1.0),
            _sample_record(worker_id="w1", affect_sec=3.0),
            _sample_record(worker_id="w2", affect_sec=10.0),
        ]
        by_worker = summarize_timings_by_worker(records, fields=("affect_sec",))
        self.assertEqual(set(by_worker.keys()), {"w1", "w2"})
        self.assertEqual(by_worker["w1"]["affect_sec"].count, 2)
        self.assertAlmostEqual(by_worker["w1"]["affect_sec"].mean, 2.0)
        self.assertEqual(by_worker["w2"]["affect_sec"].count, 1)
        self.assertAlmostEqual(by_worker["w2"]["affect_sec"].mean, 10.0)


class TestFormatTimingSummary(unittest.TestCase):
    def test_non_empty_output(self):
        records = [_sample_record()]
        summary = summarize_timings(records)
        output = format_timing_summary(summary, title="Test")
        self.assertIn("Test", output)
        self.assertIn("affect_sec", output)
        self.assertIn("count", output)

    def test_empty_summary(self):
        output = format_timing_summary({})
        self.assertEqual(output, "No timing data.\n")


class TestFormatTimingCsv(unittest.TestCase):
    def test_csv_header_and_row_count(self):
        records = [
            _sample_record(archive_id="a1"),
            _sample_record(archive_id="a2"),
        ]
        csv_output = format_timing_csv(records)
        lines = [l for l in csv_output.strip().split("\n") if l]
        self.assertEqual(len(lines), 3)  # header + 2 rows
        header = lines[0].split(",")
        self.assertIn("worker_id", header)
        self.assertIn("vad_mode", header)
        self.assertIn("s3_key", header)
        self.assertIn("audio_object_size_bytes", header)
        self.assertIn("prefetch_scheduler_wait_sec", header)
        self.assertIn("affect_sec", header)

    def test_csv_includes_derived_vad_mode(self):
        records = [_sample_record(precomputed_vad=True)]
        csv_output = format_timing_csv(records)
        lines = csv_output.strip().split("\n")
        header = lines[0].split(",")
        vad_mode_idx = header.index("vad_mode")
        row = lines[1].split(",")
        self.assertEqual(row[vad_mode_idx], "prefetched")


class TestCliTimings(unittest.TestCase):
    def _write_records(self, tmpdir, records):
        timings_dir = Path(tmpdir) / "_meta" / "timings"
        timings_dir.mkdir(parents=True)
        jsonl = timings_dir / "test_worker.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_cli_timings_csv_output(self):
        from audio_classification_playground.acoustic_events.orchestration.cli import main
        import sys

        records = [_sample_record(), _sample_record(archive_id="a2")]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_records(tmpdir, records)
            captured = StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = captured
                main(["timings", "--output", tmpdir, "--csv", "--no-split-by-vad-mode"])
            finally:
                sys.stdout = old_stdout
            output = captured.getvalue()
            lines = [l for l in output.strip().split("\n") if l]
            self.assertEqual(len(lines), 3)  # header + 2 rows
            self.assertIn("affect_sec", lines[0])

    def test_cli_timings_filter_by_audio_duration(self):
        from audio_classification_playground.acoustic_events.orchestration.cli import main
        import sys

        records = [
            _sample_record(audio_duration_sec=5.0, archive_id="short"),
            _sample_record(audio_duration_sec=60.0, archive_id="long"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_records(tmpdir, records)
            captured = StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = captured
                main([
                    "timings", "--output", tmpdir, "--csv",
                    "--min-audio-sec", "10", "--no-split-by-vad-mode",
                ])
            finally:
                sys.stdout = old_stdout
            output = captured.getvalue()
            lines = [l for l in output.strip().split("\n") if l]
            self.assertEqual(len(lines), 2)  # header + 1 row (only "long")
            self.assertIn("60.0", lines[1])


if __name__ == "__main__":
    unittest.main()
