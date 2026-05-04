import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from audio_classification_playground.acoustic_events import (
    Config,
    Event,
    MarkerItem,
    MarkerTrack,
    ProducerRun,
    RegularGridTrack,
    Vad,
)
from audio_classification_playground.acoustic_events.review.inherit import inherit_labels
from audio_classification_playground.acoustic_events.review.server import make_app
from audio_classification_playground.acoustic_events.review.storage import save_session


def event(
    event_id,
    *,
    producer_id="mock.v1",
    task="emotion",
    label="anger",
    event_type="categorical",
    start=1.0,
    end=2.0,
    source_track_ids=("emotion.anger",),
    direction=None,
    children=(),
):
    return Event(
        event_id=event_id,
        producer_id=producer_id,
        task=task,
        event_type=event_type,
        label=label,
        start_sec=start,
        end_sec=end,
        duration_sec=end - start,
        source_track_ids=source_track_ids,
        score=0.9,
        score_name="probability",
        direction=direction,
        children=children,
    )


class GenericReviewStorageTest(unittest.TestCase):
    def test_save_load_mixed_tracks_and_generic_events(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "sample.wav"
            sf.write(str(audio), np.zeros(16000, dtype=np.float32), 16000)
            tracks = [
                RegularGridTrack(
                    track_id="emotion.anger",
                    producer_id="emotion.v1",
                    task="emotion",
                    name="anger",
                    value_type="probability",
                    renderer="probability",
                    values=np.array([0.1, 0.8, 0.2]),
                    hop_sec=0.5,
                    window_sec=1.0,
                ),
                RegularGridTrack(
                    track_id="disfluency.type",
                    producer_id="disfluency.v1",
                    task="disfluency",
                    name="type",
                    value_type="probability",
                    renderer="multi_probability",
                    values=np.array([[0.1, 0.9], [0.8, 0.2]]),
                    hop_sec=0.25,
                    window_sec=0.5,
                    channels=("filled_pause", "repetition"),
                ),
                MarkerTrack(
                    track_id="vocalization.markers",
                    producer_id="vocalization.v1",
                    task="vocalization",
                    name="vocalization markers",
                    renderer="marker",
                    items=(MarkerItem(0.2, None, "laugh", 0.7),),
                ),
            ]
            producer_runs = [
                ProducerRun("emotion.v1", "emotion", "mock-emotion", config_hash="abc"),
                ProducerRun("disfluency.v1", "disfluency", "mock-disfluency", config_hash="def"),
                ProducerRun("vocalization.v1", "vocalization", "mock-vocalization", config_hash="ghi"),
            ]

            path = save_session(
                events=[event("emotion.v1.categorical.000001")],
                tracks=tracks,
                producer_runs=producer_runs,
                vad=Vad(intervals=((0.0, 1.0),)),
                audio_path=audio,
                session_dir=root / "sessions",
            )

            self.assertRegex(path.name, r"^\d{8}T\d{6}Z__[0-9a-f]{8}\.json$")
            data = json.loads(path.read_text())
            self.assertEqual(data["event_schema"], "acoustic_events.v1")
            self.assertIn("session_fingerprint", data)
            self.assertNotIn("blocks", data)
            self.assertIn("tracks_meta", data)
            self.assertEqual(data["tracks_meta"]["disfluency.type"]["channels"], ["filled_pause", "repetition"])
            self.assertEqual(data["tracks_meta"]["vocalization.markers"]["items"][0]["label"], "laugh")

            with self.assertRaises(FileNotFoundError):
                make_app(path)

    def test_tracks_only_session_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "sample.wav"
            sf.write(str(audio), np.zeros(8000, dtype=np.float32), 8000)
            path = save_session(
                events=[],
                tracks=[
                    RegularGridTrack(
                        track_id="emotion.anger",
                        producer_id="emotion.v1",
                        task="emotion",
                        name="anger",
                        value_type="probability",
                        renderer="probability",
                        values=np.array([0.1, 0.2]),
                        hop_sec=0.5,
                        window_sec=1.0,
                    )
                ],
                producer_runs=[ProducerRun("emotion.v1", "emotion", "mock")],
                vad=Vad(intervals=((0.0, 0.5),)),
                audio_path=audio,
                session_dir=root / "sessions",
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["events"], [])

    def test_affect_blocks_are_producer_scoped_when_config_is_supplied(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "sample.wav"
            sf.write(str(audio), np.zeros(16000, dtype=np.float32), 16000)
            path = save_session(
                events=[],
                tracks=[
                    RegularGridTrack(
                        track_id="affect.arousal",
                        producer_id="affect.default",
                        task="affect",
                        name="arousal",
                        value_type="continuous",
                        renderer="line",
                        values=np.zeros(8),
                        hop_sec=0.25,
                        window_sec=1.0,
                    )
                ],
                vad=Vad(intervals=((0.0, 2.0),)),
                config=Config(min_speech_block_sec=0.1),
                audio_path=audio,
                session_dir=root / "sessions",
            )
            data = json.loads(path.read_text())
            self.assertNotIn("blocks", data)
            self.assertEqual(data["producer_runs"][0]["producer_id"], "affect.default")
            self.assertEqual(data["producer_runs"][0]["outputs"]["blocks"][0]["start_sec"], 0.0)

    def test_inheritance_uses_semantic_key_and_joint_child_signature(self):
        prev_events = [
            event("old-emotion", start=10, end=12),
            event(
                "old-a",
                task="affect",
                label="arousal_deviation",
                event_type="deviation",
                source_track_ids=("affect.arousal",),
                direction="+",
                start=20,
                end=22,
            ),
            event(
                "old-v",
                task="affect",
                label="valence_deviation",
                event_type="deviation",
                source_track_ids=("affect.valence",),
                direction="-",
                start=20,
                end=22,
            ),
            event(
                "old-joint",
                task="affect",
                label="joint",
                event_type="joint",
                source_track_ids=("affect.arousal", "affect.valence"),
                start=20,
                end=22,
                children=("old-a", "old-v"),
            ),
        ]
        prev = {
            "session_id": "prev",
            "events": [e.as_dict() for e in prev_events],
            "labels": {
                "old-emotion": {"verdict": "tp"},
                "old-joint": {"verdict": "fp"},
            },
        }

        inherited = inherit_labels(
            prev_session=prev,
            new_events=[
                event("new-emotion", start=10.2, end=12.2).as_dict(),
                event(
                    "new-a",
                    task="affect",
                    label="arousal_deviation",
                    event_type="deviation",
                    source_track_ids=("affect.arousal",),
                    direction="+",
                    start=20.1,
                    end=22.1,
                ).as_dict(),
                event(
                    "new-d",
                    task="affect",
                    label="dominance_deviation",
                    event_type="deviation",
                    source_track_ids=("affect.dominance",),
                    direction="-",
                    start=20.1,
                    end=22.1,
                ).as_dict(),
                event(
                    "new-joint",
                    task="affect",
                    label="joint",
                    event_type="joint",
                    source_track_ids=("affect.arousal", "affect.dominance"),
                    start=20.1,
                    end=22.1,
                    children=("new-a", "new-d"),
                ).as_dict(),
            ],
        )

        self.assertIn("new-emotion", inherited)
        self.assertNotIn("new-joint", inherited)


if __name__ == "__main__":
    unittest.main()
