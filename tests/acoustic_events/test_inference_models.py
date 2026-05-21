import unittest
from unittest.mock import patch

from audio_classification_playground.acoustic_events.inference import models


class ModelSuiteLoadSelectionTest(unittest.TestCase):
    def test_empty_tasks_to_load_means_no_gpu_models(self):
        constructed = []

        def make_constructor(name):
            def _ctor(*args, **kwargs):
                constructed.append(name)
                return object()

            return _ctor

        with patch.object(models, "configure_torch_matmul"), patch.object(
            models, "AffectPredictor", side_effect=make_constructor("affect")
        ), patch.object(
            models, "DisfluencyPredictor", side_effect=make_constructor("disfluency")
        ), patch.object(
            models, "EmotionPredictor", side_effect=make_constructor("emotion")
        ), patch.object(
            models, "VadDetector", side_effect=make_constructor("vad")
        ):
            suite = models.ModelSuite(
                affect_backbone="wavlm",
                disfluency_backbone="wavlm",
                tasks_to_load=(),
                load_vad=False,
            )

        self.assertEqual(constructed, [])
        self.assertIsNone(suite.affect)
        self.assertIsNone(suite.disfluency)
        self.assertIsNone(suite.emotion)

    def test_empty_tasks_to_load_can_still_load_sync_vad(self):
        constructed = []

        def make_constructor(name):
            def _ctor(*args, **kwargs):
                constructed.append(name)
                return object()

            return _ctor

        with patch.object(models, "configure_torch_matmul"), patch.object(
            models, "AffectPredictor", side_effect=make_constructor("affect")
        ), patch.object(
            models, "DisfluencyPredictor", side_effect=make_constructor("disfluency")
        ), patch.object(
            models, "EmotionPredictor", side_effect=make_constructor("emotion")
        ), patch.object(
            models, "VadDetector", side_effect=make_constructor("vad")
        ):
            suite = models.ModelSuite(
                affect_backbone="wavlm",
                disfluency_backbone="wavlm",
                tasks_to_load=(),
                load_vad=True,
            )

        self.assertEqual(constructed, ["vad"])
        self.assertIsNotNone(suite.vad)

    def test_none_tasks_to_load_keeps_legacy_default_models(self):
        constructed = []

        def make_constructor(name):
            def _ctor(*args, **kwargs):
                constructed.append(name)
                return object()

            return _ctor

        with patch.object(models, "configure_torch_matmul"), patch.object(
            models, "AffectPredictor", side_effect=make_constructor("affect")
        ), patch.object(
            models, "DisfluencyPredictor", side_effect=make_constructor("disfluency")
        ), patch.object(
            models, "EmotionPredictor", side_effect=make_constructor("emotion")
        ), patch.object(
            models, "VadDetector", side_effect=make_constructor("vad")
        ):
            models.ModelSuite(
                affect_backbone="wavlm",
                disfluency_backbone="wavlm",
                tasks_to_load=None,
                load_vad=False,
            )

        self.assertEqual(constructed, ["affect", "disfluency", "emotion"])


if __name__ == "__main__":
    unittest.main()
