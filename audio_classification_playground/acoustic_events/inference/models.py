"""Persistent model wrappers for reuse across multiple inference calls.

Each class loads a model once in ``__init__`` and implements ``__call__``
matching the predictor/detector callable interface expected by the
inference runners.  This allows an orchestrator to keep models resident
in GPU memory across thousands of files.

Example::

    from audio_classification_playground.acoustic_events.inference.models import ModelSuite

    models = ModelSuite(
        affect_backbone="wavlm",
        disfluency_backbone="whisper",
        batch_size=512,
        device="cuda",
    )
    # models.affect, models.disfluency, models.emotion are persistent
    # callables. models.vad is loaded lazily when requested.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np

from ...vox_profile.wavlm_inference import (
    autocast_context,
    compile_wavlm_backbone,
    validate_autocast_dtype,
)
from .artifacts import SAMPLE_RATE
from .audio import writable_contiguous_float32
from .emotion2vec import (
    make_direct_emotion2vec_scorer,
    predict_emotion2vec_scores,
    predict_emotion2vec_scores_from_audio,
)
from .emotion_runtime import (
    DEFAULT_EMOTION_COMPILE_MODE,
    OPTIMIZED_EMOTION_BATCH_SIZE,
    resolve_emotion_runtime_settings,
    torch_matmul_precision,
)
from .log import get_logger
from .runners import (
    DEFAULT_AFFECT_MODELS,
    DEFAULT_DISFLUENCY_MODELS,
    DEFAULT_EMOTION_MODEL,
    EMOTION_WINDOW_SEC,
    DEFAULT_VAD_MIN_SILENCE_SEC,
    DEFAULT_VAD_MIN_SPEECH_SEC,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    _load_affect_wrapper,
    _load_disfluency_wrapper,
    resolve_task_batch_sizes,
)

LOGGER = get_logger()
ProgressFn = Callable[[str], None]


def _batches(windows: np.ndarray, batch_size: int, task: str):
    if batch_size <= 0:
        raise ValueError(f"{task} batch size must be positive")
    n = len(windows)
    for start in range(0, n, batch_size):
        yield windows[start : min(start + batch_size, n)]


def configure_torch_matmul(*, allow_tf32: bool) -> None:
    """Apply process-wide torch matmul precision knobs for inference workers."""
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except Exception:
        LOGGER.warning("Could not configure torch matmul settings", exc_info=True)


class AffectPredictor:
    """Persistent Vox-Profile dimensional affect model.

    Matches ``Callable[[np.ndarray], dict[str, np.ndarray]]``.
    """

    def __init__(
        self,
        backbone: str,
        model_id: str | None = None,
        device: str | None = None,
        batch_size: int = 512,
        wavlm_autocast_dtype: str | None = None,
        wavlm_compile: bool = False,
        wavlm_compile_mode: str = "reduce-overhead",
        wavlm_compile_dynamic: bool = False,
        wavlm_stream_layer_sum: bool = False,
    ) -> None:
        import torch

        self.backbone = backbone
        self.batch_size = batch_size
        self.wavlm_autocast_dtype = (
            validate_autocast_dtype(wavlm_autocast_dtype)
            if backbone == "wavlm"
            else None
        )
        self.wavlm_compile = bool(wavlm_compile and backbone == "wavlm")
        self.wavlm_stream_layer_sum = bool(wavlm_stream_layer_sum and backbone == "wavlm")
        resolved_id = model_id or DEFAULT_AFFECT_MODELS[backbone]
        self.model_id = resolved_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = _load_affect_wrapper(backbone)
        self._model = wrapper.from_pretrained(resolved_id).to(self._device).eval()
        if self.wavlm_stream_layer_sum:
            self._model.wavlm_stream_layer_sum = True
        if self.wavlm_compile:
            compile_wavlm_backbone(
                self._model,
                mode=wavlm_compile_mode,
                dynamic=wavlm_compile_dynamic,
            )
        LOGGER.info(
            "AffectPredictor loaded: %s on %s autocast=%s compile=%s stream_layer_sum=%s",
            resolved_id,
            self._device,
            self.wavlm_autocast_dtype,
            self.wavlm_compile,
            self.wavlm_stream_layer_sum,
        )

    def __call__(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        arousal, valence, dominance = [], [], []
        with torch.inference_mode():
            for batch_np in _batches(windows, self.batch_size, "affect"):
                batch = torch.from_numpy(writable_contiguous_float32(batch_np))
                if self.backbone != "wavlm":
                    batch = batch.to(self._device)
                with autocast_context(torch, self._device, self.wavlm_autocast_dtype):
                    a, v, d = self._model(batch)
                arousal.append(a.detach().float().reshape(-1))
                valence.append(v.detach().float().reshape(-1))
                dominance.append(d.detach().float().reshape(-1))
        return {
            "arousal": torch.cat(arousal).cpu().numpy(),
            "valence": torch.cat(valence).cpu().numpy(),
            "dominance": torch.cat(dominance).cpu().numpy(),
        }


class DisfluencyPredictor:
    """Persistent Vox-Profile disfluency model.

    Matches ``Callable[[np.ndarray], dict[str, np.ndarray]]``.
    """

    def __init__(
        self,
        backbone: str,
        model_id: str | None = None,
        device: str | None = None,
        batch_size: int = 512,
        wavlm_autocast_dtype: str | None = None,
        wavlm_compile: bool = False,
        wavlm_compile_mode: str = "reduce-overhead",
        wavlm_compile_dynamic: bool = False,
        wavlm_stream_layer_sum: bool = False,
    ) -> None:
        import torch

        self.backbone = backbone
        self.batch_size = batch_size
        self.wavlm_autocast_dtype = (
            validate_autocast_dtype(wavlm_autocast_dtype)
            if backbone == "wavlm"
            else None
        )
        self.wavlm_compile = bool(wavlm_compile and backbone == "wavlm")
        self.wavlm_stream_layer_sum = bool(wavlm_stream_layer_sum and backbone == "wavlm")
        resolved_id = model_id or DEFAULT_DISFLUENCY_MODELS[backbone]
        self.model_id = resolved_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = _load_disfluency_wrapper(backbone)
        self._model = wrapper.from_pretrained(resolved_id).to(self._device).eval()
        if self.wavlm_stream_layer_sum:
            self._model.wavlm_stream_layer_sum = True
        if self.wavlm_compile:
            compile_wavlm_backbone(
                self._model,
                mode=wavlm_compile_mode,
                dynamic=wavlm_compile_dynamic,
            )
        LOGGER.info(
            "DisfluencyPredictor loaded: %s on %s autocast=%s compile=%s stream_layer_sum=%s",
            resolved_id,
            self._device,
            self.wavlm_autocast_dtype,
            self.wavlm_compile,
            self.wavlm_stream_layer_sum,
        )

    def __call__(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        fluency, dysfluency = [], []
        with torch.inference_mode():
            for batch_np in _batches(windows, self.batch_size, "disfluency"):
                batch = torch.from_numpy(writable_contiguous_float32(batch_np))
                if self.backbone != "wavlm":
                    batch = batch.to(self._device)
                with autocast_context(torch, self._device, self.wavlm_autocast_dtype):
                    f, d = self._model(batch, return_feature=False)
                fluency.append(f.detach().float())
                dysfluency.append(d.detach().float())
        return {
            "fluency_logits": torch.cat(fluency, dim=0).cpu().numpy(),
            "disfluency_type_logits": torch.cat(dysfluency, dim=0).cpu().numpy(),
        }


class EmotionPredictor:
    """Persistent emotion2vec model.

    Matches ``Callable[[np.ndarray], tuple[np.ndarray, Sequence[str]]]``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMOTION_MODEL,
        sample_rate: int = SAMPLE_RATE,
        batch_size: int = 512,
        device: str | None = None,
        autocast_dtype: str | None = None,
        compile_model: bool = False,
        compile_mode: str = DEFAULT_EMOTION_COMPILE_MODE,
        allow_tf32: bool = False,
        warmup: bool = False,
    ) -> None:
        from funasr import AutoModel

        self.model_id = model_id
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self._device = device
        self.autocast_dtype = autocast_dtype
        self.compile_model = bool(compile_model)
        self.compile_mode = compile_mode
        self.allow_tf32 = bool(allow_tf32)
        auto_kwargs = {
            "model": model_id,
            "batch_size": batch_size,
            "disable_update": True,
            "disable_pbar": True,
        }
        if device is not None:
            auto_kwargs["device"] = device
        with torch_matmul_precision(allow_tf32=self.allow_tf32):
            self._model = AutoModel(**auto_kwargs)
            self._direct_scorer = make_direct_emotion2vec_scorer(
                self._model,
                sample_rate=self.sample_rate,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
            )
            if warmup:
                self.warmup()
        LOGGER.info(
            "EmotionPredictor loaded: %s autocast=%s compile=%s tf32=%s warmup=%s",
            model_id,
            autocast_dtype,
            self.compile_model,
            self.allow_tf32,
            bool(warmup),
        )

    def warmup(self) -> None:
        """Run one fixed-shape dummy batch through the direct scorer."""
        if self._direct_scorer is None:
            return
        import torch

        window_samples = int(round(EMOTION_WINDOW_SEC * self.sample_rate))
        batch = torch.zeros(
            (int(self.batch_size), window_samples),
            dtype=torch.float32,
            device=self._direct_scorer.device,
        )
        with torch.inference_mode(), torch_matmul_precision(allow_tf32=self.allow_tf32):
            self._direct_scorer(batch, autocast_dtype=self.autocast_dtype)

    def __call__(self, windows: np.ndarray) -> tuple[np.ndarray, Sequence[str]]:
        with torch_matmul_precision(allow_tf32=self.allow_tf32):
            if self._direct_scorer is not None:
                return self._direct_scorer.predict_windows(
                    windows,
                    batch_size=self.batch_size,
                    autocast_dtype=self.autocast_dtype,
                )
            return predict_emotion2vec_scores(
                self._model,
                windows,
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
                autocast_dtype=self.autocast_dtype,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
            )

    def predict_audio(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        window_sec: float,
        hop_sec: float,
        progress: ProgressFn | None = None,
    ) -> tuple[np.ndarray, Sequence[str]]:
        if int(sample_rate) != int(self.sample_rate):
            raise ValueError(
                f"EmotionPredictor loaded for {self.sample_rate} Hz, got {sample_rate} Hz"
            )
        with torch_matmul_precision(allow_tf32=self.allow_tf32):
            if self._direct_scorer is not None:
                return self._direct_scorer.predict_audio(
                    samples,
                    sample_rate=self.sample_rate,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                    batch_size=self.batch_size,
                    autocast_dtype=self.autocast_dtype,
                    progress=progress,
                )
            return predict_emotion2vec_scores_from_audio(
                self._model,
                samples,
                sample_rate=self.sample_rate,
                window_sec=window_sec,
                hop_sec=hop_sec,
                batch_size=self.batch_size,
                autocast_dtype=self.autocast_dtype,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
                progress=progress,
            )


class VadDetector:
    """Persistent Silero VAD model (CPU).

    Matches ``Callable[[np.ndarray, int], list[tuple[float, float]]]``.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
        min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
        min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
    ) -> None:
        import torch

        self.threshold = float(threshold)
        self.min_speech_sec = float(min_speech_sec)
        self.min_silence_sec = float(min_silence_sec)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            onnx=False,
        )
        self._model = model.to("cpu")
        self._get_speech_timestamps = utils[0]
        LOGGER.info("VadDetector loaded (CPU)")

    def __call__(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float]]:
        import torch

        timestamps = self._get_speech_timestamps(
            torch.from_numpy(writable_contiguous_float32(samples)),
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=int(self.min_speech_sec * 1000),
            min_silence_duration_ms=int(self.min_silence_sec * 1000),
            return_seconds=False,
        )
        sr = float(sample_rate)
        return [(float(t["start"]) / sr, float(t["end"]) / sr) for t in timestamps]


class ModelSuite:
    """Convenience holder for persistent inference models.

    Example::

        suite = ModelSuite(
            affect_backbone="wavlm",
            disfluency_backbone="whisper",
            batch_size=512,
            device="cuda",
        )
        # suite.affect   -> AffectPredictor
        # suite.disfluency -> DisfluencyPredictor
        # suite.emotion  -> EmotionPredictor
        # suite.vad      -> VadDetector (loaded lazily unless load_vad=True)
    """

    def __init__(
        self,
        *,
        affect_backbone: str,
        disfluency_backbone: str,
        batch_size: int = 512,
        affect_batch_size: int | None = None,
        disfluency_batch_size: int | None = None,
        emotion_batch_size: int | None = None,
        device: str | None = None,
        vad_threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
        vad_min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
        vad_min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
        load_vad: bool = True,
        wavlm_autocast_dtype: str | None = None,
        wavlm_compile: bool = False,
        wavlm_compile_mode: str = "reduce-overhead",
        wavlm_compile_dynamic: bool = False,
        wavlm_stream_layer_sum: bool = False,
        emotion_autocast_dtype: str | None = None,
        emotion_compile: bool = False,
        emotion_compile_mode: str = DEFAULT_EMOTION_COMPILE_MODE,
        emotion_runtime_mode: str | None = "auto",
        allow_tf32: bool = False,
    ) -> None:
        configure_torch_matmul(allow_tf32=allow_tf32)
        emotion_settings = resolve_emotion_runtime_settings(
            mode=emotion_runtime_mode,
            default_mode="auto",
            device=device,
            autocast_dtype=emotion_autocast_dtype,
            compile_model=emotion_compile,
            compile_mode=emotion_compile_mode,
            allow_tf32=allow_tf32,
        )
        batches = resolve_task_batch_sizes(
            batch_size=batch_size,
            affect_batch_size=affect_batch_size,
            disfluency_batch_size=disfluency_batch_size,
            emotion_batch_size=(
                OPTIMIZED_EMOTION_BATCH_SIZE
                if emotion_batch_size is None and emotion_settings.mode == "optimized"
                else emotion_batch_size
            ),
        )
        self.affect = AffectPredictor(
            backbone=affect_backbone,
            device=device,
            batch_size=batches["affect"],
            wavlm_autocast_dtype=wavlm_autocast_dtype,
            wavlm_compile=wavlm_compile,
            wavlm_compile_mode=wavlm_compile_mode,
            wavlm_compile_dynamic=wavlm_compile_dynamic,
            wavlm_stream_layer_sum=wavlm_stream_layer_sum,
        )
        self.disfluency = DisfluencyPredictor(
            backbone=disfluency_backbone,
            device=device,
            batch_size=batches["disfluency"],
            wavlm_autocast_dtype=wavlm_autocast_dtype,
            wavlm_compile=wavlm_compile,
            wavlm_compile_mode=wavlm_compile_mode,
            wavlm_compile_dynamic=wavlm_compile_dynamic,
            wavlm_stream_layer_sum=wavlm_stream_layer_sum,
        )
        self.emotion = EmotionPredictor(
            batch_size=batches["emotion"],
            device=emotion_settings.device,
            autocast_dtype=emotion_settings.autocast_dtype,
            compile_model=emotion_settings.compile_model,
            compile_mode=emotion_settings.compile_mode,
            allow_tf32=emotion_settings.allow_tf32,
            warmup=emotion_settings.warmup,
        )
        self._vad_config = dict(
            threshold=vad_threshold,
            min_speech_sec=vad_min_speech_sec,
            min_silence_sec=vad_min_silence_sec,
        )
        self._vad: VadDetector | None = None
        if load_vad:
            self._vad = VadDetector(**self._vad_config)

    @property
    def vad(self) -> VadDetector:
        """Lazily load the CPU VAD model when requested."""
        if self._vad is None:
            self._vad = VadDetector(**self._vad_config)
        return self._vad
