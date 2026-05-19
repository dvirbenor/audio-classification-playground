"""Fast emotion2vec inference helpers.

FunASR's public ``AutoModel.generate`` API accepts a list of waveforms, but the
``Emotion2vec.inference`` implementation loops over that list and runs the
transformer one waveform at a time.  The orchestration pipeline frames audio
into fixed-size 16 kHz windows, so we can safely run those windows as a real
tensor batch and keep the same projection/softmax postprocessing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext

import numpy as np

from .audio import frame_audio, frame_audio_geometry, writable_contiguous_float32

ProgressFn = Callable[[str], None]


def predict_emotion2vec_scores(
    auto_model,
    windows: np.ndarray,
    *,
    sample_rate: int,
    batch_size: int,
    autocast_dtype: str | None = None,
    compile_model: bool = False,
    compile_mode: str = "default",
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, Sequence[str]]:
    """Return raw emotion2vec scores and labels for framed audio windows.

    Uses a direct batched model path for the standard FunASR Emotion2vec model.
    If the loaded model does not expose those internals, falls back to
    ``AutoModel.generate`` so custom/older FunASR models remain supported.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(windows) == 0:
        raise ValueError("emotion2vec received no windows")
    _validate_autocast_dtype(autocast_dtype)

    if _supports_direct_batched_scores(auto_model, sample_rate):
        return _predict_direct_batched(
            auto_model,
            windows,
            batch_size=batch_size,
            autocast_dtype=autocast_dtype,
            compile_model=compile_model,
            compile_mode=compile_mode,
            progress=progress,
        )
    return _predict_via_generate(
        auto_model,
        windows,
        sample_rate=sample_rate,
        batch_size=batch_size,
        autocast_dtype=autocast_dtype,
        progress=progress,
    )


def predict_emotion2vec_scores_from_audio(
    auto_model,
    samples: np.ndarray,
    *,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    autocast_dtype: str | None = None,
    compile_model: bool = False,
    compile_mode: str = "default",
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, Sequence[str]]:
    """Return emotion2vec scores for the same windows as ``frame_audio``.

    The direct path moves the decoded audio to the model device once and
    builds overlapping windows with tensor strides, avoiding repeated host to
    device copies of the same samples. Unsupported FunASR variants fall back
    to the standard framed-window path.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _validate_autocast_dtype(autocast_dtype)

    if _supports_direct_batched_scores(auto_model, sample_rate):
        return _predict_direct_batched_from_audio(
            auto_model,
            samples,
            sample_rate=sample_rate,
            window_sec=window_sec,
            hop_sec=hop_sec,
            batch_size=batch_size,
            autocast_dtype=autocast_dtype,
            compile_model=compile_model,
            compile_mode=compile_mode,
            progress=progress,
        )

    windows = frame_audio(
        samples,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    return predict_emotion2vec_scores(
        auto_model,
        windows,
        sample_rate=sample_rate,
        batch_size=batch_size,
        autocast_dtype=autocast_dtype,
        compile_model=compile_model,
        compile_mode=compile_mode,
        progress=progress,
    )


def _supports_direct_batched_scores(auto_model, sample_rate: int) -> bool:
    if int(sample_rate) != 16_000:
        return False
    model = getattr(auto_model, "model", None)
    kwargs = getattr(auto_model, "kwargs", {}) or {}
    tokenizer = kwargs.get("tokenizer")
    return (
        model is not None
        and hasattr(model, "cfg")
        and hasattr(model, "extract_features")
        and getattr(model, "proj", None) is not None
        and tokenizer is not None
        and hasattr(tokenizer, "token_list")
    )


class _Emotion2vecScoreCore:
    """Tiny torch module for the actual hot-path classifier."""

    def __new__(cls, *args, **kwargs):
        import torch

        class ScoreCore(torch.nn.Module):
            def __init__(self, model, unuse_mask, keep_index_tensor, normalize: bool):
                super().__init__()
                self.model = model
                self.normalize = bool(normalize)
                self.register_buffer("unuse_mask", unuse_mask)
                self.register_buffer("keep_index_tensor", keep_index_tensor)

            def forward(self, batch):
                import torch.nn.functional as F

                if self.normalize:
                    batch = F.layer_norm(batch, batch.shape[1:])
                feats = self.model.extract_features(batch, padding_mask=None)
                x = feats["x"].mean(dim=1)
                logits = self.model.proj(x)
                logits = logits.masked_fill(self.unuse_mask.unsqueeze(0), -torch.inf)
                scores = torch.softmax(logits, dim=-1)
                return scores.index_select(1, self.keep_index_tensor)

        return ScoreCore(*args, **kwargs)


class DirectEmotion2vecScorer:
    """Reusable direct emotion2vec scorer.

    This caches label masks on the model device and, when requested, compiles
    the actual ``extract_features -> proj -> softmax`` scorer rather than the
    FunASR wrapper object.
    """

    def __init__(
        self,
        auto_model,
        *,
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        import torch

        self.auto_model = auto_model
        self.compiled = bool(compile_model)
        model = auto_model.model
        kwargs = getattr(auto_model, "kwargs", {}) or {}
        tokenizer = kwargs["tokenizer"]
        labels = list(tokenizer.token_list)
        keep_indices = [
            idx for idx, label in enumerate(labels) if not str(label).startswith("unuse")
        ]
        if not keep_indices:
            raise ValueError("emotion2vec tokenizer has no usable labels")

        self.device = _model_device(model, kwargs)
        self.selected_labels = [labels[idx] for idx in keep_indices]
        unuse_mask = torch.tensor(
            [str(label).startswith("unuse") for label in labels],
            dtype=torch.bool,
            device=self.device,
        )
        keep_index_tensor = torch.tensor(keep_indices, dtype=torch.long, device=self.device)
        model.eval()
        core = _Emotion2vecScoreCore(
            model,
            unuse_mask,
            keep_index_tensor,
            bool(_cfg_get(getattr(model, "cfg", None), "normalize", False)),
        ).to(self.device).eval()
        if self.compiled:
            _validate_compile_mode_for_device(compile_mode, self.device)
            core = torch.compile(core, mode=compile_mode)
        self.core = core

    def __call__(self, batch, *, autocast_dtype: str | None = None):
        import torch

        if self.compiled:
            _mark_cudagraph_step_begin(torch, self.device)
        with _autocast_context(torch, self.device, autocast_dtype):
            scores = self.core(batch)
        scores = scores.to(dtype=torch.float32)
        return scores.clone() if self.compiled else scores

    def predict_windows(
        self,
        windows: np.ndarray,
        *,
        batch_size: int,
        autocast_dtype: str | None,
        progress: ProgressFn | None = None,
    ) -> tuple[np.ndarray, Sequence[str]]:
        import torch

        if not self.compiled:
            all_scores: list[np.ndarray] = []
            with torch.inference_mode():
                for _, batch_np in _batches(windows, batch_size, progress, "emotion"):
                    batch = torch.from_numpy(writable_contiguous_float32(batch_np)).to(self.device)
                    scores = self(batch, autocast_dtype=autocast_dtype)
                    all_scores.append(scores.detach().cpu().numpy())
            return np.concatenate(all_scores, axis=0).astype(np.float32, copy=False), self.selected_labels

        out = torch.empty(
            (len(windows), len(self.selected_labels)),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            for start, batch_np in _batches(windows, batch_size, progress, "emotion"):
                end = start + len(batch_np)
                batch = torch.from_numpy(writable_contiguous_float32(batch_np)).to(self.device)
                out[start:end] = self(batch, autocast_dtype=autocast_dtype)
        return out.detach().cpu().numpy(), self.selected_labels

    def predict_audio(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int,
        window_sec: float,
        hop_sec: float,
        batch_size: int,
        autocast_dtype: str | None,
        progress: ProgressFn | None = None,
    ) -> tuple[np.ndarray, Sequence[str]]:
        import torch
        import torch.nn.functional as F

        audio_np = writable_contiguous_float32(samples)
        n_frames, window_samples, hop_samples, pad_needed = frame_audio_geometry(
            len(audio_np),
            sample_rate=sample_rate,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )
        if not self.compiled:
            all_scores: list[np.ndarray] = []
            with torch.inference_mode():
                audio = torch.from_numpy(audio_np).to(self.device)
                if pad_needed:
                    audio = F.pad(audio, (0, pad_needed))
                windows = audio.as_strided(
                    size=(n_frames, window_samples),
                    stride=(hop_samples, 1),
                )
                for start in range(0, n_frames, batch_size):
                    end = min(start + batch_size, n_frames)
                    if progress is not None:
                        progress(f"emotion batch {start}:{end} / {n_frames}")
                    batch = windows[start:end].contiguous()
                    scores = self(batch, autocast_dtype=autocast_dtype)
                    all_scores.append(scores.detach().cpu().numpy())
            return np.concatenate(all_scores, axis=0).astype(np.float32, copy=False), self.selected_labels

        out = torch.empty(
            (n_frames, len(self.selected_labels)),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            audio = torch.from_numpy(audio_np).to(self.device)
            if pad_needed:
                audio = F.pad(audio, (0, pad_needed))
            windows = audio.as_strided(
                size=(n_frames, window_samples),
                stride=(hop_samples, 1),
            )
            for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                if progress is not None:
                    progress(f"emotion batch {start}:{end} / {n_frames}")
                batch = windows[start:end].contiguous()
                out[start:end] = self(batch, autocast_dtype=autocast_dtype)
        return out.detach().cpu().numpy(), self.selected_labels


def make_direct_emotion2vec_scorer(
    auto_model,
    *,
    sample_rate: int,
    compile_model: bool = False,
    compile_mode: str = "default",
) -> DirectEmotion2vecScorer | None:
    if not _supports_direct_batched_scores(auto_model, sample_rate):
        return None
    return DirectEmotion2vecScorer(
        auto_model,
        compile_model=compile_model,
        compile_mode=compile_mode,
    )


def _predict_direct_batched(
    auto_model,
    windows: np.ndarray,
    *,
    batch_size: int,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    scorer = DirectEmotion2vecScorer(
        auto_model,
        compile_model=compile_model,
        compile_mode=compile_mode,
    )
    return scorer.predict_windows(
        windows,
        batch_size=batch_size,
        autocast_dtype=autocast_dtype,
        progress=progress,
    )


def _predict_direct_batched_from_audio(
    auto_model,
    samples: np.ndarray,
    *,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    scorer = DirectEmotion2vecScorer(
        auto_model,
        compile_model=compile_model,
        compile_mode=compile_mode,
    )
    return scorer.predict_audio(
        samples,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        autocast_dtype=autocast_dtype,
        progress=progress,
    )


def _predict_via_generate(
    auto_model,
    windows: np.ndarray,
    *,
    sample_rate: int,
    batch_size: int,
    autocast_dtype: str | None,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    import torch

    all_scores = []
    labels = None
    for start, batch_np in _batches(windows, batch_size, progress, "emotion"):
        batch = [np.ascontiguousarray(batch_np[i], dtype=np.float32) for i in range(len(batch_np))]
        model = getattr(auto_model, "model", None)
        device = _model_device(model, getattr(auto_model, "kwargs", {}) or {}) if model is not None else None
        with _autocast_context(torch, device, autocast_dtype):
            results = auto_model.generate(
                input=batch,
                fs=sample_rate,
                batch_size=len(batch),
                granularity="utterance",
                extract_embedding=False,
                disable_pbar=True,
            )
        if labels is None:
            labels = list(results[0]["labels"])
        all_scores.extend(result["scores"] for result in results)
    if labels is None:
        raise ValueError("emotion2vec produced no results")
    return np.asarray(all_scores, dtype=np.float32), labels


def _batches(
    windows: np.ndarray,
    batch_size: int,
    progress: ProgressFn | None,
    task: str,
):
    n = len(windows)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if progress is not None:
            progress(f"{task} batch {start}:{end} / {n}")
        yield start, windows[start:end]


def _model_device(model, kwargs: dict):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(kwargs.get("device", "cpu"))


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _validate_autocast_dtype(dtype: str | None) -> None:
    if dtype is None:
        return
    if dtype not in {"fp16", "float16", "bf16", "bfloat16"}:
        raise ValueError("autocast_dtype must be one of: fp16, float16, bf16, bfloat16")


def _autocast_context(torch, device, dtype: str | None):
    if dtype is None or device is None or getattr(device, "type", None) != "cuda":
        return nullcontext()
    resolved = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    return torch.autocast("cuda", dtype=resolved)


def _mark_cudagraph_step_begin(torch, device) -> None:
    if device is None or getattr(device, "type", None) != "cuda":
        return
    compiler = getattr(torch, "compiler", None)
    mark = getattr(compiler, "cudagraph_mark_step_begin", None)
    if mark is not None:
        mark()


def _validate_compile_mode_for_device(mode: str, device) -> None:
    if mode != "reduce-overhead" or getattr(device, "type", None) != "cuda":
        return
    raise ValueError(
        "emotion2vec torch.compile mode 'reduce-overhead' is not supported on CUDA "
        "for this FunASR path because it uses CUDA Graph replay and can fail across "
        "repeated batches with overwritten internal tensors. Use "
        "--emotion-compile-mode default."
    )
