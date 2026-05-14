"""Fast emotion2vec inference helpers.

FunASR's public ``AutoModel.generate`` API accepts a list of waveforms, but the
``Emotion2vec.inference`` implementation loops over that list and runs the
transformer one waveform at a time.  The orchestration pipeline frames audio
into fixed-size 16 kHz windows, so we can safely run those windows as a real
tensor batch and keep the same projection/softmax postprocessing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

ProgressFn = Callable[[str], None]


def predict_emotion2vec_scores(
    auto_model,
    windows: np.ndarray,
    *,
    sample_rate: int,
    batch_size: int,
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

    if _supports_direct_batched_scores(auto_model, sample_rate):
        return _predict_direct_batched(auto_model, windows, batch_size=batch_size, progress=progress)
    return _predict_via_generate(
        auto_model,
        windows,
        sample_rate=sample_rate,
        batch_size=batch_size,
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


def _predict_direct_batched(
    auto_model,
    windows: np.ndarray,
    *,
    batch_size: int,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    import torch
    import torch.nn.functional as F

    model = auto_model.model
    kwargs = getattr(auto_model, "kwargs", {}) or {}
    tokenizer = kwargs["tokenizer"]
    labels = list(tokenizer.token_list)
    keep_indices = [idx for idx, label in enumerate(labels) if not str(label).startswith("unuse")]
    if not keep_indices:
        raise ValueError("emotion2vec tokenizer has no usable labels")

    device = _model_device(model, kwargs)
    unuse_mask = torch.tensor(
        [str(label).startswith("unuse") for label in labels],
        dtype=torch.bool,
        device=device,
    )
    selected_labels = [labels[idx] for idx in keep_indices]

    model.eval()
    all_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start, batch_np in _batches(windows, batch_size, progress, "emotion"):
            batch = torch.from_numpy(np.ascontiguousarray(batch_np, dtype=np.float32)).to(device)
            if _cfg_get(getattr(model, "cfg", None), "normalize", False):
                batch = F.layer_norm(batch, batch.shape[1:])

            feats = model.extract_features(batch, padding_mask=None)
            x = feats["x"].mean(dim=1)
            logits = model.proj(x)
            logits = logits.masked_fill(unuse_mask.unsqueeze(0), -torch.inf)
            scores = torch.softmax(logits, dim=-1)
            all_scores.append(scores[:, keep_indices].detach().cpu().numpy())

    return np.concatenate(all_scores, axis=0).astype(np.float32, copy=False), selected_labels


def _predict_via_generate(
    auto_model,
    windows: np.ndarray,
    *,
    sample_rate: int,
    batch_size: int,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    all_scores = []
    labels = None
    for start, batch_np in _batches(windows, batch_size, progress, "emotion"):
        batch = [np.ascontiguousarray(batch_np[i], dtype=np.float32) for i in range(len(batch_np))]
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
