import librosa
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    labels,
    index_to_category,
    high_level_categories,
    category_to_index,
    num_high_level_categories,
)
from .inference import DEFAULT_WINDOW_SIZE, DEFAULT_HOP_SIZE, DEFAULT_SAMPLE_RATE


def visualize_frame_level_output(
    raw_audio: np.ndarray,
    framewise_output: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    top_k: int = 10,
) -> None:
    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_result_mat = framewise_output[:, sorted_indexes[0:top_k]]

    # Plot result
    stft = librosa.core.stft(
        y=raw_audio, n_fft=window_size, hop_length=hop_size, window="hann", center=True
    )
    frames_num = stft.shape[-1]
    frames_per_second = sample_rate // hop_size

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin="lower", aspect="auto", cmap="jet")
    axs[0].set_ylabel("Frequency bins")
    axs[0].set_title("Log spectrogram")
    axs[1].matshow(
        top_result_mat.T, origin="upper", aspect="auto", cmap="jet", vmin=0, vmax=1
    )
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0:top_k]])
    axs[1].yaxis.grid(color="k", linestyle="solid", linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel("Seconds")
    axs[1].xaxis.set_ticks_position("bottom")

    plt.tight_layout()
    plt.show()


def aggregate_to_high_level(framewise_output: np.ndarray) -> np.ndarray:
    """
    Aggregate frame-level predictions to high-level categories (Music, Speech, SFX).

    Sums the probabilities of all low-level classes belonging to each high-level category.

    Args:
        framewise_output: Array of shape (num_frames, num_classes) with probabilities.

    Returns:
        Array of shape (num_frames, 3) with aggregated probabilities for
        [Music, Speech, SFX].
    """
    num_frames = framewise_output.shape[0]
    high_level_output = np.zeros((num_frames, num_high_level_categories))

    for class_idx, category in enumerate(index_to_category):
        category_idx = category_to_index[category]
        high_level_output[:, category_idx] += framewise_output[:, class_idx]

    return high_level_output


def visualize_high_level_categories(
    raw_audio: np.ndarray,
    framewise_output: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """
    Visualize audio classification results using high-level categories.

    Groups the 527 AudioSet classes into three categories (Music, Speech, SFX)
    and displays their aggregated probabilities over time.

    Args:
        raw_audio: Raw audio waveform.
        framewise_output: Array of shape (num_frames, num_classes) with predictions.
        window_size: STFT window size.
        hop_size: STFT hop size.
        sample_rate: Audio sample rate.
    """
    # Aggregate to high-level categories
    high_level_output = aggregate_to_high_level(framewise_output)

    # Compute spectrogram
    stft = librosa.core.stft(
        y=raw_audio, n_fft=window_size, hop_length=hop_size, window="hann", center=True
    )
    frames_num = stft.shape[-1]
    frames_per_second = sample_rate // hop_size

    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

    # Top: Log spectrogram
    axs[0].matshow(np.log(np.abs(stft)), origin="lower", aspect="auto", cmap="jet")
    axs[0].set_ylabel("Frequency bins")
    axs[0].set_title("Log spectrogram")

    # Bottom: High-level categories
    axs[1].matshow(
        high_level_output.T, origin="upper", aspect="auto", cmap="jet", vmin=0, vmax=1
    )
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, num_high_level_categories))
    axs[1].yaxis.set_ticklabels(high_level_categories)
    axs[1].yaxis.grid(color="k", linestyle="solid", linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel("Seconds")
    axs[1].xaxis.set_ticks_position("bottom")
    axs[1].set_title("High-level categories (Music, Speech, SFX)")

    plt.tight_layout()
    plt.show()
