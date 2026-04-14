"""Visualization utilities for synthetic audio and multi-label ground truth."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .label_generator import (
    LABEL_NAMES,
    DEFAULT_HOP_SIZE,
    frame_to_time,
    load_labels
)
from .audio_loader import DEFAULT_SAMPLE_RATE


# Color scheme for classes
COLORS = {
    "speech": "#FFD700",  # Gold/yellow
    "music": "#4A90D9",   # Blue
    "sfx": "#7CB342"      # Green
}


def plot_labels(
    labels: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE,
    figsize: Tuple[float, float] = (14, 4),
    title: str = "Multi-Label Ground Truth",
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot multi-label ground truth as stacked activity bars.
    
    Args:
        labels: Labels array of shape (num_frames, 3)
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_frames = labels.shape[0]
    duration = frame_to_time(num_frames, sample_rate, hop_size)
    times = np.array([frame_to_time(i, sample_rate, hop_size) for i in range(num_frames)])
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    for idx, (ax, name) in enumerate(zip(axes, LABEL_NAMES)):
        color = COLORS[name]
        activity = labels[:, idx]
        
        # Fill where active
        ax.fill_between(times, 0, activity, color=color, alpha=0.7, step='mid')
        ax.set_ylabel(name.capitalize(), fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Off', 'On'])
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=10)
    axes[-1].set_xlim(0, duration)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_labels_combined(
    labels: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE,
    figsize: Tuple[float, float] = (14, 3),
    title: str = "Multi-Label Activity",
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot multi-label ground truth as overlaid colored regions.
    
    Args:
        labels: Labels array of shape (num_frames, 3)
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_frames = labels.shape[0]
    duration = frame_to_time(num_frames, sample_rate, hop_size)
    times = np.array([frame_to_time(i, sample_rate, hop_size) for i in range(num_frames)])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each class at different heights (music on top, speech in middle, sfx at bottom)
    heights = {"music": 2.5, "speech": 1.5, "sfx": 0.5}
    
    for idx, name in enumerate(LABEL_NAMES):
        color = COLORS[name]
        activity = labels[:, idx]
        height = heights[name]
        
        # Fill where active
        ax.fill_between(
            times, 
            height - 0.4, 
            height + 0.4,
            where=activity == 1,
            color=color, 
            alpha=0.8,
            step='mid',
            label=name.capitalize()
        )
    
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, duration)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['SFX', 'Speech', 'Music'])
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    handles = [mpatches.Patch(color=COLORS[name], label=name.capitalize()) 
               for name in LABEL_NAMES]
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_with_waveform(
    audio: np.ndarray,
    labels: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE,
    figsize: Tuple[float, float] = (14, 6),
    title: str = "Synthetic Audio with Labels",
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot audio waveform with multi-label activity overlay.
    
    Args:
        audio: Audio waveform
        labels: Labels array of shape (num_frames, 3)
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_frames = labels.shape[0]
    duration = len(audio) / sample_rate
    audio_times = np.linspace(0, duration, len(audio))
    label_times = np.array([frame_to_time(i, sample_rate, hop_size) for i in range(num_frames)])
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Waveform
    axes[0].plot(audio_times, audio, color='gray', linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Labels as colored regions (music on top, speech in middle, sfx at bottom)
    heights = {"music": 2.5, "speech": 1.5, "sfx": 0.5}
    
    for idx, name in enumerate(LABEL_NAMES):
        color = COLORS[name]
        activity = labels[:, idx]
        height = heights[name]
        
        axes[1].fill_between(
            label_times, 
            height - 0.35, 
            height + 0.35,
            where=activity == 1,
            color=color, 
            alpha=0.8,
            step='mid'
        )
    
    axes[1].set_ylim(0, 3.5)
    axes[1].set_xlim(0, duration)
    axes[1].set_yticks([0.5, 1.5, 2.5])
    axes[1].set_yticklabels(['SFX', 'Speech', 'Music'])
    axes[1].set_xlabel('Time (seconds)')
    axes[1].grid(True, axis='x', linestyle='--', alpha=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Legend
    handles = [mpatches.Patch(color=COLORS[name], label=name.capitalize()) 
               for name in LABEL_NAMES]
    axes[1].legend(handles=handles, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE,
    figsize: Tuple[float, float] = (14, 8),
    title: str = "Ground Truth vs Predictions",
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Compare ground truth labels with model predictions.
    
    Args:
        ground_truth: Ground truth labels (num_frames, 3) binary
        predictions: Model predictions (num_frames, 3) probabilities
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_frames = ground_truth.shape[0]
    duration = frame_to_time(num_frames, sample_rate, hop_size)
    times = np.array([frame_to_time(i, sample_rate, hop_size) for i in range(num_frames)])
    
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    for idx, name in enumerate(LABEL_NAMES):
        color = COLORS[name]
        
        # Ground truth
        gt = ground_truth[:, idx]
        axes[idx, 0].fill_between(times, 0, gt, color=color, alpha=0.7, step='mid')
        axes[idx, 0].set_ylabel(f'{name.capitalize()}\n(GT)')
        axes[idx, 0].set_ylim(-0.1, 1.1)
        axes[idx, 0].grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # Predictions
        pred = predictions[:, idx]
        axes[idx, 1].fill_between(times, 0, pred, color=color, alpha=0.7)
        axes[idx, 1].plot(times, pred, color=color, linewidth=0.5, alpha=0.8)
        axes[idx, 1].set_ylabel(f'{name.capitalize()}\n(Pred)')
        axes[idx, 1].set_ylim(-0.1, 1.1)
        axes[idx, 1].grid(True, axis='x', linestyle='--', alpha=0.3)
    
    axes[0, 0].set_title('Ground Truth')
    axes[0, 1].set_title('Predictions')
    axes[-1, 0].set_xlabel('Time (seconds)')
    axes[-1, 1].set_xlabel('Time (seconds)')
    
    for ax in axes.flat:
        ax.set_xlim(0, duration)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_from_files(
    labels_path: Path,
    audio_path: Optional[Path] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE,
    show: bool = True,
) -> plt.Figure:
    """Load and visualize labels from file.
    
    Args:
        labels_path: Path to labels.npy
        audio_path: Optional path to audio file for waveform display
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        show: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    labels = load_labels(labels_path)
    
    if audio_path and audio_path.exists():
        import torchaudio
        audio, sr = torchaudio.load(str(audio_path))
        audio = audio.squeeze().numpy()
        
        if sr != sample_rate:
            print(f"Warning: Audio sample rate ({sr}) differs from expected ({sample_rate})")
        
        fig = plot_with_waveform(audio, labels, sample_rate, hop_size)
    else:
        fig = plot_labels(labels, sample_rate, hop_size)
    
    if show:
        plt.show()
    
    return fig
