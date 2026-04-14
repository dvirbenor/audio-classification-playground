"""Main generator for synthetic multi-label audio dataset."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torchaudio
import torch

from .audio_loader import (
    load_audio_sources,
    AudioSources,
    DEFAULT_SAMPLE_RATE
)
from .timeline_generator import (
    Timeline,
    ActualTimeline,
    generate_timeline,
    generate_structured_timeline,
    validate_timeline
)
from .audio_mixer import mix_audio, render_info_to_actual_timeline
from .label_generator import (
    generate_labels,
    generate_labels_from_actual,
    save_labels,
    validate_labels,
    labels_to_dict,
    DEFAULT_HOP_SIZE
)


@dataclass
class GenerationResult:
    """Result of synthetic audio generation."""
    
    audio_path: Path
    labels_path: Path
    timeline_path: Path
    actual_timeline_path: Path  # Path to actual timeline (ground truth)
    audio: np.ndarray
    labels: np.ndarray
    timeline: Timeline  # Planned timeline
    actual_timeline: ActualTimeline  # What was actually rendered
    sample_rate: int
    hop_size: int
    
    @property
    def num_frames(self) -> int:
        return self.labels.shape[0]
    
    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


def generate_synthetic_audio(
    output_dir: Path,
    resources_dir: Optional[Path] = None,
    duration: float = 180.0,
    seed: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE,
    structured: bool = True,
    validate: bool = True,
    verbose: bool = True
) -> GenerationResult:
    """Generate synthetic multi-label audio with ground truth annotations.
    
    This function orchestrates the entire generation pipeline:
    1. Load audio sources from resources directory
    2. Generate a planned timeline with entrance times and max durations
    3. Mix audio according to timeline (audio plays at natural duration, up to max)
    4. Generate frame-accurate labels from actual rendered content
    5. Save all outputs (including both planned and actual timelines)
    
    The labels are generated from the ACTUAL timeline (what was rendered),
    not the planned timeline, ensuring ground truth accuracy.
    
    Args:
        output_dir: Directory to save outputs (will be created if needed)
        resources_dir: Directory containing speech/, music/, sfx/ subdirs
                      (defaults to 'resources/' in workspace root)
        duration: Total audio duration in seconds (default 180 = 3 minutes)
        seed: Random seed for reproducibility
        sample_rate: Audio sample rate (default 32kHz)
        hop_size: Hop size for frame-level labels (default 320 = 10ms)
        structured: Use structured timeline for guaranteed variety
        validate: Whether to validate labels after generation
        verbose: Print progress information
        
    Returns:
        GenerationResult with paths and data (including actual timeline)
    """
    # Resolve paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if resources_dir is None:
        # Default to resources/ in workspace root
        resources_dir = Path(__file__).parent.parent.parent / "resources"
    else:
        resources_dir = Path(resources_dir)
    
    if verbose:
        print(f"Generating {duration}s synthetic audio...")
        print(f"Resources: {resources_dir}")
        print(f"Output: {output_dir}")
    
    # Step 1: Load audio sources (now with metadata)
    if verbose:
        print("\n[1/5] Loading audio sources...")
    
    sources = load_audio_sources(
        resources_dir,
        target_sample_rate=sample_rate,
        normalize=True
    )
    
    num_speech = len(sources.speech)
    num_music = len(sources.music)
    num_sfx = len(sources.sfx)
    
    if verbose:
        print(f"Loaded: {num_speech} speech, {num_music} music, {num_sfx} SFX samples")
        # Show sample durations
        for category in ['speech', 'music', 'sfx']:
            samples = getattr(sources, category)
            durations = [f"{s.duration:.1f}s" for s in samples[:3]]
            if len(samples) > 3:
                durations.append("...")
            print(f"  {category}: [{', '.join(durations)}]")
    
    if num_speech == 0 or num_music == 0 or num_sfx == 0:
        raise ValueError(
            f"Missing audio samples. Need at least 1 of each type. "
            f"Found: speech={num_speech}, music={num_music}, sfx={num_sfx}"
        )
    
    # Step 2: Generate planned timeline (entrance times + max durations)
    if verbose:
        print("\n[2/5] Generating planned timeline...")
    
    if structured:
        timeline = generate_structured_timeline(
            duration=duration,
            seed=seed,
            num_speech_samples=num_speech,
            num_music_samples=num_music,
            num_sfx_samples=num_sfx
        )
    else:
        timeline = generate_timeline(
            duration=duration,
            seed=seed,
            num_speech_samples=num_speech,
            num_music_samples=num_music,
            num_sfx_samples=num_sfx
        )
    
    if verbose:
        print(f"Created planned timeline: {len(timeline.base_segments)} base segments, "
              f"{len(timeline.sfx_events)} SFX events")
    
    # Step 3: Mix audio (returns actual render information)
    if verbose:
        print("\n[3/5] Mixing audio layers...")
    
    audio, render_info = mix_audio(
        timeline=timeline,
        sources=sources,
        sample_rate=sample_rate,
        seed=seed
    )
    
    # Convert render info to actual timeline
    actual_timeline = render_info_to_actual_timeline(render_info)
    
    if verbose:
        print(f"Mixed audio: {len(audio)} samples ({len(audio) / sample_rate:.2f}s)")
        print(f"Actual timeline: {len(actual_timeline.segments)} segments rendered, "
              f"{len(actual_timeline.sfx_events)} SFX events")
    
    # Step 4: Generate labels from ACTUAL timeline (ground truth)
    if verbose:
        print("\n[4/5] Generating frame-level labels from actual content...")
    
    labels = generate_labels_from_actual(
        actual_timeline=actual_timeline,
        sample_rate=sample_rate,
        hop_size=hop_size
    )
    
    if verbose:
        stats = labels_to_dict(labels)
        print(f"Labels: {stats['num_frames']} frames")
        print(f"  Speech: {stats['speech_percentage']:.1f}%")
        print(f"  Music: {stats['music_percentage']:.1f}%")
        print(f"  SFX: {stats['sfx_percentage']:.1f}%")
    
    # Validate labels against actual timeline
    if validate:
        if not validate_labels(labels, actual_timeline, sample_rate, hop_size):
            raise ValueError("Label validation failed!")
    
    # Step 5: Save outputs
    if verbose:
        print("\n[5/5] Saving outputs...")
    
    audio_path = output_dir / "synthetic_audio.wav"
    labels_path = output_dir / "labels.npy"
    timeline_path = output_dir / "timeline_planned.json"  # Planned timeline
    actual_timeline_path = output_dir / "timeline_actual.json"  # Ground truth
    
    # Save audio
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    torchaudio.save(str(audio_path), audio_tensor, sample_rate)
    
    # Save labels
    save_labels(labels, labels_path)
    
    # Save both timelines
    timeline.save(timeline_path)
    actual_timeline.save(actual_timeline_path)
    
    if verbose:
        print(f"Saved: {audio_path.name}, {labels_path.name}")
        print(f"Saved: {timeline_path.name} (planned), {actual_timeline_path.name} (actual)")
        print("\nGeneration complete!")
    
    return GenerationResult(
        audio_path=audio_path,
        labels_path=labels_path,
        timeline_path=timeline_path,
        actual_timeline_path=actual_timeline_path,
        audio=audio,
        labels=labels,
        timeline=timeline,
        actual_timeline=actual_timeline,
        sample_rate=sample_rate,
        hop_size=hop_size
    )


def generate_test_audio(
    output_dir: Path,
    resources_dir: Optional[Path] = None,
    duration: float = 10.0,
    seed: int = 42
) -> GenerationResult:
    """Generate a short test audio for validation.
    
    This creates a brief audio clip for testing the pipeline.
    
    Args:
        output_dir: Directory to save outputs
        resources_dir: Directory containing audio samples
        duration: Duration in seconds (default 10s for quick testing)
        seed: Random seed for reproducibility
        
    Returns:
        GenerationResult with paths and data
    """
    return generate_synthetic_audio(
        output_dir=output_dir,
        resources_dir=resources_dir,
        duration=duration,
        seed=seed,
        structured=True,
        validate=True,
        verbose=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic multi-label audio")
    parser.add_argument("--output", "-o", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--resources", "-r", type=str, default=None,
                       help="Resources directory with audio samples")
    parser.add_argument("--duration", "-d", type=float, default=180.0,
                       help="Duration in seconds (default: 180)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--test", action="store_true",
                       help="Generate short test audio (10s)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    resources_dir = Path(args.resources) if args.resources else None
    
    if args.test:
        result = generate_test_audio(output_dir, resources_dir, seed=args.seed or 42)
    else:
        result = generate_synthetic_audio(
            output_dir=output_dir,
            resources_dir=resources_dir,
            duration=args.duration,
            seed=args.seed
        )
    
    print(f"\nResults:")
    print(f"  Audio: {result.audio_path} ({result.duration:.2f}s)")
    print(f"  Labels: {result.labels_path} ({result.num_frames} frames)")
    print(f"  Timeline: {result.timeline_path}")
