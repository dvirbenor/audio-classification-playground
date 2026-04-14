"""Label generator for frame-accurate multi-label annotations."""

import numpy as np
from pathlib import Path
from typing import Optional, Union

from .timeline_generator import Timeline, ActualTimeline
from .audio_loader import DEFAULT_SAMPLE_RATE

# Default hop size matching PANNs inference
DEFAULT_HOP_SIZE = 320  # 320 samples @ 32kHz = 10ms per frame


# Label column indices
SPEECH_IDX = 0
MUSIC_IDX = 1
SFX_IDX = 2

LABEL_NAMES = ["speech", "music", "sfx"]


def time_to_frame(time_seconds: float, sample_rate: int, hop_size: int) -> int:
    """Convert time in seconds to frame index.
    
    Args:
        time_seconds: Time in seconds
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        
    Returns:
        Frame index
    """
    sample_idx = int(time_seconds * sample_rate)
    return sample_idx // hop_size


def frame_to_time(frame_idx: int, sample_rate: int, hop_size: int) -> float:
    """Convert frame index to time in seconds.
    
    Args:
        frame_idx: Frame index
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        
    Returns:
        Time in seconds
    """
    sample_idx = frame_idx * hop_size
    return sample_idx / sample_rate


def compute_num_frames(duration: float, sample_rate: int, hop_size: int) -> int:
    """Compute the number of frames for a given duration.
    
    Args:
        duration: Duration in seconds
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        
    Returns:
        Number of frames
    """
    num_samples = int(duration * sample_rate)
    return num_samples // hop_size


def generate_labels(
    timeline: Timeline,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE
) -> np.ndarray:
    """Generate frame-level multi-label annotations from planned timeline.
    
    NOTE: For accurate labels based on actual audio content, use 
    generate_labels_from_actual() with an ActualTimeline instead.
    
    This creates a binary label array where each row represents a 10ms frame
    and each column represents a class:
    - Column 0: Speech active (0 or 1)
    - Column 1: Music active (0 or 1)
    - Column 2: SFX active (0 or 1)
    
    Args:
        timeline: Planned timeline with base segments and SFX events
        sample_rate: Audio sample rate (default 32kHz)
        hop_size: Hop size in samples (default 320 = 10ms frames)
        
    Returns:
        Labels array of shape (num_frames, 3) with binary values
    """
    num_frames = compute_num_frames(timeline.duration, sample_rate, hop_size)
    labels = np.zeros((num_frames, 3), dtype=np.int8)
    
    # Process base segments (speech and music)
    for segment in timeline.base_segments:
        start_frame = time_to_frame(segment.start, sample_rate, hop_size)
        end_frame = time_to_frame(segment.end, sample_rate, hop_size)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, num_frames)
        
        if segment.type == "speech":
            labels[start_frame:end_frame, SPEECH_IDX] = 1
        elif segment.type == "music":
            labels[start_frame:end_frame, MUSIC_IDX] = 1
    
    # Process SFX events (precise frame-level labeling)
    for event in timeline.sfx_events:
        start_frame = time_to_frame(event.time, sample_rate, hop_size)
        end_frame = time_to_frame(event.end, sample_rate, hop_size)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, num_frames)
        
        # Mark SFX as active for these specific frames only
        labels[start_frame:end_frame, SFX_IDX] = 1
    
    return labels


def generate_labels_from_actual(
    actual_timeline: ActualTimeline,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE
) -> np.ndarray:
    """Generate frame-level multi-label annotations from actual rendered timeline.
    
    This is the preferred method for label generation as it reflects what was
    actually mixed into the audio (actual durations based on audio samples),
    not what was planned (max durations in the timeline).
    
    This creates a binary label array where each row represents a 10ms frame
    and each column represents a class:
    - Column 0: Speech active (0 or 1)
    - Column 1: Music active (0 or 1)
    - Column 2: SFX active (0 or 1)
    
    Args:
        actual_timeline: ActualTimeline with segments/events that were actually rendered
        sample_rate: Audio sample rate (default 32kHz)
        hop_size: Hop size in samples (default 320 = 10ms frames)
        
    Returns:
        Labels array of shape (num_frames, 3) with binary values
    """
    num_frames = compute_num_frames(actual_timeline.duration, sample_rate, hop_size)
    labels = np.zeros((num_frames, 3), dtype=np.int8)
    
    # Process actual segments (speech and music)
    for segment in actual_timeline.segments:
        start_frame = time_to_frame(segment.start, sample_rate, hop_size)
        end_frame = time_to_frame(segment.end, sample_rate, hop_size)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, num_frames)
        
        if segment.type == "speech":
            labels[start_frame:end_frame, SPEECH_IDX] = 1
        elif segment.type == "music":
            labels[start_frame:end_frame, MUSIC_IDX] = 1
    
    # Process actual SFX events (precise frame-level labeling)
    for event in actual_timeline.sfx_events:
        start_frame = time_to_frame(event.start, sample_rate, hop_size)
        end_frame = time_to_frame(event.end, sample_rate, hop_size)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, num_frames)
        
        # Mark SFX as active for these specific frames only
        labels[start_frame:end_frame, SFX_IDX] = 1
    
    return labels


def labels_to_dict(labels: np.ndarray) -> dict:
    """Convert labels array to a dictionary for inspection.
    
    Args:
        labels: Labels array of shape (num_frames, 3)
        
    Returns:
        Dictionary with label statistics
    """
    num_frames = labels.shape[0]
    
    # Count frames per class
    speech_frames = int(labels[:, SPEECH_IDX].sum())
    music_frames = int(labels[:, MUSIC_IDX].sum())
    sfx_frames = int(labels[:, SFX_IDX].sum())
    
    # Count multi-label combinations
    combinations = {}
    for frame in range(num_frames):
        key = tuple(labels[frame].tolist())
        combinations[key] = combinations.get(key, 0) + 1
    
    return {
        "num_frames": num_frames,
        "speech_frames": speech_frames,
        "music_frames": music_frames,
        "sfx_frames": sfx_frames,
        "speech_percentage": speech_frames / num_frames * 100,
        "music_percentage": music_frames / num_frames * 100,
        "sfx_percentage": sfx_frames / num_frames * 100,
        "combinations": {
            str(k): v for k, v in sorted(combinations.items())
        }
    }


def save_labels(labels: np.ndarray, path: Path) -> None:
    """Save labels to numpy file.
    
    Args:
        labels: Labels array
        path: Output path (should end with .npy)
    """
    np.save(str(path), labels)


def load_labels(path: Path) -> np.ndarray:
    """Load labels from numpy file.
    
    Args:
        path: Path to .npy file
        
    Returns:
        Labels array
    """
    return np.load(str(path))


def validate_labels(
    labels: np.ndarray,
    timeline: Union[Timeline, ActualTimeline],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE
) -> bool:
    """Validate that labels correctly represent the timeline.
    
    Args:
        labels: Generated labels array
        timeline: Source timeline (Timeline or ActualTimeline)
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        
    Returns:
        True if validation passes
    """
    expected_frames = compute_num_frames(timeline.duration, sample_rate, hop_size)
    
    # Check shape
    if labels.shape != (expected_frames, 3):
        print(f"Shape mismatch: expected ({expected_frames}, 3), got {labels.shape}")
        return False
    
    # Check values are binary
    if not np.all((labels == 0) | (labels == 1)):
        print("Labels contain non-binary values")
        return False
    
    # Get segments and sfx_events depending on timeline type
    if isinstance(timeline, ActualTimeline):
        segments = timeline.segments
        sfx_events = timeline.sfx_events
    else:
        segments = timeline.base_segments
        sfx_events = timeline.sfx_events
    
    # Spot-check a few segments
    for segment in segments[:3]:
        mid_frame = time_to_frame((segment.start + segment.end) / 2, sample_rate, hop_size)
        mid_frame = min(mid_frame, expected_frames - 1)
        
        if segment.type == "speech":
            if labels[mid_frame, SPEECH_IDX] != 1:
                print(f"Speech segment not properly labeled at frame {mid_frame}")
                return False
        elif segment.type == "music":
            if labels[mid_frame, MUSIC_IDX] != 1:
                print(f"Music segment not properly labeled at frame {mid_frame}")
                return False
    
    # Spot-check SFX events
    for event in sfx_events[:3]:
        # Get start time (different attribute name depending on type)
        if isinstance(timeline, ActualTimeline):
            start_time = event.start
        else:
            start_time = event.time
        
        mid_time = start_time + event.duration / 2
        mid_frame = time_to_frame(mid_time, sample_rate, hop_size)
        mid_frame = min(mid_frame, expected_frames - 1)
        
        if labels[mid_frame, SFX_IDX] != 1:
            print(f"SFX event not properly labeled at frame {mid_frame}")
            return False
    
    print(f"Label validation passed: {expected_frames} frames, 3 classes")
    return True


def get_active_classes_at_time(
    labels: np.ndarray,
    time_seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_size: int = DEFAULT_HOP_SIZE
) -> list[str]:
    """Get the active classes at a specific time.
    
    Args:
        labels: Labels array
        time_seconds: Time to query
        sample_rate: Audio sample rate
        hop_size: Hop size in samples
        
    Returns:
        List of active class names
    """
    frame_idx = time_to_frame(time_seconds, sample_rate, hop_size)
    frame_idx = min(frame_idx, labels.shape[0] - 1)
    
    active = []
    for i, name in enumerate(LABEL_NAMES):
        if labels[frame_idx, i] == 1:
            active.append(name)
    
    return active
