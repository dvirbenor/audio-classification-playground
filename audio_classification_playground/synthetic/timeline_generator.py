"""Timeline generator for creating layered audio composition timelines."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import json
from pathlib import Path

import numpy as np


TransitionType = Literal["abrupt", "crossfade"]
SegmentType = Literal["speech", "music"]


@dataclass
class BaseSegment:
    """A continuous audio segment (speech or music).
    
    Note: In planned timelines, 'end' represents max duration.
    The actual playback may be shorter based on audio sample length.
    """
    
    start: float  # Start time in seconds
    end: float  # End time in seconds (max duration in planned timeline)
    type: SegmentType  # 'speech' or 'music'
    transition: TransitionType = "abrupt"  # Transition type at the end
    sample_idx: int = 0  # Which sample to use from the category
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SFXEvent:
    """A short SFX event at a specific time.
    
    Note: In planned timelines, 'duration' represents max duration.
    The actual playback may be shorter based on audio sample length.
    """
    
    time: float  # Start time in seconds
    duration: float  # Duration in seconds (max duration in planned timeline)
    sample_idx: int = 0  # Which SFX sample to use
    
    @property
    def end(self) -> float:
        return self.time + self.duration


@dataclass
class ActualSegment:
    """A segment that was actually rendered to audio.
    
    Unlike BaseSegment, this represents what was actually played,
    including the actual duration based on the audio sample length.
    """
    
    start: float  # Actual start time in seconds
    end: float  # Actual end time in seconds
    type: SegmentType  # 'speech' or 'music'
    sample_idx: int = 0  # Which sample was used
    sample_name: str = ""  # Name of the audio file used
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ActualSFXEvent:
    """An SFX event that was actually rendered to audio.
    
    Unlike SFXEvent, this represents what was actually played,
    including the actual duration based on the audio sample length.
    """
    
    start: float  # Actual start time in seconds
    end: float  # Actual end time in seconds
    sample_idx: int = 0  # Which sample was used
    sample_name: str = ""  # Name of the audio file used
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ActualTimeline:
    """Timeline representing what was actually rendered.
    
    This is the ground truth for label generation - it contains
    the exact timing of what was actually mixed into the audio.
    """
    
    segments: List[ActualSegment] = field(default_factory=list)
    sfx_events: List[ActualSFXEvent] = field(default_factory=list)
    duration: float = 180.0  # Total duration in seconds
    
    def to_dict(self) -> dict:
        """Convert actual timeline to dictionary for JSON serialization."""
        return {
            "duration": self.duration,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "type": seg.type,
                    "sample_idx": seg.sample_idx,
                    "sample_name": seg.sample_name
                }
                for seg in self.segments
            ],
            "sfx_events": [
                {
                    "start": evt.start,
                    "end": evt.end,
                    "sample_idx": evt.sample_idx,
                    "sample_name": evt.sample_name
                }
                for evt in self.sfx_events
            ]
        }
    
    def save(self, path: Path) -> None:
        """Save actual timeline to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActualTimeline":
        """Load actual timeline from dictionary."""
        timeline = cls(duration=data.get("duration", 180.0))
        
        for seg_data in data.get("segments", []):
            timeline.segments.append(ActualSegment(
                start=seg_data["start"],
                end=seg_data["end"],
                type=seg_data["type"],
                sample_idx=seg_data.get("sample_idx", 0),
                sample_name=seg_data.get("sample_name", "")
            ))
        
        for evt_data in data.get("sfx_events", []):
            timeline.sfx_events.append(ActualSFXEvent(
                start=evt_data["start"],
                end=evt_data["end"],
                sample_idx=evt_data.get("sample_idx", 0),
                sample_name=evt_data.get("sample_name", "")
            ))
        
        return timeline
    
    @classmethod
    def load(cls, path: Path) -> "ActualTimeline":
        """Load actual timeline from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class Timeline:
    """Complete timeline with base segments and SFX events."""
    
    base_segments: List[BaseSegment] = field(default_factory=list)
    sfx_events: List[SFXEvent] = field(default_factory=list)
    duration: float = 180.0  # Total duration in seconds
    
    def to_dict(self) -> dict:
        """Convert timeline to dictionary for JSON serialization."""
        return {
            "duration": self.duration,
            "base_segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "type": seg.type,
                    "transition": seg.transition,
                    "sample_idx": seg.sample_idx
                }
                for seg in self.base_segments
            ],
            "sfx_events": [
                {
                    "time": evt.time,
                    "duration": evt.duration,
                    "sample_idx": evt.sample_idx
                }
                for evt in self.sfx_events
            ]
        }
    
    def save(self, path: Path) -> None:
        """Save timeline to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Timeline":
        """Load timeline from dictionary."""
        timeline = cls(duration=data.get("duration", 180.0))
        
        for seg_data in data.get("base_segments", []):
            timeline.base_segments.append(BaseSegment(
                start=seg_data["start"],
                end=seg_data["end"],
                type=seg_data["type"],
                transition=seg_data.get("transition", "abrupt"),
                sample_idx=seg_data.get("sample_idx", 0)
            ))
        
        for evt_data in data.get("sfx_events", []):
            timeline.sfx_events.append(SFXEvent(
                time=evt_data["time"],
                duration=evt_data["duration"],
                sample_idx=evt_data.get("sample_idx", 0)
            ))
        
        return timeline
    
    @classmethod
    def load(cls, path: Path) -> "Timeline":
        """Load timeline from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_timeline(
    duration: float = 180.0,
    seed: Optional[int] = None,
    num_speech_samples: int = 2,
    num_music_samples: int = 2,
    num_sfx_samples: int = 4,
    min_segment_duration: float = 5.0,
    max_segment_duration: float = 15.0,
    sfx_min_duration: float = 0.5,
    sfx_max_duration: float = 3.0,
    sfx_density: float = 0.15,  # Probability of SFX per second (increased for more variety)
    overlap_probability: float = 0.35,  # Probability of speech+music overlap
    crossfade_probability: float = 0.5,  # Probability of crossfade transition
) -> Timeline:
    """Generate a random layered timeline.
    
    The timeline consists of:
    - Base segments: continuous speech/music layers (can overlap)
    - SFX events: short sound effects at specific time points
    
    Note: Default segment durations (5-15s) are designed to work well with
    audio samples of up to 15 seconds. Shorter segments mean more segments
    and more variety in the output.
    
    Args:
        duration: Total timeline duration in seconds
        seed: Random seed for reproducibility
        num_speech_samples: Number of speech samples available
        num_music_samples: Number of music samples available
        num_sfx_samples: Number of SFX samples available
        min_segment_duration: Minimum duration for base segments (default 5s)
        max_segment_duration: Maximum duration for base segments (default 15s)
        sfx_min_duration: Minimum SFX duration
        sfx_max_duration: Maximum SFX duration
        sfx_density: Expected SFX events per second
        overlap_probability: Probability of creating overlapping speech+music
        crossfade_probability: Probability of using crossfade transitions
        
    Returns:
        Generated Timeline object
    """
    rng = np.random.default_rng(seed)
    timeline = Timeline(duration=duration)
    
    # Generate base segments
    # Strategy: Alternate between speech-only, music-only, and overlapping periods
    current_time = 0.0
    segment_types: List[SegmentType] = ["speech", "music"]
    
    while current_time < duration:
        # Decide segment type and whether to overlap
        primary_type = segment_types[int(rng.integers(0, 2))]
        create_overlap = rng.random() < overlap_probability
        
        # Generate segment duration
        seg_duration = rng.uniform(min_segment_duration, max_segment_duration)
        seg_duration = min(seg_duration, duration - current_time)
        
        if seg_duration < min_segment_duration / 2:
            break  # Don't create very short segments at the end
        
        # Determine transition type
        transition: TransitionType = "crossfade" if rng.random() < crossfade_probability else "abrupt"
        
        # Create primary segment
        primary_segment = BaseSegment(
            start=current_time,
            end=current_time + seg_duration,
            type=primary_type,
            transition=transition,
            sample_idx=int(rng.integers(0, num_speech_samples if primary_type == "speech" else num_music_samples))
        )
        timeline.base_segments.append(primary_segment)
        
        # Create overlapping segment if needed
        if create_overlap:
            secondary_type: SegmentType = "music" if primary_type == "speech" else "speech"
            
            # Overlap starts partway through primary segment
            overlap_start = current_time + rng.uniform(seg_duration * 0.2, seg_duration * 0.5)
            # Overlap extends beyond primary segment
            overlap_duration = rng.uniform(min_segment_duration, max_segment_duration)
            overlap_end = min(overlap_start + overlap_duration, duration)
            
            if overlap_end - overlap_start >= min_segment_duration / 2:
                secondary_segment = BaseSegment(
                    start=overlap_start,
                    end=overlap_end,
                    type=secondary_type,
                    transition="crossfade" if rng.random() < crossfade_probability else "abrupt",
                    sample_idx=int(rng.integers(0, num_music_samples if secondary_type == "music" else num_speech_samples))
                )
                timeline.base_segments.append(secondary_segment)
        
        current_time += seg_duration
    
    # Sort segments by start time
    timeline.base_segments.sort(key=lambda s: s.start)
    
    # Generate SFX events
    # Place SFX at random points throughout the timeline
    expected_sfx_count = int(duration * sfx_density)
    actual_sfx_count = rng.poisson(expected_sfx_count)
    
    # Ensure at least a few SFX events
    actual_sfx_count = max(actual_sfx_count, 3)
    
    for _ in range(actual_sfx_count):
        sfx_time = rng.uniform(0, duration - sfx_max_duration)
        sfx_duration = rng.uniform(sfx_min_duration, sfx_max_duration)
        
        # Ensure SFX doesn't extend beyond timeline
        sfx_duration = min(sfx_duration, duration - sfx_time)
        
        sfx_event = SFXEvent(
            time=sfx_time,
            duration=sfx_duration,
            sample_idx=int(rng.integers(0, num_sfx_samples))
        )
        timeline.sfx_events.append(sfx_event)
    
    # Sort SFX events by time
    timeline.sfx_events.sort(key=lambda e: e.time)
    
    return timeline


def _is_music_active_in_range(segments: List[BaseSegment], start: float, end: float) -> bool:
    """Check if any music segment overlaps with a time range.
    
    Args:
        segments: List of base segments to check
        start: Start of time range (in seconds)
        end: End of time range (in seconds)
        
    Returns:
        True if music overlaps with the given range
    """
    for seg in segments:
        if seg.type == "music":
            # Check if ranges overlap
            if seg.start < end and seg.end > start:
                return True
    return False


RegionType = Literal["speech", "music", "speech_music", "sfx_only", "silence"]


def generate_structured_timeline(
    duration: float = 180.0,
    seed: Optional[int] = None,
    num_speech_samples: int = 2,
    num_music_samples: int = 2,
    num_sfx_samples: int = 4,
    min_segment_duration: float = 5.0,
    max_segment_duration: float = 15.0,
    sfx_region_duration: float = 4.0,
    silence_duration: float = 3.0,
) -> Timeline:
    """Generate a structured timeline with explicit region types.
    
    Uses region-based approach to guarantee proper SFX placement:
    - Speech regions: speech only, may have SFX
    - Music regions: music only, NO SFX allowed
    - Speech+Music regions: both playing, NO SFX allowed
    - SFX-only regions: guaranteed isolated SFX, no speech/music
    - Silence regions: pure silence, no audio
    
    This ensures SFX ONLY appears during speech-only or sfx-only regions.
    
    Args:
        duration: Total timeline duration in seconds
        seed: Random seed for reproducibility
        num_speech_samples: Number of speech samples available
        num_music_samples: Number of music samples available
        num_sfx_samples: Number of SFX samples available
        min_segment_duration: Minimum segment duration (default 5s)
        max_segment_duration: Maximum segment duration (default 15s)
        sfx_region_duration: Duration for SFX-only regions (default 4s)
        silence_duration: Duration for silence regions (default 3s)
        
    Returns:
        Generated Timeline object
    """
    rng = np.random.default_rng(seed)
    timeline = Timeline(duration=duration)
    
    # ========================================
    # STEP 1: Plan region sequence
    # ========================================
    
    # Calculate how many SFX-only regions we need (roughly every 40-50 seconds)
    num_sfx_regions = max(3, int(duration / 45))
    
    # Calculate approximate positions for SFX-only regions (evenly distributed)
    sfx_region_positions = []
    for i in range(num_sfx_regions):
        # Position at roughly 1/(n+1), 2/(n+1), etc. of the duration
        target_pos = duration * (i + 1) / (num_sfx_regions + 1)
        sfx_region_positions.append(target_pos)
    
    # Define base region type weights (excluding sfx_only which is explicitly placed)
    base_region_weights = {
        "speech": 0.35,
        "music": 0.35,
        "speech_music": 0.20,
        "silence": 0.10,
    }
    
    # Build the region sequence
    regions: List[tuple] = []  # (start, end, region_type)
    current_time = 0.0
    last_region_type: Optional[RegionType] = None
    sfx_regions_placed = 0
    
    while current_time < duration - 2:
        remaining = duration - current_time
        
        # Check if we should place an SFX-only region here
        # Place SFX region if we've reached or passed the target position
        place_sfx_region = False
        if sfx_regions_placed < num_sfx_regions:
            target_pos = sfx_region_positions[sfx_regions_placed]
            # Place SFX region if current_time is at or past the target position
            # OR if adding another regular segment would skip past the target
            if current_time >= target_pos or (current_time + min_segment_duration > target_pos):
                place_sfx_region = True
        
        if place_sfx_region:
            # Place SFX-only region
            region_duration = min(sfx_region_duration, remaining)
            if region_duration >= 2:
                regions.append((current_time, current_time + region_duration, "sfx_only"))
                current_time += region_duration
                sfx_regions_placed += 1
                last_region_type = "sfx_only"
                continue
        
        # Pick a regular region type
        # Avoid repeating the same type more than twice
        available_types = list(base_region_weights.keys())
        
        # Remove last type if it was used twice in a row
        if len(regions) >= 2:
            last_two = [r[2] for r in regions[-2:]]
            if last_two[0] == last_two[1] and last_two[0] in available_types:
                available_types.remove(last_two[0])
        
        # Don't put silence after sfx_only (would be too much gap)
        if last_region_type == "sfx_only" and "silence" in available_types:
            available_types.remove("silence")
        
        # Calculate weights for available types
        weights = [base_region_weights[t] for t in available_types]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Pick region type
        region_type: RegionType = rng.choice(available_types, p=weights)
        
        # Determine duration
        if region_type == "silence":
            region_duration = min(silence_duration, remaining)
        else:
            region_duration = rng.uniform(min_segment_duration, max_segment_duration)
            region_duration = min(region_duration, remaining)
        
        if region_duration < 2:
            break
        
        regions.append((current_time, current_time + region_duration, region_type))
        current_time += region_duration
        last_region_type = region_type
    
    # Debug: print region summary
    sfx_only_count = sum(1 for _, _, t in regions if t == "sfx_only")
    print(f"  Generated {len(regions)} regions: {sfx_only_count} sfx_only regions")
    
    # ========================================
    # STEP 2: Create segments for each region
    # ========================================
    
    sfx_only_regions: List[tuple] = []  # Track for SFX placement
    
    for start, end, region_type in regions:
        region_duration = end - start
        
        if region_type == "speech":
            # Add speech segment
            timeline.base_segments.append(BaseSegment(
                start=start,
                end=end,
                type="speech",
                transition="crossfade" if rng.random() < 0.5 else "abrupt",
                sample_idx=int(rng.integers(0, num_speech_samples))
            ))
            
        elif region_type == "music":
            # Add music segment
            timeline.base_segments.append(BaseSegment(
                start=start,
                end=end,
                type="music",
                transition="crossfade" if rng.random() < 0.5 else "abrupt",
                sample_idx=int(rng.integers(0, num_music_samples))
            ))
            
        elif region_type == "speech_music":
            # Add both speech and music (overlapping)
            timeline.base_segments.append(BaseSegment(
                start=start,
                end=end,
                type="speech",
                transition="crossfade",
                sample_idx=int(rng.integers(0, num_speech_samples))
            ))
            # Music slightly offset for natural feel
            music_start = start + region_duration * 0.05
            music_end = end - region_duration * 0.05
            timeline.base_segments.append(BaseSegment(
                start=music_start,
                end=music_end,
                type="music",
                transition="crossfade",
                sample_idx=int(rng.integers(0, num_music_samples))
            ))
            
        elif region_type == "sfx_only":
            # Track for SFX placement (no base segments)
            sfx_only_regions.append((start, end))
            
        # "silence" - add nothing
    
    # Sort segments
    timeline.base_segments.sort(key=lambda s: s.start)
    
    # ========================================
    # STEP 3: Add SFX events
    # ========================================
    
    # 3A) Add SFX in sfx_only regions (GUARANTEED)
    for sfx_start, sfx_end in sfx_only_regions:
        sfx_duration_region = sfx_end - sfx_start
        # Add 1-2 SFX events
        num_sfx = 1 if sfx_duration_region < 3 else 2
        for i in range(num_sfx):
            offset = (i + 0.5) / num_sfx * (sfx_duration_region - 1.5)
            sfx_time = sfx_start + 0.5 + offset
            sfx_dur = rng.uniform(0.5, min(2.5, sfx_duration_region * 0.4))
            
            if sfx_time + sfx_dur <= sfx_end:
                timeline.sfx_events.append(SFXEvent(
                    time=sfx_time,
                    duration=sfx_dur,
                    sample_idx=int(rng.integers(0, num_sfx_samples))
                ))
    
    # 3B) Add SFX during speech-only regions (optional, with validation)
    speech_only_regions = []
    for start, end, region_type in regions:
        if region_type == "speech":
            speech_only_regions.append((start, end))
    
    for speech_start, speech_end in speech_only_regions:
        speech_duration = speech_end - speech_start
        if speech_duration < 4:
            continue
        
        # 40% chance of adding SFX to a speech region
        if rng.random() < 0.4:
            num_sfx = rng.integers(1, 3)
            for _ in range(num_sfx):
                sfx_time = speech_start + rng.uniform(0.5, speech_duration - 1.5)
                sfx_dur = rng.uniform(0.5, min(2.0, speech_duration * 0.2))
                sfx_end_time = sfx_time + sfx_dur
                
                # Double-check no music overlap (should be impossible but verify)
                if not _is_music_active_in_range(timeline.base_segments, sfx_time, sfx_end_time):
                    if sfx_end_time <= speech_end:
                        timeline.sfx_events.append(SFXEvent(
                            time=sfx_time,
                            duration=sfx_dur,
                            sample_idx=int(rng.integers(0, num_sfx_samples))
                        ))
    
    # Sort SFX events
    timeline.sfx_events.sort(key=lambda e: e.time)
    
    # ========================================
    # STEP 4: Validate (raises error if invalid)
    # ========================================
    
    validate_timeline(timeline)
    
    return timeline


def validate_timeline(timeline: Timeline) -> None:
    """Validate timeline constraints. Raises ValueError if invalid.
    
    Checks:
    - No SFX event overlaps with any music segment
    - At least one SFX event exists
    
    Args:
        timeline: Timeline to validate
        
    Raises:
        ValueError: If any constraint is violated
    """
    violations = []
    
    # Check for SFX overlapping with music
    for sfx in timeline.sfx_events:
        if _is_music_active_in_range(timeline.base_segments, sfx.time, sfx.end):
            violations.append(
                f"SFX at {sfx.time:.2f}-{sfx.end:.2f}s overlaps with music"
            )
    
    # Check for at least some SFX
    if len(timeline.sfx_events) == 0:
        violations.append("No SFX events in timeline")
    
    if violations:
        error_msg = "Timeline validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        raise ValueError(error_msg)
