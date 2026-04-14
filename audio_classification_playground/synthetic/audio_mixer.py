"""Audio mixer for layered audio composition with precise timing."""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from .audio_loader import AudioSources, AudioSample, prepare_audio_segment, DEFAULT_SAMPLE_RATE
from .timeline_generator import (
    Timeline, BaseSegment, SFXEvent,
    ActualTimeline, ActualSegment, ActualSFXEvent
)


# Mixing weights for different combinations
SPEECH_WEIGHT = 0.6  # When mixed with music
MUSIC_WEIGHT = 0.4  # When mixed with speech
SFX_WEIGHT = 0.5  # SFX injection weight


@dataclass
class RenderedSegment:
    """Information about a rendered audio segment."""
    
    start: float  # Actual start time in seconds
    end: float  # Actual end time in seconds (based on what was rendered)
    type: str  # 'speech' or 'music'
    sample_idx: int  # Which sample was used
    sample_name: str  # Name of the sample used
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RenderedSFX:
    """Information about a rendered SFX event."""
    
    start: float  # Actual start time in seconds
    end: float  # Actual end time in seconds
    sample_idx: int  # Which sample was used
    sample_name: str  # Name of the sample used
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RenderInfo:
    """Complete information about what was actually rendered."""
    
    segments: List[RenderedSegment]
    sfx_events: List[RenderedSFX]
    duration: float  # Total timeline duration


def create_fade_envelope(
    length: int,
    fade_in_samples: int = 0,
    fade_out_samples: int = 0
) -> np.ndarray:
    """Create a fade envelope for smooth transitions.
    
    Args:
        length: Total length of the envelope
        fade_in_samples: Number of samples for fade-in
        fade_out_samples: Number of samples for fade-out
        
    Returns:
        Envelope array with values between 0 and 1
    """
    envelope = np.ones(length, dtype=np.float32)
    
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples, dtype=np.float32)
        envelope[:fade_in_samples] = fade_in
    
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples, dtype=np.float32)
        envelope[-fade_out_samples:] = fade_out
    
    return envelope


def time_to_samples(time_seconds: float, sample_rate: int) -> int:
    """Convert time in seconds to sample index."""
    return int(time_seconds * sample_rate)


def render_base_segment(
    segment: BaseSegment,
    sources: AudioSources,
    sample_rate: int,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, int, int, RenderedSegment]:
    """Render a base segment to audio samples using natural duration (no looping).
    
    The audio plays at its natural duration, capped by the segment's max duration.
    
    Args:
        segment: The segment to render (end is treated as max duration)
        sources: Audio sources container
        sample_rate: Sample rate
        rng: Random number generator
        
    Returns:
        Tuple of (audio array, start sample, actual end sample, rendered info)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Get AudioSample for this category
    audio_sample = sources.get_sample_by_idx(segment.type, segment.sample_idx)
    
    # Compute effective duration: min(audio natural duration, segment max duration)
    max_duration = segment.duration
    effective_duration = min(audio_sample.duration, max_duration)
    
    # Prepare audio WITHOUT looping - use natural duration up to max
    audio = prepare_audio_segment(
        audio_sample.audio,
        effective_duration,
        sample_rate,
        loop=False  # No looping - use natural duration
    )
    
    # Apply crossfade if needed
    if segment.transition == "crossfade":
        fade_samples = int(1.0 * sample_rate)  # 1 second fade
        fade_samples = min(fade_samples, len(audio) // 4)
        envelope = create_fade_envelope(len(audio), fade_in_samples=fade_samples, fade_out_samples=fade_samples)
        audio = audio * envelope
    else:
        # Small fade to avoid clicks even for abrupt transitions
        fade_samples = int(0.01 * sample_rate)  # 10ms micro-fade
        envelope = create_fade_envelope(len(audio), fade_in_samples=fade_samples, fade_out_samples=fade_samples)
        audio = audio * envelope
    
    start_sample = time_to_samples(segment.start, sample_rate)
    end_sample = start_sample + len(audio)
    
    # Create render info with actual times
    actual_end_time = segment.start + (len(audio) / sample_rate)
    rendered = RenderedSegment(
        start=segment.start,
        end=actual_end_time,
        type=segment.type,
        sample_idx=segment.sample_idx % len(getattr(sources, segment.type)),
        sample_name=audio_sample.name
    )
    
    return audio, start_sample, end_sample, rendered


def render_sfx_event(
    event: SFXEvent,
    sources: AudioSources,
    sample_rate: int,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, int, RenderedSFX]:
    """Render an SFX event to audio samples using natural duration.
    
    Args:
        event: The SFX event to render (duration is treated as max)
        sources: Audio sources container
        sample_rate: Sample rate
        rng: Random number generator
        
    Returns:
        Tuple of (audio array, start sample, rendered info)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    sfx_samples = sources.sfx
    if not sfx_samples:
        raise ValueError("No SFX samples available")
    
    # Get AudioSample
    sample_idx = event.sample_idx % len(sfx_samples)
    audio_sample = sfx_samples[sample_idx]
    
    # Use natural duration, capped by event's max duration
    max_samples = time_to_samples(event.duration, sample_rate)
    if audio_sample.num_samples >= max_samples:
        audio = audio_sample.audio[:max_samples].copy()
    else:
        # Use the full SFX sample (natural duration)
        audio = audio_sample.audio.copy()
    
    # Apply small fade for smooth injection
    fade_samples = int(0.02 * sample_rate)  # 20ms fade
    fade_samples = min(fade_samples, len(audio) // 4)
    envelope = create_fade_envelope(len(audio), fade_in_samples=fade_samples, fade_out_samples=fade_samples)
    audio = audio * envelope
    
    start_sample = time_to_samples(event.time, sample_rate)
    
    # Create render info with actual times
    actual_duration = len(audio) / sample_rate
    rendered = RenderedSFX(
        start=event.time,
        end=event.time + actual_duration,
        sample_idx=sample_idx,
        sample_name=audio_sample.name
    )
    
    return audio, start_sample, rendered


def compute_overlap_weights_from_render_info(
    render_info: RenderInfo,
    num_samples: int,
    sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample weights for speech and music based on actual rendered overlaps.
    
    Args:
        render_info: Information about what was actually rendered
        num_samples: Total number of samples
        sample_rate: Sample rate
        
    Returns:
        Tuple of (speech_weights, music_weights) arrays
    """
    speech_weights = np.ones(num_samples, dtype=np.float32)
    music_weights = np.ones(num_samples, dtype=np.float32)
    
    # Find overlapping regions based on actual rendered segments
    speech_active = np.zeros(num_samples, dtype=bool)
    music_active = np.zeros(num_samples, dtype=bool)
    
    for segment in render_info.segments:
        start_sample = time_to_samples(segment.start, sample_rate)
        end_sample = time_to_samples(segment.end, sample_rate)
        end_sample = min(end_sample, num_samples)
        
        if segment.type == "speech":
            speech_active[start_sample:end_sample] = True
        else:
            music_active[start_sample:end_sample] = True
    
    # Where both are active, apply mixing weights
    overlap = speech_active & music_active
    speech_weights[overlap] = SPEECH_WEIGHT
    music_weights[overlap] = MUSIC_WEIGHT
    
    return speech_weights, music_weights


def mix_audio(
    timeline: Timeline,
    sources: AudioSources,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, RenderInfo]:
    """Mix audio according to timeline with layered approach.
    
    Audio samples play at their natural duration (up to max), without looping.
    Returns both the mixed audio and information about what was actually rendered.
    
    Args:
        timeline: Timeline with base segments and SFX events (end times are max durations)
        sources: Audio sources container
        sample_rate: Target sample rate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mixed audio as numpy array, RenderInfo with actual playback info)
    """
    rng = np.random.default_rng(seed)
    
    # Initialize buffers
    num_samples = time_to_samples(timeline.duration, sample_rate)
    speech_buffer = np.zeros(num_samples, dtype=np.float32)
    music_buffer = np.zeros(num_samples, dtype=np.float32)
    sfx_buffer = np.zeros(num_samples, dtype=np.float32)
    
    # Track what was actually rendered
    rendered_segments: List[RenderedSegment] = []
    rendered_sfx: List[RenderedSFX] = []
    
    # Render base segments to appropriate buffers
    for segment in timeline.base_segments:
        try:
            audio, start_sample, end_sample, rendered = render_base_segment(
                segment, sources, sample_rate, rng
            )
            
            # Ensure we don't exceed buffer length
            end_sample = min(end_sample, num_samples)
            audio_length = end_sample - start_sample
            
            if segment.type == "speech":
                # Add to speech buffer (mix if already present)
                speech_buffer[start_sample:end_sample] += audio[:audio_length]
            else:
                # Add to music buffer
                music_buffer[start_sample:end_sample] += audio[:audio_length]
            
            # Track what was rendered
            rendered_segments.append(rendered)
                
        except Exception as e:
            print(f"Warning: Failed to render segment: {e}")
    
    # Render SFX events
    for event in timeline.sfx_events:
        try:
            audio, start_sample, rendered = render_sfx_event(event, sources, sample_rate, rng)
            
            end_sample = min(start_sample + len(audio), num_samples)
            audio_length = end_sample - start_sample
            
            sfx_buffer[start_sample:end_sample] += audio[:audio_length]
            
            # Track what was rendered
            rendered_sfx.append(rendered)
            
        except Exception as e:
            print(f"Warning: Failed to render SFX event: {e}")
    
    # Create RenderInfo
    render_info = RenderInfo(
        segments=rendered_segments,
        sfx_events=rendered_sfx,
        duration=timeline.duration
    )
    
    # Compute overlap-aware weights based on actual rendered content
    speech_weights, music_weights = compute_overlap_weights_from_render_info(
        render_info, num_samples, sample_rate
    )
    
    # Apply weights and mix
    weighted_speech = speech_buffer * speech_weights
    weighted_music = music_buffer * music_weights
    weighted_sfx = sfx_buffer * SFX_WEIGHT
    
    # Combine all layers
    mixed = weighted_speech + weighted_music + weighted_sfx
    
    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 0:
        mixed = mixed / max_val * 0.95  # Leave some headroom
    
    return mixed, render_info


def render_info_to_actual_timeline(render_info: RenderInfo) -> ActualTimeline:
    """Convert RenderInfo to ActualTimeline for label generation and storage.
    
    Args:
        render_info: The render information from mix_audio
        
    Returns:
        ActualTimeline with the same data in a format suitable for label generation
    """
    actual_timeline = ActualTimeline(duration=render_info.duration)
    
    # Convert rendered segments
    for seg in render_info.segments:
        actual_timeline.segments.append(ActualSegment(
            start=seg.start,
            end=seg.end,
            type=seg.type,
            sample_idx=seg.sample_idx,
            sample_name=seg.sample_name
        ))
    
    # Convert rendered SFX events
    for sfx in render_info.sfx_events:
        actual_timeline.sfx_events.append(ActualSFXEvent(
            start=sfx.start,
            end=sfx.end,
            sample_idx=sfx.sample_idx,
            sample_name=sfx.sample_name
        ))
    
    return actual_timeline
