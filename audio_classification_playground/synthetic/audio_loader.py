"""Audio loader for loading and preprocessing audio samples from resources directory."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torchaudio
import torch

# Default sample rate matching PANNs inference
DEFAULT_SAMPLE_RATE = 32000


@dataclass
class AudioSample:
    """Container for a single audio sample with metadata."""
    
    audio: np.ndarray  # Audio waveform (mono, float32)
    duration: float  # Duration in seconds
    name: str = ""  # Optional filename/identifier
    
    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        return len(self.audio)


@dataclass
class AudioSources:
    """Container for loaded audio sources by category."""
    
    speech: List[AudioSample] = field(default_factory=list)
    music: List[AudioSample] = field(default_factory=list)
    sfx: List[AudioSample] = field(default_factory=list)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    
    def get_random_sample(self, category: str, rng: Optional[np.random.Generator] = None) -> AudioSample:
        """Get a random sample from the specified category.
        
        Args:
            category: One of 'speech', 'music', 'sfx'
            rng: Optional random number generator for reproducibility
            
        Returns:
            AudioSample with audio waveform and metadata
        """
        samples = getattr(self, category)
        if not samples:
            raise ValueError(f"No samples available for category: {category}")
        
        if rng is None:
            rng = np.random.default_rng()
        
        idx = rng.integers(0, len(samples))
        return samples[idx]
    
    def get_sample_by_idx(self, category: str, idx: int) -> AudioSample:
        """Get a specific sample by index from the specified category.
        
        Args:
            category: One of 'speech', 'music', 'sfx'
            idx: Sample index (will wrap around if out of bounds)
            
        Returns:
            AudioSample with audio waveform and metadata
        """
        samples = getattr(self, category)
        if not samples:
            raise ValueError(f"No samples available for category: {category}")
        
        # Wrap index to valid range
        idx = idx % len(samples)
        return samples[idx]


def load_audio_file(
    file_path: Path,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True
) -> Tuple[np.ndarray, float]:
    """Load and preprocess a single audio file.
    
    Args:
        file_path: Path to the audio file
        target_sample_rate: Target sample rate for resampling
        normalize: Whether to normalize audio to prevent clipping
        
    Returns:
        Tuple of (audio waveform as numpy array (mono, float32), duration in seconds)
    """
    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(str(file_path))
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
    
    # Convert to numpy
    audio = waveform.squeeze().numpy().astype(np.float32)
    
    # Calculate duration
    duration = len(audio) / target_sample_rate
    
    # Normalize to prevent clipping when mixing
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9  # Leave some headroom
    
    return audio, duration


def discover_audio_files(directory: Path) -> List[Path]:
    """Discover all audio files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of paths to audio files
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    files = []
    
    if directory.exists():
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                files.append(file_path)
    
    return sorted(files)


def load_audio_sources(
    resources_dir: Path,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True
) -> AudioSources:
    """Load all audio sources from the resources directory.
    
    Args:
        resources_dir: Path to resources directory containing speech/, music/, sfx/ subdirs
        target_sample_rate: Target sample rate for all audio
        normalize: Whether to normalize audio levels
        
    Returns:
        AudioSources container with loaded audio samples (with metadata)
    """
    resources_dir = Path(resources_dir)
    sources = AudioSources(sample_rate=target_sample_rate)
    
    categories = ['speech', 'music', 'sfx']
    
    for category in categories:
        category_dir = resources_dir / category
        audio_files = discover_audio_files(category_dir)
        
        if not audio_files:
            print(f"Warning: No audio files found in {category_dir}")
            continue
        
        loaded_samples: List[AudioSample] = []
        for file_path in audio_files:
            try:
                audio, duration = load_audio_file(
                    file_path,
                    target_sample_rate=target_sample_rate,
                    normalize=normalize
                )
                sample = AudioSample(
                    audio=audio,
                    duration=duration,
                    name=file_path.name
                )
                loaded_samples.append(sample)
                print(f"Loaded {category}: {file_path.name} ({duration:.2f}s)")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        setattr(sources, category, loaded_samples)
    
    return sources


def prepare_audio_segment(
    audio: np.ndarray,
    target_duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    loop: bool = True
) -> np.ndarray:
    """Prepare an audio segment to match target duration.
    
    Args:
        audio: Input audio waveform
        target_duration: Target duration in seconds
        sample_rate: Sample rate of the audio
        loop: Whether to loop audio if too short
        
    Returns:
        Audio segment of the specified duration
    """
    target_samples = int(target_duration * sample_rate)
    current_samples = len(audio)
    
    if current_samples >= target_samples:
        # Trim to target length
        return audio[:target_samples]
    elif loop:
        # Loop to fill target length
        repeats = int(np.ceil(target_samples / current_samples))
        looped = np.tile(audio, repeats)
        return looped[:target_samples]
    else:
        # Pad with zeros
        padded = np.zeros(target_samples, dtype=audio.dtype)
        padded[:current_samples] = audio
        return padded
