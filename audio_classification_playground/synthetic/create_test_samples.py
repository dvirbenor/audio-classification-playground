"""Create synthetic test audio samples for pipeline testing.

This script generates placeholder audio files that can be used to test
the synthetic audio generation pipeline. These are NOT real speech/music/sfx
samples, but synthetic audio that allows testing the pipeline.

IMPORTANT: The synthetic audio generator now uses natural duration playback:
- Audio samples play at their natural duration (up to a max specified in timeline)
- Samples are NOT looped - if a 10s sample is placed in a 30s segment, only 10s plays
- Labels are generated from actual rendered content, not planned durations

The test samples created here have varied durations (5-25s) to properly test
this natural duration behavior. Replace with real samples for actual use.
"""

import numpy as np
from pathlib import Path
import torchaudio
import torch


SAMPLE_RATE = 32000


def generate_speech_like(duration: float, seed: int = 0) -> np.ndarray:
    """Generate speech-like audio (amplitude modulated noise + formants).
    
    This is NOT real speech, just a placeholder for testing.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(duration * SAMPLE_RATE)
    
    # Create noise
    noise = rng.normal(0, 1, num_samples).astype(np.float32)
    
    # Apply formant-like filtering (simple bandpass simulation)
    # Just low-pass for simplicity
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 3000 / (SAMPLE_RATE / 2), btype='low')
    filtered = filtfilt(b, a, noise).astype(np.float32)
    
    # Amplitude modulation (syllable-like rhythm)
    t = np.linspace(0, duration, num_samples)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # ~4 Hz modulation
    modulation = modulation.astype(np.float32)
    
    audio = filtered * modulation
    
    # Normalize
    audio = audio / np.abs(audio).max() * 0.8
    
    return audio


def generate_music_like(duration: float, seed: int = 0) -> np.ndarray:
    """Generate music-like audio (harmonics + rhythm).
    
    This is NOT real music, just a placeholder for testing.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples)
    
    # Base frequency (musical note)
    base_freq = 220  # A3
    
    # Create harmonic series
    audio = np.zeros(num_samples, dtype=np.float32)
    for harmonic in range(1, 6):
        freq = base_freq * harmonic
        amplitude = 1.0 / harmonic
        audio += amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    # Add some chord changes
    chord_freqs = [261.63, 329.63, 392.00]  # C major chord
    for freq in chord_freqs:
        audio += 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    # Add rhythm (beat-like modulation)
    beat_rate = 2  # 120 BPM / 60
    beat_mod = 0.3 + 0.7 * (np.sin(2 * np.pi * beat_rate * t) > 0).astype(np.float32)
    audio = audio * beat_mod
    
    # Normalize
    audio = audio / np.abs(audio).max() * 0.8
    
    return audio


def generate_sfx_like(duration: float, sfx_type: str = "burst", seed: int = 0) -> np.ndarray:
    """Generate SFX-like audio (short transient sounds).
    
    This is NOT real SFX, just a placeholder for testing.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples)
    
    if sfx_type == "burst":
        # Short noise burst with fast decay
        noise = rng.normal(0, 1, num_samples).astype(np.float32)
        envelope = np.exp(-t * 8).astype(np.float32)
        audio = noise * envelope
        
    elif sfx_type == "beep":
        # Short beep tone
        freq = rng.uniform(800, 2000)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Apply envelope
        envelope = np.exp(-t * 3).astype(np.float32)
        audio = audio * envelope
        
    elif sfx_type == "click":
        # Very short click
        audio = np.zeros(num_samples, dtype=np.float32)
        click_samples = min(int(0.01 * SAMPLE_RATE), num_samples)
        audio[:click_samples] = rng.normal(0, 1, click_samples).astype(np.float32)
        envelope = np.exp(-t * 20).astype(np.float32)
        audio = audio * envelope
        
    else:
        # Default: noise burst
        noise = rng.normal(0, 1, num_samples).astype(np.float32)
        envelope = np.exp(-t * 5).astype(np.float32)
        audio = noise * envelope
    
    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.9
    
    return audio


def create_test_samples(resources_dir: Path, varied_durations: bool = True) -> None:
    """Create all test sample files.
    
    Args:
        resources_dir: Path to resources directory
        varied_durations: If True, create samples with varied durations (5-25s)
                         to test natural duration playback. If False, use fixed
                         durations (30s speech, 60s music) for backward compatibility.
    """
    resources_dir = Path(resources_dir)
    
    # Create directories
    (resources_dir / "speech").mkdir(parents=True, exist_ok=True)
    (resources_dir / "music").mkdir(parents=True, exist_ok=True)
    (resources_dir / "sfx").mkdir(parents=True, exist_ok=True)
    
    print("Creating test audio samples...")
    if varied_durations:
        print("(Using varied durations to test natural duration playback)")
    
    # Speech samples - varied durations to test natural playback
    if varied_durations:
        speech_durations = [8.0, 12.0, 18.0, 25.0]  # Varied: 8s, 12s, 18s, 25s
    else:
        speech_durations = [30.0, 30.0]  # Fixed: 30s each
    
    for i, duration in enumerate(speech_durations):
        audio = generate_speech_like(duration, seed=i)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        path = resources_dir / "speech" / f"sample_{i+1}.wav"
        torchaudio.save(str(path), audio_tensor, SAMPLE_RATE)
        print(f"Created: {path} ({duration:.1f}s)")
    
    # Music samples - varied durations to test natural playback
    if varied_durations:
        music_durations = [10.0, 15.0, 22.0, 35.0]  # Varied: 10s, 15s, 22s, 35s
    else:
        music_durations = [60.0, 60.0]  # Fixed: 60s each
    
    for i, duration in enumerate(music_durations):
        audio = generate_music_like(duration, seed=i + 10)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        path = resources_dir / "music" / f"sample_{i+1}.wav"
        torchaudio.save(str(path), audio_tensor, SAMPLE_RATE)
        print(f"Created: {path} ({duration:.1f}s)")
    
    # SFX samples (1-5 seconds each) - already short, natural duration matters
    sfx_types = ["burst", "beep", "click", "burst", "beep", "click"]
    sfx_durations = [1.5, 2.0, 0.5, 3.0, 4.5, 1.0]
    
    for i, (sfx_type, duration) in enumerate(zip(sfx_types, sfx_durations)):
        audio = generate_sfx_like(duration, sfx_type=sfx_type, seed=i + 20)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        path = resources_dir / "sfx" / f"sample_{i+1}.wav"
        torchaudio.save(str(path), audio_tensor, SAMPLE_RATE)
        print(f"Created: {path} ({duration:.1f}s)")
    
    print(f"\nAll test samples created in {resources_dir}")
    print("NOTE: These are synthetic placeholder samples for testing only.")
    print("Replace with real speech/music/SFX samples for actual use.")
    if varied_durations:
        print("\nSamples have varied durations to test natural duration playback.")
        print("The generator will play each sample at its actual duration (not looped).")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create test audio samples")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Resources directory (default: resources/)")
    parser.add_argument("--fixed-durations", action="store_true",
                       help="Use fixed durations (30s speech, 60s music) instead of varied")
    
    args = parser.parse_args()
    
    if args.output:
        resources_dir = Path(args.output)
    else:
        # Default to resources/ in workspace root
        resources_dir = Path(__file__).parent.parent.parent / "resources"
    
    create_test_samples(resources_dir, varied_durations=not args.fixed_durations)
