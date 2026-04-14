"""Download real audio samples from public datasets.

OPTIONAL: This script is provided as a convenience for downloading sample audio.
You can provide your own audio samples instead by placing them in:
  - resources/speech/  (speech audio files)
  - resources/music/   (music audio files)  
  - resources/sfx/     (sound effects files)

NATURAL DURATION PLAYBACK:
The synthetic audio generator now uses natural duration playback:
- Audio samples play at their actual duration (not looped)
- Samples shorter than the timeline segment will end early
- Labels are generated from actual rendered content

This means samples of ANY duration work correctly - short samples (e.g., 5-15s)
are often better than long samples as they provide more variety.

Datasets used by this script:
- Speech: LibriSpeech (via direct FLAC download)
- Music: GTZAN or Free Music Archive  
- SFX: ESC-50 environmental sounds
"""

import io
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import urllib.request
import ssl

import numpy as np
import torch
import torchaudio
import soundfile as sf

# Workaround for SSL issues on some systems
ssl._create_default_https_context = ssl._create_unverified_context

SAMPLE_RATE = 32000


def download_to_bytes(url: str, desc: str = "") -> bytes:
    """Download a file and return bytes."""
    print(f"  Downloading {desc}...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        print(f"OK ({len(data) // 1024} KB)")
        return data
    except Exception as e:
        print(f"FAILED: {e}")
        raise


def resample_and_save(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    output_path: Path,
    normalize: bool = True,
    max_duration: Optional[float] = None
) -> None:
    """Resample audio and save to file."""
    # Ensure float32
    if audio.dtype != np.float32:
        if np.issubdtype(audio.dtype, np.integer):
            # Convert int to float
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)
    
    # Convert to tensor
    if audio.ndim == 1:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    else:
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.shape[0] > audio_tensor.shape[1]:
            audio_tensor = audio_tensor.T
    
    # Convert to mono
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    
    # Trim to max duration before resampling
    if max_duration:
        max_samples = int(max_duration * orig_sr)
        audio_tensor = audio_tensor[:, :max_samples]
    
    # Resample
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio_tensor = resampler(audio_tensor)
    
    # Normalize
    if normalize:
        max_val = audio_tensor.abs().max()
        if max_val > 0:
            audio_tensor = audio_tensor / max_val * 0.9
    
    # Save
    torchaudio.save(str(output_path), audio_tensor, target_sr)


def load_audio_from_bytes(data: bytes, format_hint: str = "flac") -> Tuple[np.ndarray, int]:
    """Load audio from bytes using soundfile."""
    with io.BytesIO(data) as f:
        audio, sr = sf.read(f)
    return audio, sr


def download_speech_samples(resources_dir: Path, num_samples: int = 2) -> None:
    """Download speech samples from LibriSpeech via OpenSLR.
    
    Uses direct FLAC file URLs from the LibriSpeech test-clean subset.
    """
    speech_dir = resources_dir / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Downloading Speech Samples (LibriSpeech) ===")
    
    # Direct links to LibriSpeech FLAC files from test-clean
    # These are hosted on OpenSLR
    base_url = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main"
    
    # Sample URLs from the parquet metadata
    # We'll use HuggingFace's raw file hosting
    sample_urls = [
        ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/audio/1272-128104-0000.flac", "speaker_1272"),
        ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/audio/1462-170142-0000.flac", "speaker_1462"),
        ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/audio/1673-143396-0000.flac", "speaker_1673"),
    ]
    
    count = 0
    for url, speaker in sample_urls[:num_samples]:
        try:
            data = download_to_bytes(url, f"speech {count + 1}")
            audio, sr = load_audio_from_bytes(data)
            
            # These are short samples, use them directly
            output_path = speech_dir / f"sample_{count + 1}_{speaker}.wav"
            resample_and_save(audio, sr, SAMPLE_RATE, output_path)
            
            duration = len(audio) / sr
            print(f"  Saved: {output_path.name} ({duration:.1f}s)")
            count += 1
            
        except Exception as e:
            print(f"  Error with speech sample: {e}")
    
    if count < num_samples:
        print(f"  Note: Only got {count}/{num_samples} speech samples")
    print(f"Downloaded {count} speech samples")


def download_music_samples(resources_dir: Path, num_samples: int = 2) -> None:
    """Download music samples from public sources.
    
    Uses Free Music Archive or similar CC-licensed sources.
    """
    music_dir = resources_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Downloading Music Samples ===")
    
    # Sample music from Free Music Archive (CC-licensed)
    # These are small preview clips that are freely available
    sample_urls = [
        ("https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/blues.00000.wav", "blues"),
        ("https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/classical.00000.wav", "classical"),
        ("https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/jazz.00000.wav", "jazz"),
    ]
    
    count = 0
    for url, genre in sample_urls[:num_samples]:
        try:
            data = download_to_bytes(url, f"music/{genre}")
            audio, sr = load_audio_from_bytes(data)
            
            output_path = music_dir / f"sample_{count + 1}_{genre}.wav"
            resample_and_save(audio, sr, SAMPLE_RATE, output_path, max_duration=60.0)
            
            duration = min(len(audio) / sr, 60.0)
            print(f"  Saved: {output_path.name} ({duration:.1f}s)")
            count += 1
            
        except Exception as e:
            print(f"  Error with music sample: {e}")
    
    if count < num_samples:
        print(f"  Note: Only got {count}/{num_samples} music samples")
    print(f"Downloaded {count} music samples")


def download_sfx_samples(resources_dir: Path, num_samples: int = 6) -> None:
    """Download SFX samples from ESC-50 via HuggingFace.
    
    ESC-50 contains 5-second environmental sound clips.
    """
    sfx_dir = resources_dir / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Downloading SFX Samples (ESC-50) ===")
    
    # ESC-50 sample files from HuggingFace
    # Format: folder/file_id-category.wav
    sample_urls = [
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-100032-A-0.wav", "dog"),
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-100038-A-14.wav", "chirping_birds"),
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-100210-A-36.wav", "vacuum_cleaner"),
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-101296-A-19.wav", "thunderstorm"),
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-103298-A-22.wav", "siren"),
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-104089-A-23.wav", "car_horn"),
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-110389-A-0.wav", "dog_2"),
        ("https://huggingface.co/datasets/ashraq/esc50/resolve/main/audio/1-116765-A-41.wav", "train"),
    ]
    
    count = 0
    for url, category in sample_urls[:num_samples]:
        try:
            data = download_to_bytes(url, f"sfx/{category}")
            audio, sr = load_audio_from_bytes(data)
            
            output_path = sfx_dir / f"sample_{count + 1}_{category}.wav"
            resample_and_save(audio, sr, SAMPLE_RATE, output_path)
            
            duration = len(audio) / sr
            print(f"  Saved: {output_path.name} ({duration:.1f}s)")
            count += 1
            
        except Exception as e:
            print(f"  Error with SFX sample: {e}")
    
    if count < num_samples:
        print(f"  Note: Only got {count}/{num_samples} SFX samples")
    print(f"Downloaded {count} SFX samples")


def clear_existing_samples(resources_dir: Path) -> None:
    """Remove existing sample files."""
    for subdir in ["speech", "music", "sfx"]:
        dir_path = resources_dir / subdir
        if dir_path.exists():
            for f in dir_path.glob("*.wav"):
                f.unlink()
            for f in dir_path.glob("*.mp3"):
                f.unlink()
            for f in dir_path.glob("*.flac"):
                f.unlink()


def download_all_samples(
    resources_dir: Path,
    num_speech: int = 2,
    num_music: int = 2,
    num_sfx: int = 6,
    clear_existing: bool = True
) -> None:
    """Download all sample types from public datasets.
    
    Args:
        resources_dir: Path to resources directory
        num_speech: Number of speech samples to download
        num_music: Number of music samples to download
        num_sfx: Number of SFX samples to download
        clear_existing: Whether to remove existing samples first
    """
    resources_dir = Path(resources_dir)
    
    print("=" * 60)
    print("Downloading Real Audio Samples from Public Datasets")
    print("=" * 60)
    print(f"Target: {resources_dir}")
    print(f"Speech: {num_speech} (LibriSpeech)")
    print(f"Music: {num_music} (GTZAN)")
    print(f"SFX: {num_sfx} (ESC-50)")
    
    if clear_existing:
        print("\nClearing existing samples...")
        clear_existing_samples(resources_dir)
    
    # Download each type
    download_speech_samples(resources_dir, num_speech)
    download_music_samples(resources_dir, num_music)
    download_sfx_samples(resources_dir, num_sfx)
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    # Summary
    for subdir in ["speech", "music", "sfx"]:
        dir_path = resources_dir / subdir
        files = list(dir_path.glob("*.wav"))
        print(f"\n{subdir}: {len(files)} files")
        for f in sorted(files):
            audio, sr = torchaudio.load(str(f))
            duration = audio.shape[1] / sr
            print(f"  - {f.name} ({duration:.1f}s)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real audio samples")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Resources directory (default: resources/)")
    parser.add_argument("--speech", type=int, default=2,
                       help="Number of speech samples")
    parser.add_argument("--music", type=int, default=2,
                       help="Number of music samples")
    parser.add_argument("--sfx", type=int, default=6,
                       help="Number of SFX samples")
    parser.add_argument("--keep-existing", action="store_true",
                       help="Keep existing samples instead of clearing")
    
    args = parser.parse_args()
    
    if args.output:
        resources_dir = Path(args.output)
    else:
        resources_dir = Path(__file__).parent.parent.parent / "resources"
    
    download_all_samples(
        resources_dir,
        num_speech=args.speech,
        num_music=args.music,
        num_sfx=args.sfx,
        clear_existing=not args.keep_existing
    )
