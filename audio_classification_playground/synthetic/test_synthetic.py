"""Test script for synthetic audio generation."""

import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent / "audio-classification-playground"))

from synthetic import generate_test_audio, visualize_from_files


def main():
    output_dir = Path("../../outputs")
    resources_dir = Path("../../resources")
    
    print("=" * 60)
    print("Testing Synthetic Audio Generation Pipeline")
    print("=" * 60)
    
    # Generate test audio (10 seconds for quick testing)
    result = generate_test_audio(
        output_dir=output_dir,
        resources_dir=resources_dir,
        duration=10.0,
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("Generation Results")
    print("=" * 60)
    print(f"Audio file: {result.audio_path}")
    print(f"  Duration: {result.duration:.2f} seconds")
    print(f"  Samples: {len(result.audio)}")
    print(f"Labels file: {result.labels_path}")
    print(f"  Shape: {result.labels.shape}")
    print(f"  Frames: {result.num_frames}")
    print(f"Timeline file: {result.timeline_path}")
    print(f"  Base segments: {len(result.timeline.base_segments)}")
    print(f"  SFX events: {len(result.timeline.sfx_events)}")
    
    # Show label statistics
    from synthetic.label_generator import labels_to_dict
    stats = labels_to_dict(result.labels)
    print("\nLabel Statistics:")
    print(f"  Speech active: {stats['speech_percentage']:.1f}% of frames")
    print(f"  Music active: {stats['music_percentage']:.1f}% of frames")
    print(f"  SFX active: {stats['sfx_percentage']:.1f}% of frames")
    print("\nClass combinations (frame counts):")
    for combo, count in stats['combinations'].items():
        print(f"  {combo}: {count} frames")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
