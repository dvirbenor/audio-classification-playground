"""One-time migration: move flat inference error JSON files to per-archive subdirs.

Before this migration:
  _meta/inference_errors/models/{uuid}.json

After this migration:
  _meta/inference_errors/models/{session_id}/{archive_id}/{uuid}.json

Run from a pod with EFS mounted:
  python scripts/migrate_inference_errors_to_per_archive.py \
      /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference
"""
import json
import sys
from pathlib import Path


def migrate(output_base: Path) -> None:
    errors_root = output_base / "_meta" / "inference_errors"
    if not errors_root.is_dir():
        print(f"No inference errors directory at {errors_root}")
        return

    moved = 0
    skipped = 0
    failed = 0

    # Walk all immediate subdirs (task_group dirs like "models", "all", etc.)
    # and the root itself, looking for flat UUID-named JSON files.
    dirs_to_scan = [errors_root] + [d for d in errors_root.iterdir() if d.is_dir()]
    for task_dir in dirs_to_scan:
        for f in list(task_dir.glob("*.json")):
            if f.name.startswith("."):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sid = data["session_id"]
                aid = data["archive_id"]
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                print(f"  SKIP (parse error) {f.name}: {exc}")
                failed += 1
                continue

            target_dir = task_dir / sid / aid
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f.name
            if target.exists():
                f.unlink()
                skipped += 1
                continue
            f.rename(target)
            moved += 1
            if moved % 500 == 0:
                print(f"  moved {moved}...")

    print(f"Done: moved={moved} skipped(duplicate)={skipped} failed={failed}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_base>")
        sys.exit(1)
    migrate(Path(sys.argv[1]))
