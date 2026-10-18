
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data-source" / "minecraft-segmentation"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".xml", ".csv"}
IGNORED_DIRS = {".git", ".cache", "__pycache__"}


def count_files_by_extension(root: Path) -> Counter:
    """Count files by extension without opening image contents."""
    counts = Counter()

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        counts[path.suffix.lower() or "[no extension]"] += 1

    return counts


def print_directory_structure(root: Path, max_depth: int = 3) -> None:
    """Print a compact directory tree."""
    print("\n=== DIRECTORY STRUCTURE ===")
    print(root)

    for path in sorted(root.rglob("*")):
        if not path.is_dir() and not path.is_file():
            continue

        relative = path.relative_to(root)

        if any(part in IGNORED_DIRS for part in relative.parts):
            continue

        if len(relative.parts) > max_depth:
            continue

        indent = "  " * len(relative.parts)
        suffix = "/" if path.is_dir() else ""
        print(f"{indent}{path.name}{suffix}")


def inspect_metadata_file(path: Path) -> None:
    """Print a small preview of likely metadata or annotation files."""
    print(f"\n--- Sample file: {path.relative_to(DATASET_ROOT)} ---")

    try:
        if path.stat().st_size > 1_000_000:
            print("Skipped preview: file exceeds 1 MB.")
            return

        text = path.read_text(encoding="utf-8-sig", errors="replace")

        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
            preview = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
            preview = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        else:
            preview = text

        print(preview[:2500])
        if len(preview) > 2500:
            print("\n... preview truncated ...")

    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"Could not preview file: {exc}")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}\n"
            "Check the download location before continuing."
        )

    print("MobMask Dataset Inspector")
    print("=" * 60)
    print(f"Dataset root: {DATASET_ROOT}")

    extension_counts = count_files_by_extension(DATASET_ROOT)

    print("\n=== FILE COUNTS BY EXTENSION ===")
    for extension, count in extension_counts.most_common():
        print(f"{extension:18} {count:>8}")

    image_count = sum(
        count for extension, count in extension_counts.items()
        if extension in IMAGE_EXTENSIONS
    )

    print(f"\nTotal recognized image files: {image_count}")

    print_directory_structure(DATASET_ROOT)

    candidates = []

    for path in DATASET_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        if path.suffix.lower() not in ANNOTATION_EXTENSIONS:
            continue

        name = path.name.lower()
        likely_metadata = (
            path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".xml"}
            or any(
                word in name
                for word in (
                    "annotation", "mask", "label", "segment",
                    "class", "metadata", "dataset", "info",
                )
            )
        )

        if likely_metadata:
            candidates.append(path)

    print(f"\n=== POSSIBLE ANNOTATION / METADATA FILES: {len(candidates)} ===")

    for path in candidates[:30]:
        print(f"{path.relative_to(DATASET_ROOT)} ({path.stat().st_size:,} bytes)")

    if len(candidates) > 30:
        print(f"... and {len(candidates) - 30} more files")

    print("\n=== SAMPLE ANNOTATION / METADATA CONTENTS ===")

    for path in candidates[:5]:
        inspect_metadata_file(path)

    print("\nInspection complete.")
    print("No files were modified.")


if __name__ == "__main__":
    main()
