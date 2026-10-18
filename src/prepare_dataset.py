from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LABEL_IDS = set(range(11))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_archive_pairs(archive_path: Path) -> list[tuple[str, str]]:
    """Return (image member, mask member) pairs for imgs/ and masks/ layouts."""
    pairs = []
    with zipfile.ZipFile(archive_path) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        member_set = set(members)
        for name in sorted(members):
            p = PurePosixPath(name)
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parts = list(p.parts)
            try:
                idx = parts.index("imgs")
            except ValueError:
                continue
            mask_parts = parts.copy()
            mask_parts[idx] = "masks"
            mask_name = str(PurePosixPath(*mask_parts))
            if mask_name in member_set:
                pairs.append((name, mask_name))
    return pairs


def split_hashes(hashes: list[str], seed: int) -> dict[str, str]:
    unique_hashes = sorted(set(hashes))
    rng = random.Random(seed)
    rng.shuffle(unique_hashes)
    n = len(unique_hashes)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    # The remainder goes to test, so totals always sum to n.
    train_set = set(unique_hashes[:n_train])
    val_set = set(unique_hashes[n_train:n_train + n_val])
    result = {}
    for h in unique_hashes:
        if h in train_set:
            result[h] = "train"
        elif h in val_set:
            result[h] = "validation"
        else:
            result[h] = "test"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and split MobMask zipped semantic-segmentation data.")
    parser.add_argument("--root", default="data-source/minecraft-segmentation", help="Dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/manifest.csv")
    parser.add_argument("--audit", default="reports/dataset_audit.json")
    args = parser.parse_args()

    root = Path(args.root)
    archives = [root / "data" / "low_seg.zip", root / "data" / "high_seg.zip"]
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(f"Missing archive: {archive}")

    records = []
    pair_count_by_archive = Counter()
    bad_records = []
    pixel_counts = Counter()
    mask_shapes = Counter()
    unique_image_hashes = set()
    duplicate_image_count = 0

    # Hash image bytes to detect exact duplicates before splitting.
    # Each mask is decoded and audited for shape and class IDs.
    seen_hashes = set()
    for archive_path in archives:
        print(f"\nScanning {archive_path.name} ...")
        pairs = find_archive_pairs(archive_path)
        print(f"  paired image/mask files: {len(pairs):,}")
        with zipfile.ZipFile(archive_path) as zf:
            for i, (image_member, mask_member) in enumerate(pairs, start=1):
                try:
                    image_bytes = zf.read(image_member)
                    mask_bytes = zf.read(mask_member)
                    image_hash = sha256_bytes(image_bytes)
                    with Image.open(__import__("io").BytesIO(image_bytes)) as im:
                        im.verify()
                    with Image.open(__import__("io").BytesIO(image_bytes)) as im:
                        image_size = im.size
                        image_mode = im.mode
                    with Image.open(__import__("io").BytesIO(mask_bytes)) as m:
                        m.load()
                        mask = np.asarray(m)
                        mask_size = m.size
                        mask_mode = m.mode
                    if mask.ndim != 2:
                        # Do not silently convert RGB masks: that could alter label encoding.
                        raise ValueError(f"Expected single-channel mask, got shape {mask.shape}")
                    observed = set(int(v) for v in np.unique(mask))
                    unknown = sorted(observed - LABEL_IDS)
                    missing = sorted(LABEL_IDS - observed)
                    if unknown:
                        raise ValueError(f"Unexpected label IDs: {unknown}")
                    if image_size != mask_size:
                        raise ValueError(f"Image/mask size mismatch: {image_size} vs {mask_size}")
                    if image_size != (256, 256):
                        # Record non-native dimensions as an audit warning, not an automatic failure.
                        pass

                    is_duplicate = image_hash in seen_hashes
                    if is_duplicate:
                        duplicate_image_count += 1
                    else:
                        seen_hashes.add(image_hash)
                        unique_image_hashes.add(image_hash)

                    pixel_counts.update({int(k): int(v) for k, v in zip(*np.unique(mask, return_counts=True))})
                    mask_shapes[str(mask.shape)] += 1
                    records.append({
                        "archive": archive_path.name,
                        "image_member": image_member,
                        "mask_member": mask_member,
                        "image_sha256": image_hash,
                        "split": "",
                        "image_width": image_size[0],
                        "image_height": image_size[1],
                        "image_mode": image_mode,
                        "mask_mode": mask_mode,
                        "mask_shape": f"{mask.shape[1]}x{mask.shape[0]}",
                        "observed_label_ids": ";".join(map(str, sorted(observed))),
                        "missing_label_ids": ";".join(map(str, missing)),
                        "exact_duplicate_image": str(is_duplicate).lower()
                    })
                except (KeyError, UnidentifiedImageError, OSError, ValueError, zipfile.BadZipFile) as exc:
                    bad_records.append({
                        "archive": archive_path.name,
                        "image_member": image_member,
                        "mask_member": mask_member,
                        "error": str(exc)
                    })
                pair_count_by_archive[archive_path.name] += 1
                if i % 1000 == 0:
                    print(f"  checked {i:,}/{len(pairs):,}")

    # Split by unique image hash so exact duplicate images cannot cross splits.
    split_by_hash = split_hashes([r["image_sha256"] for r in records], args.seed)
    for r in records:
        r["split"] = split_by_hash[r["image_sha256"]]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "archive", "image_member", "mask_member", "image_sha256", "split",
        "image_width", "image_height", "image_mode", "mask_mode", "mask_shape",
        "observed_label_ids", "missing_label_ids", "exact_duplicate_image"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    total_pixels = sum(pixel_counts.values())
    audit = {
        "dataset": "klima7/minecraft-segmentation",
        "task": "semantic_segmentation",
        "seed": args.seed,
        "label_mapping_status": "unverified",
        "assumed_label_ids": sorted(LABEL_IDS),
        "archives": dict(pair_count_by_archive),
        "valid_pairs": len(records),
        "invalid_pairs": len(bad_records),
        "unique_exact_image_hashes": len(unique_image_hashes),
        "duplicate_image_occurrences": duplicate_image_count,
        "split_counts": dict(Counter(r["split"] for r in records)),
        "mask_shapes": dict(mask_shapes),
        "pixel_counts_by_label_id": {str(k): v for k, v in sorted(pixel_counts.items())},
        "pixel_percent_by_label_id": {
            str(k): round(v / total_pixels * 100, 5) if total_pixels else 0
            for k, v in sorted(pixel_counts.items())
        },
        "invalid_examples": bad_records[:100],
        "notes": [
            "Splits are deterministic and grouped by exact image SHA-256.",
            "Near-duplicate scenes are not detected by this script.",
            "Class names and ignore-index semantics remain unverified.",
            "Label IDs outside 0..10 are reported as invalid; update LABEL_IDS only after investigating the dataset."
        ]
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print("\n=== AUDIT SUMMARY ===")
    print(f"Valid pairs: {len(records):,}")
    print(f"Invalid pairs: {len(bad_records):,}")
    print(f"Unique exact image hashes: {len(unique_image_hashes):,}")
    print(f"Duplicate image occurrences: {duplicate_image_count:,}")
    print(f"Split counts: {dict(Counter(r['split'] for r in records))}")
    print(f"Pixel counts: {dict(sorted(pixel_counts.items()))}")
    print(f"Manifest: {out_path}")
    print(f"Audit JSON: {audit_path}")
    if bad_records:
        print("WARNING: invalid pairs found; inspect reports/dataset_audit.json before training.")


if __name__ == "__main__":
    main()
