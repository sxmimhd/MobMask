
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
ARCHIVE_DIR = ROOT / "data-source" / "minecraft-segmentation"
REPORT_DIR = ROOT / "reports"

ARCHIVES = ["low_seg.zip", "high_seg.zip"]


def read_mask(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member) as stream:
        data = stream.read()

    with Image.open(BytesIO(data)) as image:
        image.load()
        return np.asarray(image.convert("L")).copy()


def main() -> None:
    df = pd.read_csv(MANIFEST)

    archives = {}
    try:
        for name in ARCHIVES:
            archives[name] = zipfile.ZipFile(
                ARCHIVE_DIR / "data" / name, "r"
            )

        conflict_groups = []
        pairwise_results = []

        duplicates = df.groupby("image_sha256")

        for image_hash, group in duplicates:
            if len(group) < 2:
                continue

            variants = {}

            for _, row in group.iterrows():
                mask = read_mask(
                    archives[row["archive"]],
                    row["mask_member"],
                )

                # Hash decoded pixel data, not the compressed PNG bytes.
                signature = hashlib.sha256(
                    str(mask.shape).encode() + mask.tobytes()
                ).hexdigest()

                if signature not in variants:
                    variants[signature] = {
                        "mask": mask,
                        "records": [],
                    }

                variants[signature]["records"].append({
                    "archive": row["archive"],
                    "image_member": row["image_member"],
                    "mask_member": row["mask_member"],
                    "split": row["split"],
                })

            if len(variants) < 2:
                continue

            variant_summaries = []

            for signature, item in variants.items():
                mask = item["mask"]
                labels, counts = np.unique(mask, return_counts=True)

                variant_summaries.append({
                    "mask_signature": signature,
                    "record_count": len(item["records"]),
                    "records": item["records"],
                    "label_pixel_counts": {
                        str(int(label)): int(count)
                        for label, count in zip(labels, counts)
                    },
                })

            conflict_groups.append({
                "image_sha256": image_hash,
                "total_records": int(len(group)),
                "mask_variant_count": len(variants),
                "variants": variant_summaries,
            })

            # Compare each distinct mask variant with every other variant.
            variant_items = list(variants.items())

            for i in range(len(variant_items)):
                sig_a, item_a = variant_items[i]
                mask_a = item_a["mask"]

                for j in range(i + 1, len(variant_items)):
                    sig_b, item_b = variant_items[j]
                    mask_b = item_b["mask"]

                    if mask_a.shape != mask_b.shape:
                        pairwise_results.append({
                            "image_sha256": image_hash,
                            "mask_a": sig_a,
                            "mask_b": sig_b,
                            "shape_a": str(mask_a.shape),
                            "shape_b": str(mask_b.shape),
                            "different_pixels": None,
                            "different_pixel_percent": None,
                            "status": "shape_mismatch",
                        })
                        continue

                    different = mask_a != mask_b
                    different_pixels = int(different.sum())
                    total_pixels = int(different.size)

                    pairwise_results.append({
                        "image_sha256": image_hash,
                        "mask_a": sig_a,
                        "mask_b": sig_b,
                        "shape_a": str(mask_a.shape),
                        "shape_b": str(mask_b.shape),
                        "different_pixels": different_pixels,
                        "different_pixel_percent": round(
                            100 * different_pixels / total_pixels, 4
                        ),
                        "status": "compared",
                    })

    finally:
        for archive in archives.values():
            archive.close()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    group_path = REPORT_DIR / "mask_conflict_groups.json"
    pairs_path = REPORT_DIR / "mask_conflict_comparisons.csv"

    group_path.write_text(
        json.dumps(conflict_groups, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(pairwise_results).to_csv(
        pairs_path, index=False
    )

    comparable = [
        item for item in pairwise_results
        if item["status"] == "compared"
    ]

    percentages = [
        item["different_pixel_percent"] for item in comparable
    ]

    print("\n=== MASK CONFLICT ANALYSIS ===")
    print(f"Conflicting image groups: {len(conflict_groups):,}")
    print(f"Distinct mask comparisons: {len(pairwise_results):,}")

    if percentages:
        values = np.asarray(percentages)
        print(f"Minimum disagreement: {values.min():.4f}%")
        print(f"Median disagreement: {np.median(values):.4f}%")
        print(f"Maximum disagreement: {values.max():.4f}%")
        print(
            "Comparisons with <= 1% disagreement:",
            f"{int((values <= 1).sum()):,}",
        )
        print(
            "Comparisons with > 10% disagreement:",
            f"{int((values > 10).sum()):,}",
        )

    print(f"\nDetailed groups: {group_path.relative_to(ROOT)}")
    print(f"Pixel comparisons: {pairs_path.relative_to(ROOT)}")

    if comparable:
        print("\nLargest disagreements:")
        result = pd.DataFrame(comparable)
        print(
            result.sort_values(
                "different_pixel_percent", ascending=False
            ).head(10).to_string(index=False)
        )


if __name__ == "__main__":
    main()
