
from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from io import BytesIO


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.csv"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "duplicate_audit.json"

ARCHIVE_DIR = ROOT / "data-source" / "minecraft-segmentation"
ARCHIVE_NAMES = ["low_seg.zip", "high_seg.zip"]


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    required = {
        "archive",
        "image_member",
        "mask_member",
        "image_sha256",
        "split",
        "exact_duplicate_image",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    # Group exact duplicate images by SHA-256.
    groups = {
        image_hash: group
        for image_hash, group in df.groupby("image_sha256")
        if len(group) > 1
    }

    duplicate_rows = sum(len(group) for group in groups.values())
    repeated_occurrences = duplicate_rows - len(groups)

    split_leakage = []
    cross_archive_groups = []
    mask_conflicts = []
    same_mask_groups = []

    # Open archives once and cache masks only for duplicate groups.
    archives: dict[str, zipfile.ZipFile] = {}
    try:
        for name in ARCHIVE_NAMES:
            path = ARCHIVE_DIR / "data" / name
            if not path.exists():
                raise FileNotFoundError(f"Dataset archive not found: {path}")
            archives[name] = zipfile.ZipFile(path, "r")

        for image_hash, group in groups.items():
            splits = sorted(group["split"].dropna().unique().tolist())
            sources = sorted(group["archive"].dropna().unique().tolist())

            records = group[
                ["archive", "image_member", "mask_member", "split"]
            ].to_dict("records")

            if len(splits) > 1:
                split_leakage.append(
                    {
                        "image_sha256": image_hash,
                        "splits": splits,
                        "records": records,
                    }
                )

            if len(sources) > 1:
                cross_archive_groups.append(
                    {
                        "image_sha256": image_hash,
                        "archives": sources,
                        "records": records,
                    }
                )

            # Compare decoded mask pixel arrays for every image duplicate.
            mask_signatures = defaultdict(list)

            for _, row in group.iterrows():
                archive_name = row["archive"]
                member = row["mask_member"]

                with archives[archive_name].open(member) as stream:
                    mask_bytes = stream.read()

                with Image.open(BytesIO(mask_bytes)) as mask_image:
                    mask_image.load()
                    mask = mask_image.convert("L")

                    # Include dimensions in the signature.
                    signature = (
                        mask.size,
                        mask.tobytes(),
                    )

                mask_signatures[signature].append(
                    {
                        "archive": archive_name,
                        "image_member": row["image_member"],
                        "mask_member": member,
                        "split": row["split"],
                    }
                )

            if len(mask_signatures) > 1:
                mask_conflicts.append(
                    {
                        "image_sha256": image_hash,
                        "mask_variants": [
                            {
                                "record_count": len(records_for_mask),
                                "records": records_for_mask,
                            }
                            for records_for_mask in mask_signatures.values()
                        ],
                    }
                )
            else:
                same_mask_groups.append(image_hash)

    finally:
        for archive in archives.values():
            archive.close()

    split_counts = df["split"].value_counts().to_dict()

    report = {
        "manifest_rows": int(len(df)),
        "unique_image_hashes": int(df["image_sha256"].nunique()),
        "duplicate_groups": int(len(groups)),
        "rows_in_duplicate_groups": int(duplicate_rows),
        "repeated_occurrences_beyond_unique": int(repeated_occurrences),
        "split_counts": {
            str(key): int(value) for key, value in split_counts.items()
        },
        "duplicate_groups_crossing_splits": int(len(split_leakage)),
        "duplicate_groups_across_archives": int(len(cross_archive_groups)),
        "duplicate_groups_with_conflicting_masks": int(len(mask_conflicts)),
        "duplicate_groups_with_identical_masks": int(len(same_mask_groups)),
        "split_leakage_examples": split_leakage[:20],
        "cross_archive_examples": cross_archive_groups[:20],
        "mask_conflict_examples": mask_conflicts[:20],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== MOBMASK DUPLICATE AUDIT ===")
    print(f"Manifest rows: {report['manifest_rows']:,}")
    print(f"Unique image hashes: {report['unique_image_hashes']:,}")
    print(f"Duplicate groups: {report['duplicate_groups']:,}")
    print(
        "Repeated occurrences beyond unique:",
        f"{report['repeated_occurrences_beyond_unique']:,}",
    )
    print(
        "Duplicate groups crossing splits:",
        report["duplicate_groups_crossing_splits"],
    )
    print(
        "Duplicate groups across archives:",
        report["duplicate_groups_across_archives"],
    )
    print(
        "Duplicate groups with conflicting masks:",
        report["duplicate_groups_with_conflicting_masks"],
    )
    print(
        "Duplicate groups with identical masks:",
        report["duplicate_groups_with_identical_masks"],
    )
    print(f"\nDetailed report: {REPORT_PATH.relative_to(ROOT)}")

    if split_leakage:
        print("\nWARNING: Exact duplicate images cross dataset splits.")
    else:
        print("\nPASS: No exact duplicate image groups cross dataset splits.")

    if mask_conflicts:
        print(
            "WARNING: Some identical images have different masks. "
            "Inspect the report before training."
        )
    else:
        print("PASS: All exact duplicate image groups have consistent masks.")


if __name__ == "__main__":
    main()
