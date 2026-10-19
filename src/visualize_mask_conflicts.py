
from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data-source" / "minecraft-segmentation"
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.csv"
GROUPS_PATH = ROOT / "reports" / "mask_conflict_groups.json"
COMPARISONS_PATH = ROOT / "reports" / "mask_conflict_comparisons.csv"
OUTPUT_PATH = ROOT / "reports" / "figures" / "mask_conflict_contact_sheet.png"

ARCHIVE_NAMES = ["low_seg.zip", "high_seg.zip"]
TOP_N = 12
PANEL_SIZE = 192
LABEL_HEIGHT = 42
PADDING = 12

# Visualization-only palette. It does not imply semantic class names.
PALETTE = [
    (35, 35, 35),
    (230, 75, 75),
    (75, 160, 230),
    (80, 190, 120),
    (230, 190, 70),
    (170, 100, 220),
    (70, 210, 200),
    (230, 130, 65),
    (130, 160, 90),
    (220, 110, 170),
    (200, 200, 200),
]


def read_bytes(archive, member):
    with archive.open(member) as stream:
        return stream.read()


def read_mask(archive, member):
    with Image.open(BytesIO(read_bytes(archive, member))) as image:
        image.load()
        return np.asarray(image.convert("L")).copy()


def colorize(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in np.unique(mask):
        if int(label) < len(PALETTE):
            rgb[mask == label] = PALETTE[int(label)]
        else:
            rgb[mask == label] = (255, 255, 255)
    return Image.fromarray(rgb, mode="RGB")


def fit_image(image):
    image = image.convert("RGB")
    image.thumbnail((PANEL_SIZE, PANEL_SIZE))
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), "white")
    x = (PANEL_SIZE - image.width) // 2
    y = (PANEL_SIZE - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main():
    groups = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    comparisons = pd.read_csv(COMPARISONS_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    # Pick one representative comparison per image, ranked by disagreement.
    comparisons = comparisons[
        comparisons["status"] == "compared"
    ].sort_values("different_pixel_percent", ascending=False)

    worst = comparisons.drop_duplicates("image_sha256").head(TOP_N)

    archives = {}
    try:
        for name in ARCHIVE_NAMES:
            archives[name] = zipfile.ZipFile(
                DATA_ROOT / "data" / name, "r"
            )

        group_by_hash = {
            item["image_sha256"]: item for item in groups
        }

        panel_count = len(worst)
        cell_width = PANEL_SIZE + PADDING
        row_height = PANEL_SIZE + LABEL_HEIGHT + PADDING
        sheet_width = PADDING + 4 * cell_width
        sheet_height = PADDING + panel_count * row_height

        sheet = Image.new("RGB", (sheet_width, sheet_height), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()

        headers = [
            "Original image",
            "Mask A",
            "Mask B",
            "Disagreement",
        ]

        for col, header in enumerate(headers):
            x = PADDING + col * cell_width
            draw.text((x, 2), header, fill="black", font=font)

        for row_index, (_, comparison) in enumerate(worst.iterrows()):
            image_hash = comparison["image_sha256"]
            group = group_by_hash[image_hash]
            variants = group["variants"]

            sig_a = comparison["mask_a"]
            sig_b = comparison["mask_b"]

            variant_a = next(v for v in variants
                             if v["mask_signature"] == sig_a)
            variant_b = next(v for v in variants
                             if v["mask_signature"] == sig_b)

            record_a = variant_a["records"][0]
            record_b = variant_b["records"][0]

            mask_a = read_mask(
                archives[record_a["archive"]],
                record_a["mask_member"],
            )
            mask_b = read_mask(
                archives[record_b["archive"]],
                record_b["mask_member"],
            )

            # Both records should contain the same exact image.
            image_bytes = read_bytes(
                archives[record_a["archive"]],
                record_a["image_member"],
            )
            with Image.open(BytesIO(image_bytes)) as original:
                original.load()
                original_panel = fit_image(original)

            difference = mask_a != mask_b
            diff_visual = np.zeros((*difference.shape, 3), dtype=np.uint8)
            diff_visual[~difference] = (35, 35, 35)
            diff_visual[difference] = (255, 50, 50)

            panels = [
                original_panel,
                fit_image(colorize(mask_a)),
                fit_image(colorize(mask_b)),
                fit_image(Image.fromarray(diff_visual)),
            ]

            y = PADDING + row_index * row_height + 18

            for col, panel in enumerate(panels):
                x = PADDING + col * cell_width
                sheet.paste(panel, (x, y))

            percent = float(comparison["different_pixel_percent"])
            group_label = (
                f"#{row_index + 1} | {percent:.2f}% pixels differ"
            )
            draw.text(
                (PADDING, y + PANEL_SIZE + 3),
                group_label,
                fill="black",
                font=font,
            )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(OUTPUT_PATH)

        print("=== CONFLICT VISUALIZATION ===")
        print(f"Groups shown: {panel_count}")
        print(f"Output: {OUTPUT_PATH.relative_to(ROOT)}")

    finally:
        for archive in archives.values():
            archive.close()


if __name__ == "__main__":
    main()
