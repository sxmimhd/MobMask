
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data-source" / "minecraft-segmentation" / "data"
FIGURES_DIR = ROOT / "reports" / "figures"

ARCHIVES = ["low_seg.zip", "high_seg.zip"]

# A distinct display color for each possible label ID.
PALETTE = [
    (0, 0, 0),
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (255, 225, 25),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 128, 128),
]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    total_pixels = Counter()
    image_presence = Counter()
    mask_shapes = Counter()
    all_samples = []

    for archive_name in ARCHIVES:
        archive_path = DATA_DIR / archive_name
        print(f"\n{'=' * 60}\n{archive_name}\n{'=' * 60}")

        with ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            image_names = sorted(
                n for n in names
                if n.startswith("imgs/") and n.lower().endswith(".png")
            )
            mask_names = {
                n for n in names
                if n.startswith("masks/") and n.lower().endswith(".png")
            }

            stems = [
                Path(n).stem for n in image_names
                if f"masks/{Path(n).stem}.png" in mask_names
            ]
            print(f"Matched pairs: {len(stems):,}")

            for index, stem in enumerate(stems):
                mask_path = f"masks/{stem}.png"

                with Image.open(BytesIO(archive.read(mask_path))) as im:
                    mask = np.asarray(im.convert("L"))

                mask_shapes[mask.shape] += 1
                values, counts = np.unique(mask, return_counts=True)

                for value, count in zip(values, counts):
                    value = int(value)
                    total_pixels[value] += int(count)
                    image_presence[value] += 1

                # Save a few representative masks for visual inspection.
                if index in (0, 1, 2, len(stems) // 2, len(stems) - 1):
                    image_path = f"imgs/{stem}.png"
                    with Image.open(BytesIO(archive.read(image_path))) as im:
                        original = im.convert("RGB").copy()

                    colored = np.zeros(
                        (*mask.shape, 3), dtype=np.uint8
                    )
                    for value in values:
                        if value < len(PALETTE):
                            colored[mask == value] = PALETTE[int(value)]
                        else:
                            # Unknown IDs get a visible fallback color.
                            colored[mask == value] = (255, 255, 255)

                    colored_image = Image.fromarray(colored)
                    overlay = Image.blend(original, colored_image, 0.55)
                    all_samples.append(
                        (archive_name, stem, original, colored_image, overlay)
                    )

                if (index + 1) % 5000 == 0:
                    print(f"  Audited {index + 1:,} masks...")

    print("\nMASK SHAPES")
    for shape, count in sorted(mask_shapes.items()):
        print(f"{shape}: {count:,} masks")

    print("\nLABEL DISTRIBUTION")
    print("ID | Pixel count | Pixel % | Images containing label")
    grand_total = sum(total_pixels.values())

    for label_id, pixels in sorted(total_pixels.items()):
        percentage = 100 * pixels / grand_total
        print(
            f"{label_id:2} | {pixels:12,} | {percentage:7.3f}% "
            f"| {image_presence[label_id]:,}"
        )

    # Create a categorical mask visualization.
    cell_w, cell_h, label_h = 256, 256, 24
    columns = 3
    rows = len(all_samples)
    sheet = Image.new(
        "RGB",
        (columns * cell_w, rows * (cell_h + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for row, (archive, stem, original, mask, overlay) in enumerate(all_samples):
        y = row * (cell_h + label_h)
        draw.text((5, y + 4), f"{archive} | {stem}", fill="black")

        for col, panel in enumerate((original, mask, overlay)):
            sheet.paste(panel.resize((cell_w, cell_h)), (col * cell_w, y + label_h))

        draw.text((5, y + label_h + cell_h - 16), "Original", fill="white")
        draw.text((cell_w + 5, y + label_h + cell_h - 16), "Colored mask", fill="white")
        draw.text((2 * cell_w + 5, y + label_h + cell_h - 16), "Overlay", fill="white")

    output = FIGURES_DIR / "mask_label_audit.png"
    sheet.save(output)
    print(f"\nSaved visualization: {output}")


if __name__ == "__main__":
    main()
