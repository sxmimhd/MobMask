
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data-source" / "minecraft-segmentation" / "data"
OUTPUT = ROOT / "reports" / "figures" / "mask_inspection.png"
SAMPLES_PER_ARCHIVE = 3


def inspect_archive(archive_path: Path):
    print(f"\n{'=' * 65}\n{archive_path.name}\n{'=' * 65}")

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())

        image_names = {
            name for name in names
            if name.startswith("imgs/")
            and name.lower().endswith(".png")
        }
        mask_names = {
            name for name in names
            if name.startswith("masks/")
            and name.lower().endswith(".png")
        }

        image_stems = {Path(name).stem for name in image_names}
        mask_stems = {Path(name).stem for name in mask_names}

        matched = sorted(image_stems & mask_stems)
        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)

        print(f"Images: {len(image_names):,}")
        print(f"Masks: {len(mask_names):,}")
        print(f"Matched pairs: {len(matched):,}")
        print(f"Images without masks: {len(missing_masks):,}")
        print(f"Masks without images: {len(missing_images):,}")

        samples = []

        for stem in matched[:SAMPLES_PER_ARCHIVE]:
            image_path = f"imgs/{stem}.png"
            mask_path = f"masks/{stem}.png"

            image = Image.open(
                BytesIO(archive.read(image_path))
            ).convert("RGB")

            raw_mask = Image.open(
                BytesIO(archive.read(mask_path))
            )
            mask = raw_mask.copy()

            mask_array = np.asarray(mask)

            print(f"\nSample: {stem}")
            print(
                f"  Image: size={image.size}, mode={image.mode}"
            )
            print(
                f"  Mask:  size={mask.size}, mode={mask.mode}, "
                f"array_shape={mask_array.shape}, "
                f"dtype={mask_array.dtype}"
            )

            if mask_array.ndim == 2:
                values, counts = np.unique(
                    mask_array, return_counts=True
                )
                order = np.argsort(counts)[::-1][:20]
                print("  Most common mask values:")
                print([
                    (int(values[i]), int(counts[i]))
                    for i in order
                ])
            else:
                colors, counts = np.unique(
                    mask_array.reshape(-1, mask_array.shape[-1]),
                    axis=0,
                    return_counts=True,
                )
                order = np.argsort(counts)[::-1][:12]
                print("  Most common mask colors (RGB/RGBA):")
                print([
                    (
                        tuple(int(v) for v in colors[i]),
                        int(counts[i]),
                    )
                    for i in order
                ])

            samples.append((archive_path.stem, stem, image, mask))

        return samples


def make_contact_sheet(samples):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cell_w, cell_h = 320, 250
    label_h = 28
    columns = 3
    rows = len(samples)

    sheet = Image.new(
        "RGB",
        (columns * cell_w, rows * (cell_h + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for row, (archive_name, stem, image, mask) in enumerate(samples):
        y = row * (cell_h + label_h)
        draw.text(
            (8, y + 5),
            f"{archive_name} | {stem}",
            fill="black",
        )

        image.thumbnail((cell_w, cell_h))
        mask_rgb = mask.convert("RGB")
        mask_rgb.thumbnail((cell_w, cell_h))

        # Resize mask only for display if its dimensions differ.
        display_mask = mask_rgb.resize(image.size)

        overlay = Image.blend(image, display_mask, 0.5)

        for col, panel in enumerate((image, display_mask, overlay)):
            x = col * cell_w
            panel.thumbnail((cell_w, cell_h))
            sheet.paste(panel, (x, y + label_h))

        draw.text((8, y + label_h + cell_h - 18), "Original", fill="black")
        draw.text((cell_w + 8, y + label_h + cell_h - 18), "Mask", fill="black")
        draw.text((2 * cell_w + 8, y + label_h + cell_h - 18), "Overlay", fill="black")

    sheet.save(OUTPUT)
    print(f"\nContact sheet saved: {OUTPUT}")


def main():
    all_samples = []

    for filename in ("low_seg.zip", "high_seg.zip"):
        archive_path = DATA / filename
        if not archive_path.exists():
            print(f"Missing archive: {archive_path}")
            continue
        all_samples.extend(inspect_archive(archive_path))

    if all_samples:
        make_contact_sheet(all_samples)


if __name__ == "__main__":
    main()
