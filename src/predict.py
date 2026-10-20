
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from train_segmentation import UNet

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
ARCHIVE_ROOT = ROOT / "data-source" / "minecraft-segmentation"
CHECKPOINT = ROOT / "checkpoints" / "mobmask_unet_best.pt"
OUTPUT = ROOT / "runs" / "predictions"


def main():
    df = pd.read_csv(MANIFEST)
    row = df[df["split"].str.lower() == "test"].iloc[0]

    archive_path = ARCHIVE_ROOT / row["archive"]
    if not archive_path.exists():
        # Some manifests store paths such as data/low_seg.zip.
        archive_path = ARCHIVE_ROOT / "data" / Path(row["archive"]).name

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}\n"
            f"Manifest archive value: {row['archive']}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(row["image_member"]) as file:
            image = Image.open(file).convert("RGB").copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )

    num_classes = checkpoint.get("num_classes", 11)
    model = UNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original_size = image.size
    resized = image.resize((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        prediction = model(tensor).argmax(dim=1)[0].cpu().numpy()

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        palette[i] = [
            (i * 67 + 50) % 256,
            (i * 113 + 90) % 256,
            (i * 173 + 130) % 256,
        ]
    palette[0] = [35, 35, 35]

    mask = Image.fromarray(palette[prediction]).resize(
        original_size, Image.Resampling.NEAREST
    )
    overlay = Image.blend(image, mask, 0.45)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    mask.save(OUTPUT / "test_prediction_mask.png")
    overlay.save(OUTPUT / "test_prediction_overlay.png")

    print(f"Test image: {row['image_member']}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device: {device}")
    print(f"Saved: {OUTPUT / 'test_prediction_mask.png'}")
    print(f"Saved: {OUTPUT / 'test_prediction_overlay.png'}")


if __name__ == "__main__":
    main()
