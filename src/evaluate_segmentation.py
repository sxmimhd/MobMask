
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from train_segmentation import (
    ROOT,
    MinecraftSegmentationDataset,
    UNet,
    calculate_metrics,
)


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


def colorize(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        rgb[mask == class_id] = PALETTE[int(class_id)]
    return Image.fromarray(rgb)


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    color_mask = colorize(mask).convert("RGB")
    image = image.convert("RGB")
    return Image.blend(image, color_mask, alpha)


@torch.no_grad()
def main():
    config = yaml.safe_load(
        (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    )

    device = torch.device(
        "cuda"
        if config["hardware"]["device"] == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    num_classes = int(config["dataset"]["num_classes"])
    checkpoint_path = (
        ROOT / config["outputs"]["checkpoints_dir"] / "mobmask_unet_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run src/train_segmentation.py first."
        )

    manifest = pd.read_csv(ROOT / config["outputs"]["manifest"])
    test_df = manifest[manifest["split"] == "test"].copy()

    if test_df.empty:
        raise RuntimeError("The test split is empty.")

    dataset = MinecraftSegmentationDataset(
        test_df, ROOT / config["dataset"]["root"]
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = UNet(num_classes=num_classes).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.int64, device=device
    )
    total_loss = 0.0
    total_images = 0
    total_inference_seconds = 0.0
    saved_examples = []
    max_examples = 12
    criterion = nn.CrossEntropyLoss()

    reports_dir = ROOT / config["outputs"]["reports_dir"]
    figures_dir = ROOT / config["outputs"]["figures_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_inference_seconds += time.perf_counter() - start

            loss = criterion(logits, masks)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

            encoded = (
                masks.reshape(-1) * num_classes + predictions.reshape(-1)
            )
            confusion += torch.bincount(
                encoded,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

            for i in range(images.size(0)):
                if len(saved_examples) >= max_examples:
                    break

                image_array = (
                    images[i].detach().cpu().permute(1, 2, 0).numpy()
                    * 255.0
                ).clip(0, 255).astype(np.uint8)

                image_pil = Image.fromarray(image_array)
                target = masks[i].detach().cpu().numpy()
                prediction = predictions[i].detach().cpu().numpy()

                saved_examples.append(
                    (image_pil, target.copy(), prediction.copy())
                )

    finally:
        dataset.close()

    metrics = calculate_metrics(confusion.cpu())
    metrics["test_loss"] = total_loss / max(total_images, 1)
    metrics["test_images"] = len(test_df)
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    metrics["best_validation_miou"] = float(
        checkpoint["best_val_mean_iou"]
    )
    metrics["device"] = str(device)
    metrics["total_inference_seconds"] = total_inference_seconds
    metrics["milliseconds_per_image"] = (
        1000 * total_inference_seconds / max(total_images, 1)
    )
    metrics["images_per_second"] = (
        total_images / max(total_inference_seconds, 1e-9)
    )
    metrics["class_names_status"] = "unverified_class_ids"

    metrics_path = reports_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Build a contact sheet: input, ground truth, prediction, overlay.
    panel_size = 256
    label_height = 28
    padding = 10
    columns = 4
    cell_width = panel_size + padding
    row_height = panel_size + label_height + padding
    sheet = Image.new(
        "RGB",
        (
            padding + columns * cell_width,
            padding + len(saved_examples) * row_height,
        ),
        "#eeeeee",
    )
    draw = ImageDraw.Draw(sheet)
    headers = ["Input", "Ground truth", "Prediction", "Overlay"]

    for col, header in enumerate(headers):
        draw.text(
            (padding + col * cell_width, 2),
            header,
            fill="black",
        )

    for index, (image, target, prediction) in enumerate(saved_examples):
        panels = [
            image,
            colorize(target),
            colorize(prediction),
            overlay_mask(image, prediction),
        ]
        y = padding + index * row_height + 18

        for col, panel in enumerate(panels):
            x = padding + col * cell_width
            panel = panel.resize((panel_size, panel_size))
            sheet.paste(panel, (x, y))

        draw.text(
            (padding, y + panel_size + 3),
            f"Test example {index + 1}",
            fill="black",
        )

    figure_path = figures_dir / "test_predictions.png"
    sheet.save(figure_path)

    print("\n=== MOBMASK TEST RESULTS ===")
    print(f"Test images: {len(test_df):,}")
    print(f"Checkpoint epoch: {metrics['checkpoint_epoch']}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Pixel accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"Milliseconds/image: {metrics['milliseconds_per_image']:.2f}")
    print(f"Images/second: {metrics['images_per_second']:.2f}")

    print("\nPer-class metrics (IDs remain unverified):")
    for class_id in range(num_classes):
        iou = metrics["per_class_iou"][class_id]
        dice = metrics["per_class_dice"][class_id]
        print(
            f"Class {class_id:02d}: "
            f"IoU={iou if iou is not None else 'N/A'} | "
            f"Dice={dice if dice is not None else 'N/A'}"
        )

    print(f"\nMetrics: {metrics_path}")
    print(f"Predictions: {figure_path}")


if __name__ == "__main__":
    main()
