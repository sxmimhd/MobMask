
from __future__ import annotations

import json
import random
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MinecraftSegmentationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, archive_root: Path):
        self.df = manifest.reset_index(drop=True)
        self.archive_root = archive_root
        self.archives = {}
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _archive(self, name: str):
        if name not in self.archives:
            self.archives[name] = zipfile.ZipFile(
                self.archive_root / "data" / name, "r"
            )
        return self.archives[name]

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        archive = self._archive(row["archive"])

        with archive.open(row["image_member"]) as f:
            image_bytes = f.read()
        with archive.open(row["mask_member"]) as f:
            mask_bytes = f.read()

        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        with Image.open(BytesIO(mask_bytes)) as mask:
            mask = mask.convert("L")
            mask_array = np.asarray(mask, dtype=np.int64)

        # Native images/masks are 256x256; avoid resizing categorical masks.
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array.copy())

        return image_tensor, mask_tensor

    def close(self):
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 11, base_channels: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(3, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, 2, stride=2
        )
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


def calculate_metrics(confusion: torch.Tensor) -> dict:
    confusion = confusion.double()
    tp = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - tp
    target_count = confusion.sum(1)

    valid_iou = union > 0
    iou = torch.full_like(tp, float("nan"))
    iou[valid_iou] = tp[valid_iou] / union[valid_iou]

    denominator = confusion.sum(0) + confusion.sum(1)
    dice = torch.full_like(tp, float("nan"))
    valid_dice = denominator > 0
    dice[valid_dice] = 2 * tp[valid_dice] / denominator[valid_dice]

    total = confusion.sum().clamp_min(1)
    return {
        "pixel_accuracy": (tp.sum() / total).item(),
        "mean_iou": torch.nanmean(iou).item(),
        "mean_dice": torch.nanmean(dice).item(),
        "per_class_iou": [
            None if torch.isnan(value) else value.item() for value in iou
        ],
        "per_class_dice": [
            None if torch.isnan(value) else value.item() for value in dice
        ],
        "target_pixels_per_class": target_count.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_images = 0
    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.int64, device=device
    )

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)

        predictions = logits.argmax(dim=1)
        encoded = masks.reshape(-1) * num_classes + predictions.reshape(-1)
        confusion += torch.bincount(
            encoded,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    metrics = calculate_metrics(confusion.cpu())
    metrics["loss"] = total_loss / max(total_images, 1)
    return metrics


def main():
    config = yaml.safe_load(
        (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    )

    seed = int(config["project"]["seed"])
    seed_everything(seed)

    num_classes = int(config["dataset"]["num_classes"])
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    workers = int(config["training"]["workers"])

    device = torch.device(
        "cuda" if config["hardware"]["device"] == "cuda"
        and torch.cuda.is_available() else "cpu"
    )

    manifest_path = ROOT / config["outputs"]["manifest"]
    df = pd.read_csv(manifest_path)

    # Use only the existing, audited splits.
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    if train_df.empty or val_df.empty:
        raise RuntimeError("Training or validation split is empty.")

    valid_ids = set(config["dataset"]["label_ids"])
    for subset_name, subset in [("train", train_df), ("validation", val_df)]:
        observed = set()
        for value in subset["observed_label_ids"].dropna():
            observed.update(int(x) for x in str(value).split(";") if x)
        unexpected = observed - valid_ids
        if unexpected:
            raise ValueError(
                f"Unexpected mask IDs in {subset_name}: {sorted(unexpected)}"
            )

    archive_root = ROOT / config["dataset"]["root"]
    train_ds = MinecraftSegmentationDataset(train_df, archive_root)
    val_ds = MinecraftSegmentationDataset(val_df, archive_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    checkpoint_dir = ROOT / config["outputs"]["checkpoints_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "mobmask_unet_best.pt"

    reports_dir = ROOT / config["outputs"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / "training_history.json"

    history = []
    best_miou = -1.0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train images: {len(train_ds):,}")
    print(f"Validation images: {len(val_ds):,}")
    print(f"Classes: {num_classes}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            start = time.time()

            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                seen += images.size(0)

            train_loss = running_loss / max(seen, 1)
            val_metrics = evaluate(
                model, val_loader, criterion, device, num_classes
            )
            scheduler.step(val_metrics["mean_iou"])

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                "val_mean_iou": val_metrics["mean_iou"],
                "val_mean_dice": val_metrics["mean_dice"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - start,
            }
            history.append(record)
            history_path.write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"mIoU={val_metrics['mean_iou']:.4f} | "
                f"Dice={val_metrics['mean_dice']:.4f} | "
                f"pixel_acc={val_metrics['pixel_accuracy']:.4f} | "
                f"{record['epoch_seconds']:.1f}s"
            )

            if val_metrics["mean_iou"] > best_miou:
                best_miou = val_metrics["mean_iou"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_mean_iou": best_miou,
                        "config": config,
                        "num_classes": num_classes,
                    },
                    best_path,
                )
                print(f"  Saved best checkpoint: {best_path.name}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        print(f"\nBest validation mIoU: {best_miou:.4f}")
        print(f"Checkpoint: {best_path}")
        print(f"History: {history_path}")

    finally:
        train_ds.close()
        val_ds.close()


if __name__ == "__main__":
    main()
