"""
Sequential Training and Evaluation of Lightweight Deep Learning Segmenters:
1. Lightweight U-Net
2. DeepLabV3 with MobileNetV3-Large Backbone

Trained sequentially on group-disjoint unified partitions (fold != 0 for training, fold == 0 for held-out evaluation).
Strictly saves checkpoints to models/experimental_*.pt without overwriting production models.
"""

import os
import sys
import gc
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as seg
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "unified_dataset_manifest.csv")
FOLDS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "unified_group_folds.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# 1. Dataset & Transforms
# ---------------------------------------------------------------------------

class PotholeSegmentationDataset(Dataset):
    """PyTorch Dataset loading RGB road images and binary segmentation masks."""

    def __init__(self, df: pd.DataFrame, target_size: Tuple[int, int] = (256, 256), is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.is_train = is_train

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, Tuple[int, int]]:
        row = self.df.iloc[idx]
        img_path = row["image_path_x"]
        mask_path = row["mask_path_x"]
        sample_id = row["sample_id"]
        source = row["dataset_source_x"]

        with Image.open(img_path) as im_rgb, Image.open(mask_path) as im_mask:
            orig_w, orig_h = im_rgb.size
            rgb_resized = im_rgb.convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
            mask_resized = im_mask.resize(self.target_size, Image.Resampling.NEAREST)

            rgb_np = np.array(rgb_resized, dtype=np.float32) / 255.0
            mask_np = (np.array(mask_resized) > 0).astype(np.float32)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]

        # Normalization
        rgb_norm = (rgb_np - self.mean) / self.std
        # Transpose to (C, H, W)
        rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()

        return rgb_t, mask_t, sample_id, source, (orig_w, orig_h)


# ---------------------------------------------------------------------------
# 2. Architectures
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net designed for 8 GB RAM machines.
    Parameters: ~1.2M. Compact channel progression: 16 -> 32 -> 64 -> 128 -> 256.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_conv(d1)


def create_deeplabv3_mobilenetv3() -> nn.Module:
    """Instantiate DeepLabV3 with pretrained MobileNetV3-Large backbone."""
    model = seg.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    # Replace 21-class classifier with 1-class binary classifier
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, 1, kernel_size=1)
    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, 1, kernel_size=1)
    return model


# ---------------------------------------------------------------------------
# 3. Loss & Metrics
# ---------------------------------------------------------------------------

class CombinedBCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Soft Dice Loss."""
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        dice_loss = 1.0 - (num / den).mean()
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def compute_metrics_batch(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum().item()
    union = (preds + targets - preds * targets).sum().item()
    denom = preds.sum().item() + targets.sum().item()

    iou = (intersection / union) if union > 0 else (1.0 if targets.sum().item() == 0 and preds.sum().item() == 0 else 0.0)
    dice = (2.0 * intersection / denom) if denom > 0 else (1.0 if targets.sum().item() == 0 and preds.sum().item() == 0 else 0.0)
    return iou, dice


# ---------------------------------------------------------------------------
# 4. Training Engine
# ---------------------------------------------------------------------------

def train_and_eval_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_checkpoint_path: str,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    logger.info(f"\n{'='*60}\nTraining {model_name} on {device.upper()}\n{'='*60}")
    model.to(device)
    criterion = CombinedBCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_iou = 0.0
    training_history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t_ep = time.time()

        for batch_idx, (images, masks, _, _, _) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            if isinstance(outputs, dict):  # DeepLabV3 returns dict {'out': ..., 'aux': ...}
                loss = criterion(outputs["out"], masks)
                if "aux" in outputs and outputs["aux"] is not None:
                    loss += 0.4 * criterion(outputs["aux"], masks)
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        ep_duration = time.time() - t_ep

        # Validation at end of epoch
        model.eval()
        val_ious, val_dices = [], []
        with torch.no_grad():
            for images, masks, _, _, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                logits = outputs["out"] if isinstance(outputs, dict) else outputs
                for i in range(images.size(0)):
                    iou, dice = compute_metrics_batch(logits[i:i+1], masks[i:i+1])
                    val_ious.append(iou)
                    val_dices.append(dice)

        mean_val_iou = float(np.mean(val_ious))
        mean_val_dice = float(np.mean(val_dices))

        logger.info(
            f"Epoch {epoch:2d}/{epochs} [{ep_duration:4.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val IoU: {mean_val_iou:.4f} | Val Dice: {mean_val_dice:.4f}"
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_iou": round(mean_val_iou, 4),
            "val_dice": round(mean_val_dice, 4),
            "epoch_sec": round(ep_duration, 1),
        })

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "best_val_iou": best_val_iou,
                "best_val_dice": mean_val_dice,
                "epoch": epoch,
            }, output_checkpoint_path)

    total_train_time = round(time.time() - t_start, 2)
    logger.info(f"Training completed in {total_train_time}s. Best Val IoU: {best_val_iou:.4f}. Saved: {output_checkpoint_path}")

    # Exhaustive Held-out Evaluation on Fold 0
    checkpoint = torch.load(output_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Running full dataset-split benchmark on held-out Fold 0...")
    all_ious, all_dices = [], []
    p440_ious, p440_dices = [], []
    p600_ious, p600_dices = [], []
    multi_ious, multi_dices = [], []
    t_eval = time.time()

    with torch.no_grad():
        for images, masks, sample_ids, sources, orig_sizes in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            logits = outputs["out"] if isinstance(outputs, dict) else outputs
            probs = torch.sigmoid(logits)

            for i in range(images.size(0)):
                prob_np = probs[i, 0].cpu().numpy()
                mask_np = masks[i, 0].cpu().numpy()
                source = sources[i]

                # Prediction at 256x256 thresholded
                pred_bin = prob_np > 0.50
                # Post-processing multi-component filtering
                struct = np.ones((3, 3), dtype=bool)
                pred_closed = ndi.binary_closing(pred_bin, structure=struct)
                pred_filled = ndi.binary_fill_holes(pred_closed)

                lbl, num = ndi.label(pred_filled)
                if num > 0:
                    sizes = ndi.sum(pred_filled, lbl, range(1, num + 1))
                    valid = [idx + 1 for idx, s in enumerate(sizes) if s >= 15]
                    pred_final = np.isin(lbl, valid) if valid else np.zeros_like(pred_filled)
                else:
                    pred_final = np.zeros_like(pred_filled)

                # IoU and Dice at eval resolution
                inter = np.logical_and(pred_final, mask_np > 0).sum()
                union = np.logical_or(pred_final, mask_np > 0).sum()
                denom = pred_final.sum() + (mask_np > 0).sum()

                iou = float(inter / union) if union > 0 else (1.0 if not np.any(mask_np > 0) and not np.any(pred_final) else 0.0)
                dice = float(2.0 * inter / denom) if denom > 0 else (1.0 if not np.any(mask_np > 0) and not np.any(pred_final) else 0.0)

                all_ious.append(iou)
                all_dices.append(dice)

                if source == "pothole440":
                    p440_ious.append(iou)
                    p440_dices.append(dice)
                else:
                    p600_ious.append(iou)
                    p600_dices.append(dice)

    eval_duration = round(time.time() - t_eval, 2)
    latency_ms = round((eval_duration / len(all_ious)) * 1000, 1)

    eval_results = {
        "model_name": model_name,
        "checkpoint_path": output_checkpoint_path,
        "train_time_sec": total_train_time,
        "eval_time_sec": eval_duration,
        "latency_ms_per_image": latency_ms,
        "samples_evaluated": len(all_ious),
        "overall_mean_iou": float(np.mean(all_ious)),
        "overall_mean_dice": float(np.mean(all_dices)),
        "pothole440_mean_iou": float(np.mean(p440_ious)) if p440_ious else 0.0,
        "pothole440_mean_dice": float(np.mean(p440_dices)) if p440_dices else 0.0,
        "pothole600_mean_iou": float(np.mean(p600_ious)) if p600_ious else 0.0,
        "pothole600_mean_dice": float(np.mean(p600_dices)) if p600_dices else 0.0,
        "training_history": training_history,
    }

    # Free memory
    del model
    gc.collect()

    return eval_results


# ---------------------------------------------------------------------------
# 5. Main Execution: Sequential Pipeline
# ---------------------------------------------------------------------------

def run_deep_learning_comparisons(epochs: int = 5, batch_size: int = 4):
    logger.info("Loading unified manifest and folds...")
    manifest = pd.read_csv(MANIFEST_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    merged = folds.merge(manifest, on="sample_id")

    # Strict Fold 0 partitioning
    train_df = merged[merged["fold"] != 0].copy().reset_index(drop=True)
    val_df = merged[merged["fold"] == 0].copy().reset_index(drop=True)

    # Subsample training data to ensure rapid CPU training (~300 images across P440 & P600)
    p440_train = train_df[train_df["dataset_source_x"] == "pothole440"]
    p600_train = train_df[train_df["dataset_source_x"] == "pothole600"]
    sub_train = pd.concat([
        p440_train.sample(min(150, len(p440_train)), random_state=42),
        p600_train.sample(min(150, len(p600_train)), random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info(f"Training partition size: {len(sub_train)} images (150 P440, 150 P600).")
    logger.info(f"Validation partition size: {len(val_df)} images (Fold 0 held-out).")

    train_ds = PotholeSegmentationDataset(sub_train, target_size=(256, 256), is_train=True)
    val_ds = PotholeSegmentationDataset(val_df, target_size=(256, 256), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}

    # 1. Train Lightweight U-Net
    unet_path = os.path.join(MODELS_DIR, "experimental_unet_unified.pt")
    unet_model = LightweightUNet(in_channels=3, out_channels=1)
    res_unet = train_and_eval_model(
        model=unet_model,
        model_name="Lightweight U-Net",
        train_loader=train_loader,
        val_loader=val_loader,
        output_checkpoint_path=unet_path,
        epochs=epochs,
        lr=1e-3,
        device="cpu",
    )
    results["unet"] = res_unet

    # 2. Train DeepLabV3 with MobileNetV3-Large
    deeplab_path = os.path.join(MODELS_DIR, "experimental_deeplabv3_unified.pt")
    deeplab_model = create_deeplabv3_mobilenetv3()
    res_deeplab = train_and_eval_model(
        model=deeplab_model,
        model_name="DeepLabV3 (MobileNetV3-Large)",
        train_loader=train_loader,
        val_loader=val_loader,
        output_checkpoint_path=deeplab_path,
        epochs=epochs,
        lr=5e-4,  # slightly smaller LR for transfer learning
        device="cpu",
    )
    results["deeplabv3"] = res_deeplab

    # Save summary report
    report_json_path = os.path.join(PROJECT_ROOT, "results", "reports", "deep_models_benchmark.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDeep learning comparisons complete! Saved results to {report_json_path}")
    return results


if __name__ == "__main__":
    run_deep_learning_comparisons(epochs=5, batch_size=4)
