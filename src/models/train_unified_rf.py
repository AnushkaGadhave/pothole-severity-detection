"""
Train Retrained Random Forest Baseline on Unified Dataset.

Uses the unified dataset manifest and leakage-free group partitions (fold != 0 for training,
fold == 0 for held-out evaluation).
Saves model strictly to experimental path: models/experimental_rf_unified.joblib.
Preserves production model models/rgb_segmenter.joblib untouched.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.segmentation import extract_pixel_features
from src.models.train_segmenter import compute_iou_and_dice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "unified_dataset_manifest.csv")
FOLDS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "unified_group_folds.csv")
EXPERIMENTAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "experimental_rf_unified.joblib")
EXPERIMENTAL_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "experimental_rf_unified_metadata.json")


def train_unified_rf(
    manifest_path: str = MANIFEST_PATH,
    folds_path: str = FOLDS_PATH,
    output_model_path: str = EXPERIMENTAL_MODEL_PATH,
    output_meta_path: str = EXPERIMENTAL_METADATA_PATH,
    val_fold: int = 0,
    max_train_images: int = 120,
    pos_samples_per_img: int = 500,
    neg_samples_per_img: int = 1000,
    scale: float = 0.25,
    threshold: float = 0.40,
    random_state: int = 42,
) -> Dict[str, Any]:
    logger.info("--- Starting Retrained Random Forest Baseline on Unified Dataset ---")
    manifest = pd.read_csv(manifest_path)
    folds = pd.read_csv(folds_path)

    merged = folds.merge(manifest, on="sample_id")

    # 1. Strict leakage-free partition
    train_df = merged[merged["fold"] != val_fold].copy()
    val_df = merged[merged["fold"] == val_fold].copy()

    train_groups = set(train_df["group_id_x"])
    val_groups = set(val_df["group_id_x"])
    overlap = train_groups & val_groups
    assert len(overlap) == 0, f"Critical Data Leakage detected: {overlap}"

    logger.info(
        f"Partition verified: Train={len(train_df)} samples ({len(train_groups)} groups), "
        f"Val={len(val_df)} samples ({len(val_groups)} groups). Overlap=0."
    )

    # 2. Sample balanced training images across Pothole440 and Pothole-600
    p440_train = train_df[train_df["dataset_source_x"] == "pothole440"]
    p600_train = train_df[train_df["dataset_source_x"] == "pothole600"]

    n_p440 = min(len(p440_train), max_train_images // 2)
    n_p600 = min(len(p600_train), max_train_images // 2)

    sample_train = pd.concat([
        p440_train.sample(n_p440, random_state=random_state),
        p600_train.sample(n_p600, random_state=random_state)
    ]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    logger.info(f"Extracting features from {len(sample_train)} training images ({n_p440} P440, {n_p600} P600)...")

    X_list, y_list = [], []
    t0 = time.time()

    for _, row in sample_train.iterrows():
        img_p = row["image_path_x"]
        msk_p = row["mask_path_x"]

        if not os.path.exists(img_p) or not os.path.exists(msk_p):
            continue

        with Image.open(img_p) as ir, Image.open(msk_p) as im:
            rgb = np.array(ir.convert("RGB"))
            mask = np.array(im)

        feats, (th, tw), _ = extract_pixel_features(rgb, scale=scale)
        mask_small = np.array(Image.fromarray(mask).resize((tw, th), Image.Resampling.NEAREST)) > 0
        y_flat = mask_small.ravel().astype(np.uint8)

        pos_idx = np.where(y_flat == 1)[0]
        neg_idx = np.where(y_flat == 0)[0]

        if len(pos_idx) == 0:
            continue

        n_pos = min(len(pos_idx), pos_samples_per_img)
        n_neg = min(len(neg_idx), neg_samples_per_img)

        cp = np.random.choice(pos_idx, n_pos, replace=False)
        cn = np.random.choice(neg_idx, n_neg, replace=False)

        X_list.append(feats[np.concatenate([cp, cn])])
        y_list.append(y_flat[np.concatenate([cp, cn])])

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    logger.info(f"Training dataset ready: {X_train.shape[0]} pixel samples, positive ratio: {y_train.mean():.3f} (time: {time.time() - t0:.2f}s).")

    # 3. Fit Random Forest
    t1 = time.time()
    logger.info("Fitting Random Forest (50 trees, max_depth=12)...")
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    train_time = round(time.time() - t1, 2)
    logger.info(f"Random Forest fit completed in {train_time}s.")

    # 4. Save experimental model (do not overwrite production)
    pkg = {
        "classifier": clf,
        "scale": scale,
        "threshold": threshold,
        "metadata": {
            "model_type": "RandomForest_Unified_Retrained",
            "train_samples": len(sample_train),
            "train_pixels": int(X_train.shape[0]),
            "train_time_sec": train_time,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    }
    joblib.dump(pkg, output_model_path)
    logger.info(f"Experimental model saved to: {output_model_path}")

    # 5. Evaluate on held-out validation partition (Fold 0)
    logger.info(f"Evaluating retrained RF on {len(val_df)} held-out validation samples...")
    t2 = time.time()
    ious, dices = [], []
    p440_ious, p600_ious = [], []

    import scipy.ndimage as ndi
    struct_3x3 = np.ones((3, 3), dtype=bool)

    for _, row in val_df.iterrows():
        img_p = row["image_path_x"]
        msk_p = row["mask_path_x"]
        source = row["dataset_source_x"]

        if not os.path.exists(img_p) or not os.path.exists(msk_p):
            continue

        with Image.open(img_p) as ir, Image.open(msk_p) as im:
            rgb = np.array(ir.convert("RGB"))
            gt_mask = np.array(im)
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[:, :, 0]

        orig_h, orig_w = rgb.shape[:2]
        feats, (th, tw), _ = extract_pixel_features(rgb, scale=scale)
        probs = clf.predict_proba(feats)[:, 1].reshape((th, tw))
        pred_bin = probs > threshold

        pred_closed = ndi.binary_closing(pred_bin, structure=struct_3x3)
        pred_filled = ndi.binary_fill_holes(pred_closed)

        # Multi-component retention
        eff_min_size = max(10, min(50, int(th * tw * 0.001)))
        lbl, num = ndi.label(pred_filled)
        if num > 0:
            sizes = ndi.sum(pred_filled, lbl, range(1, num + 1))
            valid_idx = [idx + 1 for idx, s in enumerate(sizes) if s >= eff_min_size]
            pred_small = np.isin(lbl, valid_idx) if valid_idx else np.zeros((th, tw), dtype=bool)
        else:
            pred_small = np.zeros((th, tw), dtype=bool)

        pred_full = np.array(Image.fromarray((pred_small.astype(np.uint8) * 255)).resize((orig_w, orig_h), Image.Resampling.NEAREST))
        iou, dice = compute_iou_and_dice(pred_full, gt_mask)
        ious.append(iou)
        dices.append(dice)

        if source == "pothole440":
            p440_ious.append(iou)
        else:
            p600_ious.append(iou)

    eval_time = round(time.time() - t2, 2)
    mean_latency_ms = round((eval_time / len(val_df)) * 1000, 1)

    eval_summary = {
        "model_name": "RandomForest (Unified Retrained)",
        "model_path": output_model_path,
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
        "latency_ms_per_image": mean_latency_ms,
        "val_samples_evaluated": len(ious),
        "overall_mean_iou": float(np.mean(ious)),
        "overall_mean_dice": float(np.mean(dices)),
        "pothole440_mean_iou": float(np.mean(p440_ious)) if p440_ious else 0.0,
        "pothole600_mean_iou": float(np.mean(p600_ious)) if p600_ious else 0.0,
    }

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RETRAINED RANDOM FOREST EVALUATION RESULTS (HELD-OUT FOLD 0)")
    print("=" * 60)
    print(f"Overall Mean IoU:   {eval_summary['overall_mean_iou']:.4f}")
    print(f"Overall Mean Dice:  {eval_summary['overall_mean_dice']:.4f}")
    print(f"Pothole440 IoU:     {eval_summary['pothole440_mean_iou']:.4f}")
    print(f"Pothole-600 IoU:    {eval_summary['pothole600_mean_iou']:.4f}")
    print(f"CPU Latency:        {eval_summary['latency_ms_per_image']:.1f} ms/image")
    print(f"Training Time:      {eval_summary['train_time_sec']:.1f} s")
    print("=" * 60 + "\n")

    return eval_summary


if __name__ == "__main__":
    train_unified_rf()
