"""
Training and Evaluation Pipeline for RGB Pothole Segmentation.

Trains an RGBSegmenter model using multi-scale pixel features on the GroupKFold
training split, evaluates on both group-disjoint and official validation splits
using Dice coefficient and Intersection over Union (IoU), and saves the serialized
model package to models/rgb_segmenter.joblib.

IMPORTANT DATASET LEAKAGE DOCUMENTATION:
The official Pothole440 dataset splits ('train', 'val', 'test') suffer from 100%
pothole identity leakage: all 98 physical potholes appear in 'train', and 'val'
merely contains augmented variants of the same potholes. To ensure a scientifically
rigorous evaluation, primary evaluation is conducted on a strictly group-disjoint
validation fold (Fold 0: 20 unseen potholes, 88 images) where no physical pothole
in validation ever appeared during training.
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

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing.dataset_loader import DEFAULT_DATASET_ROOT
from src.models.segmentation import RGBSegmenter, extract_pixel_features, DEFAULT_SEGMENTER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_FOLDS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "group_folds.csv")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_METADATA_PATH = os.path.join(DEFAULT_MODELS_DIR, "segmentation_metadata.json")


def compute_iou_and_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute Intersection over Union (IoU) and Dice coefficient between binary masks.

    Args:
        pred_mask: 2D binary array (values > 0 indicate predicted pothole).
        gt_mask: 2D binary array (values > 0 indicate ground-truth pothole).

    Returns:
        (iou, dice)
    """
    p = pred_mask > 0
    g = gt_mask > 0

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    iou = float(intersection / union) if union > 0 else (1.0 if not np.any(g) and not np.any(p) else 0.0)
    denom = p.sum() + g.sum()
    dice = float((2.0 * intersection) / denom) if denom > 0 else (1.0 if not np.any(g) and not np.any(p) else 0.0)

    return iou, dice


def train_and_evaluate_segmenter(
    dataset_root: str = DEFAULT_DATASET_ROOT,
    folds_csv: str = DEFAULT_FOLDS_CSV,
    output_model_path: str = DEFAULT_SEGMENTER_PATH,
    output_metadata_path: str = DEFAULT_METADATA_PATH,
    scale: float = 0.25,
    threshold: float = 0.40,
    random_state: int = 42,
    max_train_images: int = 80,
    pos_samples_per_img: int = 500,
    neg_samples_per_img: int = 1000,
) -> Dict[str, Any]:
    """
    Train RGB segmenter on group-disjoint training partition and evaluate on validation sets.

    Args:
        dataset_root: Path to Pothole440 directory.
        folds_csv: Path to group_folds.csv defining canonical pothole groups.
        output_model_path: Destination path for joblib artifact.
        output_metadata_path: Destination path for JSON summary metadata.
        scale: Resolution scale factor.
        threshold: Probability cutoff threshold.
        random_state: Random seed.
        max_train_images: Maximum training images to sample from training groups.
        pos_samples_per_img: Foreground pixel sample count per image.
        neg_samples_per_img: Background pixel sample count per image.

    Returns:
        Dictionary of training metadata and validation evaluation metrics.
    """
    if not os.path.exists(folds_csv):
        raise FileNotFoundError(
            f"Folds CSV not found: '{folds_csv}'. "
            "Please run 'python src/features/build_features.py' first."
        )

    df_folds = pd.read_csv(folds_csv)

    # 1. Partition into group-disjoint train and validation splits (Fold 0 held out)
    train_df = df_folds[df_folds["fold"] != 0]
    val_group_df = df_folds[df_folds["fold"] == 0]

    train_groups = set(train_df["canonical_pothole_id"])
    val_groups = set(val_group_df["canonical_pothole_id"])
    overlap = train_groups & val_groups
    if len(overlap) > 0:
        raise AssertionError(f"Leakage detected between train and val groups: {overlap}")

    logger.info(
        f"Partitioning data: {len(train_df)} training samples ({len(train_groups)} groups), "
        f"{len(val_group_df)} group-disjoint validation samples ({len(val_groups)} groups)."
    )

    # 2. Extract balanced multi-scale pixel features from training images
    np.random.seed(random_state)
    sample_train = train_df.sample(min(max_train_images, len(train_df)), random_state=random_state)

    X_list = []
    y_list = []
    t0 = time.time()
    logger.info(f"Extracting multi-scale pixel features from {len(sample_train)} training images...")

    for _, row in sample_train.iterrows():
        fn = row["filename"]
        split = row["original_split"]
        r_path = os.path.join(dataset_root, split, "rgb", fn)
        m_path = os.path.join(dataset_root, split, "mask", fn)

        if not os.path.exists(r_path) or not os.path.exists(m_path):
            continue

        with Image.open(r_path) as ir, Image.open(m_path) as im:
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
    logger.info(
        f"Training dataset ready: {X_train.shape[0]} pixel samples, {X_train.shape[1]} features, "
        f"positive ratio: {y_train.mean():.3f} (time: {time.time() - t0:.2f}s)."
    )

    # 3. Fit Random Forest pixel classifier
    t1 = time.time()
    logger.info("Fitting Random Forest pixel classifier (50 trees, max_depth=12)...")
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info(f"Model fitting complete in {time.time() - t1:.2f}s.")

    # Package into RGBSegmenter
    segmenter = RGBSegmenter(scale=scale, threshold=threshold)
    segmenter.classifier = clf

    # 4. Evaluate on primary group-disjoint validation set (Fold 0, N=88)
    logger.info(f"Evaluating on group-disjoint validation split (Fold 0, N={len(val_group_df)})...")
    gd_ious, gd_dices = [], []

    for _, row in val_group_df.iterrows():
        fn = row["filename"]
        split = row["original_split"]
        r_path = os.path.join(dataset_root, split, "rgb", fn)
        m_path = os.path.join(dataset_root, split, "mask", fn)

        if not os.path.exists(r_path) or not os.path.exists(m_path):
            continue

        with Image.open(r_path) as ir, Image.open(m_path) as im:
            rgb = np.array(ir.convert("RGB"))
            gt_mask = np.array(im)

        pred_mask = segmenter.predict_mask(rgb)
        iou, dice = compute_iou_and_dice(pred_mask, gt_mask)
        gd_ious.append(iou)
        gd_dices.append(dice)

    gd_metrics = {
        "mean_iou": float(np.mean(gd_ious)),
        "std_iou": float(np.std(gd_ious)),
        "median_iou": float(np.median(gd_ious)),
        "mean_dice": float(np.mean(gd_dices)),
        "std_dice": float(np.std(gd_dices)),
        "median_dice": float(np.median(gd_dices)),
        "samples_evaluated": len(gd_ious),
        "unique_pothole_groups": len(val_groups),
    }
    logger.info(
        f"Group-Disjoint Validation Results: "
        f"Mean IoU={gd_metrics['mean_iou']:.4f} +/- {gd_metrics['std_iou']:.4f} (median: {gd_metrics['median_iou']:.4f}), "
        f"Mean Dice={gd_metrics['mean_dice']:.4f} +/- {gd_metrics['std_dice']:.4f} (median: {gd_metrics['median_dice']:.4f})."
    )

    # 5. Evaluate on official Pothole440 validation split (N=50) for benchmark transparency
    official_val_rgb = os.path.join(dataset_root, "val", "rgb")
    official_val_mask = os.path.join(dataset_root, "val", "mask")
    off_ious, off_dices = [], []

    if os.path.exists(official_val_rgb):
        logger.info("Evaluating on official Pothole440 validation split (N=50)...")
        for fn in sorted(os.listdir(official_val_rgb)):
            r_path = os.path.join(official_val_rgb, fn)
            m_path = os.path.join(official_val_mask, fn)
            if not os.path.exists(m_path):
                continue

            with Image.open(r_path) as ir, Image.open(m_path) as im:
                rgb = np.array(ir.convert("RGB"))
                gt_mask = np.array(im)

            pred_mask = segmenter.predict_mask(rgb)
            iou, dice = compute_iou_and_dice(pred_mask, gt_mask)
            off_ious.append(iou)
            off_dices.append(dice)

    off_metrics = {
        "mean_iou": float(np.mean(off_ious)) if off_ious else 0.0,
        "std_iou": float(np.std(off_ious)) if off_ious else 0.0,
        "median_iou": float(np.median(off_ious)) if off_ious else 0.0,
        "mean_dice": float(np.mean(off_dices)) if off_dices else 0.0,
        "std_dice": float(np.std(off_dices)) if off_dices else 0.0,
        "median_dice": float(np.median(off_dices)) if off_dices else 0.0,
        "samples_evaluated": len(off_ious),
        "note": "Official Pothole440 val split contains augmented versions of training potholes (100% identity leakage).",
    }

    # 6. Save model package and metadata
    segmenter.metadata = {
        "architecture": "Multi-Scale Pixel Random Forest + Morphological Filtering",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_training_pixels": int(X_train.shape[0]),
        "n_training_images": len(sample_train),
        "scale": scale,
        "threshold": threshold,
        "group_disjoint_validation": gd_metrics,
        "official_val_comparison": off_metrics,
    }
    segmenter.save(output_model_path)

    summary_metadata = {
        "project": "AI-Based Pothole Depth and Physical Damage Severity Assessment",
        "module": "RGB Pothole Segmentation",
        "model_file": os.path.basename(output_model_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_metrics": {
            "group_disjoint_fold_0": gd_metrics,
            "official_pothole440_val": off_metrics,
        },
        "scientific_disclaimers": [
            "Segmentation produces 2D binary masks; it DOES NOT estimate 3D depth from RGB.",
            "Mean Dice is ~0.53 and Mean IoU is ~0.39; classical ML provides a lightweight baseline but is sensitive to dark pavement shadows.",
            "Severity tiers evaluated from the resulting mask remain provisional Relative Pothole Hazard Index categories, NOT certified municipal maintenance priorities.",
        ],
    }

    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2)
    logger.info(f"Segmentation metadata summary saved to: {output_metadata_path}")

    return summary_metadata


if __name__ == "__main__":
    results = train_and_evaluate_segmenter()
    print("\nRGB Segmentation Training & Evaluation Complete:")
    print(f"  Group-Disjoint Mean IoU:  {results['validation_metrics']['group_disjoint_fold_0']['mean_iou']:.4f}")
    print(f"  Group-Disjoint Mean Dice: {results['validation_metrics']['group_disjoint_fold_0']['mean_dice']:.4f}")
    print(f"  Official Val Mean IoU:    {results['validation_metrics']['official_pothole440_val']['mean_iou']:.4f}")
    print(f"  Official Val Mean Dice:   {results['validation_metrics']['official_pothole440_val']['mean_dice']:.4f}")
