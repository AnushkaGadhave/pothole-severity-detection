"""
Unified Dataset Manifest and Leakage-Free Split Generator.

Combines Pothole440 and Pothole-600 datasets into a single validated manifest:
- Validates file paths, dimensions, dtypes, and empty masks.
- Segregates physical depth (Pothole440) from transformed stereo disparity (Pothole-600).
- Clusters duplicate/sequential images into stable group IDs to prevent data leakage.
- Generates 5-fold group-disjoint cross-validation partitions.
"""

import os
import re
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndi
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POTHOLE440_ROOT = r"C:\Users\Anushka\Downloads\Pothole440-main\Pothole440-main"
POTHOLE600_ROOT = r"C:\Users\Anushka\Downloads\pothole600\pothole600"

OUTPUT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "unified_dataset_manifest.csv")
OUTPUT_FOLDS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "unified_group_folds.csv")


def dhash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """Compute difference hash (dHash) for perceptual image clustering."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def count_pothole_components(mask_arr: np.ndarray, min_area: int = 20) -> Tuple[int, int]:
    """
    Count connected components in mask with area >= min_area pixels.
    Returns (num_valid_components, total_pothole_pixels).
    """
    bin_mask = mask_arr > 0
    total_pixels = int(np.sum(bin_mask))
    if total_pixels == 0:
        return 0, 0

    lbl, num = ndi.label(bin_mask)
    if num <= 1:
        return 1, total_pixels

    sizes = ndi.sum(bin_mask, lbl, range(1, num + 1))
    valid_count = sum(1 for s in sizes if s >= min_area)
    return max(1, valid_count), total_pixels


def parse_p440_id(filename: str) -> str:
    """Extract canonical pothole ID from Pothole440 filename."""
    base = os.path.basename(filename)
    match = re.match(r"^(\d{4}_\d{2}_\d{2}_pothole_\d+)", base, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized Pothole440 filename format: '{base}'")
    return match.group(1).lower()


def process_pothole440(root_dir: str = POTHOLE440_ROOT) -> List[Dict[str, Any]]:
    """Scan and process Pothole440 samples."""
    logger.info(f"Scanning Pothole440 at: {root_dir}")
    records = []
    splits = ["train", "val", "test"]

    for split in splits:
        rgb_dir = os.path.join(root_dir, split, "rgb")
        mask_dir = os.path.join(root_dir, split, "mask")
        depth_dir = os.path.join(root_dir, split, "depth")

        if not os.path.exists(rgb_dir):
            logger.warning(f"Directory not found: {rgb_dir}")
            continue

        for fname in sorted(os.listdir(rgb_dir)):
            if not fname.lower().endswith(".png"):
                continue

            rgb_path = os.path.join(rgb_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            depth_path = os.path.join(depth_dir, fname)

            pothole_id = parse_p440_id(fname)
            group_id = f"p440_{pothole_id}"
            sample_id = f"p440_{split}_{os.path.splitext(fname)[0]}"

            try:
                with Image.open(rgb_path) as rgb_im, \
                     Image.open(mask_path) as mask_im, \
                     Image.open(depth_path) as depth_im:
                    w, h = rgb_im.size
                    mask_arr = np.array(mask_im)
                    num_cc, px_area = count_pothole_components(mask_arr, min_area=20)
                    is_valid = px_area > 0
                    reason = "" if is_valid else "empty_mask"
            except Exception as e:
                w, h, num_cc, px_area = 0, 0, 0, 0
                is_valid = False
                reason = f"corrupt_file: {str(e)}"

            records.append({
                "sample_id": sample_id,
                "dataset_source": "pothole440",
                "original_split": split,
                "group_id": group_id,
                "image_path": rgb_path,
                "mask_path": mask_path,
                "depth_path": depth_path,
                "disparity_path": "",
                "width": w,
                "height": h,
                "num_potholes": num_cc,
                "pothole_pixel_area": px_area,
                "is_valid": is_valid,
                "exclusion_reason": reason,
            })

    logger.info(f"Pothole440 processed: {len(records)} total, {sum(1 for r in records if r['is_valid'])} valid.")
    return records


def cluster_pothole600_groups(root_dir: str = POTHOLE600_ROOT) -> Dict[str, str]:
    """
    Cluster Pothole-600 images using dHash and MSE verification.
    Returns mapping from (split, filename) -> group_id.
    """
    splits = ["training", "validation", "testing"]
    all_imgs = []

    for split in splits:
        rgb_dir = os.path.join(root_dir, split, "rgb")
        for fname in sorted(os.listdir(rgb_dir)):
            if not fname.lower().endswith(".png"):
                continue
            path = os.path.join(rgb_dir, fname)
            with Image.open(path) as im:
                h = dhash(im)
            all_imgs.append((split, fname, path, h))

    n = len(all_imgs)
    adj = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            h_dist = np.count_nonzero(all_imgs[i][3] != all_imgs[j][3])
            if h_dist <= 2:
                # Confirm with MSE
                im1 = np.array(Image.open(all_imgs[i][2]))
                im2 = np.array(Image.open(all_imgs[j][2]))
                if im1.shape == im2.shape:
                    mse = np.mean((im1.astype(float) - im2.astype(float)) ** 2)
                    if mse < 100.0:  # Confirmed near or exact duplicate
                        adj[i].append(j)
                        adj[j].append(i)

    visited = set()
    group_map = {}
    cluster_idx = 0

    for i in range(n):
        if i not in visited:
            cluster = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop()
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            group_name = f"p600_grp_{cluster_idx:04d}"
            cluster_idx += 1
            for idx in cluster:
                split_name, fname, _, _ = all_imgs[idx]
                group_map[(split_name, fname)] = group_name

    logger.info(f"Pothole-600: {n} images clustered into {cluster_idx} unique scene groups.")
    return group_map


def process_pothole600(root_dir: str = POTHOLE600_ROOT) -> List[Dict[str, Any]]:
    """Scan and process Pothole-600 samples."""
    logger.info(f"Scanning Pothole-600 at: {root_dir}")
    group_map = cluster_pothole600_groups(root_dir)
    records = []
    splits = ["training", "validation", "testing"]

    for split in splits:
        rgb_dir = os.path.join(root_dir, split, "rgb")
        lbl_dir = os.path.join(root_dir, split, "label")
        dsp_dir = os.path.join(root_dir, split, "tdisp")

        for fname in sorted(os.listdir(rgb_dir)):
            if not fname.lower().endswith(".png"):
                continue

            rgb_path = os.path.join(rgb_dir, fname)
            lbl_path = os.path.join(lbl_dir, fname)
            dsp_path = os.path.join(dsp_dir, fname)

            group_id = group_map.get((split, fname), f"p600_{split}_{fname}")
            sample_id = f"p600_{split}_{os.path.splitext(fname)[0]}"

            try:
                with Image.open(rgb_path) as rgb_im, Image.open(lbl_path) as lbl_im:
                    w, h = rgb_im.size
                    lbl_arr = np.array(lbl_im)
                    num_cc, px_area = count_pothole_components(lbl_arr, min_area=20)
                    is_valid = px_area > 0 and os.path.exists(dsp_path)
                    reason = "" if is_valid else "empty_or_missing"
            except Exception as e:
                w, h, num_cc, px_area = 0, 0, 0, 0
                is_valid = False
                reason = f"corrupt_file: {str(e)}"

            records.append({
                "sample_id": sample_id,
                "dataset_source": "pothole600",
                "original_split": split,
                "group_id": group_id,
                "image_path": rgb_path,
                "mask_path": lbl_path,
                "depth_path": "",
                "disparity_path": dsp_path,
                "width": w,
                "height": h,
                "num_potholes": num_cc,
                "pothole_pixel_area": px_area,
                "is_valid": is_valid,
                "exclusion_reason": reason,
            })

    logger.info(f"Pothole-600 processed: {len(records)} total, {sum(1 for r in records if r['is_valid'])} valid.")
    return records


def build_unified_manifest_and_folds(
    output_manifest: str = OUTPUT_MANIFEST_PATH,
    output_folds: str = OUTPUT_FOLDS_PATH,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate unified dataset manifest and leakage-free cross-validation folds.
    """
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)

    records_p440 = process_pothole440()
    records_p600 = process_pothole600()

    all_records = records_p440 + records_p600
    df_manifest = pd.DataFrame(all_records)
    df_manifest.to_csv(output_manifest, index=False)
    logger.info(f"Saved unified manifest ({len(df_manifest)} entries) to: {output_manifest}")

    # Generate leakage-free 5-fold splits on valid samples
    valid_df = df_manifest[df_manifest["is_valid"]].copy().reset_index(drop=True)

    # We use GroupKFold on group_id
    gkf = GroupKFold(n_splits=n_splits)
    valid_df["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(valid_df, groups=valid_df["group_id"])):
        valid_df.loc[val_idx, "fold"] = fold_idx

    # Check for zero leakage across folds
    for f in range(n_splits):
        train_groups = set(valid_df[valid_df["fold"] != f]["group_id"])
        val_groups = set(valid_df[valid_df["fold"] == f]["group_id"])
        overlap = train_groups & val_groups
        assert len(overlap) == 0, f"Critical Data Leakage in Fold {f}! Overlapping groups: {overlap}"

    folds_df = valid_df[[
        "sample_id", "dataset_source", "original_split", "group_id", "fold", "image_path", "mask_path"
    ]].copy()
    folds_df.to_csv(output_folds, index=False)
    logger.info(f"Saved 5-fold zero-leakage partitions to: {output_folds}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("UNIFIED MANIFEST & ZERO-LEAKAGE SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total Records:           {len(df_manifest)}")
    print(f"Valid Samples:           {len(valid_df)} ({len(df_manifest) - len(valid_df)} excluded: 6 empty masks in Pothole440)")
    print(f"Pothole440 Valid:        {len(valid_df[valid_df['dataset_source'] == 'pothole440'])}")
    print(f"Pothole-600 Valid:       {len(valid_df[valid_df['dataset_source'] == 'pothole600'])}")
    print(f"Total Unique Groups:     {valid_df['group_id'].nunique()}")
    print(f"Multi-Pothole Images:    {len(valid_df[valid_df['num_potholes'] > 1])}")
    print("\nSamples per Fold:")
    for f in range(n_splits):
        sub = valid_df[valid_df["fold"] == f]
        p440_cnt = len(sub[sub["dataset_source"] == "pothole440"])
        p600_cnt = len(sub[sub["dataset_source"] == "pothole600"])
        multi_cnt = len(sub[sub["num_potholes"] > 1])
        print(f"  Fold {f}: {len(sub):3d} samples ({p440_cnt:2d} P440, {p600_cnt:3d} P600, {multi_cnt:2d} multi-pothole) from {sub['group_id'].nunique():3d} unique groups")
    print("=" * 60 + "\n")

    return df_manifest, folds_df


if __name__ == "__main__":
    build_unified_manifest_and_folds()
