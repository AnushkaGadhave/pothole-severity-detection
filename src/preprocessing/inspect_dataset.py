"""
Dataset inspection module for Pothole440.

This script performs memory-efficient auditing of the dataset structure,
modality alignment, augmentation variants, split leakage, and depth map encoding.
Adheres to strict safety rules: reads files without modification or moving.
"""

import os
import re
from typing import Dict, List, Set, Tuple, Any
from PIL import Image
import numpy as np


DEFAULT_DATASET_ROOT = r"C:\Users\Anushka\Downloads\Pothole440-main\Pothole440-main"
SPLITS = ["train", "val", "test"]
MODALITIES = ["rgb", "mask", "depth"]


def inspect_dataset_structure(root_dir: str = DEFAULT_DATASET_ROOT) -> Dict[str, Any]:
    """Inspect folder structure and count files by split and modality."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

    structure = {}
    mismatches = {}
    all_files_by_split = {}

    for split in SPLITS:
        structure[split] = {}
        all_files_by_split[split] = {}
        split_dir = os.path.join(root_dir, split)
        
        for mod in MODALITIES:
            mod_dir = os.path.join(split_dir, mod)
            if not os.path.exists(mod_dir):
                structure[split][mod] = 0
                all_files_by_split[split][mod] = set()
                continue
            files = sorted([f for f in os.listdir(mod_dir) if f.lower().endswith(".png")])
            structure[split][mod] = len(files)
            all_files_by_split[split][mod] = set(files)

        rgb_set = all_files_by_split[split].get("rgb", set())
        mask_set = all_files_by_split[split].get("mask", set())
        depth_set = all_files_by_split[split].get("depth", set())

        mismatches[split] = {
            "rgb_mask_match": rgb_set == mask_set,
            "rgb_depth_match": rgb_set == depth_set,
            "rgb_minus_mask": len(rgb_set - mask_set),
            "mask_minus_rgb": len(mask_set - rgb_set),
            "rgb_minus_depth": len(rgb_set - depth_set),
            "depth_minus_rgb": len(depth_set - rgb_set),
        }

    return {
        "counts": structure,
        "mismatches": mismatches,
        "files": all_files_by_split,
    }


def inspect_image_properties(root_dir: str = DEFAULT_DATASET_ROOT) -> Dict[str, Any]:
    """Inspect dimensions, modes, data types, and sizes across modalities."""
    dimensions = set()
    modality_info = {m: {"modes": set(), "dtypes": set(), "sizes_kb": []} for m in MODALITIES}

    for split in SPLITS:
        rgb_dir = os.path.join(root_dir, split, "rgb")
        if not os.path.exists(rgb_dir):
            continue
        for fname in os.listdir(rgb_dir):
            if not fname.lower().endswith(".png"):
                continue
            for mod in MODALITIES:
                fpath = os.path.join(root_dir, split, mod, fname)
                if not os.path.exists(fpath):
                    continue
                size_kb = os.path.getsize(fpath) / 1024.0
                modality_info[mod]["sizes_kb"].append(size_kb)

                with Image.open(fpath) as img:
                    dimensions.add(img.size)
                    modality_info[mod]["modes"].add(img.mode)
                    arr = np.array(img)
                    modality_info[mod]["dtypes"].add(str(arr.dtype))

    stats = {}
    for mod in MODALITIES:
        sizes = modality_info[mod]["sizes_kb"]
        stats[mod] = {
            "modes": sorted(list(modality_info[mod]["modes"])),
            "dtypes": sorted(list(modality_info[mod]["dtypes"])),
            "min_size_kb": round(float(np.min(sizes)), 1) if sizes else 0.0,
            "max_size_kb": round(float(np.max(sizes)), 1) if sizes else 0.0,
            "avg_size_kb": round(float(np.mean(sizes)), 1) if sizes else 0.0,
        }

    return {
        "unique_dimensions_wh": sorted(list(dimensions)),
        "modality_stats": stats,
    }


def analyze_augmentations_and_leakage(root_dir: str = DEFAULT_DATASET_ROOT) -> Dict[str, Any]:
    """
    Extract original pothole IDs and measure data leakage across splits.
    Pattern: YYYY_MM_DD_pothole_XX[_suffix]
    """
    id_pattern = re.compile(r"^(\d{4}_\d{2}_\d{2}_pothole_\d+)", re.IGNORECASE)
    potholes_by_split: Dict[str, Dict[str, List[str]]] = {s: {} for s in SPLITS}
    all_unique_ids: Set[str] = set()
    non_matching_files = []

    for split in SPLITS:
        rgb_dir = os.path.join(root_dir, split, "rgb")
        if not os.path.exists(rgb_dir):
            continue
        for fname in sorted(os.listdir(rgb_dir)):
            if not fname.lower().endswith(".png"):
                continue
            m = id_pattern.match(fname)
            if m:
                base_id = m.group(1)
                all_unique_ids.add(base_id)
                potholes_by_split[split].setdefault(base_id, []).append(fname)
            else:
                non_matching_files.append((split, fname))

    train_ids = set(potholes_by_split["train"].keys())
    val_ids = set(potholes_by_split["val"].keys())
    test_ids = set(potholes_by_split["test"].keys())

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    triple_overlap = train_ids & val_ids & test_ids

    val_leaked_count = sum(len(potholes_by_split["val"][pid]) for pid in train_val_overlap)
    test_leaked_count = sum(len(potholes_by_split["test"][pid]) for pid in train_test_overlap)

    val_total = sum(len(v) for v in potholes_by_split["val"].values())
    test_total = sum(len(v) for v in potholes_by_split["test"].values())

    return {
        "total_unique_potholes": len(all_unique_ids),
        "unique_potholes_by_split": {s: len(potholes_by_split[s]) for s in SPLITS},
        "total_images_by_split": {s: sum(len(v) for v in potholes_by_split[s].values()) for s in SPLITS},
        "train_val_overlap_ids": len(train_val_overlap),
        "train_test_overlap_ids": len(train_test_overlap),
        "val_test_overlap_ids": len(val_test_overlap),
        "triple_overlap_ids": len(triple_overlap),
        "val_leakage_percentage": round(val_leaked_count / val_total * 100.0, 2) if val_total else 0.0,
        "test_leakage_percentage": round(test_leaked_count / test_total * 100.0, 2) if test_total else 0.0,
        "non_matching_files": non_matching_files,
    }


def analyze_depth_and_masks(root_dir: str = DEFAULT_DATASET_ROOT) -> Dict[str, Any]:
    """Inspect depth distributions, background levels, and empty masks."""
    empty_masks = []
    global_depth_mins = []
    global_depth_maxs = []
    pothole_depth_means = []
    road_depth_means = []
    pothole_areas_px = []

    for split in SPLITS:
        rgb_dir = os.path.join(root_dir, split, "rgb")
        if not os.path.exists(rgb_dir):
            continue
        for fname in os.listdir(rgb_dir):
            if not fname.lower().endswith(".png"):
                continue
            m_path = os.path.join(root_dir, split, "mask", fname)
            d_path = os.path.join(root_dir, split, "depth", fname)

            mask = np.array(Image.open(m_path))
            depth = np.array(Image.open(d_path))

            global_depth_mins.append(int(depth.min()))
            global_depth_maxs.append(int(depth.max()))

            pothole_mask = mask > 0
            area_px = int(np.sum(pothole_mask))
            if area_px == 0:
                empty_masks.append(f"{split}/{fname}")
            else:
                pothole_areas_px.append(area_px)
                pothole_depth_means.append(float(np.mean(depth[pothole_mask])))

            bg_mask = mask == 0
            if np.any(bg_mask):
                road_depth_means.append(float(np.mean(depth[bg_mask])))

    return {
        "empty_masks_count": len(empty_masks),
        "empty_masks_list": empty_masks,
        "global_depth_min": min(global_depth_mins) if global_depth_mins else 0,
        "global_depth_max": max(global_depth_maxs) if global_depth_maxs else 0,
        "avg_depth_inside_pothole": round(float(np.mean(pothole_depth_means)), 2) if pothole_depth_means else 0.0,
        "avg_depth_outside_pothole": round(float(np.mean(road_depth_means)), 2) if road_depth_means else 0.0,
        "min_pothole_area_px": min(pothole_areas_px) if pothole_areas_px else 0,
        "max_pothole_area_px": max(pothole_areas_px) if pothole_areas_px else 0,
        "avg_pothole_area_px": round(float(np.mean(pothole_areas_px)), 1) if pothole_areas_px else 0.0,
    }


def run_full_audit(root_dir: str = DEFAULT_DATASET_ROOT) -> Dict[str, Any]:
    """Execute all audit phases and return aggregated report dictionary."""
    print("Executing Phase 1: Dataset structure & file counts...")
    struct = inspect_dataset_structure(root_dir)
    print("Executing Phase 2: Image properties & formats...")
    props = inspect_image_properties(root_dir)
    print("Executing Phase 3: Augmentations & split leakage...")
    leakage = analyze_augmentations_and_leakage(root_dir)
    print("Executing Phase 4: Depth & mask analysis...")
    depth_masks = analyze_depth_and_masks(root_dir)

    return {
        "structure": struct,
        "properties": props,
        "leakage": leakage,
        "depth_masks": depth_masks,
    }


if __name__ == "__main__":
    results = run_full_audit()
    print("\n" + "=" * 60)
    print("POTHOLE440 DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"Counts: {results['structure']['counts']}")
    print(f"Image Dimensions: {results['properties']['unique_dimensions_wh']}")
    print(f"Unique Potholes: {results['leakage']['total_unique_potholes']} (across 440 total images)")
    print(f"Val Leakage: {results['leakage']['val_leakage_percentage']}%")
    print(f"Test Leakage: {results['leakage']['test_leakage_percentage']}%")
    print(f"Empty Masks: {results['depth_masks']['empty_masks_count']}")
    print(f"Depth Min/Max: [{results['depth_masks']['global_depth_min']}, {results['depth_masks']['global_depth_max']}]")
    print(f"Depth Inside vs Outside: {results['depth_masks']['avg_depth_inside_pothole']} vs {results['depth_masks']['avg_depth_outside_pothole']}")
    print("=" * 60)
