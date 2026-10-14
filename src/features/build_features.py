"""
Feature build pipeline and leakage-safe data preparation for Pothole Severity Detection.

Coordinates:
1. Loading and validation of dataset triplets.
2. Systematic extraction of 2D geometric and 3D relative depth features.
3. Strict GroupKFold (5-fold) cross-validation partitioning by physical pothole ID.
4. Programmatic verification of group disjointness (zero identity leakage).
5. Generation of processed CSVs and the feature extraction report.
"""

import os
import sys
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocessing.dataset_loader import PotholeDatasetLoader, DEFAULT_DATASET_ROOT
from src.features.feature_extractor import extract_features_for_sample


OUTPUT_DATA_DIR = r"c:\Users\Anushka\Documents\pothole-severity-detection\data\processed"
OUTPUT_REPORT_PATH = r"c:\Users\Anushka\Documents\pothole-severity-detection\results\reports\feature_extraction_report.md"
FEATURES_CSV_PATH = os.path.join(OUTPUT_DATA_DIR, "pothole_features.csv")
FOLDS_CSV_PATH = os.path.join(OUTPUT_DATA_DIR, "group_folds.csv")


def extract_all_features(dataset_root: str = DEFAULT_DATASET_ROOT) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run loader, exclude empty masks, and extract features for all valid samples.

    Returns:
        (features_df, excluded_samples_list)
    """
    print(f"Loading and validating dataset from: {dataset_root}")
    loader = PotholeDatasetLoader(root_dir=dataset_root)
    valid_samples, excluded_samples = loader.load_dataset()

    print(f"Found {len(valid_samples)} valid samples and {len(excluded_samples)} excluded samples.")

    extracted_records = []
    for idx, sample in enumerate(valid_samples):
        meta = {
            "filename": sample["filename"],
            "original_split": sample["split"],
            "canonical_pothole_id": sample["canonical_pothole_id"],
        }
        features = extract_features_for_sample(
            mask_path=sample["mask_path"],
            depth_path=sample["depth_path"],
            sample_meta=meta,
        )
        extracted_records.append(features)

    df = pd.DataFrame(extracted_records)
    return df, excluded_samples


def assign_leakage_safe_folds(df: pd.DataFrame, n_splits: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Assign 5-fold GroupKFold splits based strictly on canonical_pothole_id.
    Programmatically verifies that no physical pothole overlaps between folds.

    Args:
        df: DataFrame containing extracted features and 'canonical_pothole_id'.
        n_splits: Number of cross-validation folds.

    Returns:
        (df_with_folds, fold_summary_dict)
    """
    if "canonical_pothole_id" not in df.columns:
        raise ValueError("DataFrame must contain 'canonical_pothole_id' for group partitioning.")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df["canonical_pothole_id"].values
    df = df.copy()
    df["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    # Strict programmatic verification of group disjointness
    fold_stats = {}
    for fold_idx in range(n_splits):
        train_groups = set(df[df["fold"] != fold_idx]["canonical_pothole_id"])
        val_groups = set(df[df["fold"] == fold_idx]["canonical_pothole_id"])
        overlap = train_groups & val_groups

        if len(overlap) > 0:
            raise AssertionError(
                f"CRITICAL LEAKAGE ERROR in fold {fold_idx}: "
                f"{len(overlap)} groups overlap between train and validation: {overlap}"
            )

        fold_stats[fold_idx] = {
            "val_samples_count": int(np.sum(df["fold"] == fold_idx)),
            "val_unique_potholes": len(val_groups),
            "train_samples_count": int(np.sum(df["fold"] != fold_idx)),
            "train_unique_potholes": len(train_groups),
            "overlap_count": len(overlap),
        }

    return df, fold_stats


def generate_feature_report(
    df: pd.DataFrame,
    excluded_samples: List[Dict[str, Any]],
    fold_stats: Dict[str, Any],
    report_path: str = OUTPUT_REPORT_PATH,
) -> None:
    """Write comprehensive markdown report on feature extraction and group splitting."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "fold"]

    # Verify all numeric features are finite
    nan_counts = df[feature_cols].isna().sum()
    infinite_counts = np.isinf(df[feature_cols]).sum()

    total_valid = len(df)
    unique_potholes = df["canonical_pothole_id"].nunique()

    report_lines = [
        "# Feature Extraction and Leakage-Safe Data Preparation Report",
        "",
        "**Project:** AI-Based Pothole Depth and Physical Damage Severity Assessment  ",
        "**Module:** `src/features/build_features.py`  ",
        "**Status:** Complete & Programmatically Verified  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Valid Processed Samples:** {total_valid} images",
        f"- **Unique Physical Potholes:** {unique_potholes} unique IDs",
        f"- **Excluded Samples:** {len(excluded_samples)} corrupted empty-mask files",
        f"- **Extracted Features:** {len(feature_cols)} numeric features",
        "- **Cross-Validation Scheme:** 5-Fold `GroupKFold` grouped strictly by `canonical_pothole_id`",
        "- **Group Disjointness:** **100% Verified** (Zero overlap between training and validation folds)",
        "- **Data Integrity:** All numeric features are 100% finite (0 NaNs, 0 Infs)",
        "",
        "---",
        "",
        "## 2. Excluded Defective Samples",
        "",
        "The following samples were programmatically validated and excluded due to containing exactly zero pothole pixels (`empty_mask`):",
        "",
        "| Split | Filename | Canonical ID | Exclusion Reason |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for ex in excluded_samples:
        report_lines.append(
            f"| `{ex['split']}` | `{ex['filename']}` | `{ex['canonical_pothole_id']}` | `{ex['reason']}` |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Extracted Feature Taxonomy",
        "",
        "Features are grouped into 2D geometric descriptors and uncalibrated 3D depth-map intensity descriptors.",
        "",
        "### A. 2D Geometric & Morphological Features (Image Space)",
        "",
        "| Feature Column | Units | Mathematical Definition | Physical Meaning |",
        "| :--- | :--- | :--- | :--- |",
        "| `mask_area_pixels` | pixels | $\\sum \\mathbb{I}(\\text{mask} > 0)$ | Projected horizontal surface area in image coordinates. |",
        "| `mask_coverage_ratio` | ratio | $\\frac{A}{H \\times W}$ | Fraction of the total image frame occupied by the pothole. |",
        "| `mask_coverage_pct` | percent | $100 \\times \\text{coverage\\_ratio}$ | Pothole area percentage of camera field of view. |",
        "| `perimeter_pixels` | pixels | $\\sum (\\text{mask} \\land \\neg \\text{eroded})$ | Boundary contour length of the asphalt failure rim. |",
        "| `circularity` | $[0, 1]$ | $\\frac{4 \\pi A}{P^2}$ | Isoperimetric quotient; 1.0 = circle, lower = jagged fracture. |",
        "| `compactness` | dimensionless | $\\frac{P^2}{A}$ | Shape irregularity index; higher for complex road cavities. |",
        "| `bbox_width` | pixels | $x_{\\max} - x_{\\min} + 1$ | Horizontal bounding box extent. |",
        "| `bbox_height` | pixels | $y_{\\max} - y_{\\min} + 1$ | Vertical bounding box extent. |",
        "| `bbox_aspect_ratio` | ratio | $\\frac{\\text{width}}{\\text{height}}$ | Cavity orientation elongation. |",
        "| `bbox_extent` | ratio | $\\frac{A}{\\text{bbox\\_width} \\times \\text{bbox\\_height}}$ | Ratio of pothole footprint to its bounding rectangle. |",
        "",
        "### B. 3D Depth-Map Intensity Features (Internal Cavity)",
        "",
        "| Feature Column | Units | Mathematical Definition | Description |",
        "| :--- | :--- | :--- | :--- |",
        "| `depth_min_intensity` | intensity | $\\min(D_{\\text{in}})$ | Minimum depth value recorded inside pothole boundary. |",
        "| `depth_max_intensity` | intensity | $\\max(D_{\\text{in}})$ | Peak cavity depth (deepest point of road depression). |",
        "| `depth_mean_intensity` | intensity | $\\text{mean}(D_{\\text{in}})$ | Average depth intensity across the basin. |",
        "| `depth_median_intensity` | intensity | $\\text{median}(D_{\\text{in}})$ | Median cavity depth (robust to bottom rubble). |",
        "| `depth_std_intensity` | intensity | $\\text{std}(D_{\\text{in}})$ | Surface roughness / vertical variation inside pothole. |",
        "| `depth_p10_intensity` | intensity | $\\text{percentile}(D_{\\text{in}}, 10)$ | 10th percentile depth. |",
        "| `depth_p25_intensity` | intensity | $\\text{percentile}(D_{\\text{in}}, 25)$ | 25th percentile (first quartile). |",
        "| `depth_p75_intensity` | intensity | $\\text{percentile}(D_{\\text{in}}, 75)$ | 75th percentile (third quartile). |",
        "| `depth_p90_intensity` | intensity | $\\text{percentile}(D_{\\text{in}}, 90)$ | 90th percentile depth. |",
        "| `depth_iqr_intensity` | intensity | $p75 - p25$ | Interquartile range of depth distribution. |",
        "",
        "### C. Road Surface Reference & Relative Proxy Features",
        "",
        "| Feature Column | Units | Mathematical Definition | Description |",
        "| :--- | :--- | :--- | :--- |",
        "| `road_mean_intensity` | intensity | $\\text{mean}(D_{\\text{road}})$ | Baseline road surface elevation in dilated annulus. |",
        "| `road_median_intensity` | intensity | $\\text{median}(D_{\\text{road}})$ | Robust baseline road surface plane elevation. |",
        "| `road_std_intensity` | intensity | $\\text{std}(D_{\\text{road}})$ | Pavement surface texture variance around cavity. |",
        "| `relative_depth_mean_intensity` | intensity | $\\text{mean}(D_{\\text{in}}) - \\text{mean}(D_{\\text{road}})$ | Mean relative cavity depression above road baseline. |",
        "| `relative_depth_median_intensity` | intensity | $\\text{median}(D_{\\text{in}}) - \\text{median}(D_{\\text{road}})$ | Robust relative depression index. |",
        "| `relative_depth_max_intensity` | intensity | $\\max(D_{\\text{in}}) - \\text{median}(D_{\\text{road}})$ | Maximum cavity depression relative to road plane. |",
        "| `relative_volume_proxy` | intensity $\\times$ px | $\\sum_{p} \\max(0, D(p) - \\text{road\\_med})$ | Dimensionless relative volumetric road displacement index. |",
        "| `relative_volume_density` | intensity | $\\frac{V_{\\text{proxy}}}{\\text{area\\_px}}$ | Average relative depression per unit area. |",
        "",
        "---",
        "",
        "## 4. Leakage-Safe 5-Fold GroupKFold Partitioning",
        "",
        "Evaluation uses 5-fold `GroupKFold` grouped strictly by `canonical_pothole_id`. Every augmented variant of a physical pothole is assigned to the same fold, ensuring zero data leakage.",
        "",
        "| Fold Index | Validation Potholes | Validation Images | Training Potholes | Training Images | Identity Overlap |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for fold_idx in sorted(fold_stats.keys()):
        stat = fold_stats[fold_idx]
        report_lines.append(
            f"| Fold {fold_idx} | {stat['val_unique_potholes']} potholes | {stat['val_samples_count']} images | "
            f"{stat['train_unique_potholes']} potholes | {stat['train_samples_count']} images | **{stat['overlap_count']} (0.0% overlap)** |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Statistical Feature Summary (Valid Dataset)",
        "",
        "| Feature | Min | Median | Mean | Max | Std |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    sample_features = [
        "mask_area_pixels", "circularity", "compactness", "bbox_aspect_ratio",
        "depth_max_intensity", "relative_depth_max_intensity", "relative_volume_proxy"
    ]
    for feat in sample_features:
        if feat in df.columns:
            s = df[feat]
            report_lines.append(
                f"| `{feat}` | {s.min():.2f} | {s.median():.2f} | {s.mean():.2f} | {s.max():.2f} | {s.std():.2f} |"
            )

    report_lines.extend([
        "",
        "---",
        "",
        "## 6. Critical Scientific Limitations & Calibration Requirements",
        "",
        "1. **Uncalibrated Grayscale vs. Physical Depth:**",
        "   - The depth maps are 8-bit PNGs (`uint8`), with discrete levels $[0, 190]$.",
        "   - They do NOT represent physical millimeters ($mm$). No linear multiplier ($mm / \\text{step}$) is known without camera calibration constants.",
        "2. **Uncalibrated Pixel Count vs. Surface Area:**",
        "   - Mask area is measured in pixels ($px$). Converting pixels to square centimeters ($cm^2$) requires the Ground Sampling Distance ($GSD$), which depends on camera height and focal length.",
        "3. **Relative Volume Proxy vs. Physical Volume ($cm^3$):**",
        "   - `relative_volume_proxy` represents discrete numerical integration: $\\sum (D_{xy} - D_{\\text{road}})$. It serves as a monotonic ranking proxy for severity, but CANNOT be cited as cubic centimeters ($cm^3$) or liters of asphalt patching material.",
        "4. **No Verified Ground Truth Severity Labels:**",
        "   - Severity labels (Low / Medium / High) are not provided in the dataset. Any classification model will evaluate against an operational proxy defined by municipal road guidelines (e.g. BMC/IRC:82) or statistical clustering.",
        "",
        "---",
        "",
        "## 7. Artifact Manifest",
        "",
        f"- **Extracted Tabular Features:** `data/processed/pothole_features.csv` ({len(df)} rows, {len(df.columns)} columns)",
        f"- **Fold Grouping Assignments:** `data/processed/group_folds.csv` ({len(df)} rows, metadata + fold assignments)",
        f"- **Validation Test Suite:** `tests/test_feature_extraction.py`",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Report written successfully to: {report_path}")


def main():
    """Main execution routine for feature building."""
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    # 1. Extract all features and record exclusions
    df_features, excluded_samples = extract_all_features()

    # 2. Assign leakage-safe GroupKFold splits and verify disjointness
    df_all, fold_stats = assign_leakage_safe_folds(df_features, n_splits=5)

    # 3. Check for non-finite values in numeric columns
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    nan_sum = df_all[numeric_cols].isna().sum().sum()
    inf_sum = np.isinf(df_all[numeric_cols]).sum().sum()

    if nan_sum > 0 or inf_sum > 0:
        raise ValueError(f"Extracted features contain {nan_sum} NaNs and {inf_sum} Infs!")

    print(f"All {len(numeric_cols)} numeric feature columns are verified 100% finite.")

    # 4. Save processed datasets
    df_all.to_csv(FEATURES_CSV_PATH, index=False)
    print(f"Features saved to: {FEATURES_CSV_PATH} ({len(df_all)} rows, {len(df_all.columns)} cols)")

    folds_df = df_all[["filename", "original_split", "canonical_pothole_id", "fold"]].copy()
    folds_df.to_csv(FOLDS_CSV_PATH, index=False)
    print(f"Fold assignments saved to: {FOLDS_CSV_PATH}")

    # 5. Generate comprehensive markdown audit report
    generate_feature_report(df_all, excluded_samples, fold_stats)

    print("\nFeature extraction and leakage-safe data preparation complete!")


if __name__ == "__main__":
    main()
