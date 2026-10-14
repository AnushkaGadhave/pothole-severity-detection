"""
Unit test suite for dataset loader, feature extraction, and leakage-safe grouping.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.dataset_loader import parse_pothole_id, validate_sample_files
from src.features.feature_extractor import (
    compute_mask_geometry,
    compute_depth_features,
)
from src.features.build_features import assign_leakage_safe_folds


class TestDatasetLoader(unittest.TestCase):
    """Tests for dataset loading, naming parsing, and validation."""

    def test_parse_pothole_id_standard(self):
        filename = "2022_05_22_pothole_10.png"
        self.assertEqual(parse_pothole_id(filename), "2022_05_22_pothole_10")

    def test_parse_pothole_id_augmentations(self):
        cases = [
            ("2022_05_22_pothole_10_aug0.png", "2022_05_22_pothole_10"),
            ("2022_05_22_pothole_10_horizantal.png", "2022_05_22_pothole_10"),
            ("2022_05_22_pothole_10_horizantal_vertical.png", "2022_05_22_pothole_10"),
            ("2022_05_22_pothole_10_horizantal_vertical_aug2.png", "2022_05_22_pothole_10"),
            ("2022_05_22_pothole_12_vertical_aug11.png", "2022_05_22_pothole_12"),
            ("2022_06_10_pothole_3_aug15.png", "2022_06_10_pothole_3"),
        ]
        for fname, expected in cases:
            self.assertEqual(parse_pothole_id(fname), expected)

    def test_parse_pothole_id_invalid(self):
        with self.assertRaises(ValueError):
            parse_pothole_id("random_road_image_001.png")
        with self.assertRaises(ValueError):
            parse_pothole_id("pothole10.png")

    def test_validate_sample_missing_file(self):
        is_valid, reason, meta = validate_sample_files(
            "nonexistent_rgb.png", "nonexistent_mask.png", "nonexistent_depth.png"
        )
        self.assertFalse(is_valid)
        self.assertTrue("missing_file" in reason)

    def test_validate_sample_empty_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path = os.path.join(tmpdir, "sample_rgb.png")
            mask_path = os.path.join(tmpdir, "sample_mask.png")
            depth_path = os.path.join(tmpdir, "sample_depth.png")

            # Create 100x100 dummy images
            Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128).save(rgb_path)
            # Empty mask (all zeros)
            Image.fromarray(np.zeros((100, 100), dtype=np.uint8)).save(mask_path)
            Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 100).save(depth_path)

            is_valid, reason, meta = validate_sample_files(rgb_path, mask_path, depth_path)
            self.assertFalse(is_valid)
            self.assertTrue("empty_mask" in reason)

    def test_validate_sample_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path = os.path.join(tmpdir, "sample_rgb.png")
            mask_path = os.path.join(tmpdir, "sample_mask.png")
            depth_path = os.path.join(tmpdir, "sample_depth.png")

            # Mismatched dimensions (100x100 vs 50x50)
            Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8)).save(rgb_path)
            Image.fromarray(np.ones((50, 50), dtype=np.uint8) * 255).save(mask_path)
            Image.fromarray(np.ones((100, 100), dtype=np.uint8)).save(depth_path)

            is_valid, reason, meta = validate_sample_files(rgb_path, mask_path, depth_path)
            self.assertFalse(is_valid)
            self.assertTrue("dimension_mismatch" in reason)


class TestFeatureExtractor(unittest.TestCase):
    """Tests for geometric and depth feature extraction functions."""

    def test_compute_mask_geometry(self):
        # Create 100x100 mask with 20x20 square in the center
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        geom = compute_mask_geometry(mask)
        self.assertEqual(geom["mask_area_pixels"], 400.0)
        self.assertAlmostEqual(geom["mask_coverage_ratio"], 0.04)
        self.assertAlmostEqual(geom["mask_coverage_pct"], 4.0)
        self.assertEqual(geom["bbox_width"], 20.0)
        self.assertEqual(geom["bbox_height"], 20.0)
        self.assertAlmostEqual(geom["bbox_aspect_ratio"], 1.0)
        self.assertAlmostEqual(geom["bbox_extent"], 1.0)
        self.assertTrue(0.0 < geom["circularity"] <= 1.0)
        self.assertTrue(geom["compactness"] > 0)

    def test_compute_mask_geometry_empty_raises(self):
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            compute_mask_geometry(empty_mask)

    def test_compute_depth_features(self):
        # 100x100 depth map: road surface is 80, pothole cavity is 150
        depth = np.ones((100, 100), dtype=np.uint8) * 80
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        depth[40:60, 40:60] = 150  # 400 pixels at intensity 150

        depth_feat = compute_depth_features(depth, mask)

        self.assertEqual(depth_feat["depth_min_intensity"], 150.0)
        self.assertEqual(depth_feat["depth_max_intensity"], 150.0)
        self.assertEqual(depth_feat["depth_mean_intensity"], 150.0)
        self.assertEqual(depth_feat["depth_std_intensity"], 0.0)
        self.assertEqual(depth_feat["road_mean_intensity"], 80.0)
        self.assertEqual(depth_feat["relative_depth_mean_intensity"], 70.0)
        self.assertEqual(depth_feat["relative_depth_max_intensity"], 70.0)

        # Expected volume proxy: 400 px * 70 intensity steps = 28000
        self.assertEqual(depth_feat["relative_volume_proxy"], 28000.0)
        self.assertEqual(depth_feat["relative_volume_density"], 70.0)

    def test_compute_depth_features_empty_raises(self):
        depth = np.ones((100, 100), dtype=np.uint8) * 100
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            compute_depth_features(depth, empty_mask)


class TestGroupKFoldLeakageSafety(unittest.TestCase):
    """Tests for GroupKFold partitioning and group disjointness."""

    def test_group_kfold_disjointness(self):
        # Create synthetic dataset with 20 unique physical potholes and 100 samples
        pothole_ids = [f"2022_05_22_pothole_{i}" for i in range(20)]
        records = []
        for i, pid in enumerate(pothole_ids):
            # 5 augmented variants per pothole
            for aug in range(5):
                records.append({
                    "filename": f"{pid}_aug{aug}.png",
                    "canonical_pothole_id": pid,
                    "mask_area_pixels": float(100 + i * 10),
                    "depth_max_intensity": float(120 + i),
                })

        df = pd.DataFrame(records)
        df_folds, fold_stats = assign_leakage_safe_folds(df, n_splits=5)

        self.assertEqual(len(fold_stats), 5)
        for fold_idx in range(5):
            train_groups = set(df_folds[df_folds["fold"] != fold_idx]["canonical_pothole_id"])
            val_groups = set(df_folds[df_folds["fold"] == fold_idx]["canonical_pothole_id"])
            overlap = train_groups & val_groups
            self.assertEqual(len(overlap), 0, f"Leakage detected in fold {fold_idx}")
            self.assertEqual(fold_stats[fold_idx]["overlap_count"], 0)

    def test_group_kfold_leakage_detection(self):
        # Manually inject overlap to verify failure detection
        df = pd.DataFrame({
            "canonical_pothole_id": ["pothole_1", "pothole_1", "pothole_2", "pothole_2"],
            "fold": [0, 1, 0, 1],  # Artificial overlap! pothole_1 is in fold 0 AND fold 1
        })
        train_groups = set(df[df["fold"] != 0]["canonical_pothole_id"])
        val_groups = set(df[df["fold"] == 0]["canonical_pothole_id"])
        overlap = train_groups & val_groups
        self.assertTrue(len(overlap) > 0)


if __name__ == "__main__":
    unittest.main()
