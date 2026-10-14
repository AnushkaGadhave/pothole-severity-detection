"""
Unit test suite for RPHI calculation, tier assignment, and leakage safety.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.severity.rphi_calculator import RPHICalculator
from src.severity.label_generator import RPHILabelGenerator


class TestRPHICalculator(unittest.TestCase):
    """Tests for the RPHICalculator class."""

    def setUp(self):
        # Create synthetic test DataFrame
        self.sample_df = pd.DataFrame({
            "relative_depth_max_intensity": [10.0, 50.0, 100.0, 150.0],
            "mask_area_pixels": [5000.0, 20000.0, 80000.0, 200000.0],
            "relative_volume_proxy": [50000.0, 500000.0, 2000000.0, 8000000.0],
            "canonical_pothole_id": ["pothole_1", "pothole_1", "pothole_2", "pothole_2"],
        })

    def test_rphi_weights_validation(self):
        # Valid weights sum to 1.0
        calc = RPHICalculator(weight_depth=0.5, weight_area=0.3, weight_volume=0.2)
        self.assertAlmostEqual(calc.weight_depth + calc.weight_area + calc.weight_volume, 1.0)

        # Invalid weights (negative)
        with self.assertRaises(ValueError):
            RPHICalculator(weight_depth=-0.1, weight_area=0.6, weight_volume=0.5)

        # Invalid weights (do not sum to 1.0)
        with self.assertRaises(ValueError):
            RPHICalculator(weight_depth=0.5, weight_area=0.5, weight_volume=0.5)

    def test_rphi_normalization_bounds(self):
        calc = RPHICalculator()
        comps = calc.fit(self.sample_df).compute_components(self.sample_df)

        for col in ["rphi_depth_score", "rphi_area_score", "rphi_volume_score", "rphi_score"]:
            vals = comps[col].values
            self.assertTrue(np.all(vals >= 0.0), f"{col} has negative values")
            self.assertTrue(np.all(vals <= 1.0), f"{col} exceeds 1.0")

        # Minimum sample should be 0.0, maximum should be 1.0
        self.assertAlmostEqual(comps["rphi_score"].iloc[0], 0.0)
        self.assertAlmostEqual(comps["rphi_score"].iloc[-1], 1.0)

    def test_rphi_reproducibility(self):
        calc = RPHICalculator()
        scores1 = calc.fit_transform(self.sample_df)
        scores2 = calc.transform(self.sample_df)
        self.assertTrue(np.array_equal(scores1, scores2))

    def test_rphi_invalid_values(self):
        calc = RPHICalculator()

        # Transforming before fitting
        with self.assertRaises(RuntimeError):
            calc.transform(self.sample_df)

        # Missing required column
        incomplete_df = self.sample_df.drop(columns=["mask_area_pixels"])
        with self.assertRaises(KeyError):
            calc.fit(incomplete_df)

    def test_configurable_weights(self):
        # High depth weight should dominate when depth increases
        calc_depth = RPHICalculator(weight_depth=0.9, weight_area=0.05, weight_volume=0.05)
        calc_area = RPHICalculator(weight_depth=0.05, weight_area=0.9, weight_volume=0.05)

        scores_depth = calc_depth.fit_transform(self.sample_df)
        scores_area = calc_area.fit_transform(self.sample_df)

        # Verify that weights change the scores
        self.assertFalse(np.allclose(scores_depth, scores_area))


class TestRPHILabelGenerator(unittest.TestCase):
    """Tests for the RPHILabelGenerator tier assignment and leakage safety."""

    def setUp(self):
        np.random.seed(42)
        # Create synthetic dataset with 10 groups, 4 samples each (40 samples)
        records = []
        for g_idx in range(10):
            pid = f"pothole_{g_idx:02d}"
            base_depth = 20.0 + g_idx * 15.0
            base_area = 10000.0 + g_idx * 20000.0
            base_vol = 100000.0 + g_idx * 500000.0
            for var in range(4):
                records.append({
                    "canonical_pothole_id": pid,
                    "relative_depth_max_intensity": base_depth + np.random.uniform(-2, 2),
                    "mask_area_pixels": base_area + np.random.uniform(-500, 500),
                    "relative_volume_proxy": base_vol + np.random.uniform(-5000, 5000),
                    "fold": g_idx % 5,
                })
        self.df = pd.DataFrame(records)

    def test_tier_assignment_boundaries(self):
        gen = RPHILabelGenerator(quantiles=(0.333, 0.667), threshold_mode="group_level")
        scores, tiers = gen.fit_predict_tier(self.df)

        self.assertEqual(len(scores), len(self.df))
        self.assertEqual(len(tiers), len(self.df))
        self.assertEqual(set(tiers), {"Low", "Medium", "High"})

        # Check exact cutoff mapping
        t1, t2 = gen.threshold_low_, gen.threshold_high_
        self.assertTrue(t1 < t2)

        for s, tier in zip(scores, tiers):
            if s <= t1:
                self.assertEqual(tier, "Low")
            elif s <= t2:
                self.assertEqual(tier, "Medium")
            else:
                self.assertEqual(tier, "High")

    def test_group_vs_image_level_thresholding(self):
        gen_grp = RPHILabelGenerator(threshold_mode="group_level").fit(self.df)
        gen_img = RPHILabelGenerator(threshold_mode="image_level").fit(self.df)

        self.assertIsInstance(gen_grp.threshold_low_, float)
        self.assertIsInstance(gen_img.threshold_low_, float)
        self.assertTrue(gen_grp.threshold_low_ < gen_grp.threshold_high_)
        self.assertTrue(gen_img.threshold_low_ < gen_img.threshold_high_)

    def test_fold_leakage_safety(self):
        gen = RPHILabelGenerator(threshold_mode="group_level")
        fold_res = gen.generate_fold_leakage_safe_labels(self.df, fold_col="fold")

        self.assertEqual(len(fold_res), len(self.df))
        self.assertTrue("rphi_fold_score" in fold_res.columns)
        self.assertTrue("rphi_fold_tier" in fold_res.columns)
        self.assertEqual(set(fold_res["rphi_fold_tier"]), {"Low", "Medium", "High"})
        self.assertTrue(np.all((fold_res["rphi_fold_score"] >= 0.0) & (fold_res["rphi_fold_score"] <= 1.0)))

    def test_labeled_dataset_file_integrity(self):
        labeled_path = r"c:\Users\Anushka\Documents\pothole-severity-detection\data\processed\pothole_features_labeled.csv"
        if os.path.exists(labeled_path):
            df_disk = pd.read_csv(labeled_path)
            self.assertEqual(len(df_disk), 434)
            self.assertEqual(len(df_disk.columns), 42)
            self.assertEqual(df_disk["canonical_pothole_id"].nunique(), 98)
            self.assertEqual(set(df_disk["rphi_tier"].unique()), {"Low", "Medium", "High"})
            self.assertEqual(set(df_disk["rphi_fold_tier"].unique()), {"Low", "Medium", "High"})
            self.assertEqual(df_disk.isna().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
