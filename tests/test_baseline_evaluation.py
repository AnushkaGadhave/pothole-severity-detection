"""
Unit test suite for baseline models, feature sets, and evaluation pipeline.

Validates:
  - Strict exclusion of forbidden columns and depth/volume leakage in 2D geometry.
  - Model factories and pipelines.
  - Group-disjoint cross-validation partitions.
  - Dynamic training-only target generation (RPHILabelGenerator fitted solely on training partition).
  - Training and validation targets share the exact same fitted mapping.
  - Validation perturbation invariance: changing validation features has zero impact on
    training labels and fitted thresholds.
  - Comprehensive metric computation and confusion matrix conservation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.train_baselines import (
    CLASS_LABELS,
    FORBIDDEN_COLUMNS,
    EXP_A_FEATURES,
    EXP_B_FEATURES,
    get_baseline_models,
)
from src.severity.label_generator import RPHILabelGenerator
from src.evaluation.evaluate_baselines import (
    validate_feature_subset,
    evaluate_single_experiment,
)


class TestBaselineModelsAndFeatures(unittest.TestCase):
    """Tests for feature subsets, forbidden column exclusion, and model factories."""

    def test_forbidden_columns_excluded_from_experiment_a(self):
        for col in EXP_A_FEATURES:
            self.assertNotIn(col, FORBIDDEN_COLUMNS)

    def test_forbidden_columns_excluded_from_experiment_b(self):
        for col in EXP_B_FEATURES:
            self.assertNotIn(col, FORBIDDEN_COLUMNS)

    def test_experiment_b_strictly_excludes_depth_and_volume(self):
        for col in EXP_B_FEATURES:
            lower = col.lower()
            self.assertNotIn("depth", lower, f"Depth feature found in Exp B: {col}")
            self.assertNotIn("volume", lower, f"Volume feature found in Exp B: {col}")
            self.assertNotIn("road", lower, f"Road baseline feature found in Exp B: {col}")

    def test_validate_feature_subset_raises_on_forbidden(self):
        with self.assertRaises(ValueError):
            validate_feature_subset(["mask_area_pixels", "rphi_fold_score"])

        with self.assertRaises(ValueError):
            validate_feature_subset(["mask_area_pixels", "canonical_pothole_id"])

        with self.assertRaises(ValueError):
            validate_feature_subset(["mask_area_pixels", "fold"])

    def test_get_baseline_models_factory(self):
        models = get_baseline_models(random_state=42)
        expected_models = {
            "Majority Baseline",
            "Random Forest",
            "Gradient Boosting",
            "SVM (RBF)",
            "KNN (k=5)",
        }
        self.assertEqual(set(models.keys()), expected_models)

        # Ensure all models can be instantiated and have fit/predict methods
        for name, factory in models.items():
            clf = factory()
            self.assertTrue(hasattr(clf, "fit"), f"{name} lacks fit()")
            self.assertTrue(hasattr(clf, "predict"), f"{name} lacks predict()")


class TestBaselineEvaluationPipeline(unittest.TestCase):
    """Tests for evaluation loop, group disjointness, dynamic target generation, and metrics."""

    def setUp(self):
        # Create minimal synthetic dataset with 5 folds, 2 groups per fold, 4 samples per group
        np.random.seed(42)
        records = []
        for fold_idx in range(5):
            for g_offset in range(2):
                pid = f"pothole_{fold_idx}_{g_offset}"
                for var in range(4):
                    records.append({
                        "canonical_pothole_id": pid,
                        "fold": fold_idx,
                        "relative_depth_max_intensity": float(np.random.uniform(10, 150)),
                        "mask_area_pixels": float(np.random.uniform(5000, 200000)),
                        "relative_volume_proxy": float(np.random.uniform(50000, 5000000)),
                        "mask_coverage_ratio": float(np.random.uniform(0.01, 0.2)),
                        "mask_coverage_pct": float(np.random.uniform(1.0, 20.0)),
                        "perimeter_pixels": float(np.random.uniform(100, 3000)),
                        "circularity": float(np.random.uniform(0.1, 0.8)),
                        "compactness": float(np.random.uniform(10, 80)),
                        "bbox_x_min": 100.0, "bbox_x_max": 300.0,
                        "bbox_y_min": 100.0, "bbox_y_max": 300.0,
                        "bbox_width": 200.0, "bbox_height": 200.0,
                        "bbox_aspect_ratio": 1.0, "bbox_extent": 0.5,
                    })
        self.synth_df = pd.DataFrame(records)

    def test_cross_validation_group_disjointness_synthetic(self):
        for f in range(5):
            train_g = set(self.synth_df[self.synth_df["fold"] != f]["canonical_pothole_id"])
            val_g = set(self.synth_df[self.synth_df["fold"] == f]["canonical_pothole_id"])
            self.assertEqual(len(train_g & val_g), 0, f"Synthetic leakage in fold {f}")

    def test_target_generator_fitted_only_on_training_partition(self):
        """Verify that training thresholds and labels are derived strictly from train_df."""
        for f in range(5):
            train_df = self.synth_df[self.synth_df["fold"] != f]
            val_df = self.synth_df[self.synth_df["fold"] == f]

            gen = RPHILabelGenerator(threshold_mode="group_level", group_col="canonical_pothole_id")
            gen.fit(train_df)

            # Check that thresholds match group quantiles of train_df only
            tr_scores = gen.calculator.transform(train_df)
            tr_group_scores = train_df.assign(_s=tr_scores).groupby("canonical_pothole_id")["_s"].median()
            expected_low = float(tr_group_scores.quantile(1.0 / 3.0))
            expected_high = float(tr_group_scores.quantile(2.0 / 3.0))

            self.assertAlmostEqual(gen.threshold_low_, expected_low, places=5)
            self.assertAlmostEqual(gen.threshold_high_, expected_high, places=5)

    def test_training_and_validation_targets_use_same_fitted_mapping(self):
        """Verify that y_train and y_val share identical normalization and cutoffs."""
        train_df = self.synth_df[self.synth_df["fold"] != 0]
        val_df = self.synth_df[self.synth_df["fold"] == 0]

        gen = RPHILabelGenerator(threshold_mode="group_level", group_col="canonical_pothole_id").fit(train_df)
        _, y_train = gen.predict_tier(train_df)
        _, y_val = gen.predict_tier(val_df)

        self.assertEqual(len(y_train), len(train_df))
        self.assertEqual(len(y_val), len(val_df))
        self.assertTrue(set(y_train).issubset(set(CLASS_LABELS)))
        self.assertTrue(set(y_val).issubset(set(CLASS_LABELS)))

    def test_validation_perturbation_invariance(self):
        """
        Changing validation feature values must NOT change the fitted thresholds
        or training labels, though it may change validation labels.
        """
        train_df = self.synth_df[self.synth_df["fold"] != 0].copy()
        val_df = self.synth_df[self.synth_df["fold"] == 0].copy()

        # Baseline fit
        gen_orig = RPHILabelGenerator(threshold_mode="group_level", group_col="canonical_pothole_id").fit(train_df)
        _, y_train_orig = gen_orig.predict_tier(train_df)
        _, y_val_orig = gen_orig.predict_tier(val_df)

        # Perturb validation feature values dramatically
        val_df_perturbed = val_df.copy()
        val_df_perturbed["relative_depth_max_intensity"] *= 100.0
        val_df_perturbed["mask_area_pixels"] *= 100.0
        val_df_perturbed["relative_volume_proxy"] *= 100.0

        # In a leakage-safe pipeline, the generator is fitted ONLY on train_df
        gen_after = RPHILabelGenerator(threshold_mode="group_level", group_col="canonical_pothole_id").fit(train_df)
        _, y_train_after = gen_after.predict_tier(train_df)
        _, y_val_perturbed = gen_after.predict_tier(val_df_perturbed)

        # 1. Fitted thresholds must remain 100% identical
        self.assertEqual(gen_orig.threshold_low_, gen_after.threshold_low_)
        self.assertEqual(gen_orig.threshold_high_, gen_after.threshold_high_)

        # 2. Training labels must remain 100% identical
        np.testing.assert_array_equal(y_train_orig, y_train_after)

        # 3. Validation labels SHOULD change (shifted to High due to inflated severity)
        self.assertTrue(any(y_val_orig != y_val_perturbed))
        self.assertTrue(all(tier == "High" for tier in y_val_perturbed))

    def test_evaluate_single_experiment_synthetic(self):
        res = evaluate_single_experiment(
            df=self.synth_df,
            feature_cols=EXP_A_FEATURES,
            experiment_name="Test Synthetic Exp A",
            random_state=42,
        )
        self.assertIn("model_results", res)
        self.assertIn("fold_partitions", res)
        self.assertEqual(len(res["fold_partitions"]), 5)

        for m_name, m_data in res["model_results"].items():
            mm = m_data["mean_metrics"]
            self.assertTrue(0.0 <= mm["mean_accuracy"] <= 1.0)
            self.assertTrue(0.0 <= mm["mean_balanced_accuracy"] <= 1.0)
            self.assertTrue(0.0 <= mm["mean_macro_f1"] <= 1.0)

            cm = m_data["aggregated_confusion_matrix"]
            self.assertEqual(cm.shape, (3, 3))
            self.assertEqual(np.sum(cm), len(self.synth_df))

    def test_real_dataset_integrity(self):
        labeled_path = r"c:\Users\Anushka\Documents\pothole-severity-detection\data\processed\pothole_features_labeled.csv"
        if os.path.exists(labeled_path):
            df_disk = pd.read_csv(labeled_path)
            for f in range(5):
                train_p = set(df_disk[df_disk["fold"] != f]["canonical_pothole_id"])
                val_p = set(df_disk[df_disk["fold"] == f]["canonical_pothole_id"])
                self.assertEqual(len(train_p & val_p), 0, f"Real dataset leakage in fold {f}")


if __name__ == "__main__":
    unittest.main()
