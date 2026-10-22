"""
Unit test suite for production model inference and PotholeSeverityPredictor.

Validates:
  - Model serialization and deserialization.
  - 2D mask geometric inference (Model B).
  - Mask + depth map RPHI formula reproduction inference (Model A).
  - Unified predict() routing.
  - Robust input validation: empty mask rejection, dimension mismatches,
    non-finite values, and non-2D arrays.
  - Missing artifact error handling.
  - Proper uncalibrated decision score representation (not probabilities).
  - File-path input support using temporary test fixtures.
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.train_baselines import CLASS_LABELS, EXP_A_FEATURES, EXP_B_FEATURES
from src.models.predict import PotholeSeverityPredictor, DEFAULT_MODELS_DIR


class TestInferencePipeline(unittest.TestCase):
    """Test suite for PotholeSeverityPredictor and inference pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initialize predictor using trained models."""
        cls.predictor = PotholeSeverityPredictor()

        # Create synthetic test mask (100x100 with a 30x30 central pothole)
        cls.valid_mask = np.zeros((100, 100), dtype=np.uint8)
        cls.valid_mask[35:65, 35:65] = 255

        # Create synthetic test depth map (road baseline ~100, cavity depression ~160)
        cls.valid_depth = np.full((100, 100), 100, dtype=np.uint8)
        cls.valid_depth[35:65, 35:65] = 160

    def test_models_loaded_successfully(self):
        """Verify that default model artifacts are found and properly deserialized."""
        self.assertIsNotNone(
            self.predictor.model_a,
            "Model A was not loaded. Run 'python src/models/train_final_model.py'."
        )
        self.assertIsNotNone(
            self.predictor.model_b,
            "Model B was not loaded. Run 'python src/models/train_final_model.py'."
        )
        self.assertEqual(list(self.predictor.model_a["features"]), list(EXP_A_FEATURES))
        self.assertEqual(list(self.predictor.model_b["features"]), list(EXP_B_FEATURES))

    def test_predict_2d_valid_mask(self):
        """Verify 2D geometry inference produces valid schema and tier."""
        result = self.predictor.predict_2d(self.valid_mask)

        self.assertEqual(result["experiment"], "B (2D Geometry Only)")
        self.assertIn(result["predicted_tier"], CLASS_LABELS)
        self.assertEqual(result["input_shape"], (100, 100))

        # Check extracted features match expected 14 features
        for feat in EXP_B_FEATURES:
            self.assertIn(feat, result["extracted_features"])
            self.assertTrue(np.isfinite(result["extracted_features"][feat]))

        # Check uncalibrated SVM decision scores
        self.assertIsNotNone(result["decision_scores"])
        for label in CLASS_LABELS:
            self.assertIn(label, result["decision_scores"])
            self.assertIsInstance(result["decision_scores"][label], float)

    def test_predict_with_depth_valid_pair(self):
        """Verify paired mask and depth map inference produces valid schema and tier."""
        result = self.predictor.predict_with_depth(self.valid_mask, self.valid_depth)

        self.assertEqual(result["experiment"], "A (RPHI Formula Reproduction)")
        self.assertIn(result["predicted_tier"], CLASS_LABELS)
        self.assertEqual(result["input_shape"], (100, 100))

        # Check constituent inputs
        for feat in EXP_A_FEATURES:
            self.assertIn(feat, result["extracted_features"])
            self.assertTrue(np.isfinite(result["extracted_features"][feat]))

        # Check supporting depth statistics
        self.assertIn("road_median_intensity", result["supporting_depth_stats"])
        self.assertIn("depth_max_intensity", result["supporting_depth_stats"])

    def test_unified_predict_dispatch(self):
        """Verify predict() routes to 2D model when depth is None and depth model when depth is provided."""
        res_2d = self.predictor.predict(self.valid_mask)
        self.assertEqual(res_2d["experiment"], "B (2D Geometry Only)")

        res_depth = self.predictor.predict(self.valid_mask, self.valid_depth)
        self.assertEqual(res_depth["experiment"], "A (RPHI Formula Reproduction)")

    def test_empty_mask_rejection(self):
        """Verify that an all-zero empty mask is rejected with a clear ValueError."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaises(ValueError) as ctx_2d:
            self.predictor.predict_2d(empty_mask)
        self.assertIn("Empty mask error", str(ctx_2d.exception))

        with self.assertRaises(ValueError) as ctx_depth:
            self.predictor.predict_with_depth(empty_mask, self.valid_depth)
        self.assertIn("Empty mask error", str(ctx_depth.exception))

    def test_dimension_mismatch_rejection(self):
        """Verify that mismatched mask and depth dimensions are rejected."""
        mismatched_depth = np.zeros((80, 80), dtype=np.uint8)

        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_with_depth(self.valid_mask, mismatched_depth)
        self.assertIn("Dimension mismatch", str(ctx.exception))

    def test_invalid_array_dimensions_rejection(self):
        """Verify that 1D, 4D, or invalid 3D arrays are rejected."""
        # 1D array
        with self.assertRaises(ValueError):
            self.predictor.predict_2d(np.zeros(100, dtype=np.uint8))

        # 4D array
        with self.assertRaises(ValueError):
            self.predictor.predict_2d(np.zeros((1, 2, 3, 4), dtype=np.uint8))

        # 3D depth with invalid channel count (e.g. 5 channels)
        with self.assertRaises(ValueError):
            self.predictor.predict_with_depth(self.valid_mask, np.zeros((100, 100, 5), dtype=np.uint8))

    def test_non_finite_depth_rejection(self):
        """Verify that depth maps with NaN or Inf are rejected."""
        depth_with_nan = self.valid_depth.astype(float).copy()
        depth_with_nan[10, 10] = np.nan

        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_with_depth(self.valid_mask, depth_with_nan)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_missing_artifacts_handling(self):
        """Verify that missing model files raise informative RuntimeError on prediction."""
        dummy_pred = PotholeSeverityPredictor(
            model_a_path="nonexistent_model_a.joblib",
            model_b_path="nonexistent_model_b.joblib",
        )
        self.assertIsNone(dummy_pred.model_a)
        self.assertIsNone(dummy_pred.model_b)

        with self.assertRaises(RuntimeError) as ctx_b:
            dummy_pred.predict_2d(self.valid_mask)
        self.assertIn("Model B (2D Geometry) is not loaded", str(ctx_b.exception))

        with self.assertRaises(RuntimeError) as ctx_a:
            dummy_pred.predict_with_depth(self.valid_mask, self.valid_depth)
        self.assertIn("Model A (RPHI Reproducer) is not loaded", str(ctx_a.exception))

    def test_file_path_inputs(self):
        """Verify that predictor accepts image file paths on disk."""
        temp_dir = tempfile.mkdtemp()
        try:
            mask_path = os.path.join(temp_dir, "test_mask.png")
            depth_path = os.path.join(temp_dir, "test_depth.png")

            Image.fromarray(self.valid_mask).save(mask_path)
            Image.fromarray(self.valid_depth).save(depth_path)

            # Predict using file paths
            res_2d = self.predictor.predict_2d(mask_path)
            self.assertIn(res_2d["predicted_tier"], CLASS_LABELS)

            res_depth = self.predictor.predict_with_depth(mask_path, depth_path)
            self.assertIn(res_depth["predicted_tier"], CLASS_LABELS)
        finally:
            shutil.rmtree(temp_dir)

    def test_decision_scores_are_hyperplane_margins(self):
        """Verify that decision scores are signed hyperplane margins, not probabilities."""
        res = self.predictor.predict_2d(self.valid_mask)
        scores = res["decision_scores"]
        score_values = list(scores.values())

        # Hyperplane margin scores can be negative and do not sum to 1.0
        self.assertTrue(any(s != 0.0 for s in score_values))
        sum_scores = sum(score_values)
        self.assertFalse(
            np.isclose(sum_scores, 1.0, atol=1e-4) and all(s >= 0 for s in score_values),
            "Decision scores must not be represented as normalized probabilities."
        )


if __name__ == "__main__":
    unittest.main()
