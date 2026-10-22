"""
Unit test suite for RGB Pothole Segmentation module and predictor integration.

Validates:
  - Multi-scale pixel feature extraction.
  - Model serialization and loading.
  - IoU and Dice metric calculations.
  - Output mask properties: 2D dimensions, uint8 dtype, binary values.
  - Input validation: rejection of non-3D arrays, invalid channel counts, non-finite values.
  - Predictor integration: predict_rgb() and unified predict(rgb=...) interface.
  - Graceful handling of uniform/non-pothole imagery.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.train_baselines import CLASS_LABELS
from src.models.segmentation import (
    RGBSegmenter,
    extract_pixel_features,
    DEFAULT_SEGMENTER_PATH,
)
from src.models.train_segmenter import compute_iou_and_dice
from src.models.predict import PotholeSeverityPredictor


class TestRGBSegmentation(unittest.TestCase):
    """Test suite for RGBSegmenter pipeline."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic RGB pothole test image."""
        # 100x150 road pavement image (light gray: ~180) with a dark central pothole (~80)
        cls.h, cls.w = 100, 150
        cls.synthetic_rgb = np.full((cls.h, cls.w, 3), 180, dtype=np.uint8)
        # Draw central dark pothole cavity
        cls.synthetic_rgb[30:70, 45:105, :] = 80

        # Initialize segmenter
        cls.segmenter = RGBSegmenter()

    def test_segmenter_loaded(self):
        """Verify that the production segmenter artifact is loaded."""
        self.assertIsNotNone(
            self.segmenter.classifier,
            "RGBSegmenter classifier was not loaded. Run 'python src/models/train_segmenter.py'."
        )

    def test_extract_pixel_features_shape(self):
        """Verify feature extraction dimensions and finite values."""
        feats, (th, tw), (oh, ow) = extract_pixel_features(self.synthetic_rgb, scale=0.25)

        self.assertEqual(oh, self.h)
        self.assertEqual(ow, self.w)
        self.assertEqual(th, int(self.h * 0.25))
        self.assertEqual(tw, int(self.w * 0.25))
        self.assertEqual(feats.shape, (th * tw, 15))
        self.assertTrue(np.isfinite(feats).all())

    def test_predict_mask_properties(self):
        """Verify output mask has identical spatial dimensions, uint8 dtype, and binary values."""
        mask = self.segmenter.predict_mask(self.synthetic_rgb)

        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_predict_mask_rejection_invalid_inputs(self):
        """Verify input validation rejects non-3D, wrong channel count, and non-finite arrays."""
        # 1D array
        with self.assertRaises(ValueError):
            self.segmenter.predict_mask(np.zeros(100, dtype=np.uint8))

        # 2D array (grayscale)
        with self.assertRaises(ValueError):
            self.segmenter.predict_mask(np.zeros((80, 120), dtype=np.uint8))

        # 4D array
        with self.assertRaises(ValueError):
            self.segmenter.predict_mask(np.zeros((1, 80, 120, 3), dtype=np.uint8))

        # 3D array with invalid channel count (e.g. 4 channels RGBA)
        with self.assertRaises(ValueError):
            self.segmenter.predict_mask(np.zeros((80, 120, 4), dtype=np.uint8))

        # Non-finite values
        rgb_with_nan = self.synthetic_rgb.astype(float).copy()
        rgb_with_nan[10, 10, 0] = np.nan
        with self.assertRaises(ValueError):
            self.segmenter.predict_mask(rgb_with_nan)

        # Non-existent file path
        with self.assertRaises(FileNotFoundError):
            self.segmenter.predict_mask("nonexistent_rgb_image.jpg")

    def test_predict_mask_invalid_type_rejection(self):
        """Verify invalid input types raise TypeError."""
        with self.assertRaises(TypeError):
            self.segmenter.predict_mask(12345)

        with self.assertRaises(TypeError):
            self.segmenter.predict_mask(["not_an_array"])

    def test_predict_mask_file_path_input(self):
        """Verify predict_mask accepts a valid file path string."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(self.synthetic_rgb)
            pil_img.save(tmp_path)

            mask = self.segmenter.predict_mask(tmp_path)
            self.assertEqual(mask.shape, (self.h, self.w))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(np.any(mask > 0))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_compute_iou_and_dice(self):
        """Verify IoU and Dice calculations on known synthetic cases."""
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m2 = np.zeros((50, 50), dtype=np.uint8)

        # Perfect match
        m1[10:30, 10:30] = 1
        m2[10:30, 10:30] = 1
        iou, dice = compute_iou_and_dice(m1, m2)
        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(dice, 1.0)

        # Disjoint
        m3 = np.zeros((50, 50), dtype=np.uint8)
        m3[35:45, 35:45] = 1
        iou_disjoint, dice_disjoint = compute_iou_and_dice(m1, m3)
        self.assertEqual(iou_disjoint, 0.0)
        self.assertEqual(dice_disjoint, 0.0)

        # Both empty
        empty1 = np.zeros((50, 50), dtype=np.uint8)
        empty2 = np.zeros((50, 50), dtype=np.uint8)
        iou_e, dice_e = compute_iou_and_dice(empty1, empty2)
        self.assertEqual(iou_e, 1.0)
        self.assertEqual(dice_e, 1.0)


class TestPredictorRGBIntegration(unittest.TestCase):
    """Test suite for PotholeSeverityPredictor RGB integration."""

    @classmethod
    def setUpClass(cls):
        """Initialize unified predictor with Random Forest engine for RGBSegmenter testing."""
        cls.predictor = PotholeSeverityPredictor(segmenter_engine="random_forest")

        # Synthetic road image with dark cavity
        cls.h, cls.w = 100, 150
        cls.pothole_rgb = np.full((cls.h, cls.w, 3), 180, dtype=np.uint8)
        cls.pothole_rgb[30:70, 45:105, :] = 80

        # Uniform bright road image with no pothole
        cls.clean_road_rgb = np.full((cls.h, cls.w, 3), 190, dtype=np.uint8)

    def test_predict_rgb_successful_flow(self):
        """Verify end-to-end RGB prediction flow."""
        res = self.predictor.predict_rgb(self.pothole_rgb, return_mask=True)

        self.assertEqual(res["input_modality"], "RGB")
        self.assertIn("segmentation_model", res)
        self.assertTrue(res["pothole_detected"])
        self.assertIn(res["predicted_tier"], CLASS_LABELS)
        self.assertIsNotNone(res["segmented_mask"])
        self.assertEqual(res["segmented_mask"].shape, (self.h, self.w))

    def test_predict_rgb_uniform_image_graceful_handling(self):
        """Verify that an image without any detected cavity returns pothole_detected=False gracefully."""
        res = self.predictor.predict_rgb(self.clean_road_rgb)

        self.assertEqual(res["input_modality"], "RGB")
        self.assertFalse(res["pothole_detected"])
        self.assertEqual(res["predicted_tier"], "None")
        self.assertIn("No pothole", res["message"])

    def test_unified_predict_dispatch_rgb(self):
        """Verify that predictor.predict(rgb=...) routes to RGB segmentation."""
        res = self.predictor.predict(rgb=self.pothole_rgb)
        self.assertEqual(res["input_modality"], "RGB")
        self.assertIn(res["predicted_tier"], CLASS_LABELS)

    def test_missing_segmenter_artifact_handling(self):
        """Verify that missing segmenter raises RuntimeError."""
        pred_no_seg = PotholeSeverityPredictor(segmenter_path="nonexistent_segmenter.joblib")
        self.assertIsNone(pred_no_seg.segmenter)

        with self.assertRaises(RuntimeError) as ctx:
            pred_no_seg.predict_rgb(self.pothole_rgb)
        self.assertIn("RGB Segmenter is not loaded", str(ctx.exception))

    def test_predict_2d_direct_integration_with_segmented_mask(self):
        """Verify that a mask produced by RGBSegmenter integrates directly with predict_2d()."""
        segmented_mask = self.predictor.segmenter.predict_mask(self.pothole_rgb)
        self.assertEqual(segmented_mask.shape, (self.h, self.w))
        self.assertEqual(segmented_mask.dtype, np.uint8)

        # Pass directly into predict_2d
        res = self.predictor.predict_2d(segmented_mask)
        self.assertIn(res["predicted_tier"], CLASS_LABELS)
        self.assertEqual(res["model_id"], "svm_2d_geometry")
        self.assertIn("extracted_features", res)
        self.assertEqual(len(res["extracted_features"]), 14)
        for feat_name, feat_val in res["extracted_features"].items():
            self.assertTrue(np.isfinite(feat_val), f"Feature {feat_name} is non-finite: {feat_val}")

    def test_predict_rgb_file_path_input(self):
        """Verify predict_rgb accepts image file paths from disk."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_img = Image.fromarray(self.pothole_rgb)
            pil_img.save(tmp_path)

            res = self.predictor.predict_rgb(tmp_path, return_mask=True)
            self.assertEqual(res["input_modality"], "RGB")
            self.assertTrue(res["pothole_detected"])
            self.assertIn(res["predicted_tier"], CLASS_LABELS)
            self.assertIsNotNone(res["segmented_mask"])
            self.assertEqual(res["segmented_mask"].shape, (self.h, self.w))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_unified_predict_no_args_raises(self):
        """Verify predict() raises ValueError when neither rgb nor mask is provided."""
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict()
        self.assertIn("requires either 'rgb' or 'mask'", str(ctx.exception))


class TestMultiPotholeSegmentation(unittest.TestCase):
    """Regression test suite for multi-pothole retention and noise rejection."""

    @classmethod
    def setUpClass(cls):
        import scipy.ndimage as ndi
        cls.ndi = ndi
        cls.h, cls.w = 200, 300
        cls.segmenter = RGBSegmenter()
        cls.predictor = PotholeSeverityPredictor(segmenter_engine="random_forest")

        # Base light-gray road pavement
        cls.base_road = np.full((cls.h, cls.w, 3), 180, dtype=np.uint8)

    def test_single_pothole_retention(self):
        """Verify single pothole is correctly detected and retained."""
        img = self.base_road.copy()
        # Single central cavity (dark patch ~60)
        img[60:140, 90:210, :] = 60

        mask = self.segmenter.predict_mask(img)
        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertTrue(np.any(mask > 0))

        lbl, num = self.ndi.label(mask > 0)
        self.assertEqual(num, 1, f"Expected 1 connected component, got {num}")

    def test_two_potholes_retention_and_legacy_comparison(self):
        """Verify two separate potholes are retained by default, while legacy mode keeps only one."""
        img = self.base_road.copy()
        # Pothole 1 (larger, left)
        img[30:90, 30:110, :] = 55
        # Pothole 2 (smaller, right, well separated)
        img[110:170, 180:260, :] = 55

        # 1. Multi-component mode (default)
        mask_multi = self.segmenter.predict_mask(img, keep_all_components=True)
        lbl_multi, num_multi = self.ndi.label(mask_multi > 0)
        self.assertEqual(num_multi, 2, f"Expected 2 separate potholes, got {num_multi}")

        # 2. Legacy single-dominant mode
        mask_single = self.segmenter.predict_mask(img, single_dominant=True)
        lbl_single, num_single = self.ndi.label(mask_single > 0)
        self.assertEqual(num_single, 1, f"Expected exactly 1 pothole in single_dominant mode, got {num_single}")

        # Multi-component mask should have strictly more foreground area than single-dominant mask
        self.assertGreater(
            np.sum(mask_multi > 0),
            np.sum(mask_single > 0),
            "Multi-pothole mask must contain more foreground area than single-dominant mask"
        )

    def test_three_potholes_retention(self):
        """Verify three distinct separated potholes are all detected and retained."""
        img = self.base_road.copy()
        # Pothole 1: top-left
        img[20:70, 30:90, :] = 50
        # Pothole 2: bottom-left
        img[120:170, 30:90, :] = 50
        # Pothole 3: middle-right
        img[70:130, 180:260, :] = 50

        mask = self.segmenter.predict_mask(img, keep_all_components=True)
        lbl, num = self.ndi.label(mask > 0)
        self.assertEqual(num, 3, f"Expected 3 separate potholes, got {num}")

    def test_noise_components_filtered_out(self):
        """Verify that genuine potholes are retained while tiny noise specks are eliminated."""
        img = self.base_road.copy()
        # Genuine substantial pothole cavity
        img[50:150, 100:200, :] = 50

        # Simulate small dark speckles (e.g. 2x2 or 3x3 oil droplets / road noise)
        img[10:13, 10:13, :] = 40
        img[10:12, 280:282, :] = 40
        img[180:183, 20:23, :] = 40

        mask = self.segmenter.predict_mask(img, min_component_size=20)
        lbl, num = self.ndi.label(mask > 0)
        # Should retain only the 1 large pothole, filtering out all tiny specks
        self.assertEqual(num, 1, f"Expected 1 valid component after noise filtering, got {num}")

        # Verify that the noise locations are background (0)
        self.assertEqual(mask[11, 11], 0)
        self.assertEqual(mask[11, 281], 0)
        self.assertEqual(mask[181, 21], 0)

    def test_empty_mask_on_uniform_pavement(self):
        """Verify all-zero mask is returned when no pothole exists."""
        uniform_road = np.full((self.h, self.w, 3), 190, dtype=np.uint8)
        mask = self.segmenter.predict_mask(uniform_road)
        self.assertEqual(np.sum(mask > 0), 0)

    def test_severity_pipeline_compatibility_with_multiple_potholes(self):
        """Verify multi-pothole mask integrates seamlessly with Model B severity classifier."""
        img = self.base_road.copy()
        img[30:90, 30:110, :] = 55
        img[110:170, 180:260, :] = 55

        res = self.predictor.predict(rgb=img)
        self.assertEqual(res["input_modality"], "RGB")
        self.assertTrue(res["pothole_detected"])
        self.assertIn(res["predicted_tier"], CLASS_LABELS)
        self.assertEqual(len(res["extracted_features"]), 14)


if __name__ == "__main__":
    unittest.main()
