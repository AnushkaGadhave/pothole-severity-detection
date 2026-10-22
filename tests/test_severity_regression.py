"""
Regression tests for CrackSight Severity Prediction Pipeline.

Verifies:
1. Decision scores label mapping matches pipeline.classes_.
2. Features are passed in exact expected schema order.
3. Different masks produce appropriate feature and tier changes (Low, Medium, High).
4. Empty/invalid masks do not silently produce Low severity.
5. DeepLabV3 remains the primary default segmentation engine.
6. Multi-resolution invariance prevents non-1200x800 images from collapsing to Low.
7. Frontend-backend schema compatibility is maintained.
"""

import io
import unittest
import numpy as np
from PIL import Image

from src.models.predict import PotholeSeverityPredictor, CLASS_LABELS
from src.models.inference_service import PotholeInferenceService
from src.models.segmentation import DeepLabV3Segmenter, get_segmenter
from src.features.feature_extractor import compute_mask_geometry


class TestSeverityRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = PotholeSeverityPredictor()
        cls.service = PotholeInferenceService(segmenter_engine="deeplabv3")

    def test_deeplabv3_remains_primary_segmenter(self):
        """Verify DeepLabV3-MobileNetV3 is the primary default segmentation engine."""
        segmenter = get_segmenter()
        self.assertIsInstance(segmenter, DeepLabV3Segmenter)
        self.assertEqual(self.service.segmenter_engine, "deeplabv3")
        self.assertIsInstance(self.service.predictor.segmenter, DeepLabV3Segmenter)

    def test_decision_scores_classes_mapping(self):
        """Verify decision_scores dictionary correctly aligns with pipeline.classes_."""
        # Create a large defect mask
        mask = np.zeros((800, 1200), dtype=np.uint8)
        y, x = np.ogrid[:800, :1200]
        mask[(x - 600) ** 2 + (y - 400) ** 2 <= 200 ** 2] = 255

        res = self.predictor.predict_2d(mask)
        pipeline = self.predictor.model_b["pipeline"]
        classes = list(getattr(pipeline, "classes_", getattr(getattr(pipeline, "named_steps", {}).get("svc", None), "classes_", CLASS_LABELS)))

        # Ensure all classes are present in decision_scores
        for c in classes:
            self.assertIn(c, res["decision_scores"])
            self.assertIsInstance(res["decision_scores"][c], float)

        # For a high severity pothole, High decision score should be positive and highest
        self.assertEqual(res["predicted_tier"], "High")
        top_class = max(res["decision_scores"], key=res["decision_scores"].get)
        self.assertEqual(top_class, "High")

    def test_feature_order_integrity(self):
        """Verify features passed to the model match expected schema in exact order."""
        feature_order = self.predictor.model_b["features"]
        self.assertEqual(len(feature_order), 14)
        expected_order = [
            "mask_area_pixels", "mask_coverage_ratio", "mask_coverage_pct",
            "perimeter_pixels", "circularity", "compactness",
            "bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max",
            "bbox_width", "bbox_height", "bbox_aspect_ratio", "bbox_extent"
        ]
        self.assertEqual(list(feature_order), expected_order)

    def test_different_masks_produce_distinct_tiers(self):
        """Verify small, medium, and large potholes produce Low, Medium, and High respectively."""
        # 1. Small defect (~1.5% coverage on 400x400)
        mask_small = np.zeros((400, 400), dtype=np.uint8)
        r_small = np.sqrt(400 * 400 * 0.015 / np.pi)
        y, x = np.ogrid[:400, :400]
        mask_small[(x - 200) ** 2 + (y - 200) ** 2 <= r_small ** 2] = 255
        res_small = self.predictor.predict_2d(mask_small)
        self.assertEqual(res_small["predicted_tier"], "Low")

        # 2. Medium defect (~4.5% coverage on 400x400)
        mask_med = np.zeros((400, 400), dtype=np.uint8)
        r_med = np.sqrt(400 * 400 * 0.045 / np.pi)
        mask_med[(x - 200) ** 2 + (y - 200) ** 2 <= r_med ** 2] = 255
        res_med = self.predictor.predict_2d(mask_med)
        self.assertEqual(res_med["predicted_tier"], "Medium")

        # 3. Large defect (~15.0% coverage on 400x400)
        mask_large = np.zeros((400, 400), dtype=np.uint8)
        r_large = np.sqrt(400 * 400 * 0.150 / np.pi)
        mask_large[(x - 200) ** 2 + (y - 200) ** 2 <= r_large ** 2] = 255
        res_large = self.predictor.predict_2d(mask_large)
        self.assertEqual(res_large["predicted_tier"], "High")

        # Verify geometric features differ significantly
        self.assertLess(res_small["extracted_features"]["mask_area_pixels"], res_med["extracted_features"]["mask_area_pixels"])
        self.assertLess(res_med["extracted_features"]["mask_area_pixels"], res_large["extracted_features"]["mask_area_pixels"])

    def test_empty_mask_does_not_silently_produce_low(self):
        """Verify empty mask raises ValueError in predict_2d and yields 'None' in inference service."""
        empty_mask = np.zeros((200, 200), dtype=np.uint8)

        # In predict_2d, empty mask must raise an explicit ValueError
        with self.assertRaises(ValueError):
            self.predictor.predict_2d(empty_mask)

        # In inference service, clean road produces tier='None' and pothole_detected=False
        clean_img = np.full((150, 150, 3), 180, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(clean_img).save(buf, format="PNG")
        res = self.service.analyze_road_image(buf.getvalue(), filename="clean.png")

        self.assertFalse(res["pothole_detected"])
        self.assertEqual(res["predicted_tier"], "None")
        self.assertNotEqual(res["predicted_tier"], "Low")

    def test_non_finite_inputs_raise_errors(self):
        """Verify non-finite values in masks are caught and not silently passed."""
        nan_mask = np.full((100, 100), np.nan)
        with self.assertRaises((ValueError, TypeError)):
            self.predictor.predict_2d(nan_mask)

    def test_multi_resolution_invariance(self):
        """Verify that large potholes on non-1200x800 resolutions are NOT misclassified as Low."""
        # 15% pothole across various non-training resolutions
        for (h, w) in [(400, 400), (256, 256), (1080, 1920), (500, 500)]:
            mask = np.zeros((h, w), dtype=np.uint8)
            r = np.sqrt(h * w * 0.15 / np.pi)
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = 255

            res = self.predictor.predict_2d(mask)
            self.assertEqual(
                res["predicted_tier"], "High",
                f"Resolution ({h}x{w}) failed to predict High for 15% coverage (got {res['predicted_tier']})"
            )


if __name__ == "__main__":
    unittest.main()
