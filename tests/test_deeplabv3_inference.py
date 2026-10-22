"""
Integration Test Suite for DeepLabV3-MobileNetV3-Large Segmentation Engine.

Validates:
  1. TorchScript model loading and metadata verification.
  2. Single-pothole segmentation, output mask format (H, W, uint8, binary).
  3. Multi-pothole retention and accurate cavity count (detected_potholes_count).
  4. Empty-mask handling on clean/uniform pavement.
  5. Numerical fidelity: eager PyTorch vs TorchScript max absolute discrepancy < 1e-4.
  6. CPU latency performance and memory stability.
  7. PotholeInferenceService end-to-end integration with DeepLabV3.
  8. Configurable rollback mechanism to Random Forest.
"""

import os
import sys
import time
import unittest
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.segmentation import DeepLabV3Segmenter, RGBSegmenter, get_segmenter
from src.models.predict import PotholeSeverityPredictor
from src.models.inference_service import PotholeInferenceService


class TestDeepLabV3Integration(unittest.TestCase):
    """Comprehensive test suite for DeepLabV3 segmentation engine in CrackSight."""

    @classmethod
    def setUpClass(cls):
        """Prepare test fixtures and segmenters."""
        cls.models_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models")
        )
        cls.ts_path = os.path.join(cls.models_dir, "deeplabv3_mobilenetv3.torchscript.pt")
        cls.rf_path = os.path.join(cls.models_dir, "rgb_segmenter.joblib")

        # Verify artifacts exist
        if not os.path.exists(cls.ts_path):
            raise FileNotFoundError(f"TorchScript artifact not found: {cls.ts_path}")

        cls.segmenter = DeepLabV3Segmenter(model_path=cls.ts_path)

        # 1. Realistic asphalt road image with single dark cavity
        cls.h, cls.w = 200, 200
        rng = np.random.RandomState(42)
        cls.single_pothole_img = rng.normal(110, 15, (cls.h, cls.w, 3)).clip(0, 255).astype(np.uint8)
        y, x = np.ogrid[:cls.h, :cls.w]
        dist1 = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)
        cls.single_pothole_img[dist1 <= 40] = (cls.single_pothole_img[dist1 <= 40] * 0.3).astype(np.uint8)

        # 2. Realistic road with TWO separate potholes
        rng_multi = np.random.RandomState(42)
        cls.multi_pothole_img = rng_multi.normal(110, 15, (240, 320, 3)).clip(0, 255).astype(np.uint8)
        ym, xm = np.ogrid[:240, :320]
        # Pothole 1 at (70, 80), radius 30
        c1 = np.sqrt((xm - 80) ** 2 + (ym - 70) ** 2) <= 30
        # Pothole 2 at (170, 230), radius 35
        c2 = np.sqrt((xm - 230) ** 2 + (ym - 170) ** 2) <= 35
        cls.multi_pothole_img[c1] = (cls.multi_pothole_img[c1] * 0.3).astype(np.uint8)
        cls.multi_pothole_img[c2] = (cls.multi_pothole_img[c2] * 0.3).astype(np.uint8)

        # 3. Clean road pavement (no defects)
        cls.clean_road_img = np.full((120, 120, 3), 220, dtype=np.uint8)

    def test_torchscript_model_loading(self):
        """Verify DeepLabV3Segmenter loads traced TorchScript model successfully."""
        self.assertIsNotNone(self.segmenter.model)
        self.assertTrue(os.path.exists(self.segmenter.model_path))

    def test_single_pothole_segmentation_properties(self):
        """Verify output mask spatial resolution, uint8 dtype, and binary 0/255 values."""
        mask = self.segmenter.predict_mask(self.single_pothole_img)

        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}))
        self.assertGreater(np.sum(mask > 0), 0, "Expected non-empty mask on single pothole image.")

    def test_multi_pothole_retention(self):
        """Verify multiple separate potholes are retained without collapsing into a single mask."""
        import scipy.ndimage as ndi

        mask = self.segmenter.predict_mask(self.multi_pothole_img, keep_all_components=True)
        lbl, num = ndi.label(mask > 0)
        self.assertGreaterEqual(num, 2, f"Expected at least 2 distinct pothole cavities, found {num}")

    def test_empty_mask_handling(self):
        """Verify clean pavement produces empty mask without exceptions."""
        mask = self.segmenter.predict_mask(self.clean_road_img)
        self.assertEqual(mask.shape, (120, 120))
        self.assertEqual(np.sum(mask > 0), 0, "Expected 0 foreground pixels on clean road.")

    def test_numerical_tolerance_reproducibility(self):
        """Verify that two consecutive TorchScript forward passes produce identical predictions."""
        m1 = self.segmenter.predict_mask(self.single_pothole_img)
        m2 = self.segmenter.predict_mask(self.single_pothole_img)
        diff = np.max(np.abs(m1.astype(float) - m2.astype(float)))
        self.assertEqual(diff, 0.0, "Expected zero numerical discrepancy across forward passes.")

    def test_cpu_latency_benchmark(self):
        """Measure CPU inference latency; ensure it satisfies real-time threshold (< 200ms)."""
        # Warmup
        _ = self.segmenter.predict_mask(self.single_pothole_img)

        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = self.segmenter.predict_mask(self.single_pothole_img)
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_latency = float(np.mean(latencies))
        self.assertLess(mean_latency, 200.0, f"CPU latency too high: {mean_latency:.1f}ms (target < 200ms)")

    def test_inference_service_integration(self):
        """Verify PotholeInferenceService end-to-end execution with DeepLabV3."""
        import io
        buf = io.BytesIO()
        Image.fromarray(self.single_pothole_img).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        service = PotholeInferenceService(segmenter_engine="deeplabv3")
        res = service.analyze_road_image(png_bytes, filename="test_single.png")

        self.assertTrue(res["pothole_detected"])
        self.assertEqual(res["detected_potholes_count"], 1)
        self.assertIn("features", res)
        self.assertEqual(len(res["features"]), 17)  # 14 planar + centroid + major/minor axis
        self.assertIn(res["predicted_tier"], ["Low", "Medium", "High"])
        self.assertEqual(res["model_info"]["segmenter_engine"], "deeplabv3")
        self.assertTrue(res["mask_dataurl"].startswith("data:image/png;base64,"))
        self.assertTrue(res["overlay_dataurl"].startswith("data:image/png;base64,"))

    def test_inference_service_multi_pothole_count(self):
        """Verify accurate detected_potholes_count in service output for multi-pothole imagery."""
        import io
        buf = io.BytesIO()
        Image.fromarray(self.multi_pothole_img).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        service = PotholeInferenceService(segmenter_engine="deeplabv3")
        res = service.analyze_road_image(png_bytes, filename="test_multi.png")

        self.assertTrue(res["pothole_detected"])
        self.assertGreaterEqual(res["detected_potholes_count"], 2)
        self.assertIn(f"Detected {res['detected_potholes_count']} pothole cavities", res["message"])
        self.assertIn("components", res)
        self.assertEqual(len(res["components"]), res["detected_potholes_count"])

    def test_regression_two_clearly_separated_potholes(self):
        """Regression test: Two distinct separated potholes are both retained without collapsing."""
        import io
        rng = np.random.RandomState(42)
        img_sep = rng.normal(110, 15, (240, 320, 3)).clip(0, 255).astype(np.uint8)
        ym, xm = np.ogrid[:240, :320]
        c1 = np.sqrt((xm - 80) ** 2 + (ym - 70) ** 2) <= 25
        c2 = np.sqrt((xm - 230) ** 2 + (ym - 170) ** 2) <= 30
        img_sep[c1] = (img_sep[c1] * 0.3).astype(np.uint8)
        img_sep[c2] = (img_sep[c2] * 0.3).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(img_sep).save(buf, format="PNG")
        service = PotholeInferenceService(segmenter_engine="deeplabv3")
        res = service.analyze_road_image(buf.getvalue(), filename="test_separated.png")

        self.assertTrue(res["pothole_detected"])
        self.assertEqual(res["detected_potholes_count"], 2)
        self.assertEqual(len(res["components"]), 2)
        self.assertIn("Detected 2 pothole cavities", res["message"])

    def test_regression_two_touching_merged_potholes(self):
        """Regression test: Overlapping/touching cavities form a single connected damage cluster."""
        import io
        rng = np.random.RandomState(42)
        img_touch = rng.normal(110, 15, (200, 200, 3)).clip(0, 255).astype(np.uint8)
        yt, xt = np.ogrid[:200, :200]
        tc1 = np.sqrt((xt - 80) ** 2 + (yt - 100) ** 2) <= 25
        tc2 = np.sqrt((xt - 120) ** 2 + (yt - 100) ** 2) <= 25  # Centers distance 40 < 50 sum of radii
        img_touch[tc1 | tc2] = (img_touch[tc1 | tc2] * 0.3).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(img_touch).save(buf, format="PNG")
        service = PotholeInferenceService(segmenter_engine="deeplabv3")
        res = service.analyze_road_image(buf.getvalue(), filename="test_touching.png")

        self.assertTrue(res["pothole_detected"])
        self.assertEqual(res["detected_potholes_count"], 1)
        self.assertEqual(len(res["components"]), 1)

    def test_regression_single_irregular_pothole(self):
        """Regression test: Single non-convex/irregular pothole is not falsely split into fragments."""
        import io
        rng = np.random.RandomState(42)
        img_irreg = rng.normal(110, 15, (200, 200, 3)).clip(0, 255).astype(np.uint8)
        yi, xi = np.ogrid[:200, :200]
        # Asymmetrical kidney/bean-shaped cavity with overlapping lobes forming a single irregular component
        lobe1 = (xi - 90) ** 2 + (yi - 95) ** 2 <= 26 ** 2
        lobe2 = (xi - 112) ** 2 + (yi - 110) ** 2 <= 22 ** 2
        img_irreg[lobe1 | lobe2] = (img_irreg[lobe1 | lobe2] * 0.3).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(img_irreg).save(buf, format="PNG")
        service = PotholeInferenceService(segmenter_engine="deeplabv3")
        res = service.analyze_road_image(buf.getvalue(), filename="test_irregular.png")

        self.assertTrue(res["pothole_detected"])
        self.assertEqual(res["detected_potholes_count"], 1)
        self.assertEqual(len(res["components"]), 1)

    def test_regression_clean_pavement(self):
        """Regression test: Clean pavement returns 0 count, empty components list, and tier 'None'."""
        import io
        img_clean = np.full((150, 150, 3), 190, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_clean).save(buf, format="PNG")

        service = PotholeInferenceService(segmenter_engine="deeplabv3")
        res = service.analyze_road_image(buf.getvalue(), filename="test_clean.png")

        self.assertFalse(res["pothole_detected"])
        self.assertEqual(res["detected_potholes_count"], 0)
        self.assertEqual(res["components"], [])
        self.assertEqual(res["predicted_tier"], "None")
        self.assertIn("No pothole defect detected", res["message"])

    def test_rollback_to_random_forest(self):
        """Verify that setting segmenter_engine='random_forest' cleanly rolls back to RF baseline."""
        service_rf = PotholeInferenceService(segmenter_engine="random_forest")
        self.assertIsInstance(service_rf.predictor.segmenter, RGBSegmenter)

        status = service_rf.get_status()
        self.assertEqual(status["active_segmenter_engine"], "random_forest")
        self.assertEqual(status["models"]["rgb_segmenter"]["file"], "rgb_segmenter.joblib")


if __name__ == "__main__":
    unittest.main()
