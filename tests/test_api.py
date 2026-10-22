"""
Unit and Integration Tests for CrackSight Backend API and Inference Service.
"""

import os
import sys
import io
import unittest
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api import app
from src.models.inference_service import PotholeInferenceService


class TestCrackSightAPI(unittest.TestCase):
    """Test suite covering FastAPI endpoints and PotholeInferenceService."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.service = PotholeInferenceService.get_instance()

        # Create a realistic road image patch with dark central pothole cavity
        rng = np.random.RandomState(42)
        cls.test_rgb = rng.normal(110, 15, (200, 200, 3)).clip(0, 255).astype(np.uint8)
        y, x = np.ogrid[:200, :200]
        dist_from_center = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)
        mask = dist_from_center <= 40
        cls.test_rgb[mask] = (cls.test_rgb[mask] * 0.3).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(cls.test_rgb).save(buf, format="PNG")
        cls.test_png_bytes = buf.getvalue()

        # Blank uniform white image (tests empty mask handling)
        cls.blank_rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
        buf_blank = io.BytesIO()
        Image.fromarray(cls.blank_rgb).save(buf_blank, format="PNG")
        cls.blank_png_bytes = buf_blank.getvalue()

    def test_root_endpoint(self):
        """Test GET / returns online status and endpoint map."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_health_check_endpoint(self):
        """Test GET /api/health verifies model readiness."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["models"]["rgb_segmenter"]["loaded"])
        self.assertTrue(data["models"]["svm_2d_geometry"]["loaded"])
        self.assertTrue(data["models"]["svm_rphi_reproducer"]["loaded"])

    def test_models_metadata_endpoint(self):
        """Test GET /api/models returns architectures and validation scores."""
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertIn("pipeline", data)
        self.assertIn("scientific_integrity", data)

    def test_analyze_valid_rgb_image(self):
        """Test POST /api/analyze with valid synthetic road image."""
        files = {"file": ("test_road.png", self.test_png_bytes, "image/png")}
        res = self.client.post("/api/analyze", files=files)
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertIn("analysis_id", data)
        self.assertTrue(data["analysis_id"].startswith("ANL-"))
        self.assertEqual(data["filename"], "test_road.png")
        self.assertEqual(data["dimensions"]["width"], 200)
        self.assertEqual(data["dimensions"]["height"], 200)
        self.assertTrue(data["pothole_detected"])
        self.assertIn(data["predicted_tier"], ["High", "Medium", "Low"])

        # Check features
        feats = data["features"]
        self.assertGreater(feats["mask_area_pixels"], 0)
        self.assertIn("circularity", feats)
        self.assertIn("compactness", feats)
        self.assertIn("perimeter_pixels", feats)
        self.assertIn("bbox_width", feats)
        self.assertIn("bbox_height", feats)

        # Check image base64 data URLs
        self.assertTrue(data["mask_dataurl"].startswith("data:image/png;base64,"))
        self.assertTrue(data["overlay_dataurl"].startswith("data:image/png;base64,"))

    def test_analyze_real_pothole_asset(self):
        """Test POST /api/analyze with public/assets/sample_pothole_rgb.png."""
        asset_path = os.path.join(os.path.dirname(__file__), "..", "public", "assets", "sample_pothole_rgb.png")
        if not os.path.exists(asset_path):
            self.skipTest(f"Asset not found: {asset_path}")

        with open(asset_path, "rb") as f:
            real_bytes = f.read()

        files = {"file": ("sample_pothole_rgb.png", real_bytes, "image/png")}
        res = self.client.post("/api/analyze", files=files)
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertTrue(data["pothole_detected"])
        self.assertEqual(data["predicted_tier"], "High")
        self.assertGreater(data["features"]["mask_area_pixels"], 50000)

    def test_analyze_empty_mask_handling(self):
        """Test POST /api/analyze with blank image returns graceful pothole_detected=False."""
        files = {"file": ("blank_road.png", self.blank_png_bytes, "image/png")}
        res = self.client.post("/api/analyze", files=files)
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertFalse(data["pothole_detected"])
        self.assertEqual(data["predicted_tier"], "None")
        self.assertEqual(data["features"]["mask_area_pixels"], 0.0)
        self.assertIn("No pothole", data["message"])

    def test_analyze_invalid_file_type(self):
        """Test POST /api/analyze rejects text files."""
        files = {"file": ("test.txt", b"plain text is not an image", "text/plain")}
        res = self.client.post("/api/analyze", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file format", res.json()["detail"])

    def test_analyze_empty_file(self):
        """Test POST /api/analyze rejects 0-byte file."""
        files = {"file": ("empty.png", b"", "image/png")}
        res = self.client.post("/api/analyze", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"])

    def test_analyze_with_optional_location(self):
        """Test POST /api/analyze preserves valid location metadata when provided."""
        files = {"file": ("test_road.png", self.test_png_bytes, "image/png")}
        data = {
            "latitude": "18.9220",
            "longitude": "72.8258",
            "address": "Marine Drive, Mumbai"
        }
        res = self.client.post("/api/analyze", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        res_json = res.json()
        self.assertIn("location", res_json)
        self.assertAlmostEqual(res_json["location"]["latitude"], 18.9220, places=3)
        self.assertAlmostEqual(res_json["location"]["longitude"], 72.8258, places=3)
        self.assertEqual(res_json["location"]["address"], "Marine Drive, Mumbai")

    def test_analyze_with_invalid_out_of_bounds_coordinates(self):
        """Test POST /api/analyze ignores invalid out-of-bounds coordinates without failing inference."""
        files = {"file": ("test_road.png", self.test_png_bytes, "image/png")}
        # Latitude must be -90..90, longitude -180..180
        data = {
            "latitude": "125.0",
            "longitude": "240.0",
            "address": "Invalid Coordinates Place"
        }
        res = self.client.post("/api/analyze", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        res_json = res.json()
        self.assertNotIn("location", res_json)
        self.assertTrue(res_json["pothole_detected"])

    def test_analyze_without_location_metadata(self):
        """Test POST /api/analyze without location succeeds without location payload."""
        files = {"file": ("test_road.png", self.test_png_bytes, "image/png")}
        res = self.client.post("/api/analyze", files=files)
        self.assertEqual(res.status_code, 200)
        res_json = res.json()
        self.assertNotIn("location", res_json)
        self.assertTrue(res_json["pothole_detected"])

    def test_location_metadata_does_not_affect_model_prediction(self):
        """Verify that attached location metadata is strictly non-interfering metadata."""
        files_no_loc = {"file": ("test_road.png", self.test_png_bytes, "image/png")}
        res1 = self.client.post("/api/analyze", files=files_no_loc)
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()

        files_with_loc = {"file": ("test_road.png", self.test_png_bytes, "image/png")}
        loc_data = {"latitude": "19.0760", "longitude": "72.8777", "address": "Mumbai"}
        res2 = self.client.post("/api/analyze", files=files_with_loc, data=loc_data)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()

        # Model predictions and extracted features must be strictly identical
        self.assertEqual(data1["predicted_tier"], data2["predicted_tier"])
        self.assertEqual(data1["features"]["mask_area_pixels"], data2["features"]["mask_area_pixels"])
        self.assertEqual(data1["features"]["circularity"], data2["features"]["circularity"])
        self.assertEqual(data1["features"]["perimeter_pixels"], data2["features"]["perimeter_pixels"])
        self.assertEqual(data1["features"]["compactness"], data2["features"]["compactness"])
        self.assertNotIn("location", data1)
        self.assertIn("location", data2)


if __name__ == "__main__":
    unittest.main()
