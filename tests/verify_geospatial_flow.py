"""
End-to-end verification of the geospatial location workflow and inference proxy.
"""
import requests
import json

def test_api_proxy():
    print("1. Testing /api/health via Vite dev server proxy...")
    r = requests.get("http://localhost:5174/api/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    health = r.json()
    assert health["status"] == "healthy"
    print("   [OK] Health check passed:", health["status"], "Models loaded:", list(health["models"].keys()))

    print("2. Testing /api/analyze with sample road image and tagged location...")
    with open("public/assets/sample_pothole_rgb.png", "rb") as f:
        files = {"file": ("sample_pothole_rgb.png", f, "image/png")}
        data = {
            "latitude": "18.9557",
            "longitude": "72.8127",
            "address": "Marine Drive, Mumbai"
        }
        res = requests.post("http://localhost:5174/api/analyze", files=files, data=data)

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    result = res.json()
    assert result["pothole_detected"] is True
    assert "location" in result
    assert abs(result["location"]["latitude"] - 18.9557) < 0.001
    assert abs(result["location"]["longitude"] - 72.8127) < 0.001
    assert result["location"]["address"] == "Marine Drive, Mumbai"
    print(f"   [OK] Inference result: ID={result['analysis_id']}, Tier={result['predicted_tier']}, Area={result['features']['mask_area_pixels']} px")
    print(f"   [OK] Location metadata preserved: {result['location']}")

    print("3. Testing /api/analyze without location (optional flow)...")
    with open("public/assets/sample_pothole_rgb.png", "rb") as f:
        files = {"file": ("sample_pothole_rgb.png", f, "image/png")}
        res_no_loc = requests.post("http://localhost:5174/api/analyze", files=files)
    assert res_no_loc.status_code == 200
    res_no_loc_json = res_no_loc.json()
    assert "location" not in res_no_loc_json
    print("   [OK] Optional flow verified: no location attached, inference identical.")

    print("4. Testing prediction invariance between geotagged and unlocated runs...")
    assert result["predicted_tier"] == res_no_loc_json["predicted_tier"]
    assert result["features"]["mask_area_pixels"] == res_no_loc_json["features"]["mask_area_pixels"]
    assert result["features"]["circularity"] == res_no_loc_json["features"]["circularity"]
    print("   [OK] Invariance confirmed: location metadata is strictly non-interfering.")

    print("\nAll end-to-end verification tests passed successfully!")

if __name__ == "__main__":
    test_api_proxy()
