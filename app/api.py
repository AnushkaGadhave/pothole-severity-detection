"""
CrackSight REST API Server.

High-performance ASGI application serving real-time computer-vision pothole
segmentation, planar geometry profiling, and provisional severity classification.
Built on Starlette + Uvicorn for lightweight, robust local inference.
"""

import os
import sys
import logging
from typing import Optional
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.datastructures import UploadFile

# Ensure project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.inference_service import PotholeInferenceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cracksight.api")

# Initialize inference service singleton on startup
service = PotholeInferenceService.get_instance()


async def root(request: Request) -> JSONResponse:
    """Root metadata endpoint."""
    return JSONResponse({
        "project": "CrackSight AI-Powered Road Inspection",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "analyze": "/api/analyze",
        },
    })


async def health_check(request: Request) -> JSONResponse:
    """
    Health check and model readiness diagnostics.
    Returns real-time status of loaded models, uptime, and validation benchmarks.
    """
    try:
        diag = service.get_status()
        return JSONResponse(diag, status_code=200)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)


async def get_models_metadata(request: Request) -> JSONResponse:
    """
    Return comprehensive architectural and cross-validation metadata
    for the trained models in the repository.
    """
    status_data = service.get_status()
    return JSONResponse({
        "models": status_data["models"],
        "pipeline": {
            "name": "CrackSight CV-088",
            "segmenter": "Lightweight Classical Multi-Scale Pixel Classifier (11.5 MB)",
            "severity_engine": "Support Vector Machine (RBF Kernel) on 14 Planar Features",
            "evaluation": "Group-Disjoint 5-Fold Cross-Validation on 98 Physical Locations",
        },
        "scientific_integrity": status_data["scientific_disclaimers"],
    }, status_code=200)


async def analyze_pothole_image(request: Request) -> JSONResponse:
    """
    Analyze an uploaded road image using the actual trained model pipeline:
    1. Segments 2D pothole region via trained RGBSegmenter (Random Forest multi-scale pixel classifier).
    2. Postprocesses mask with morphological closing and hole-filling.
    3. Computes 14 image-derived geometric planar features.
    4. Evaluates provisional severity tier via trained Model B (SVM 2D Geometry).
    5. Returns base64 PNG binary mask, composite overlay, features, and metadata.
    """
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse({"detail": f"Failed to parse multipart form data: {str(e)}"}, status_code=400)

    file = form.get("file") or form.get("image")
    if file is None or not isinstance(file, UploadFile):
        return JSONResponse({"detail": "Missing required 'file' or 'image' parameter in multipart form upload."}, status_code=400)

    # 1. Validate content type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
    if file.content_type and file.content_type.lower() not in allowed_types:
        return JSONResponse({
            "detail": f"Unsupported file format '{file.content_type}'. Supported formats: PNG, JPG, JPEG, WEBP."
        }, status_code=400)

    # 2. Read image bytes
    try:
        contents = await file.read()
    except Exception as e:
        return JSONResponse({"detail": f"Failed to read uploaded image bytes: {str(e)}"}, status_code=400)

    if not contents or len(contents) == 0:
        return JSONResponse({"detail": "Uploaded file is empty (0 bytes)."}, status_code=400)

    if len(contents) > 20 * 1024 * 1024:
        return JSONResponse({"detail": "File size exceeds maximum allowable limit of 20MB."}, status_code=413)

    # 3. Read optional depth file
    depth_file = form.get("depth_file")
    depth_bytes = None
    if depth_file is not None and isinstance(depth_file, UploadFile):
        try:
            depth_bytes = await depth_file.read()
        except Exception as e:
            logger.warning(f"Could not read depth file: {e}")

    # 4. Optional location metadata parameters
    lat_val = form.get("latitude")
    lng_val = form.get("longitude")
    addr_val = form.get("address")
    location_payload = None
    if lat_val is not None and lng_val is not None:
        try:
            lat_f = float(lat_val)
            lng_f = float(lng_val)
            if -90.0 <= lat_f <= 90.0 and -180.0 <= lng_f <= 180.0:
                location_payload = {
                    "latitude": lat_f,
                    "longitude": lng_f,
                    "address": str(addr_val) if addr_val else None,
                }
        except (ValueError, TypeError):
            pass

    # 5. Execute model inference using singleton service
    try:
        result = service.analyze_road_image(
            image_bytes=contents,
            filename=file.filename or "uploaded_road_image.png",
            depth_bytes=depth_bytes,
        )
        if location_payload:
            result["location"] = location_payload
        return JSONResponse(result, status_code=200)
    except ValueError as ve:
        logger.warning(f"Validation error during inference: {ve}")
        return JSONResponse({"detail": str(ve)}, status_code=400)
    except RuntimeError as re:
        logger.error(f"Model runtime error: {re}")
        return JSONResponse({"detail": str(re)}, status_code=503)
    except Exception as e:
        logger.exception(f"Unexpected inference exception: {e}")
        return JSONResponse({
            "detail": "An error occurred while executing model inference on the provided image."
        }, status_code=500)


# Routes
routes = [
    Route("/", root, methods=["GET"]),
    Route("/api/health", health_check, methods=["GET"]),
    Route("/api/models", get_models_metadata, methods=["GET"]),
    Route("/api/analyze", analyze_pothole_image, methods=["POST"]),
]

# Middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

# ASGI Application
app = Starlette(debug=False, routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
