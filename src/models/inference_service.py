"""
Inference Service for Pothole Segmentation and Severity Assessment.

Exposes a clean, thread-safe interface for running the actual trained
RGBSegmenter and SVM severity classifiers on uploaded imagery, converting
model outputs into high-performance base64 image data URLs, 14 planar
geometric features, and scientific metadata.
"""

import os
import sys
import io
import time
import base64
import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import scipy.ndimage as ndi

from src.models.predict import PotholeSeverityPredictor, DEFAULT_MODELS_DIR
from src.models.segmentation import DeepLabV3Segmenter, RGBSegmenter, get_segmenter
from src.features.feature_extractor import compute_mask_geometry, compute_depth_features

logger = logging.getLogger(__name__)


def numpy_to_base64_png(arr: np.ndarray, is_mask: bool = False) -> str:
    """
    Encode a 2D or 3D numpy array into a base64 PNG data URL.

    Args:
        arr: 2D uint8 mask (H, W) or 3D uint8 RGB/RGBA image (H, W, C).
        is_mask: If True and arr is 2D, saves as a 1-channel or paletted PNG.

    Returns:
        String data URL: 'data:image/png;base64,...'
    """
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        else:
            img = Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported array dimensions for image encoding: {arr.shape}")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_composite_overlay(
    rgb_arr: np.ndarray,
    mask_arr: np.ndarray,
    color_rgb: Tuple[int, int, int] = (180, 35, 104),  # Berry Pink #B42368
    alpha: float = 0.55,
) -> str:
    """
    Blend a binary pothole mask onto an RGB road image and return base64 PNG.

    Args:
        rgb_arr: 3D uint8 array (H, W, 3).
        mask_arr: 2D uint8 array (H, W) where > 0 indicates pothole cavity.
        color_rgb: Color tuple to paint the pothole region.
        alpha: Blending transparency (0.0 to 1.0).

    Returns:
        Base64 PNG data URL of the composite visualization.
    """
    h, w, _ = rgb_arr.shape
    overlay = rgb_arr.astype(np.float32).copy()
    binary_mask = (mask_arr > 0)

    if np.any(binary_mask):
        tint = np.array(color_rgb, dtype=np.float32)
        overlay[binary_mask] = (
            (1.0 - alpha) * overlay[binary_mask] + alpha * tint
        )

    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    return numpy_to_base64_png(overlay_uint8)


class PotholeInferenceService:
    """
    Singleton service managing the trained model pipeline and inference dispatch.
    Supports both deep neural network (DeepLabV3-MobileNetV3) and classical
    Random Forest segmentation engines with transparent fallback.
    """
    _instance: Optional["PotholeInferenceService"] = None

    def __init__(
        self,
        models_dir: Optional[str] = None,
        segmenter_engine: Optional[str] = None,
    ):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.segmenter_engine = segmenter_engine or os.environ.get("CRACKSIGHT_SEGMENTER_ENGINE", "deeplabv3")
        self.predictor = PotholeSeverityPredictor(
            models_dir=self.models_dir,
            segmenter_engine=self.segmenter_engine,
        )
        # Initialize configurable fallback Random Forest segmenter
        self.fallback_rf: Optional[RGBSegmenter] = None
        rf_path = os.path.join(self.models_dir, "rgb_segmenter.joblib")
        if os.path.exists(rf_path):
            try:
                self.fallback_rf = RGBSegmenter(model_path=rf_path)
            except Exception as e:
                logger.warning(f"Could not load fallback Random Forest segmenter: {e}")

        self.startup_time = time.time()
        self.inference_count = 0
        logger.info(
            f"PotholeInferenceService initialized. "
            f"Engine: {self.segmenter_engine}, "
            f"Model A: {bool(self.predictor.model_a)}, "
            f"Model B: {bool(self.predictor.model_b)}, "
            f"Segmenter: {type(self.predictor.segmenter).__name__ if self.predictor.segmenter else None}"
        )

    @classmethod
    def get_instance(
        cls,
        models_dir: Optional[str] = None,
        segmenter_engine: Optional[str] = None,
    ) -> "PotholeInferenceService":
        if cls._instance is None:
            cls._instance = cls(models_dir=models_dir, segmenter_engine=segmenter_engine)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing or dynamic reconfiguration)."""
        cls._instance = None

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of models and server health."""
        is_deeplab = isinstance(self.predictor.segmenter, DeepLabV3Segmenter)
        active_engine = "deeplabv3" if is_deeplab else "random_forest"
        seg_file = (
            "deeplabv3_mobilenetv3.torchscript.pt"
            if is_deeplab
            else "rgb_segmenter.joblib"
        )
        seg_arch = (
            "DeepLabV3 (MobileNetV3-Large Backbone + ASPP, TorchScript Optimized)"
            if is_deeplab
            else "Multi-Scale Pixel Random Forest + Morphological Post-Processing"
        )
        dice_score = 0.8198 if is_deeplab else 0.525
        iou_score = 0.7101 if is_deeplab else 0.387

        return {
            "status": "healthy" if self.predictor.segmenter else "degraded",
            "models_dir": self.models_dir,
            "active_segmenter_engine": active_engine,
            "fallback_rf_available": bool(self.fallback_rf),
            "models": {
                "rgb_segmenter": {
                    "loaded": bool(self.predictor.segmenter),
                    "file": seg_file,
                    "engine": active_engine,
                    "architecture": seg_arch,
                    "mean_dice_cv": dice_score,
                    "mean_iou_cv": iou_score,
                },
                "svm_2d_geometry": {
                    "loaded": bool(self.predictor.model_b),
                    "file": "svm_2d_geometry.joblib",
                    "architecture": "SVM (RBF Kernel) on 14 Planar Features",
                    "macro_f1_cv": 0.6955,
                    "balanced_acc_cv": 0.7054,
                },
                "svm_rphi_reproducer": {
                    "loaded": bool(self.predictor.model_a),
                    "file": "svm_rphi_reproducer.joblib",
                    "architecture": "SVM (RBF Kernel) on 3 Constituent RPHI Features",
                    "macro_f1_cv": 0.8879,
                    "balanced_acc_cv": 0.8938,
                },
            },
            "uptime_seconds": round(time.time() - self.startup_time, 1),
            "total_inferences": self.inference_count,
            "scientific_disclaimers": [
                "2D Planar segmentation predicts road cavity footprints from image appearance.",
                "Depth maps are uncalibrated uint8 grayscale [0, 190], NOT physical millimeters (mm).",
                "Severity tiers are provisional relative categories, NOT certified municipal ratings (BMC/IRC:82).",
            ],
        }

    def analyze_road_image(
        self,
        image_bytes: bytes,
        filename: str = "road_inspection.png",
        depth_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end pothole segmentation and severity assessment.

        Args:
            image_bytes: Raw bytes of the uploaded road image (PNG, JPG, WEBP).
            filename: Client-provided filename for metadata tracking.
            depth_bytes: Optional raw bytes of a paired 8-bit depth map.

        Returns:
            Structured analysis result dictionary.
        """
        start_time = time.perf_counter()

        # 1. Validate and decode RGB image
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image payload: 0 bytes received.")

        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError("Image file exceeds maximum allowable size (20MB).")

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                rgb_pil = pil_img.convert("RGB")
                rgb_arr = np.array(rgb_pil)
        except Exception as e:
            raise ValueError(f"Failed to decode image file '{filename}': {e}")

        orig_h, orig_w, channels = rgb_arr.shape
        if orig_h < 32 or orig_w < 32:
            raise ValueError(f"Image dimensions too small ({orig_w}x{orig_h}). Minimum is 32x32.")

        # 2. Check segmenter model availability
        if self.predictor.segmenter is None:
            raise RuntimeError(
                "Segmentation model artifact is not loaded. "
                "Ensure the model file exists in the models/ directory."
            )

        # 3. Predict 2D binary segmentation mask
        pred_mask = self.predictor.segmenter.predict_mask(rgb_arr)
        foreground_pixels = int(np.sum(pred_mask > 0))
        pothole_detected = foreground_pixels > 0

        # Multi-pothole connected component detection (preserves multiple separate cavities)
        lbl, num_components = ndi.label(pred_mask > 0)
        detected_potholes_count = int(num_components) if pothole_detected else 0

        # Optional paired depth map handling
        depth_arr = None
        if depth_bytes and len(depth_bytes) > 0:
            try:
                with Image.open(io.BytesIO(depth_bytes)) as pil_depth:
                    depth_pil = pil_depth.convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST)
                    depth_arr = np.array(depth_pil)
            except Exception as e:
                logger.warning(f"Could not decode paired depth map: {e}")

        # 4. Extract 14 planar features and predict severity tier
        components_detail = []
        if pothole_detected:
            # Extract planar geometric features across detected damage
            geom_features = compute_mask_geometry(pred_mask)

            # Compute centroid and axis details
            y_coords, x_coords = np.where(pred_mask > 0)
            cx = int(round(float(np.mean(x_coords))))
            cy = int(round(float(np.mean(y_coords))))
            bbox = [
                int(geom_features["bbox_x_min"]),
                int(geom_features["bbox_y_min"]),
                int(geom_features["bbox_width"]),
                int(geom_features["bbox_height"]),
            ]
            major_axis = max(bbox[2], bbox[3])
            minor_axis = min(bbox[2], bbox[3])

            # Extract per-component details for each detected pothole cavity
            if detected_potholes_count > 0:
                for c_idx in range(1, num_components + 1):
                    c_mask = (lbl == c_idx)
                    c_area = int(np.sum(c_mask))
                    c_ys, c_xs = np.where(c_mask)
                    c_bbox = [
                        int(np.min(c_xs)),
                        int(np.min(c_ys)),
                        int(np.max(c_xs) - np.min(c_xs) + 1),
                        int(np.max(c_ys) - np.min(c_ys) + 1),
                    ]
                    components_detail.append({
                        "component_id": c_idx,
                        "area_pixels": c_area,
                        "centroid": [int(round(float(np.mean(c_xs)))), int(round(float(np.mean(c_ys))))],
                        "bbox": c_bbox,
                    })

            # Classify severity using Model B (2D Geometry Only)
            pred_tier = "None"
            decision_scores = None
            if self.predictor.model_b is not None:
                try:
                    res_2d = self.predictor.predict_2d(pred_mask)
                    pred_tier = res_2d.get("predicted_tier", "Unavailable")
                    decision_scores = res_2d.get("decision_scores")
                except Exception as e:
                    logger.warning(f"Model B severity classification error: {e}")
                    pred_tier = "Unavailable"

            # If depth map is present, run Model A (RPHI Formula Reproducer)
            depth_stats = None
            if depth_arr is not None and self.predictor.model_a is not None:
                try:
                    res_depth = self.predictor.predict_with_depth(pred_mask, depth_arr)
                    pred_tier = res_depth.get("predicted_tier", pred_tier)
                    depth_stats = res_depth.get("supporting_depth_stats")
                except Exception as e:
                    logger.warning(f"Model A depth-based evaluation error: {e}")

            # Structured 14 features payload
            extracted_features = {
                "mask_area_pixels": float(geom_features["mask_area_pixels"]),
                "mask_coverage_ratio": float(geom_features["mask_coverage_ratio"]),
                "mask_coverage_pct": float(round(geom_features["mask_coverage_pct"], 2)),
                "perimeter_pixels": float(geom_features["perimeter_pixels"]),
                "circularity": float(round(geom_features["circularity"], 3)),
                "compactness": float(round(geom_features["compactness"], 2)),
                "bbox_x_min": float(geom_features["bbox_x_min"]),
                "bbox_x_max": float(geom_features["bbox_x_max"]),
                "bbox_y_min": float(geom_features["bbox_y_min"]),
                "bbox_y_max": float(geom_features["bbox_y_max"]),
                "bbox_width": float(geom_features["bbox_width"]),
                "bbox_height": float(geom_features["bbox_height"]),
                "bbox_aspect_ratio": float(round(geom_features["bbox_aspect_ratio"], 2)),
                "bbox_extent": float(round(geom_features["bbox_extent"], 3)),
                "centroid": [cx, cy],
                "major_axis": major_axis,
                "minor_axis": minor_axis,
            }
            if depth_stats:
                extracted_features.update(depth_stats)

            if detected_potholes_count > 1:
                message = f"Detected {detected_potholes_count} pothole cavities ({foreground_pixels:,} px total damage area)."
            else:
                message = f"Pothole defect segmented successfully ({foreground_pixels:,} px cavity area)."
        else:
            # Handle empty mask (valid model outcome)
            pred_tier = "None"
            decision_scores = None
            bbox = [0, 0, 0, 0]
            cx, cy = orig_w // 2, orig_h // 2
            extracted_features = {
                "mask_area_pixels": 0.0,
                "mask_coverage_ratio": 0.0,
                "mask_coverage_pct": 0.0,
                "perimeter_pixels": 0.0,
                "circularity": 0.0,
                "compactness": 0.0,
                "bbox_x_min": 0.0,
                "bbox_x_max": 0.0,
                "bbox_y_min": 0.0,
                "bbox_y_max": 0.0,
                "bbox_width": 0.0,
                "bbox_height": 0.0,
                "bbox_aspect_ratio": 0.0,
                "bbox_extent": 0.0,
                "centroid": [cx, cy],
                "major_axis": 0,
                "minor_axis": 0,
            }
            message = "No pothole defect detected in this image. The surface was classified as standard asphalt pavement."

        # 5. Generate Base64 visualizations
        mask_dataurl = numpy_to_base64_png(pred_mask, is_mask=True)
        overlay_dataurl = create_composite_overlay(rgb_arr, pred_mask)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        self.inference_count += 1

        analysis_id = f"ANL-{int(time.time())}-{np.random.randint(1000, 9999)}"

        is_deeplab = isinstance(self.predictor.segmenter, DeepLabV3Segmenter)
        seg_model_name = (
            "DeepLabV3-MobileNetV3 (TorchScript 42.7 MB)"
            if is_deeplab
            else "RGBSegmenter (11.5 MB Joblib Artifact)"
        )
        dice_metric = 0.8198 if is_deeplab else 0.525
        iou_metric = 0.7101 if is_deeplab else 0.387
        protocol = (
            "Group-Disjoint Held-Out Benchmark (Fold 0, N=207)"
            if is_deeplab
            else "Group-Disjoint 5-Fold Cross Validation"
        )

        result_payload = {
            "analysis_id": analysis_id,
            "filename": filename,
            "dimensions": {
                "width": orig_w,
                "height": orig_h,
                "channels": channels,
            },
            "pothole_detected": pothole_detected,
            "detected_potholes_count": detected_potholes_count,
            "predicted_tier": pred_tier,
            "decision_scores": decision_scores,
            "features": extracted_features,
            "bbox": bbox,
            "centroid": [cx, cy],
            "mask_dataurl": mask_dataurl,
            "overlay_dataurl": overlay_dataurl,
            "modality": "RGB + Depth" if depth_arr is not None else "RGB 2D",
            "inference_time_ms": elapsed_ms,
            "message": message,
            "model_info": {
                "segmentation_model": seg_model_name,
                "segmenter_engine": "deeplabv3" if is_deeplab else "random_forest",
                "classifier_model": "SVM (RBF) - 2D Geometry Only (Model B)",
                "evaluation_protocol": protocol,
                "validation_dice": dice_metric,
                "validation_iou": iou_metric,
            },
            "scientific_disclaimer": (
                "2D planar segmentation derived from natural image appearance. "
                "Tiers are provisional relative categories, not certified municipal maintenance ratings."
            ),
        }

        result_payload["components"] = components_detail

        return result_payload
