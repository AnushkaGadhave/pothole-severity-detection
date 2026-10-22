"""
Inference module for Pothole Severity Detection.

Provides the PotholeSeverityPredictor interface to load trained model artifacts
and predict provisional Relative Pothole Hazard Index (RPHI) tiers from:
  1. 2D binary pothole segmentation masks (Model B: 2D Geometry Only)
  2. Paired 2D binary masks and 8-bit depth maps (Model A: RPHI Formula Reproducer)
  3. Raw RGB images via integrated RGBSegmenter pipeline + Model B

IMPORTANT SCIENTIFIC LIMITATIONS:
1. Target tiers ('Low', 'Medium', 'High') are provisional relative ranking categories
   within the Pothole440 dataset population. They DO NOT correspond to certified municipal
   maintenance urgency (e.g. BMC criteria) or civil engineering standards (IRC:82, ASTM D6433).
2. The Go!SCAN depth maps are uncalibrated 8-bit grayscale PNGs [0, 190]. They DO NOT
   provide calibrated physical depth in millimeters (mm) or physical volume in liters.
3. RGB segmentation uses a lightweight classical multi-scale pixel classifier (Mean Dice ~0.53,
   Mean IoU ~0.39 on group-disjoint validation). It provides a fast CPU baseline and is sensitive
   to shadows and dark road artifacts; it DOES NOT estimate 3D depth from RGB.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import joblib
import numpy as np
from PIL import Image

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.feature_extractor import (
    compute_mask_geometry,
    compute_depth_features,
)
from src.models.train_baselines import (
    CLASS_LABELS,
    EXP_A_FEATURES,
    EXP_B_FEATURES,
)
from src.models.segmentation import (
    RGBSegmenter,
    DeepLabV3Segmenter,
    get_segmenter,
    DEFAULT_SEGMENTER_PATH,
    DEFAULT_DEEPLAB_PATH,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
DEFAULT_MODEL_A_PATH = os.path.join(DEFAULT_MODELS_DIR, "svm_rphi_reproducer.joblib")
DEFAULT_MODEL_B_PATH = os.path.join(DEFAULT_MODELS_DIR, "svm_2d_geometry.joblib")


def _load_array_from_input(
    input_data: Union[np.ndarray, str, os.PathLike],
    expected_name: str = "Input",
) -> np.ndarray:
    """
    Load and validate a 2D numpy array from an in-memory array or image file path.

    Args:
        input_data: Numpy array or file path to an image.
        expected_name: Name of the input for descriptive error messages.

    Returns:
        2D numpy array.
    """
    if isinstance(input_data, (str, os.PathLike)):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"{expected_name} file not found: '{input_data}'")
        try:
            with Image.open(input_data) as img:
                arr = np.array(img)
        except Exception as e:
            raise ValueError(f"Failed to load {expected_name} image from '{input_data}': {e}")
    elif isinstance(input_data, np.ndarray):
        arr = input_data
    else:
        raise TypeError(
            f"{expected_name} must be a file path string or numpy array. Got: {type(input_data).__name__}"
        )

    # Handle 3-channel grayscale / RGB images by reducing to 2D
    if arr.ndim == 3:
        if arr.shape[2] in (3, 4):
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"{expected_name} must be 2-dimensional. Got shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"{expected_name} must be a 2D array. Got shape: {arr.shape}")

    return arr


class PotholeSeverityPredictor:
    """
    Production inference engine for Pothole Severity Detection.

    Supports:
      - 2D Geometry Only inference via Model B (requires binary mask).
      - RPHI Formula Reproduction inference via Model A (requires binary mask + depth map).
      - End-to-end RGB inference via RGBSegmenter + Model B.
    """

    def __init__(
        self,
        model_a_path: Optional[str] = None,
        model_b_path: Optional[str] = None,
        segmenter_path: Optional[str] = None,
        models_dir: Optional[str] = None,
        segmenter_engine: Optional[str] = None,
    ):
        """
        Initialize predictor by loading available model artifacts.

        Args:
            model_a_path: Path to serialized Model A (RPHI reproducer).
            model_b_path: Path to serialized Model B (2D geometry only).
            segmenter_path: Path to serialized segmenter model (joblib or torchscript).
            models_dir: Directory containing default model files.
            segmenter_engine: Desired segmenter engine ('deeplabv3' or 'random_forest').
        """
        base_dir = models_dir or DEFAULT_MODELS_DIR
        self.models_dir = base_dir
        self.model_a_path = model_a_path or os.path.join(base_dir, "svm_rphi_reproducer.joblib")
        self.model_b_path = model_b_path or os.path.join(base_dir, "svm_2d_geometry.joblib")
        self.segmenter_path = segmenter_path
        self.segmenter_engine = segmenter_engine or os.environ.get("CRACKSIGHT_SEGMENTER_ENGINE", "deeplabv3")

        self.model_a: Optional[Dict[str, Any]] = None
        self.model_b: Optional[Dict[str, Any]] = None
        self.segmenter: Optional[Union[RGBSegmenter, DeepLabV3Segmenter]] = None

        self._load_available_models()

    def _load_available_models(self) -> None:
        """Load available model files, recording availability."""
        if os.path.exists(self.model_a_path):
            try:
                self.model_a = joblib.load(self.model_a_path)
                self._validate_model_package(self.model_a, "Model A", EXP_A_FEATURES)
                logger.info(f"Loaded Model A from: {self.model_a_path}")
            except Exception as e:
                logger.warning(f"Could not load Model A from '{self.model_a_path}': {e}")
                self.model_a = None

        if os.path.exists(self.model_b_path):
            try:
                self.model_b = joblib.load(self.model_b_path)
                self._validate_model_package(self.model_b, "Model B", EXP_B_FEATURES)
                logger.info(f"Loaded Model B from: {self.model_b_path}")
            except Exception as e:
                logger.warning(f"Could not load Model B from '{self.model_b_path}': {e}")
                self.model_b = None

        if self.segmenter_path is not None:
            if os.path.exists(self.segmenter_path):
                try:
                    if self.segmenter_path.endswith(".pt"):
                        self.segmenter = DeepLabV3Segmenter(model_path=self.segmenter_path)
                    else:
                        self.segmenter = RGBSegmenter(model_path=self.segmenter_path)
                    logger.info(f"Loaded segmenter from: {self.segmenter_path}")
                except Exception as e:
                    logger.warning(f"Could not load segmenter from '{self.segmenter_path}': {e}")
                    self.segmenter = None
            else:
                logger.warning(f"Specified segmenter file does not exist: '{self.segmenter_path}'")
                self.segmenter = None
        else:
            try:
                self.segmenter = get_segmenter(engine=self.segmenter_engine, models_dir=self.models_dir)
                logger.info(f"Loaded default segmenter: {type(self.segmenter).__name__}")
            except Exception as e:
                logger.warning(f"Could not load default segmenter: {e}")
                self.segmenter = None

        if self.model_a is None and self.model_b is None and self.segmenter is None:
            logger.warning(
                "No trained model artifacts loaded. "
                "Ensure models are trained via 'python src/models/train_final_model.py'."
            )

    @staticmethod
    def _validate_model_package(
        pkg: Any, name: str, expected_features: List[str]
    ) -> None:
        """Validate structure and feature schema of a loaded model package."""
        if not isinstance(pkg, dict):
            raise TypeError(f"{name} artifact must be a dictionary. Got: {type(pkg).__name__}")

        if "pipeline" not in pkg:
            raise KeyError(f"{name} artifact is missing the 'pipeline' key.")

        pipeline = pkg["pipeline"]
        if not hasattr(pipeline, "predict"):
            raise TypeError(f"{name} pipeline does not have a predict() method.")

        if "features" not in pkg:
            raise KeyError(f"{name} artifact is missing 'features' schema.")

        pkg_features = list(pkg["features"])
        if pkg_features != list(expected_features):
            raise ValueError(
                f"{name} feature mismatch. Expected: {expected_features}, Got: {pkg_features}"
            )

    def predict_2d(self, mask: Union[np.ndarray, str, os.PathLike]) -> Dict[str, Any]:
        """
        Predict provisional severity tier from a 2D binary pothole segmentation mask.

        Uses Model B (14 geometric features, 100% depth-free).

        Args:
            mask: 2D binary numpy array or image file path.

        Returns:
            Dictionary containing predicted tier, uncalibrated decision scores,
            extracted geometric features, and scientific disclaimers.
        """
        if self.model_b is None:
            raise RuntimeError(
                f"Model B (2D Geometry) is not loaded. Checked path: '{self.model_b_path}'. "
                "Please run 'python src/models/train_final_model.py' to train and save models."
            )

        mask_arr = _load_array_from_input(mask, expected_name="Mask")

        # Validate non-empty mask
        binary_mask = mask_arr > 0
        if not np.any(binary_mask):
            raise ValueError(
                "Empty mask error: Supplied pothole mask contains 0 foreground pixels. "
                "Cannot extract geometric features or predict severity."
            )

        # Extract 2D geometric features
        geom_features = compute_mask_geometry(mask_arr)

        # Validate finite values
        for k, v in geom_features.items():
            if not np.isfinite(v):
                raise ValueError(f"Extracted feature '{k}' is non-finite ({v}).")

        # Canonical resolution projection (Model B training reference: 800x1200)
        # Prevents resolution-dependent domain shift to Low on non-1200x800 imagery
        target_h, target_w = 800, 1200
        if mask_arr.shape == (target_h, target_w):
            classifier_geom = geom_features
        else:
            pil_canonical = Image.fromarray((binary_mask.astype(np.uint8) * 255)).resize(
                (target_w, target_h), Image.Resampling.NEAREST
            )
            classifier_geom = compute_mask_geometry(np.array(pil_canonical))

        # Construct feature vector in exact schema order using canonical geometry
        feature_order = self.model_b["features"]
        X = np.array([[classifier_geom[f] for f in feature_order]], dtype=np.float64)

        pipeline = self.model_b["pipeline"]
        pred_tier = str(pipeline.predict(X)[0])

        # Extract uncalibrated SVM decision scores if available
        decision_scores = None
        if hasattr(pipeline, "decision_function"):
            raw_scores = pipeline.decision_function(X)[0]
            classes = list(getattr(pipeline, "classes_", getattr(getattr(pipeline, "named_steps", {}).get("svc", None), "classes_", CLASS_LABELS)))
            decision_scores = {
                classes[i]: float(raw_scores[i]) for i in range(len(classes))
            }

        return {
            "model_id": self.model_b.get("model_id", "svm_2d_geometry"),
            "model_name": self.model_b.get("model_name", "SVM (RBF) - 2D Geometry Only"),
            "experiment": "B (2D Geometry Only)",
            "predicted_tier": pred_tier,
            "decision_scores": decision_scores,
            "extracted_features": geom_features,
            "constituent_inputs_used": feature_order,
            "input_shape": mask_arr.shape,
            "scientific_disclaimer": (
                "Provisional tier from 2D footprint geometry. Does not measure physical "
                "cavity depth or certified municipal maintenance urgency (BMC / IRC:82)."
            ),
        }

    def predict_with_depth(
        self,
        mask: Union[np.ndarray, str, os.PathLike],
        depth: Union[np.ndarray, str, os.PathLike],
    ) -> Dict[str, Any]:
        """
        Predict provisional severity tier from a binary mask and paired depth map.

        Uses Model A (3-feature RPHI Formula Reproducer).

        Args:
            mask: 2D binary numpy array or image file path.
            depth: 2D uint8 depth map numpy array or image file path.

        Returns:
            Dictionary containing predicted tier, uncalibrated decision scores,
            extracted constituent features, and scientific disclaimers.
        """
        if self.model_a is None:
            raise RuntimeError(
                f"Model A (RPHI Reproducer) is not loaded. Checked path: '{self.model_a_path}'. "
                "Please run 'python src/models/train_final_model.py' to train and save models."
            )

        mask_arr = _load_array_from_input(mask, expected_name="Mask")
        depth_arr = _load_array_from_input(depth, expected_name="Depth Map")

        # Validate dimension alignment
        if mask_arr.shape != depth_arr.shape:
            raise ValueError(
                f"Dimension mismatch: Mask shape {mask_arr.shape} does not match "
                f"Depth map shape {depth_arr.shape}."
            )

        # Validate non-empty mask
        binary_mask = mask_arr > 0
        if not np.any(binary_mask):
            raise ValueError(
                "Empty mask error: Supplied pothole mask contains 0 foreground pixels. "
                "Cannot extract cavity depth features or predict severity."
            )

        # Check finite depth values
        if not np.isfinite(depth_arr).all():
            raise ValueError("Depth map contains non-finite values (NaN or Inf).")

        # Extract features
        geom_features = compute_mask_geometry(mask_arr)
        depth_features = compute_depth_features(depth_arr, mask_arr)

        combined_features = {}
        combined_features.update(geom_features)
        combined_features.update(depth_features)

        # Validate finite values across required features
        feature_order = self.model_a["features"]
        for f in feature_order:
            val = combined_features[f]
            if not np.isfinite(val):
                raise ValueError(f"Extracted feature '{f}' is non-finite ({val}).")

        # Canonical resolution projection for Model A (training reference: 800x1200)
        target_h, target_w = 800, 1200
        if mask_arr.shape == (target_h, target_w):
            classifier_geom = geom_features
            classifier_depth = depth_features
        else:
            pil_canonical_mask = Image.fromarray((binary_mask.astype(np.uint8) * 255)).resize(
                (target_w, target_h), Image.Resampling.NEAREST
            )
            pil_canonical_depth = Image.fromarray(depth_arr).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
            classifier_geom = compute_mask_geometry(np.array(pil_canonical_mask))
            classifier_depth = compute_depth_features(np.array(pil_canonical_depth), np.array(pil_canonical_mask))

        classifier_combined = {}
        classifier_combined.update(classifier_geom)
        classifier_combined.update(classifier_depth)

        # Construct feature vector
        X = np.array([[classifier_combined[f] for f in feature_order]], dtype=np.float64)

        pipeline = self.model_a["pipeline"]
        pred_tier = str(pipeline.predict(X)[0])

        # Extract uncalibrated SVM decision scores if available
        decision_scores = None
        if hasattr(pipeline, "decision_function"):
            raw_scores = pipeline.decision_function(X)[0]
            classes = list(getattr(pipeline, "classes_", getattr(getattr(pipeline, "named_steps", {}).get("svc", None), "classes_", CLASS_LABELS)))
            decision_scores = {
                classes[i]: float(raw_scores[i]) for i in range(len(classes))
            }

        return {
            "model_id": self.model_a.get("model_id", "svm_rphi_reproducer"),
            "model_name": self.model_a.get("model_name", "SVM (RBF) - RPHI Formula Reproducer"),
            "experiment": "A (RPHI Formula Reproduction)",
            "predicted_tier": pred_tier,
            "decision_scores": decision_scores,
            "extracted_features": {
                f: combined_features[f] for f in feature_order
            },
            "supporting_depth_stats": {
                "road_median_intensity": combined_features["road_median_intensity"],
                "depth_max_intensity": combined_features["depth_max_intensity"],
                "depth_mean_intensity": combined_features["depth_mean_intensity"],
            },
            "constituent_inputs_used": feature_order,
            "input_shape": mask_arr.shape,
            "scientific_disclaimer": (
                "Provisional RPHI tier from constituent formula inputs. Depth is uncalibrated "
                "grayscale [0, 190], NOT physical depth in millimeters or certified maintenance priority."
            ),
        }

    def predict_rgb(
        self,
        rgb: Union[np.ndarray, str, os.PathLike],
        return_mask: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict provisional severity tier directly from an unsegmented RGB image.

        1. Segments a 2D binary pothole mask using the integrated RGBSegmenter pipeline.
        2. Evaluates 2D mask footprint geometry through Model B (2D Geometry Only).

        Args:
            rgb: 3D uint8 RGB numpy array (H, W, 3) or image file path.
            return_mask: If True, includes the segmented 2D mask in the result dictionary.

        Returns:
            Dictionary containing predicted tier, uncalibrated decision scores,
            extracted 2D geometric features, and segmentation metadata.
        """
        if self.segmenter is None:
            raise RuntimeError(
                f"RGB Segmenter is not loaded. Checked path: '{self.segmenter_path}'. "
                "Please run 'python src/models/train_segmenter.py' to train and save the segmenter."
            )

        # 1. Segment RGB image into binary mask
        segmented_mask = self.segmenter.predict_mask(rgb)

        import scipy.ndimage as ndi
        lbl, num_components = ndi.label(segmented_mask > 0)
        pothole_detected = int(np.sum(segmented_mask > 0)) > 0
        detected_count = int(num_components) if pothole_detected else 0
        is_deeplab = isinstance(self.segmenter, DeepLabV3Segmenter)
        model_name = (
            "DeepLabV3-MobileNetV3 (TorchScript)"
            if is_deeplab
            else "RGBSegmenter (Multi-Scale Pixel RF)"
        )

        # 2. Check if a pothole was detected
        if not pothole_detected:
            return {
                "input_modality": "RGB",
                "segmentation_model": model_name,
                "predicted_tier": "None",
                "pothole_detected": False,
                "detected_potholes_count": 0,
                "message": "No pothole cavity detected in the provided RGB image.",
                "extracted_features": {},
                "decision_scores": None,
                "segmented_mask": segmented_mask if return_mask else None,
                "scientific_disclaimer": (
                    "Automated 2D segmentation from RGB. No pothole was detected by the segmenter. "
                    "Does NOT estimate 3D depth from RGB."
                ),
            }

        # 3. Predict severity tier using Model B (2D Geometry Only)
        pred_res = self.predict_2d(segmented_mask)

        pred_res["input_modality"] = "RGB"
        pred_res["segmentation_model"] = model_name
        pred_res["pothole_detected"] = True
        pred_res["detected_potholes_count"] = detected_count
        pred_res["segmentation_validation_note"] = (
            "Segmenter evaluated on group-disjoint validation: Mean Dice ~0.82, Mean IoU ~0.71."
            if is_deeplab
            else "Segmenter evaluated on group-disjoint validation: Mean Dice ~0.53, Mean IoU ~0.39."
        )
        if return_mask:
            pred_res["segmented_mask"] = segmented_mask

        return pred_res

    def predict(
        self,
        mask: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        depth: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        rgb: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Dict[str, Any]:
        """
        Unified prediction entry point.

        - If 'rgb' is provided: executes end-to-end RGB segmentation + Model B.
        - If 'mask' and 'depth' are provided: executes Model A (RPHI Formula Reproducer).
        - If only 'mask' is provided: executes Model B (2D Geometry Only).

        Args:
            mask: Optional 2D binary mask (numpy array or file path).
            depth: Optional 2D depth map (numpy array or file path).
            rgb: Optional 3D RGB image (numpy array or file path).

        Returns:
            Structured prediction dictionary.
        """
        if rgb is not None:
            return self.predict_rgb(rgb)
        elif mask is not None:
            if depth is not None:
                if self.model_a is None:
                    raise RuntimeError(
                        "Depth map was provided, but Model A (depth-based RPHI reproducer) "
                        f"is not loaded from '{self.model_a_path}'."
                    )
                return self.predict_with_depth(mask, depth)
            else:
                if self.model_b is None:
                    raise RuntimeError(
                        "Only mask was provided, but Model B (2D Geometry Only) "
                        f"is not loaded from '{self.model_b_path}'."
                    )
                return self.predict_2d(mask)
        else:
            raise ValueError(
                "Unified predict() requires either 'rgb' or 'mask' input. Neither was provided."
            )


if __name__ == "__main__":
    predictor = PotholeSeverityPredictor()
    print("PotholeSeverityPredictor initialized.")
    print(f"  Model A (RPHI Reproducer): {'LOADED' if predictor.model_a else 'NOT LOADED'}")
    print(f"  Model B (2D Geometry):    {'LOADED' if predictor.model_b else 'NOT LOADED'}")
    print(f"  RGB Segmenter:            {'LOADED' if predictor.segmenter else 'NOT LOADED'}")
