"""
RGB Pothole Segmentation Module.

Implements an efficient classical machine-learning pixel classifier (Random Forest)
combined with multi-scale color, contrast, texture gradients, spatial priors,
and morphological postprocessing to segment potholes directly from RGB imagery.

IMPORTANT SCIENTIFIC LIMITATIONS:
1. This module provides a fast, CPU-efficient 2D localization baseline (Mean Dice ~0.53,
   Mean IoU ~0.39 on group-disjoint validation). It is NOT a deep neural network (e.g., U-Net
   or Mask R-CNN) and may produce false positives on dark road patches, shadows, or oil stains.
2. It produces a 2D binary segmentation mask. It DOES NOT estimate 3D depth from RGB.
3. Target severity tiers evaluated from the resulting mask remain provisional Relative
   Pothole Hazard Index (RPHI) categories, NOT certified municipal maintenance priorities.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Union, Tuple
import joblib
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
DEFAULT_SEGMENTER_PATH = os.path.join(DEFAULT_MODELS_DIR, "rgb_segmenter.joblib")


def extract_pixel_features(
    rgb_img: np.ndarray,
    scale: float = 0.25,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Extract multi-scale color, contrast, gradient, and spatial features for each pixel.

    Downsamples the input image by `scale` to ensure fast inference and large receptive fields,
    extracting 15 feature channels per pixel:
      - Normalized RGB channels (3)
      - Opponent color differences: R - G, R - B (2)
      - Grayscale luminance (1)
      - Gaussian blurs at small (1.5), medium (4.0), and large (8.0) scales (3)
      - Local relative contrast: Luminance - Gaussian blur (2)
      - Sobel gradient magnitude (1)
      - Normalized centered spatial coordinates (x, y) and radial distance (3)

    Args:
        rgb_img: 3D uint8 numpy array of shape (H, W, 3).
        scale: Resolution scaling factor for feature extraction (default: 0.25).

    Returns:
        (feature_matrix, (scaled_h, scaled_w), (orig_h, orig_w))
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError(f"Expected 3D RGB image with 3 channels. Got shape: {rgb_img.shape}")

    orig_h, orig_w, _ = rgb_img.shape
    tw = max(int(orig_w * scale), 16)
    th = max(int(orig_h * scale), 16)

    pil_img = Image.fromarray(rgb_img).resize((tw, th), Image.Resampling.BILINEAR)
    small = np.array(pil_img, dtype=np.float32) / 255.0

    r = small[:, :, 0]
    g = small[:, :, 1]
    b = small[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Opponent color differences
    rg_diff = r - g
    rb_diff = r - b

    # Multi-scale Gaussian smoothing
    blur_s = ndi.gaussian_filter(gray, sigma=1.5)
    blur_m = ndi.gaussian_filter(gray, sigma=4.0)
    blur_l = ndi.gaussian_filter(gray, sigma=8.0)

    # Local relative contrast (cavities are darker than surrounding pavement)
    contrast_m = gray - blur_m
    contrast_l = gray - blur_l

    # Edge texture / gradient magnitude
    gy = ndi.sobel(gray, axis=0)
    gx = ndi.sobel(gray, axis=1)
    grad_mag = np.hypot(gx, gy)

    # Normalized spatial coordinates relative to image center
    y_coords, x_coords = np.mgrid[0:th, 0:tw]
    norm_x = (x_coords / tw) - 0.5
    norm_y = (y_coords / th) - 0.5
    dist_center = np.hypot(norm_x, norm_y)

    feature_matrix = np.stack([
        r.ravel(),
        g.ravel(),
        b.ravel(),
        gray.ravel(),
        rg_diff.ravel(),
        rb_diff.ravel(),
        blur_s.ravel(),
        blur_m.ravel(),
        blur_l.ravel(),
        contrast_m.ravel(),
        contrast_l.ravel(),
        grad_mag.ravel(),
        norm_x.ravel(),
        norm_y.ravel(),
        dist_center.ravel(),
    ], axis=-1)

    return feature_matrix, (th, tw), (orig_h, orig_w)


class RGBSegmenter:
    """
    Lightweight, CPU-compatible RGB Pothole Segmentation pipeline.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scale: float = 0.25,
        threshold: float = 0.40,
    ):
        """
        Initialize the segmenter.

        Args:
            model_path: Path to serialized segmenter artifact (optional).
            scale: Downscaling factor for feature extraction (default: 0.25).
            threshold: Probability cutoff for pothole classification (default: 0.40).
        """
        self.model_path = model_path or DEFAULT_SEGMENTER_PATH
        self.scale = scale
        self.threshold = threshold
        self.classifier: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

        if os.path.exists(self.model_path):
            self.load(self.model_path)

    def load(self, model_path: str) -> "RGBSegmenter":
        """Load trained classifier from a joblib file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Segmentation model artifact not found: '{model_path}'")

        data = joblib.load(model_path)
        if isinstance(data, dict):
            self.classifier = data.get("classifier")
            self.metadata = data.get("metadata", {})
            self.scale = data.get("scale", self.scale)
            self.threshold = data.get("threshold", self.threshold)
        else:
            self.classifier = data

        self.model_path = model_path
        logger.info(f"Loaded RGBSegmenter from: {model_path}")
        return self

    def save(self, output_path: str) -> None:
        """Serialize segmenter and metadata to disk."""
        if self.classifier is None:
            raise RuntimeError("Cannot save an unfitted segmenter.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pkg = {
            "classifier": self.classifier,
            "scale": self.scale,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }
        joblib.dump(pkg, output_path)
        logger.info(f"RGBSegmenter saved to: {output_path}")

    def predict_mask(
        self,
        rgb_input: Union[np.ndarray, str, os.PathLike],
        threshold: Optional[float] = None,
        min_component_size: Optional[int] = None,
        min_component_ratio: float = 0.001,
        keep_all_components: bool = True,
        single_dominant: bool = False,
    ) -> np.ndarray:
        """
        Segment a pothole binary mask from an RGB image.

        Supports multi-pothole detection by retaining all connected components
        that meet or exceed a configurable, resolution-aware minimum area threshold.

        Args:
            rgb_input: 3D uint8 numpy array (H, W, 3) or file path to RGB image.
            threshold: Probability threshold cutoff (overrides default if provided).
            min_component_size: Minimum pixel count in downsampled space to retain.
                                If None, computed as max(10, th * tw * min_component_ratio).
            min_component_ratio: Minimum area ratio of downsampled image for resolution-aware sizing.
            keep_all_components: If True (default), retains all valid potholes in the image.
            single_dominant: If True, legacy mode retaining only the single largest component.

        Returns:
            2D uint8 binary mask of shape (H, W) where 255 indicates pothole, 0 background.
        """
        if self.classifier is None:
            raise RuntimeError(
                f"RGBSegmenter is not loaded or fitted. Checked path: '{self.model_path}'. "
                "Train the segmenter via 'python src/models/train_segmenter.py'."
            )

        # 1. Load and validate RGB image
        if isinstance(rgb_input, (str, os.PathLike)):
            if not os.path.exists(rgb_input):
                raise FileNotFoundError(f"RGB image file not found: '{rgb_input}'")
            with Image.open(rgb_input) as img:
                rgb_arr = np.array(img.convert("RGB"))
        elif isinstance(rgb_input, np.ndarray):
            rgb_arr = rgb_input
        else:
            raise TypeError(f"Expected file path or numpy array. Got: {type(rgb_input).__name__}")

        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise ValueError(f"RGB image must have shape (H, W, 3). Got: {rgb_arr.shape}")

        if not np.isfinite(rgb_arr).all():
            raise ValueError("RGB image contains non-finite values (NaN / Inf).")

        # 2. Extract multi-scale pixel features
        cutoff = threshold if threshold is not None else self.threshold
        feats, (th, tw), (orig_h, orig_w) = extract_pixel_features(rgb_arr, scale=self.scale)

        # 3. Predict pothole probability
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(feats)[:, 1].reshape((th, tw))
            pred_bin = probs > cutoff
        else:
            pred_bin = self.classifier.predict(feats).reshape((th, tw)).astype(bool)

        # 4. Morphological refinement: closing and hole-filling
        struct_3x3 = np.ones((3, 3), dtype=bool)
        pred_closed = ndi.binary_closing(pred_bin, structure=struct_3x3)
        pred_filled = ndi.binary_fill_holes(pred_closed)

        # 5. Connected component filtering: retain all valid pothole components
        eff_min_size = (
            min_component_size
            if min_component_size is not None
            else max(10, min(50, int(th * tw * min_component_ratio)))
        )

        lbl, num = ndi.label(pred_filled)
        if num > 0:
            sizes = ndi.sum(pred_filled, lbl, range(1, num + 1))
            if keep_all_components and not single_dominant:
                valid_indices = [
                    idx + 1 for idx, size in enumerate(sizes) if size >= eff_min_size
                ]
                if valid_indices:
                    pred_final_small = np.isin(lbl, valid_indices)
                else:
                    pred_final_small = np.zeros((th, tw), dtype=bool)
            else:
                max_idx = int(np.argmax(sizes)) + 1
                if sizes[max_idx - 1] >= eff_min_size:
                    pred_final_small = (lbl == max_idx)
                else:
                    pred_final_small = np.zeros((th, tw), dtype=bool)
        else:
            pred_final_small = np.zeros((th, tw), dtype=bool)

        # 6. Upsample binary mask to original image dimensions
        pil_mask_small = Image.fromarray((pred_final_small.astype(np.uint8) * 255))
        pil_mask_full = pil_mask_small.resize((orig_w, orig_h), Image.Resampling.NEAREST)
        final_mask = np.array(pil_mask_full, dtype=np.uint8)

        return final_mask


DEFAULT_DEEPLAB_PATH = os.path.join(
    DEFAULT_MODELS_DIR, "deeplabv3_mobilenetv3.torchscript.pt"
)


class DeepLabV3Segmenter:
    """
    High-performance DeepLabV3-MobileNetV3-Large segmentation engine.

    Compatible drop-in replacement for RGBSegmenter, providing predict_mask()
    with multi-component retention and fast CPU inference via TorchScript.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        target_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.50,
    ):
        """
        Initialize the DeepLabV3 segmenter.

        Args:
            model_path: Path to serialized TorchScript artifact.
            target_size: Processing resolution (W, H) for deep model forward pass.
            threshold: Sigmoid probability cutoff for pothole classification.
        """
        self.model_path = model_path or DEFAULT_DEEPLAB_PATH
        self.target_size = target_size
        self.threshold = threshold
        self.model: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if os.path.exists(self.model_path):
            self.load(self.model_path)

    def load(self, model_path: str) -> "DeepLabV3Segmenter":
        """Load traced TorchScript model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DeepLabV3 artifact not found: '{model_path}'")
        try:
            import torch
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            self.model_path = model_path
            logger.info(f"Loaded DeepLabV3Segmenter (TorchScript) from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load TorchScript DeepLabV3 model from '{model_path}': {e}")
            raise
        return self

    def predict_mask(
        self,
        rgb_input: Union[np.ndarray, str, os.PathLike],
        threshold: Optional[float] = None,
        min_component_size: Optional[int] = None,
        min_component_ratio: float = 0.001,
        keep_all_components: bool = True,
        single_dominant: bool = False,
    ) -> np.ndarray:
        """
        Segment a pothole binary mask from an RGB image using DeepLabV3.

        Args:
            rgb_input: 3D uint8 numpy array (H, W, 3) or file path to RGB image.
            threshold: Probability cutoff threshold (overrides default if provided).
            min_component_size: Minimum pixel count in downsampled space to retain.
            min_component_ratio: Minimum area ratio for resolution-aware sizing.
            keep_all_components: If True (default), retains all valid potholes in the image.
            single_dominant: If True, legacy mode retaining only the single largest component.

        Returns:
            2D uint8 binary mask of shape (H, W) where 255 indicates pothole, 0 background.
        """
        if self.model is None:
            raise RuntimeError(
                f"DeepLabV3Segmenter is not loaded or fitted. Checked path: '{self.model_path}'."
            )

        # 1. Load and validate RGB image
        if isinstance(rgb_input, (str, os.PathLike)):
            if not os.path.exists(rgb_input):
                raise FileNotFoundError(f"RGB image file not found: '{rgb_input}'")
            with Image.open(rgb_input) as img:
                rgb_arr = np.array(img.convert("RGB"))
        elif isinstance(rgb_input, np.ndarray):
            rgb_arr = rgb_input
        else:
            raise TypeError(f"Expected file path or numpy array. Got: {type(rgb_input).__name__}")

        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise ValueError(f"RGB image must have shape (H, W, 3). Got: {rgb_arr.shape}")

        if not np.isfinite(rgb_arr).all():
            raise ValueError("RGB image contains non-finite values (NaN / Inf).")

        orig_h, orig_w = rgb_arr.shape[:2]
        cutoff = threshold if threshold is not None else self.threshold

        # 2. Resize to processing resolution and normalize
        import torch
        pil_resized = Image.fromarray(rgb_arr).resize(self.target_size, Image.Resampling.BILINEAR)
        arr_norm = (np.array(pil_resized, dtype=np.float32) / 255.0 - self.mean) / self.std
        tensor_input = torch.from_numpy(arr_norm.transpose(2, 0, 1)).unsqueeze(0).float()

        # 3. Forward pass through TorchScript engine
        with torch.no_grad():
            logits = self.model(tensor_input)
            if isinstance(logits, dict):
                logits = logits.get("out", logits)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # 4. Binary thresholding & morphological refinement
        pred_bin = probs > cutoff
        pred_filled = ndi.binary_fill_holes(pred_bin)

        # 5. Connected component filtering
        th, tw = self.target_size[1], self.target_size[0]
        eff_min_size = (
            min_component_size
            if min_component_size is not None
            else max(10, min(50, int(th * tw * min_component_ratio)))
        )

        lbl, num = ndi.label(pred_filled)
        if num > 0:
            sizes = ndi.sum(pred_filled, lbl, range(1, num + 1))
            if keep_all_components and not single_dominant:
                valid_indices = [
                    idx + 1 for idx, size in enumerate(sizes) if size >= eff_min_size
                ]
                if valid_indices:
                    pred_final_small = np.isin(lbl, valid_indices)
                else:
                    pred_final_small = np.zeros((th, tw), dtype=bool)
            else:
                max_idx = int(np.argmax(sizes)) + 1
                if sizes[max_idx - 1] >= eff_min_size:
                    pred_final_small = (lbl == max_idx)
                else:
                    pred_final_small = np.zeros((th, tw), dtype=bool)
        else:
            pred_final_small = np.zeros((th, tw), dtype=bool)

        # 6. Upsample to original image resolution
        pil_mask_small = Image.fromarray((pred_final_small.astype(np.uint8) * 255))
        pil_mask_full = pil_mask_small.resize((orig_w, orig_h), Image.Resampling.NEAREST)
        final_mask = np.array(pil_mask_full, dtype=np.uint8)

        return final_mask


def get_segmenter(
    engine: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> Union[DeepLabV3Segmenter, RGBSegmenter]:
    """
    Factory creating the configured segmentation engine.

    Configuration Priority:
      1. Explicit argument `engine` ('deeplabv3' or 'random_forest'/'rf').
      2. Environment variable `CRACKSIGHT_SEGMENTER_ENGINE`.
      3. Default: 'deeplabv3' if artifact exists, with automatic fallback to 'random_forest'.

    Args:
        engine: Desired segmentation engine ('deeplabv3' or 'random_forest').
        models_dir: Directory containing model artifacts.

    Returns:
        Fitted or loadable Segmenter instance.
    """
    base_dir = models_dir or DEFAULT_MODELS_DIR
    selected_engine = (
        engine or os.environ.get("CRACKSIGHT_SEGMENTER_ENGINE", "deeplabv3")
    ).lower().strip()

    deeplab_file = os.path.join(base_dir, "deeplabv3_mobilenetv3.torchscript.pt")
    rf_file = os.path.join(base_dir, "rgb_segmenter.joblib")

    if selected_engine in ("deeplabv3", "deep", "nn") and os.path.exists(deeplab_file):
        try:
            return DeepLabV3Segmenter(model_path=deeplab_file)
        except Exception as e:
            logger.warning(
                f"Could not initialize DeepLabV3Segmenter ({e}). "
                "Falling back to Random Forest RGBSegmenter."
            )

    return RGBSegmenter(model_path=rf_file)
