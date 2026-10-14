"""
Feature extraction module for Pothole Severity Detection.

Extracts measurable 2D geometric and uncalibrated 3D depth-map intensity
descriptors from paired pothole masks and depth maps.

IMPORTANT SCIENTIFIC LIMITATIONS:
1. The depth maps in Pothole440 are 8-bit grayscale PNGs (uint8, range [0, 190]).
2. No physical calibration factor (mm / intensity level) or camera intrinsic/
   extrinsic parameters (focal length, sensor height, tilt angle) are available.
3. Therefore, all depth metrics are reported in uncalibrated 'intensity levels',
   and all area/volume metrics are reported in 'pixels' or 'relative proxy units'.
   DO NOT interpret or report these metrics as millimeters (mm) or cubic centimeters (cm^3).
"""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def compute_mask_geometry(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute 2D geometric and morphological features from a binary mask.

    Args:
        mask: 2D numpy array where pothole pixels are > 0.

    Returns:
        Dictionary of geometric feature names and values.
    """
    binary_mask = (mask > 0).astype(bool)
    area_pixels = float(np.sum(binary_mask))

    if area_pixels == 0:
        raise ValueError("Cannot extract geometric features from an empty mask.")

    h, w = mask.shape
    total_image_pixels = float(h * w)
    coverage_ratio = area_pixels / total_image_pixels
    coverage_pct = coverage_ratio * 100.0

    # Perimeter via morphological gradient (boundary pixels)
    struct = ndi.generate_binary_structure(2, 1)
    eroded = ndi.binary_erosion(binary_mask, structure=struct)
    boundary_mask = binary_mask ^ eroded
    perimeter_pixels = float(np.sum(boundary_mask))
    # Safeguard for degenerate 1-pixel cases
    perimeter_pixels = max(perimeter_pixels, 1.0)

    # Circularity: 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
    circularity = (4.0 * math.pi * area_pixels) / (perimeter_pixels ** 2)
    circularity = max(0.0, min(circularity, 1.0))

    # Compactness: Perimeter^2 / Area (higher for irregular/elongated shapes)
    compactness = (perimeter_pixels ** 2) / area_pixels

    # Bounding Box metrics
    y_indices, x_indices = np.where(binary_mask)
    bbox_y_min = int(np.min(y_indices))
    bbox_y_max = int(np.max(y_indices))
    bbox_x_min = int(np.min(x_indices))
    bbox_x_max = int(np.max(x_indices))

    bbox_width = float(bbox_x_max - bbox_x_min + 1)
    bbox_height = float(bbox_y_max - bbox_y_min + 1)
    bbox_aspect_ratio = bbox_width / max(bbox_height, 1.0)
    bbox_area = bbox_width * bbox_height
    bbox_extent = area_pixels / max(bbox_area, 1.0)

    return {
        "mask_area_pixels": area_pixels,
        "mask_coverage_ratio": coverage_ratio,
        "mask_coverage_pct": coverage_pct,
        "perimeter_pixels": perimeter_pixels,
        "circularity": circularity,
        "compactness": compactness,
        "bbox_x_min": float(bbox_x_min),
        "bbox_x_max": float(bbox_x_max),
        "bbox_y_min": float(bbox_y_min),
        "bbox_y_max": float(bbox_y_max),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "bbox_extent": bbox_extent,
    }


def compute_depth_features(
    depth: np.ndarray, mask: np.ndarray, dilation_iterations: int = 15
) -> Dict[str, float]:
    """
    Extract depth statistics inside the pothole cavity and compare against
    the local surrounding road surface.

    Empirical Evidence on Grayscale-to-Depth Direction:
      In Pothole440, the road plane has lower intensity (mean ~82-120), whereas
      the interior of the pothole cavity increases in intensity (mean ~145-190).
      Thus, higher pixel values indicate deeper depressions into the pavement.

    Args:
        depth: 2D uint8 numpy array of depth intensities.
        mask: 2D binary mask (> 0 indicates pothole region).
        dilation_iterations: Morphological dilation radius for road margin.

    Returns:
        Dictionary of depth statistics and relative proxy metrics.
    """
    binary_mask = (mask > 0).astype(bool)
    if not np.any(binary_mask):
        raise ValueError("Cannot extract depth features from an empty mask.")

    pothole_depths = depth[binary_mask].astype(np.float64)

    # Depth statistics inside cavity
    depth_min = float(np.min(pothole_depths))
    depth_max = float(np.max(pothole_depths))
    depth_mean = float(np.mean(pothole_depths))
    depth_median = float(np.median(pothole_depths))
    depth_std = float(np.std(pothole_depths))
    depth_p10 = float(np.percentile(pothole_depths, 10))
    depth_p25 = float(np.percentile(pothole_depths, 25))
    depth_p75 = float(np.percentile(pothole_depths, 75))
    depth_p90 = float(np.percentile(pothole_depths, 90))
    depth_iqr = depth_p75 - depth_p25

    # Road surface reference region: dilated band outside pothole
    struct = ndi.generate_binary_structure(2, 1)
    dilated_mask = ndi.binary_dilation(binary_mask, structure=struct, iterations=dilation_iterations)
    road_band = dilated_mask & (~binary_mask)

    # Fallback to entire outside region if dilated band is too small
    if np.sum(road_band) < 50:
        road_band = ~binary_mask

    road_depths = depth[road_band].astype(np.float64)
    road_mean = float(np.mean(road_depths))
    road_median = float(np.median(road_depths))
    road_std = float(np.std(road_depths))
    road_min = float(np.min(road_depths))
    road_max = float(np.max(road_depths))

    # Relative intensity differences (Cavity depression relative to road plane)
    relative_depth_mean = depth_mean - road_mean
    relative_depth_median = depth_median - road_median
    relative_depth_max = depth_max - road_median

    # Relative volume proxy: integral of depth depression above local road baseline
    # Delta(p) = max(0, depth(p) - road_median)
    # Unit: intensity_levels * pixels (dimensionless proxy)
    cavity_depressions = np.maximum(0.0, pothole_depths - road_median)
    relative_volume_proxy = float(np.sum(cavity_depressions))
    relative_volume_density = float(np.mean(cavity_depressions))

    return {
        # Depth inside cavity (intensity levels)
        "depth_min_intensity": depth_min,
        "depth_max_intensity": depth_max,
        "depth_mean_intensity": depth_mean,
        "depth_median_intensity": depth_median,
        "depth_std_intensity": depth_std,
        "depth_p10_intensity": depth_p10,
        "depth_p25_intensity": depth_p25,
        "depth_p75_intensity": depth_p75,
        "depth_p90_intensity": depth_p90,
        "depth_iqr_intensity": depth_iqr,
        # Local road surface baseline (intensity levels)
        "road_mean_intensity": road_mean,
        "road_median_intensity": road_median,
        "road_std_intensity": road_std,
        "road_min_intensity": road_min,
        "road_max_intensity": road_max,
        # Relative metrics (intensity levels / dimensionless proxy)
        "relative_depth_mean_intensity": relative_depth_mean,
        "relative_depth_median_intensity": relative_depth_median,
        "relative_depth_max_intensity": relative_depth_max,
        "relative_volume_proxy": relative_volume_proxy,
        "relative_volume_density": relative_volume_density,
    }


def extract_features_for_sample(
    mask_path: str, depth_path: str, sample_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract combined geometric and relative depth features for a validated sample.

    Args:
        mask_path: Path to binary mask image.
        depth_path: Path to 8-bit depth map image.
        sample_meta: Optional dictionary of sample metadata.

    Returns:
        Complete dictionary of metadata and extracted numeric features.
    """
    with Image.open(mask_path) as m_img, Image.open(depth_path) as d_img:
        mask_arr = np.array(m_img)
        depth_arr = np.array(d_img)

    geom_features = compute_mask_geometry(mask_arr)
    depth_features = compute_depth_features(depth_arr, mask_arr)

    combined = {}
    if sample_meta:
        combined.update(sample_meta)
    combined.update(geom_features)
    combined.update(depth_features)

    return combined
