"""
Dataset loader and validation module for Pothole440.

Loads RGB images, binary masks, and depth maps, parses canonical physical
pothole identifiers from filenames, validates image integrity and dimensions,
and records explicit exclusions (such as empty masks).
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from PIL import Image
import numpy as np


DEFAULT_DATASET_ROOT = r"C:\Users\Anushka\Downloads\Pothole440-main\Pothole440-main"
SPLITS = ["train", "val", "test"]
MODALITIES = ["rgb", "mask", "depth"]

# Known defective empty-mask files identified during audit
KNOWN_EMPTY_MASKS = {
    "train/2022_05_22_pothole_11_horizantal_aug5.png",
    "train/2022_05_22_pothole_15_aug20.png",
    "train/2022_05_22_pothole_21_horizantal_vertical_aug46.png",
    "val/2022_05_22_pothole_12_vertical_aug11.png",
    "val/2022_05_22_pothole_19_vertical_aug39.png",
    "test/2022_05_22_pothole_11_vertical_aug7.png",
}


def parse_pothole_id(filename: str) -> str:
    """
    Extract the canonical physical pothole identifier from a dataset filename.

    The dataset filenames follow the pattern:
      YYYY_MM_DD_pothole_XX[_augmentation_suffix].png

    Examples:
      '2022_05_22_pothole_10.png' -> '2022_05_22_pothole_10'
      '2022_05_22_pothole_10_aug0.png' -> '2022_05_22_pothole_10'
      '2022_05_22_pothole_10_horizantal_vertical_aug2.png' -> '2022_05_22_pothole_10'

    Args:
        filename: Base filename or path.

    Returns:
        Canonical pothole ID string, e.g. '2022_05_22_pothole_10'.

    Raises:
        ValueError: If filename does not match the expected naming pattern.
    """
    base_name = os.path.basename(filename)
    pattern = r"^(\d{4}_\d{2}_\d{2}_pothole_\d+)"
    match = re.match(pattern, base_name, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Filename '{base_name}' does not match expected pattern: "
            "YYYY_MM_DD_pothole_XX[_suffix].png"
        )
    return match.group(1).lower()


def validate_sample_files(
    rgb_path: str, mask_path: str, depth_path: str
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Verify readable files, matching dimensions, expected dtypes, and non-empty mask.

    Args:
        rgb_path: Path to RGB image.
        mask_path: Path to binary mask image.
        depth_path: Path to depth map image.

    Returns:
        (is_valid, reason, metadata_dict)
    """
    for p, name in [(rgb_path, "rgb"), (mask_path, "mask"), (depth_path, "depth")]:
        if not os.path.exists(p):
            return False, f"missing_file_{name}", None

    try:
        with Image.open(rgb_path) as rgb_img, \
             Image.open(mask_path) as mask_img, \
             Image.open(depth_path) as depth_img:

            rgb_size = rgb_img.size
            mask_size = mask_img.size
            depth_size = depth_img.size

            if not (rgb_size == mask_size == depth_size):
                return False, f"dimension_mismatch (rgb:{rgb_size}, mask:{mask_size}, depth:{depth_size})", None

            mask_arr = np.array(mask_img)
            depth_arr = np.array(depth_img)

            # Check if mask has any pothole pixels
            pothole_pixels = int(np.sum(mask_arr > 0))
            if pothole_pixels == 0:
                return False, "empty_mask (zero foreground pixels)", {
                    "width": rgb_size[0],
                    "height": rgb_size[1],
                    "pothole_pixels": 0,
                }

            meta = {
                "width": rgb_size[0],
                "height": rgb_size[1],
                "pothole_pixels": pothole_pixels,
                "rgb_mode": rgb_img.mode,
                "mask_mode": mask_img.mode,
                "depth_mode": depth_img.mode,
                "mask_dtype": str(mask_arr.dtype),
                "depth_dtype": str(depth_arr.dtype),
            }
            return True, "valid", meta

    except Exception as e:
        return False, f"corrupted_or_unreadable: {str(e)}", None


class PotholeDatasetLoader:
    """
    Loads and validates samples from the Pothole440 dataset directory.
    Maintains strict read-only access to source files.
    """

    def __init__(self, root_dir: str = DEFAULT_DATASET_ROOT):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        self.valid_samples: List[Dict[str, Any]] = []
        self.excluded_samples: List[Dict[str, Any]] = []

    def load_dataset(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Scan all splits, validate file triplets, and segregate valid from excluded samples.

        Returns:
            (valid_samples, excluded_samples)
        """
        self.valid_samples = []
        self.excluded_samples = []

        for split in SPLITS:
            rgb_dir = os.path.join(self.root_dir, split, "rgb")
            mask_dir = os.path.join(self.root_dir, split, "mask")
            depth_dir = os.path.join(self.root_dir, split, "depth")

            if not os.path.exists(rgb_dir):
                continue

            for fname in sorted(os.listdir(rgb_dir)):
                if not fname.lower().endswith(".png"):
                    continue

                rgb_path = os.path.join(rgb_dir, fname)
                mask_path = os.path.join(mask_dir, fname)
                depth_path = os.path.join(depth_dir, fname)

                # Parse canonical physical pothole identifier
                try:
                    pothole_id = parse_pothole_id(fname)
                except ValueError as err:
                    self.excluded_samples.append({
                        "filename": fname,
                        "split": split,
                        "canonical_pothole_id": None,
                        "reason": f"invalid_naming: {str(err)}",
                        "rgb_path": rgb_path,
                        "mask_path": mask_path,
                        "depth_path": depth_path,
                    })
                    continue

                # Validate triplets and check for empty masks
                is_valid, reason, meta = validate_sample_files(rgb_path, mask_path, depth_path)

                sample_entry = {
                    "filename": fname,
                    "split": split,
                    "canonical_pothole_id": pothole_id,
                    "rgb_path": rgb_path,
                    "mask_path": mask_path,
                    "depth_path": depth_path,
                    "reason": reason,
                }
                if meta:
                    sample_entry.update(meta)

                if is_valid:
                    self.valid_samples.append(sample_entry)
                else:
                    self.excluded_samples.append(sample_entry)

        return self.valid_samples, self.excluded_samples
