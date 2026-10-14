"""
Relative Pothole Hazard Index (RPHI) Calculator.

Constructs a reproducible, composite index combining uncalibrated 3D cavity
depth, 2D mask surface footprint, and relative volumetric displacement.

IMPORTANT SCIENTIFIC LIMITATIONS:
1. RPHI is a provisional, dimensionless relative hazard proxy [0, 1].
2. It does NOT represent physical depth in millimeters (mm) or physical volume in
   cubic centimeters (cm^3), as the underlying Go!SCAN depth maps are uncalibrated
   8-bit integers [0, 190].
3. RPHI serves as a ranking tool to prioritize road maintenance interventions,
   not as a certified physical measurement.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


DEFAULT_WEIGHT_DEPTH = 0.40
DEFAULT_WEIGHT_AREA = 0.30
DEFAULT_WEIGHT_VOLUME = 0.30


class RPHICalculator:
    """
    Computes a composite Relative Pothole Hazard Index (RPHI) from:
      - relative_depth_max_intensity: maximum cavity depression above local road plane.
      - mask_area_pixels: projected 2D surface area footprint in image pixels.
      - relative_volume_proxy: numerical integral of cavity depth above road plane.

    Weights represent experimental hazard prioritization assumptions:
      - Depth weight (default 0.40): Primary vehicle tire impact and suspension damage risk.
      - Area weight (default 0.30): Surface footprint and collision avoidance hazard.
      - Volume proxy weight (default 0.30): Asphalt material loss and base-course distress.
    """

    def __init__(
        self,
        weight_depth: float = DEFAULT_WEIGHT_DEPTH,
        weight_area: float = DEFAULT_WEIGHT_AREA,
        weight_volume: float = DEFAULT_WEIGHT_VOLUME,
        log_transform_area: bool = True,
        log_transform_volume: bool = True,
    ):
        weights = [weight_depth, weight_area, weight_volume]
        if any(w < 0 for w in weights):
            raise ValueError(f"All weights must be non-negative. Got: {weights}")

        total_w = sum(weights)
        if not np.isclose(total_w, 1.0, atol=1e-4):
            raise ValueError(f"Weights must sum to 1.0. Current sum: {total_w:.4f}")

        self.weight_depth = float(weight_depth)
        self.weight_area = float(weight_area)
        self.weight_volume = float(weight_volume)
        self.log_transform_area = log_transform_area
        self.log_transform_volume = log_transform_volume

        self.is_fitted = False
        self.params_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "RPHICalculator":
        """
        Fit normalization extrema (min and max) on a training dataset.

        Args:
            df: DataFrame containing 'relative_depth_max_intensity',
                'mask_area_pixels', and 'relative_volume_proxy'.

        Returns:
            Self (fitted instance).
        """
        self._validate_columns(df)

        depth = df["relative_depth_max_intensity"].astype(float)
        area = df["mask_area_pixels"].astype(float)
        vol = df["relative_volume_proxy"].astype(float)

        if self.log_transform_area:
            area = np.log1p(area)
        if self.log_transform_volume:
            vol = np.log1p(vol)

        self.params_ = {
            "d_min": float(depth.min()),
            "d_max": float(depth.max()),
            "a_min": float(area.min()),
            "a_max": float(area.max()),
            "v_min": float(vol.min()),
            "v_max": float(vol.max()),
        }

        # Prevent division by zero if all values are identical
        for k in ["d", "a", "v"]:
            if np.isclose(self.params_[f"{k}_min"], self.params_[f"{k}_max"]):
                self.params_[f"{k}_max"] = self.params_[f"{k}_min"] + 1.0

        self.is_fitted = True
        return self

    def compute_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute individual normalized component scores [0, 1] and final RPHI.

        Args:
            df: Input DataFrame containing required features.

        Returns:
            DataFrame with columns: 'depth_score', 'area_score', 'volume_score', 'rphi_score'.
        """
        if not self.is_fitted:
            raise RuntimeError("RPHICalculator must be fitted before transforming data.")

        self._validate_columns(df)

        depth = df["relative_depth_max_intensity"].astype(float).values
        area = df["mask_area_pixels"].astype(float).values
        vol = df["relative_volume_proxy"].astype(float).values

        if self.log_transform_area:
            area = np.log1p(area)
        if self.log_transform_volume:
            vol = np.log1p(vol)

        d_norm = (depth - self.params_["d_min"]) / (self.params_["d_max"] - self.params_["d_min"])
        a_norm = (area - self.params_["a_min"]) / (self.params_["a_max"] - self.params_["a_min"])
        v_norm = (vol - self.params_["v_min"]) / (self.params_["v_max"] - self.params_["v_min"])

        # Clip to [0, 1] to guard against out-of-bounds samples in validation folds
        d_norm = np.clip(d_norm, 0.0, 1.0)
        a_norm = np.clip(a_norm, 0.0, 1.0)
        v_norm = np.clip(v_norm, 0.0, 1.0)

        rphi = (
            self.weight_depth * d_norm
            + self.weight_area * a_norm
            + self.weight_volume * v_norm
        )
        rphi = np.clip(rphi, 0.0, 1.0)

        return pd.DataFrame(
            {
                "rphi_depth_score": d_norm,
                "rphi_area_score": a_norm,
                "rphi_volume_score": v_norm,
                "rphi_score": rphi,
            },
            index=df.index,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return 1D array of composite RPHI scores [0, 1]."""
        comp_df = self.compute_components(df)
        return comp_df["rphi_score"].values

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return composite RPHI scores [0, 1]."""
        return self.fit(df).transform(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = [
            "relative_depth_max_intensity",
            "mask_area_pixels",
            "relative_volume_proxy",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required feature columns for RPHI: {missing}")
