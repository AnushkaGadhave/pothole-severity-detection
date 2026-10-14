"""
Relative Pothole Hazard Index (RPHI) Tier and Label Generator.

Generates provisional relative hazard tiers ('Low', 'Medium', 'High')
from RPHI continuous scores using group-level or image-level quantile thresholds.
Provides strict fold-specific, leakage-safe threshold fitting for cross-validation.

IMPORTANT DISCLAIMER:
These tiers represent a provisional relative hazard ranking index within this
dataset population. They DO NOT correspond to certified ground-truth physical
severity or municipal maintenance urgency standards (such as IRC:82 or ASTM D6433).
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.severity.rphi_calculator import RPHICalculator


DEFAULT_FEATURES_PATH = r"c:\Users\Anushka\Documents\pothole-severity-detection\data\processed\pothole_features.csv"
DEFAULT_OUTPUT_LABELED_PATH = r"c:\Users\Anushka\Documents\pothole-severity-detection\data\processed\pothole_features_labeled.csv"


class RPHILabelGenerator:
    """
    Assigns provisional relative hazard tiers (Low, Medium, High)
    based on calibrated RPHI score quantile thresholds.
    """

    def __init__(
        self,
        calculator: Optional[RPHICalculator] = None,
        quantiles: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
        threshold_mode: str = "group_level",
        group_col: str = "canonical_pothole_id",
    ):
        """
        Args:
            calculator: Optional RPHICalculator instance. If None, uses default weights.
            quantiles: (low_tier_cutoff_quantile, med_tier_cutoff_quantile). Default: (0.333, 0.667).
            threshold_mode: 'group_level' (recommended, computes quantiles from group medians
                            to prevent augmented images from skewing cutoffs) or 'image_level'.
            group_col: Column name identifying canonical physical pothole entities.
        """
        if len(quantiles) != 2 or not (0.0 < quantiles[0] < quantiles[1] < 1.0):
            raise ValueError(f"Quantiles must be two strictly increasing floats in (0, 1). Got: {quantiles}")

        if threshold_mode not in ["group_level", "image_level"]:
            raise ValueError(f"threshold_mode must be 'group_level' or 'image_level'. Got: '{threshold_mode}'")

        self.calculator = calculator if calculator is not None else RPHICalculator()
        self.quantiles = quantiles
        self.threshold_mode = threshold_mode
        self.group_col = group_col

        self.is_fitted = False
        self.threshold_low_ = 0.0
        self.threshold_high_ = 0.0

    def fit(self, df: pd.DataFrame) -> "RPHILabelGenerator":
        """
        Fit the RPHICalculator and determine cutoff thresholds from training data.

        Args:
            df: DataFrame containing required features and group_col.

        Returns:
            Self (fitted instance).
        """
        self.calculator.fit(df)
        scores = self.calculator.transform(df)

        if self.threshold_mode == "group_level":
            if self.group_col not in df.columns:
                raise KeyError(f"Group column '{self.group_col}' missing for group_level thresholding.")
            # Calculate median RPHI per canonical physical pothole to eliminate augmentation bias
            group_scores = df.assign(_rphi=scores).groupby(self.group_col)["_rphi"].median()
            self.threshold_low_ = float(group_scores.quantile(self.quantiles[0]))
            self.threshold_high_ = float(group_scores.quantile(self.quantiles[1]))
        else:
            self.threshold_low_ = float(np.quantile(scores, self.quantiles[0]))
            self.threshold_high_ = float(np.quantile(scores, self.quantiles[1]))

        self.is_fitted = True
        return self

    def predict_tier(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RPHI scores and assign discrete hazard tiers:
          - score <= threshold_low_ -> 'Low'
          - threshold_low_ < score <= threshold_high_ -> 'Medium'
          - score > threshold_high_ -> 'High'

        Args:
            df: DataFrame containing required feature columns.

        Returns:
            (rphi_scores, rphi_tier_labels)
        """
        if not self.is_fitted:
            raise RuntimeError("RPHILabelGenerator must be fitted before assigning tiers.")

        scores = self.calculator.transform(df)
        tiers = np.empty(len(scores), dtype=object)

        low_mask = scores <= self.threshold_low_
        med_mask = (scores > self.threshold_low_) & (scores <= self.threshold_high_)
        high_mask = scores > self.threshold_high_

        tiers[low_mask] = "Low"
        tiers[med_mask] = "Medium"
        tiers[high_mask] = "High"

        return scores, tiers

    def fit_predict_tier(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on df and return (scores, tiers)."""
        return self.fit(df).predict_tier(df)

    def generate_fold_leakage_safe_labels(
        self, df: pd.DataFrame, fold_col: str = "fold"
    ) -> pd.DataFrame:
        """
        Compute fold-specific, strictly leakage-safe RPHI scores and tiers.
        For every fold k, normalization extrema and quantile cutoffs are fitted
        STRICTLY on training groups (fold != k) and evaluated on validation fold (fold == k).

        Args:
            df: DataFrame with features and fold column.
            fold_col: Column containing fold index (0 to n_splits - 1).

        Returns:
            DataFrame with columns 'rphi_fold_score' and 'rphi_fold_tier'.
        """
        if fold_col not in df.columns:
            raise KeyError(f"Fold column '{fold_col}' not found in DataFrame.")

        unique_folds = sorted(df[fold_col].unique())
        fold_scores = np.zeros(len(df), dtype=float)
        fold_tiers = np.empty(len(df), dtype=object)

        for f_idx in unique_folds:
            train_mask = df[fold_col] != f_idx
            val_mask = df[fold_col] == f_idx

            train_df = df[train_mask]
            val_df = df[val_mask]

            # Create independent generator instance to prevent state leakage
            fold_gen = RPHILabelGenerator(
                calculator=RPHICalculator(
                    weight_depth=self.calculator.weight_depth,
                    weight_area=self.calculator.weight_area,
                    weight_volume=self.calculator.weight_volume,
                    log_transform_area=self.calculator.log_transform_area,
                    log_transform_volume=self.calculator.log_transform_volume,
                ),
                quantiles=self.quantiles,
                threshold_mode=self.threshold_mode,
                group_col=self.group_col,
            )

            # Fit ONLY on training split
            fold_gen.fit(train_df)

            # Predict on validation split
            v_scores, v_tiers = fold_gen.predict_tier(val_df)
            val_indices = np.where(val_mask)[0]
            fold_scores[val_indices] = v_scores
            fold_tiers[val_indices] = v_tiers

        return pd.DataFrame(
            {
                "rphi_fold_score": fold_scores,
                "rphi_fold_tier": fold_tiers,
            },
            index=df.index,
        )


def generate_labeled_dataset(
    features_csv: str = DEFAULT_FEATURES_PATH,
    output_csv: str = DEFAULT_OUTPUT_LABELED_PATH,
) -> pd.DataFrame:
    """
    Load extracted features, compute both exploratory full-dataset RPHI tiers
    and leakage-safe fold-specific tiers, and save to output_csv.

    Args:
        features_csv: Path to input pothole_features.csv.
        output_csv: Path to output pothole_features_labeled.csv.

    Returns:
        Labeled DataFrame with 42 total columns.
    """
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    df = pd.read_csv(features_csv)

    # 1. Full-dataset exploratory RPHI (group-calibrated)
    full_gen = RPHILabelGenerator(threshold_mode="group_level", group_col="canonical_pothole_id")
    exploratory_scores, exploratory_tiers = full_gen.fit_predict_tier(df)

    # 2. Strict leakage-safe 5-fold evaluation RPHI (fitted on training folds only)
    fold_labels_df = full_gen.generate_fold_leakage_safe_labels(df, fold_col="fold")

    # Assemble combined DataFrame
    labeled_df = df.copy()
    labeled_df["rphi_score"] = exploratory_scores
    labeled_df["rphi_tier"] = exploratory_tiers
    labeled_df["rphi_fold_score"] = fold_labels_df["rphi_fold_score"]
    labeled_df["rphi_fold_tier"] = fold_labels_df["rphi_fold_tier"]

    # Save to disk
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    labeled_df.to_csv(output_csv, index=False)
    print(f"Saved labeled features to: {output_csv} ({len(labeled_df)} rows, {len(labeled_df.columns)} cols)")

    return labeled_df


if __name__ == "__main__":
    labeled_data = generate_labeled_dataset()
    print("\nExploratory Tier Distribution (Full Dataset Group Cutoffs):")
    print(labeled_data["rphi_tier"].value_counts())
    print("\nFold-Specific Tier Distribution (Leakage-Safe CV Target):")
    print(labeled_data["rphi_fold_tier"].value_counts())
