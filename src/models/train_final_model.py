"""
Final production model training script for Pothole Severity Detection.

Trains and serializes the top-performing SVM (RBF) models and their metadata:
  - Model A: 3-feature RPHI Formula Reproducer ('relative_depth_max_intensity',
             'mask_area_pixels', 'relative_volume_proxy')
  - Model B: 14-feature 2D Geometry Only (all 2D mask geometric features, zero depth/volume)

Target generation uses the existing RPHILabelGenerator fitted consistently on
canonical physical pothole groups to derive global group-calibrated RPHI tiers.

IMPORTANT SCIENTIFIC LIMITATIONS:
1. Target tiers ('Low', 'Medium', 'High') are provisional Relative Pothole Hazard
   Index (RPHI) categories within this dataset population. They DO NOT represent
   certified municipal maintenance urgency or civil engineering standards (such as IRC:82 or BMC).
2. The Go!SCAN depth maps are uncalibrated 8-bit grayscale PNGs [0, 190]. They DO NOT
   provide calibrated physical depth in millimeters (mm) or physical volume in liters.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Ensure package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.train_baselines import (
    CLASS_LABELS,
    EXP_A_FEATURES,
    EXP_B_FEATURES,
)
from src.severity.label_generator import RPHILabelGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = r"c:\Users\Anushka\Documents\pothole-severity-detection\data\processed\pothole_features.csv"
DEFAULT_OUTPUT_DIR = r"c:\Users\Anushka\Documents\pothole-severity-detection\models"
MODEL_A_FILENAME = "svm_rphi_reproducer.joblib"
MODEL_B_FILENAME = "svm_2d_geometry.joblib"
METADATA_FILENAME = "model_metadata.json"


def train_and_save_final_models(
    features_csv: str = DEFAULT_FEATURES_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train production SVM models on the dataset and save serialized pipelines and metadata.

    Args:
        features_csv: Path to input pothole_features.csv.
        output_dir: Directory where trained artifacts will be saved.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary summarizing trained model metadata and artifact paths.
    """
    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Processed features CSV not found: '{features_csv}'. "
            "Please run 'python src/features/build_features.py' first to extract features."
        )

    logger.info(f"Loading feature dataset from: {features_csv}")
    df = pd.read_csv(features_csv)
    n_samples = len(df)
    n_groups = df["canonical_pothole_id"].nunique() if "canonical_pothole_id" in df.columns else None

    logger.info(f"Dataset loaded: {n_samples} samples across {n_groups} canonical pothole groups.")

    # 1. Fit target generator on full dataset using canonical group calibration
    logger.info("Fitting group-calibrated RPHILabelGenerator...")
    label_gen = RPHILabelGenerator(
        threshold_mode="group_level",
        group_col="canonical_pothole_id",
    )
    label_gen.fit(df)
    rphi_scores, target_tiers = label_gen.predict_tier(df)

    tier_counts = pd.Series(target_tiers).value_counts().to_dict()
    logger.info(f"Fitted RPHI thresholds: Low/Med={label_gen.threshold_low_:.4f}, Med/High={label_gen.threshold_high_:.4f}")
    logger.info(f"Target tier distribution: {tier_counts}")

    os.makedirs(output_dir, exist_ok=True)

    # 2. Train Model A: 3-feature RPHI Formula Reproducer
    logger.info("Training Model A: SVM (RBF) - RPHI Formula Reproducer (3 features)...")
    X_a = df[EXP_A_FEATURES].values
    pipeline_a = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel="rbf", random_state=random_state)),
    ])
    pipeline_a.fit(X_a, target_tiers)

    model_a_path = os.path.join(output_dir, MODEL_A_FILENAME)
    model_a_package = {
        "model_id": "svm_rphi_reproducer",
        "model_name": "SVM (RBF) - RPHI Formula Reproducer",
        "experiment": "A",
        "pipeline": pipeline_a,
        "features": list(EXP_A_FEATURES),
        "class_labels": list(CLASS_LABELS),
        "thresholds": {
            "threshold_low": float(label_gen.threshold_low_),
            "threshold_high": float(label_gen.threshold_high_),
        },
        "rphi_weights": {
            "weight_depth": float(label_gen.calculator.weight_depth),
            "weight_area": float(label_gen.calculator.weight_area),
            "weight_volume": float(label_gen.calculator.weight_volume),
        },
        "training_metadata": {
            "n_samples": int(n_samples),
            "n_groups": int(n_groups) if n_groups is not None else None,
            "class_distribution": tier_counts,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": "63b1889",
        },
        "scientific_disclaimer": (
            "Provisional RPHI formula reproducer. Does not measure calibrated "
            "physical depth in millimeters or certified road maintenance priority."
        ),
    }
    joblib.dump(model_a_package, model_a_path)
    logger.info(f"Model A saved to: {model_a_path}")

    # 3. Train Model B: 14-feature 2D Geometry Only
    logger.info("Training Model B: SVM (RBF) - 2D Geometry Only (14 features)...")
    X_b = df[EXP_B_FEATURES].values
    pipeline_b = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel="rbf", random_state=random_state)),
    ])
    pipeline_b.fit(X_b, target_tiers)

    model_b_path = os.path.join(output_dir, MODEL_B_FILENAME)
    model_b_package = {
        "model_id": "svm_2d_geometry",
        "model_name": "SVM (RBF) - 2D Geometry Only",
        "experiment": "B",
        "pipeline": pipeline_b,
        "features": list(EXP_B_FEATURES),
        "class_labels": list(CLASS_LABELS),
        "thresholds": {
            "threshold_low": float(label_gen.threshold_low_),
            "threshold_high": float(label_gen.threshold_high_),
        },
        "rphi_weights": {
            "weight_depth": float(label_gen.calculator.weight_depth),
            "weight_area": float(label_gen.calculator.weight_area),
            "weight_volume": float(label_gen.calculator.weight_volume),
        },
        "training_metadata": {
            "n_samples": int(n_samples),
            "n_groups": int(n_groups) if n_groups is not None else None,
            "class_distribution": tier_counts,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": "63b1889",
        },
        "scientific_disclaimer": (
            "Provisional 2D geometry triage model. Zero depth/volume features used. "
            "Does not measure physical cavity depth or certified road maintenance priority."
        ),
    }
    joblib.dump(model_b_package, model_b_path)
    logger.info(f"Model B saved to: {model_b_path}")

    # 4. Save metadata summary JSON
    metadata = {
        "project": "AI-Based Pothole Depth and Physical Damage Severity Assessment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data": {
            "path": features_csv,
            "total_samples": int(n_samples),
            "canonical_pothole_groups": int(n_groups) if n_groups is not None else None,
            "class_distribution": tier_counts,
        },
        "models": {
            "model_a": {
                "file": MODEL_A_FILENAME,
                "name": "SVM (RBF) - RPHI Formula Reproducer",
                "features": list(EXP_A_FEATURES),
                "num_features": len(EXP_A_FEATURES),
                "cv_macro_f1": 0.8879,
                "cv_balanced_acc": 0.8938,
            },
            "model_b": {
                "file": MODEL_B_FILENAME,
                "name": "SVM (RBF) - 2D Geometry Only",
                "features": list(EXP_B_FEATURES),
                "num_features": len(EXP_B_FEATURES),
                "cv_macro_f1": 0.6955,
                "cv_balanced_acc": 0.7054,
            },
        },
        "scientific_limitations": [
            "Tiers are provisional Relative Pothole Hazard Index (RPHI) categories within the Pothole440 dataset.",
            "Tiers do NOT represent certified municipal maintenance urgency (BMC) or civil engineering standards (IRC:82).",
            "Depth maps are uncalibrated 8-bit integers [0, 190], not calibrated millimeters (mm).",
            "Mask areas are image pixel counts, not calibrated physical square meters.",
        ],
    }

    meta_path = os.path.join(output_dir, METADATA_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata summary saved to: {meta_path}")

    return {
        "model_a_path": model_a_path,
        "model_b_path": model_b_path,
        "metadata_path": meta_path,
        "n_samples": n_samples,
        "tier_distribution": tier_counts,
    }


if __name__ == "__main__":
    try:
        results = train_and_save_final_models()
        print("\nFinal models successfully trained and serialized:")
        print(f"  Model A (RPHI Reproducer): {results['model_a_path']}")
        print(f"  Model B (2D Geometry):    {results['model_b_path']}")
        print(f"  Metadata Summary:         {results['metadata_path']}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
