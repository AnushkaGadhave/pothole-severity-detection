"""
Baseline model definitions and training helpers for Pothole Severity Detection.

Defines the model architectures and feature subsets for:
  - Experiment A: RPHI Formula Reproduction (relative depth max, mask area, relative volume proxy)
  - Experiment B: 2D Geometry Only (all 2D mask geometric features, zero depth/volume features)

Models:
  - Majority-class baseline (DummyClassifier)
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (RBF kernel, with StandardScaler pipeline)
  - K-Nearest Neighbors (k=5, with StandardScaler pipeline)
"""

from typing import Dict, List, Any, Callable
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Target column (strict leakage-safe CV target)
TARGET_COLUMN = "rphi_fold_tier"
CLASS_LABELS = ["Low", "Medium", "High"]

# Forbidden feature columns (identifiers, targets, and leakage sources)
FORBIDDEN_COLUMNS = [
    "filename",
    "original_split",
    "canonical_pothole_id",
    "fold",
    "rphi_score",
    "rphi_tier",
    "rphi_fold_score",
    "rphi_fold_tier",
]

# Experiment A features: Exact constituent features of RPHI formula
EXP_A_FEATURES = [
    "relative_depth_max_intensity",
    "mask_area_pixels",
    "relative_volume_proxy",
]

# Experiment B features: Strictly 2D geometric and morphological features
# 100% EXCLUDES all depth, road baseline, and volume proxy features
EXP_B_FEATURES = [
    "mask_area_pixels",
    "mask_coverage_ratio",
    "mask_coverage_pct",
    "perimeter_pixels",
    "circularity",
    "compactness",
    "bbox_x_min",
    "bbox_x_max",
    "bbox_y_min",
    "bbox_y_max",
    "bbox_width",
    "bbox_height",
    "bbox_aspect_ratio",
    "bbox_extent",
]


def get_baseline_models(random_state: int = 42) -> Dict[str, Callable[[], Any]]:
    """
    Return dictionary of factory callables for all baseline models.

    Using callables ensures clean, unfitted model instances for each fold,
    preventing state leakage between cross-validation splits.
    """
    return {
        "Majority Baseline": lambda: DummyClassifier(strategy="most_frequent"),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=random_state
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
        "SVM (RBF)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=1.0, kernel="rbf", random_state=random_state)),
        ]),
        "KNN (k=5)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }
