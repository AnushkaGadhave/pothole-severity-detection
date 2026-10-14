# Pothole Severity Detection

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-pothole--severity--detection-lightgrey.svg)](https://github.com/AnushkaGadhave/pothole-severity-detection)

A machine learning pipeline designed to analyze **pothole depth and physical road damage**, extract multi-modal geometric and morphological features, assess structural road degradation, and benchmark traditional machine learning models for civic road maintenance planning.

---

## 📌 Project Overview

Urban roadways frequently suffer from rapid degradation caused by heavy monsoons, intense traffic loads, and environmental wear. While standard detection systems identify whether a pothole is present, prioritization requires understanding **how severe** the road defect is.

This project focuses on:
- **Depth & Geometric Profiling**: Measuring pothole depth distributions, physical surface dimensions, estimated volume displacement, and perimeter irregularity from sensor or 3D depth maps.
- **Physical Road Damage Analysis**: Characterizing the physical structure of asphalt failures, structural cracking, edge deterioration, and cavity geometry.
- **Severity Classification**: Grading road damage into actionable severity tiers (e.g., *Low*, *Medium*, *High / Critical*) to assist municipal engineers in resource allocation.
- **Municipal Prioritization Integration**: Providing the foundational intelligence for a future **Brihanmumbai Municipal Corporation (BMC)** road-maintenance prioritization dashboard.

---

## ⚠️ Critical Note: Calibration & Unit Verification

> [!IMPORTANT]
> **Severity thresholds and depth-map units must be rigorously verified before model training.**
>
> 1. **Depth-Map Sensor Units**: Verify whether raw input depth maps represent physical depth in **millimeters (mm)**, **centimeters (cm)**, **meters (m)**, or normalized disparity values. Misinterpreting sensor units directly distorts volume calculations and severity classifications.
> 2. **Severity Threshold Calibration**: Pothole severity thresholds (depth boundaries defining Low, Medium, and High hazard levels) must be calibrated against municipal road engineering guidelines (e.g., IRC / BMC pavement standards) and ground truth road inspection data prior to running feature extraction or fitting models.

---

## 🔬 Methodology & Pipeline

```
[Raw Imagery / Depth Maps]
            │
            ▼
[Data Preprocessing & Filtering]
  - Noise reduction, depth normalization, ROI segmentation
            │
            ▼
[Feature Extraction]
  - Geometric: Max depth, mean depth, variance, surface area, volume, perimeter
  - Morphological: Circularity, aspect ratio, depth gradient, roughness index
            │
            ▼
[Severity Assessment & Labeling]
  - Multi-tier severity categorization (Low / Medium / High)
            │
            ▼
[Traditional ML Model Comparison]
  - Random Forest, Gradient Boosting / XGBoost, SVM, Decision Trees, KNN
            │
            ▼
[Evaluation & Benchmarking]
  - Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix, ROC-AUC
            │
            ▼
[BMC Maintenance Prioritization Dashboard]
  - Ward-level mapping, urgency ranking, repair material dispatch
```

### 1. Feature Extraction
The feature engineering module extracts quantitative indicators of physical road cavity degradation:
- **Depth Descriptors**: Maximum cavity depth, mean depression depth, depth standard deviation, and depth gradient profiles across the pothole basin.
- **Spatial & Geometric Metrics**: Projected 2D surface area ($cm^2$), perimeter length, bounding box aspect ratio, and circularity/compactness.
- **Volumetric Estimation**: Numerical integration of depth over the segmented pothole footprint to estimate aggregate asphalt displacement volume ($cm^3$).
- **Texture & Gradient Features**: Surface roughness, edge sharpness, and structural gradient distributions around the pothole rim.

### 2. Severity Assessment
Potholes are categorized into discrete severity grades based on structural risk to vehicles and commuter safety:
- **Low Severity**: Minor surface depression with negligible structural hazard to standard traffic.
- **Medium Severity**: Moderate depth and surface expansion requiring planned maintenance before base-course collapse.
- **High / Critical Severity**: Deep structural cavity presenting severe collision risk, tire damage, and vehicle loss of control; demands immediate civic emergency patching.

### 3. Comparison of Traditional ML Models
Rather than relying solely on black-box architectures, this project evaluates and benchmarks traditional, interpretable machine learning classifiers:
- **Random Forest (RF)**
- **Gradient Boosting Classifiers (XGBoost / LightGBM)**
- **Support Vector Machines (SVM)**
- **Decision Trees**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression Baselines**

All models are cross-validated and compared using comprehensive classification metrics including Precision, Recall, Macro F1-score, Cohen's Kappa, and inference latency.

---

## 🏛️ Future Scope: BMC Road-Maintenance Prioritization Dashboard

An interactive dashboard application (`app/`) is planned to support civic bodies like the **Brihanmumbai Municipal Corporation (BMC)**:
- **Ward-Level Geospatial Prioritization**: Aggregating severity scores across Mumbai's administrative wards to pinpoint high-risk corridors.
- **Maintenance Urgency Scoring**: Weighting physical severity with road traffic density, public bus routes, and monsoon flood susceptibility.
- **Resource & Material Allocation**: Recommending whether defects require rapid cold-mix patching, durable mastic asphalt application, or comprehensive cement-concrete (CC) road resurfacing.
- **Citizen Report Auditing**: Cross-verifying citizen-submitted pothole complaints with automated severity metrics to reduce response turnaround time.

---

## 📁 Repository Structure

```
pothole-severity-detection/
├── data/
│   ├── raw/                  # Original unprocessed sensor imagery & depth maps (.gitkeep)
│   └── processed/            # Cleaned, filtered, and tabularized feature sets (.gitkeep)
├── notebooks/                # Jupyter exploration and experimental notebooks (.gitkeep)
├── src/                      # Source code package
│   ├── __init__.py           # Package initializer
│   ├── init.py               # Package initialization entry
│   ├── preprocessing/        # Data filtering, sensor cleaning & depth map alignment
│   ├── features/             # Geometric, depth, and morphological feature extraction
│   ├── severity/             # Severity grading rules, threshold definitions & metrics
│   ├── models/               # Traditional ML model training, tuning & serialization
│   ├── evaluation/           # Performance benchmarking, confusion matrices & reporting
│   └── utils/                # Logging, configuration parsers, and file I/O helpers
├── tests/                    # Unit and integration test suite (.gitkeep)
├── results/
│   ├── figures/              # Generated plots, ROC curves, and visualizations (.gitkeep)
│   └── reports/              # Model comparison tables, evaluation metrics & logs (.gitkeep)
├── app/                      # BMC road-maintenance prioritization dashboard (.gitkeep)
├── .gitignore                # Git exclusion rules for data, models, results, and caches
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Environment Setup
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/AnushkaGadhave/pothole-severity-detection.git
cd pothole-severity-detection

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
