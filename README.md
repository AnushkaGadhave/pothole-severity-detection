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

---

## 🗺️ Geospatial Inspection Dashboard & Location Workflow

CrackSight includes an interactive, location-aware road inspection workflow and geospatial municipal dashboard built with **Leaflet** and **OpenStreetMap**:

### 1. Location Selection in Analysis Workflow
- **Searchable Autocomplete**: Real-time road, landmark, and street lookup powered by OpenStreetMap Nominatim with 350ms debouncing, in-memory caching, keyboard navigation (Arrow keys/Enter/Escape), and loading indicators.
- **Interactive Map Selection**: Leaflet map supporting pan, zoom, click-to-pin, and draggable markers for fine-tuning defect coordinates.
- **Reverse Geocoding**: Automatically resolves clicked or dragged pin coordinates to human-readable street names and municipal localities.
- **"Use My Location" GPS Button**: Browser geolocation (`navigator.geolocation`) triggered **strictly on explicit user demand**, never on initial page load. Handled with explicit permission prompts and graceful error states.
- **Independent & Optional Workflow**: Image segmentation and severity inference operate independently of location metadata. Users may analyze pavement images with or without tagging coordinates.

### 2. Geospatial Inspection Dashboard (`#map`)
- **Genuine Saved Inspection Records**: The spatial map plots strictly genuine, user-analyzed pavement defects stored in browser history. Zero fabricated pins, simulated markers, or synthetic records.
- **Severity-Coded Markers**: Custom SVG map pins visually categorized by model-predicted severity (High: `#B42368` Berry, Medium: `#F59E0B` Amber, Low: `#10B981` Emerald Teal).
- **Maintenance Repair Status Tracking**: Each inspection tracks maintenance repair status (`Reported`, `Under Review`, `Repair Scheduled`, `Repaired`, `Reopened`) with inline status indicators. Status updates in map popups persist immediately to browser history.
- **Multi-Faceted Spatial Filters**:
  - Filter by **Severity Tier** (High, Medium, Low).
  - Filter by **Repair Status** (Reported, Under Review, Repair Scheduled, Repaired, Reopened).
  - Filter by **Date Range** (All Dates, Today, Last 7 Days, Last 30 Days).
  - Free-text **Search** across addresses, street names, filenames, and inspection IDs.
- **Interactive Inspection Cards**: Left-hand sidebar lists filtered inspections; clicking any card flies the map camera directly to the defect marker and opens its detail popup.
- **CSV Data Export**: One-click RFC-4180 compliant CSV export of filtered inspection records (ID, filename, timestamp, lat/lng, address, severity, repair status, planar area, coverage, circularity, RPHI score).
- **Clean Empty & Unpinned States**: Graceful empty-state overlay when no geotagged records exist, alongside clear indicators for inspections analyzed without GPS tags.

### 3. Data Privacy & Provider Configuration
- **Zero Image Data Transmission**: Geocoding providers receive only search strings or coordinate pairs. Road images and binary masks are never transmitted to map or geocoding services.
- **Mapping Provider**: OpenStreetMap raster tiles (`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`).
- **Tile Attribution**: Included automatically on all map viewports: `&copy; OpenStreetMap contributors`.
- **Environment Variables**:
  - `VITE_API_BASE_URL` (optional): URL pointing to Python backend API (defaults to `/api` via Vite dev proxy).
  - `VITE_GOOGLE_MAPS_API_KEY` (optional): For teams configured to use Google Maps Platform Places Autocomplete instead of OpenStreetMap Nominatim.
- **Coordinate Validation**: Validates numerical bounds for latitude ($-90.0^\circ$ to $+90.0^\circ$) and longitude ($-180.0^\circ$ to $+180.0^\circ$). Invalid coordinates are cleanly ignored.

---

## 🏛️ BMC Road-Maintenance Prioritization Integration

The operational dashboard (`app.html`) supports civic bodies like the **Brihanmumbai Municipal Corporation (BMC)**:
- **Ward-Level Geospatial Prioritization**: Visualizing defects across Mumbai's road network to identify deteriorating corridors.
- **Maintenance Urgency Scoring**: Correlating provisional severity with real municipal road locations.
- **Triage & Status Lifecycles**: Tracking civic maintenance from initial citizen/patrol reporting through scheduling, completion, and post-monsoon reopening.


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
# Python backend dependencies
pip install -r requirements.txt

# Frontend UI & mapping dependencies
npm install
```

### 4. Running the Full-Stack Application
To run the interactive CrackSight dashboard with full computer vision inference and geospatial mapping:

**Step 1: Start the Python Backend API (Terminal 1)**
```bash
python app/api.py
# Runs Starlette ASGI API on http://127.0.0.1:8000
# Verifies models: RGBSegmenter, SVM 2D Geometry, SVM RPHI Reproducer
```

**Step 2: Start the Vite Frontend (Terminal 2)**
```bash
npm run dev
# Starts Vite dev server on http://localhost:5173
# Automatically proxies /api requests to http://127.0.0.1:8000
```

Open `http://localhost:5173` for the landing page or `http://localhost:5173/app.html` for the operational inspection workbench.

---

## 🛠️ Model Training & Inference

### 1. Training the Final Production Models
Once feature extraction and group folds are built, train and serialize the production classifiers:
```bash
# 1. Train severity classification models (SVM RBF for Model A & Model B)
python src/models/train_final_model.py

# 2. Train and evaluate RGB segmentation model (Random Forest multi-scale pixel classifier)
python src/models/train_segmenter.py
```
This produces three production artifacts under `models/`:
- **Model A (`models/svm_rphi_reproducer.joblib`)**: 3-feature constituent RPHI formula reproducer (`relative_depth_max_intensity`, `mask_area_pixels`, `relative_volume_proxy`).
- **Model B (`models/svm_2d_geometry.joblib`)**: 14-feature 2D mask geometry model (100% depth-free triage).
- **RGB Segmenter (`models/rgb_segmenter.joblib`)**: Multi-scale color, contrast, and gradient pixel classifier with morphological refinement.
- **Metadata Summaries (`models/model_metadata.json`, `models/segmentation_metadata.json`)**: Feature manifests, group quantile thresholds, and validation metrics.

### 2. Running Inference via Python API
Use the unified `PotholeSeverityPredictor` interface for inference on raw RGB images, binary masks, or paired mask + depth map images:

```python
import numpy as np
from src.models.predict import PotholeSeverityPredictor

# Initialize predictor (loads serialized models from models/)
predictor = PotholeSeverityPredictor()

# Mode 1: End-to-End RGB Image (RGBSegmenter + Model B)
# Accepts RGB numpy array (H, W, 3) or file path:
result_rgb = predictor.predict_rgb("path/to/road_pothole.jpg", return_mask=True)
print("Predicted Tier (from RGB):", result_rgb["predicted_tier"])
print("Pothole Detected:", result_rgb["pothole_detected"])

# Mode 2: 2D Geometry Only (Model B) - requires pre-segmented 2D binary mask
result_2d = predictor.predict_2d("path/to/pothole_mask.png")
print("Predicted Tier (2D Geometry):", result_2d["predicted_tier"])
print("Uncalibrated Decision Scores:", result_2d["decision_scores"])

# Mode 3: Full RPHI Constituent Model (Model A) - requires mask + depth map
result_depth = predictor.predict_with_depth(
    mask="path/to/pothole_mask.png",
    depth="path/to/pothole_depth.png",
)
print("Predicted Tier (Depth RPHI):", result_depth["predicted_tier"])
print("Constituent Features:", result_depth["extracted_features"])

# Unified entry point (routes automatically based on provided arguments):
# result = predictor.predict(rgb="path/to/rgb.jpg")
# result = predictor.predict(mask="path/to/mask.png", depth="path/to/depth.png")
```

---

## 🧪 Running Unit Tests

Run the full test suite (52 unit tests covering preprocessing, feature extraction, severity labels, baseline cross-validation, production inference, and RGB segmentation):
```bash
python -m unittest discover tests -v
```

---

## ⚠️ Scientific Limitations & Proper Interpretation

> [!WARNING]
> **1. Provisional Relative Index, Not Physical Severity:**
> Target tiers (*Low*, *Medium*, *High*) represent provisional relative ranking percentiles within the Pothole440 dataset population. They **do not** correspond to certified physical road condition standards, municipal maintenance urgency tiers (e.g. BMC repair criteria), or structural pavement distress ratings (e.g. IRC:82, ASTM D6433).
>
> **2. Uncalibrated Sensor Grayscale:**
> Depth map values in Pothole440 are uncalibrated 8-bit integers $[0, 190]$. Without verified sensor calibration factors, camera mounting height, pitch angle, or focal geometry, **physical cavity depth in millimeters (mm) and physical cavity volume in liters / cm³ cannot be measured or predicted.**
>
> **3. RGB Segmentation Baseline Scope:**
> The integrated `RGBSegmenter` is a fast, CPU-efficient multi-scale classical ML pixel classifier (evaluated on strictly group-disjoint validation: **Mean Dice = 0.5251**, **Mean IoU = 0.3872**). It provides a lightweight baseline and may be sensitive to pavement shadows or asphalt discoloration. It **does NOT** estimate 3D depth from RGB.
>
> **4. Uncalibrated Decision Scores:**
> Decision scores returned by the SVM model represent signed distances to the separating hyperplanes in feature space, **not** calibrated class probabilities.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
