# Baseline Machine Learning and Feature Importance Evaluation Report

**Project:** AI-Based Pothole Depth and Physical Damage Severity Assessment  
**Module:** `src/models/train_baselines.py` & `src/evaluation/evaluate_baselines.py`  
**Date:** October 14, 2026  
**Status:** Complete, Replicated & Programmatically Verified  

---

## 1. Executive Summary & Experimental Framework

This evaluation investigates the capability of traditional machine learning classifiers to predict provisional **Relative Pothole Hazard Index (RPHI)** tiers across the **434 valid samples** in the Pothole440 dataset.

To prevent misleading claims of physical severity prediction, two strictly separated experiments were conducted:
1. **Experiment A (RPHI Formula Reproduction):** Evaluates how effectively models can learn and reproduce the composite relative-index decision boundaries from its 3 constituent features (`relative_depth_max_intensity`, `mask_area_pixels`, `relative_volume_proxy`).
2. **Experiment B (2D Geometry Only):** Evaluates how closely **2D mask geometry alone** (excluding all depth and volume measurements) can approximate the composite hazard tiers.

### Rigorous Target Protocol (Audit Correction Applied):
- In each outer GroupKFold split, an `RPHILabelGenerator` is fitted **strictly on the training partition** (`fold != k`).
- Both `y_train` and `y_val` are generated using that **same training-fitted generator**.
- No validation data is used to fit normalization parameters or tier thresholds.
- Preprocessing (StandardScaler) and model parameters are fitted solely on training folds.
- Training and validation partitions remain 100% group-disjoint (no physical pothole appears in both partitions).

### Key Findings:
- **Experiment A (Formula Reproduction):** **SVM (RBF)** achieves the strongest performance with **0.8879 Macro-F1** and **0.8938 Balanced Accuracy**, significantly outperforming the Majority Baseline (0.1619 Macro-F1). This confirms that traditional classifiers can accurately reconstruct the composite formula boundaries.
- **Experiment B (2D Geometry Only):** Excluding all depth and volume measurements reduces performance substantially. **SVM (RBF)** achieves **0.6955 Macro-F1** (0.7054 Balanced Accuracy). While 2D geometry captures preliminary surface triage signal far above chance (0.1619), it suffers a steep drop compared to 3D-informed features.
- **Scientific Disclaimer:** High accuracy in Experiment A reflects mathematical reconstruction of an analytical index, **NOT** certified physical severity prediction or real-world structural road safety.

---

## 2. Per-Fold Target Class Distributions & Fitted Cutoffs

The target variable for each outer fold is generated dynamically by fitting group-level 33rd/67th percentile thresholds strictly on training groups (`fold != k`). Class distributions and cutoffs across all 5 folds are confirmed non-empty:

| Fold Index | Training Cutoff (Low / High) | Val Low | Val Med | Val High | Total Val Samples | Val Pothole Groups |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 0.4034 / 0.4811 | 22 | 26 | 40 | 88 | 20 |
| Fold 1 | 0.4381 / 0.5279 | 9 | 28 | 50 | 87 | 20 |
| Fold 2 | 0.4077 / 0.4916 | 32 | 34 | 21 | 87 | 20 |
| Fold 3 | 0.4276 / 0.5075 | 48 | 19 | 19 | 86 | 19 |
| Fold 4 | 0.3598 / 0.4430 | 33 | 25 | 28 | 86 | 19 |
| **All Folds** | **Dynamic** | **144** | **132** | **158** | **434** | **98** |

> **Class Distribution Observation:**  
> All 3 classes ('Low', 'Medium', 'High') are represented in every training split (minimum 96 samples per class in training) and every validation split. Folds 0, 1, and 4 have slightly higher proportions of High-hazard samples, while Fold 3 has a higher concentration of Low-hazard samples, reflecting natural cluster variation across physical road inspection dates.

---

## 3. Experiment A Results: RPHI Formula Reproduction

- **Input Features (3):** `relative_depth_max_intensity`, `mask_area_pixels`, `relative_volume_proxy`
- **Target:** Dynamically generated `y_val` from training-fitted RPHI generator
- **Objective:** Measures model capacity to reproduce provisional formula-derived labels from constituent inputs.

| Model | Macro-F1 (Mean $\pm$ Std) | Balanced Accuracy | Accuracy | Precision (L / M / H) | Recall (L / M / H) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | **0.1619 $\pm$ 0.0487** | 0.3333 $\pm$ 0.0000 | 0.3316 | 0.00 / 0.06 / 0.27 | 0.00 / 0.20 / 0.80 |
| **Random Forest** | **0.8455 $\pm$ 0.0519** | 0.8509 $\pm$ 0.0519 | 0.8686 | 0.82 / 0.80 / 0.92 | 0.85 / 0.75 / 0.96 |
| **Gradient Boosting** | **0.8709 $\pm$ 0.0371** | 0.8635 $\pm$ 0.0487 | 0.8872 | 0.91 / 0.82 / 0.94 | 0.85 / 0.77 / 0.97 |
| **SVM (RBF)** | **0.8879 $\pm$ 0.0761** | 0.8938 $\pm$ 0.0856 | 0.8991 | 1.00 / 0.78 / 0.97 | 0.78 / 0.98 / 0.92 |
| **KNN (k=5)** | **0.8764 $\pm$ 0.0985** | 0.8907 $\pm$ 0.0971 | 0.8781 | 0.92 / 0.76 / 0.98 | 0.83 / 0.90 / 0.94 |

### Aggregated Confusion Matrices (Experiment A)

**Majority Baseline:**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             0               22              122            
Actual Med             0               26              106            
Actual High            0               40              118            
```

**Random Forest:**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             128             16              0              
Actual Med             21              99              12             
Actual High            0               8               150            
```

**Gradient Boosting:**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             128             16              0              
Actual Med             16              104             12             
Actual High            0               5               153            
```

**SVM (RBF):**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             120             24              0              
Actual Med             0               128             4              
Actual High            0               16              142            
```

**KNN (k=5):**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             115             29              0              
Actual Med             8               120             4              
Actual High            0               12              146            
```

---

## 4. Experiment B Results: 2D Geometry Only

- **Input Features (14):** `mask_area_pixels`, `mask_coverage_ratio`, `mask_coverage_pct`, `perimeter_pixels`, `circularity`, `compactness`, `bbox_x_min`, `bbox_x_max`, `bbox_y_min`, `bbox_y_max`, `bbox_width`, `bbox_height`, `bbox_aspect_ratio`, `bbox_extent`
- **Excluded Features:** All depth-map and volume proxy features (100% excluded).
- **Target:** Dynamically generated `y_val` from training-fitted RPHI generator
- **Objective:** Measures how well 2D surface geometry alone approximates the composite relative hazard tiers when depth sensing is unavailable.

| Model | Macro-F1 (Mean $\pm$ Std) | Balanced Accuracy | Accuracy | Precision (L / M / H) | Recall (L / M / H) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | **0.1619 $\pm$ 0.0487** | 0.3333 $\pm$ 0.0000 | 0.3316 | 0.00 / 0.06 / 0.27 | 0.00 / 0.20 / 0.80 |
| **Random Forest** | **0.6490 $\pm$ 0.1356** | 0.6701 $\pm$ 0.1238 | 0.6726 | 0.73 / 0.47 / 0.85 | 0.69 / 0.59 / 0.73 |
| **Gradient Boosting** | **0.6372 $\pm$ 0.1239** | 0.6544 $\pm$ 0.1168 | 0.6635 | 0.64 / 0.48 / 0.84 | 0.71 / 0.52 / 0.74 |
| **SVM (RBF)** | **0.6955 $\pm$ 0.1067** | 0.7054 $\pm$ 0.0936 | 0.7281 | 0.68 / 0.55 / 0.88 | 0.74 / 0.57 / 0.80 |
| **KNN (k=5)** | **0.5775 $\pm$ 0.1292** | 0.6077 $\pm$ 0.1177 | 0.6249 | 0.68 / 0.36 / 0.66 | 0.67 / 0.31 / 0.84 |

### Aggregated Confusion Matrices (Experiment B)

**Majority Baseline:**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             0               22              122            
Actual Med             0               26              106            
Actual High            0               40              118            
```

**Random Forest:**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             100             36              8              
Actual Med             36              84              12             
Actual High            0               50              108            
```

**Gradient Boosting:**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             104             32              8              
Actual Med             48              72              12             
Actual High            0               46              112            
```

**SVM (RBF):**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             113             29              2              
Actual Med             40              80              12             
Actual High            4               31              123            
```

**KNN (k=5):**
```text
                 Predicted Low   Predicted Med   Predicted High
Actual Low             99              33              12             
Actual Med             34              40              58             
Actual High            7               19              132            
```

---

## 5. Comparative Analysis: Experiment A vs. Experiment B

| Model | Exp A Macro-F1 (Full Formula) | Exp B Macro-F1 (2D Geometry Only) | Absolute $\Delta$ F1 | Exp B Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.1619 | 0.1619 | **+0.0000** | 0.3333 |
| **Random Forest** | 0.8455 | 0.6490 | **-0.1965** | 0.6701 |
| **Gradient Boosting** | 0.8709 | 0.6372 | **-0.2337** | 0.6544 |
| **SVM (RBF)** | 0.8879 | 0.6955 | **-0.1924** | 0.7054 |
| **KNN (k=5)** | 0.8764 | 0.5775 | **-0.2989** | 0.6077 |

### Key Insights:
1. **Depth Information is Indispensable:** Excluding 3D depth and relative volume causes a substantial drop in Macro-F1 across all classifiers (typically 19 to 30 percentage points).
2. **2D Geometry Provides Meaningful Signal Above Chance:** Despite the drop, SVM (RBF) with 2D geometry (0.6955 Macro-F1) substantially outperforms the Majority Baseline (0.1619 Macro-F1). 2D footprint geometry can serve as a coarse preliminary screening tool, but cannot reliably separate Moderate from Severe potholes.
3. **Confusion Pattern in Experiment B:** Most errors in Experiment B occur between adjacent tiers (Low $\leftrightarrow$ Medium and Medium $\leftrightarrow$ High). Severe deep potholes that happen to have small surface areas are misclassified as Low or Medium when depth is absent.

---

## 6. Feature Importance & Associational Interpretation

> [!IMPORTANT]
> **Associational vs. Causal Disclaimer:**  
> Feature importance scores derived from Random Forest and Gradient Boosting reflect mathematical split utility and correlation within this dataset. They do **NOT** imply physical causality.

### A. Experiment A Feature Importance (Tree Mean Decrease in Impurity)

| Feature | Random Forest Importance | Gradient Boosting Importance | Physical Role in Formula |
| :--- | :---: | :---: | :--- |
| `relative_depth_max_intensity` | 0.1983 | 0.0480 | Constituent component ($w=0.4$) |
| `mask_area_pixels` | 0.3368 | 0.0372 | Constituent component ($w=0.3$) |
| `relative_volume_proxy` | 0.4649 | 0.9148 | Constituent component ($w=0.3$) |

### B. Experiment B Feature Importance (Top 2D Geometry Descriptors)

| Feature | Random Forest Importance | Gradient Boosting Importance | Description |
| :--- | :---: | :---: | :--- |
| `mask_coverage_ratio` | 0.1677 | 0.1307 | 2D Geometric Descriptor |
| `mask_area_pixels` | 0.1540 | 0.2581 | 2D Geometric Descriptor |
| `mask_coverage_pct` | 0.1320 | 0.1971 | 2D Geometric Descriptor |
| `bbox_width` | 0.0922 | 0.0420 | 2D Geometric Descriptor |
| `bbox_extent` | 0.0696 | 0.0866 | 2D Geometric Descriptor |
| `circularity` | 0.0693 | 0.0399 | 2D Geometric Descriptor |
| `bbox_height` | 0.0687 | 0.0306 | 2D Geometric Descriptor |
| `perimeter_pixels` | 0.0673 | 0.0588 | 2D Geometric Descriptor |
| `compactness` | 0.0584 | 0.0476 | 2D Geometric Descriptor |
| `bbox_aspect_ratio` | 0.0498 | 0.0858 | 2D Geometric Descriptor |
| `bbox_x_max` | 0.0229 | 0.0095 | 2D Geometric Descriptor |
| `bbox_x_min` | 0.0210 | 0.0080 | 2D Geometric Descriptor |
| `bbox_y_max` | 0.0139 | 0.0025 | 2D Geometric Descriptor |
| `bbox_y_min` | 0.0134 | 0.0029 | 2D Geometric Descriptor |

---

## 7. Critical Evaluation Limitations

1. **Formula Reproduction Limitation:** The high performance in Experiment A reflects a model learning an analytical decision boundary constructed from the same features. It cannot be cited as validation of physical severity prediction.
2. **Uncalibrated Sensor Units:** The underlying depth maps are uncalibrated 8-bit integers $[0, 190]$. These models cannot predict whether a pothole meets physical repair thresholds (e.g. IRC:82 $>50\text{ mm}$ or BMC $>40\text{ mm}$).
3. **Zero Road Context:** The models evaluate single-frame segmentations without traffic density, bus route status, or rainfall exposure context.

---

## 8. Artifact Manifest

- **Model Architecture Module:** `src/models/train_baselines.py`
- **Evaluation Module:** `src/evaluation/evaluate_baselines.py`
- **Test Suite:** `tests/test_baseline_evaluation.py`
- **Metrics Summary (Exp A):** `results/reports/baseline_metrics_exp_a.csv`
- **Metrics Summary (Exp B):** `results/reports/baseline_metrics_exp_b.csv`
- **Confusion Matrices (Exp A):** `results/reports/confusion_matrix_exp_a.csv`
- **Confusion Matrices (Exp B):** `results/reports/confusion_matrix_exp_b.csv`
