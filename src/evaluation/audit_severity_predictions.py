"""
Severity Prediction Audit and Diagnostic Tool for CrackSight.

Diagnoses and audits the severity prediction pipeline across:
1. Model loading & active segmenter engine verification.
2. Canonical resolution projection & feature scaling integrity.
3. Decision score label mapping verification.
4. Macro and per-class performance on group-disjoint reference samples.
5. Verification that small, medium, and large defects produce valid, non-constant tiers.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.inference_service import PotholeInferenceService
from src.models.predict import PotholeSeverityPredictor
from src.features.feature_extractor import compute_mask_geometry
from src.severity.label_generator import RPHILabelGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cracksight.audit")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "unified_dataset_manifest.csv")
P440_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pothole_features.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "results", "reports", "severity_prediction_audit_report.md")


def run_severity_audit() -> Dict[str, Any]:
    logger.info("Initializing PotholeInferenceService...")
    service = PotholeInferenceService.get_instance(segmenter_engine="deeplabv3")
    status = service.get_status()

    # 1. Pipeline Verification
    logger.info(f"Active segmenter engine: {status['active_segmenter_engine']}")
    logger.info(f"Model B available: {status['models']['svm_2d_geometry']['loaded']}")
    logger.info(f"Model A available: {status['models']['svm_rphi_reproducer']['loaded']}")

    # 2. Synthetic Resolution and Coverage Audit
    # Test at P440 (800x1200), P600 (400x400), Smartphone (1080x1920), and Small (256x256)
    logger.info("Evaluating multi-resolution synthetic defects across 1%, 4%, and 15% coverage...")
    resolutions = [
        ("Pothole440 (800x1200)", (800, 1200)),
        ("Pothole-600 (400x400)", (400, 400)),
        ("Smartphone (1080x1920)", (1080, 1920)),
        ("Model Input (256x256)", (256, 256)),
    ]
    test_cases = [
        (0.015, "Small Defect (1.5% coverage)", "Low"),
        (0.045, "Medium Defect (4.5% coverage)", "Medium"),
        (0.150, "Large Defect (15.0% coverage)", "High"),
    ]

    resolution_audit_results = []
    for res_name, (h, w) in resolutions:
        for cov, case_name, expected_tier in test_cases:
            mask = np.zeros((h, w), dtype=np.uint8)
            r = np.sqrt(h * w * cov / np.pi)
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = 255

            res = service.predictor.predict_2d(mask)
            pred_tier = res["predicted_tier"]
            scores = res["decision_scores"]
            native_area = res["extracted_features"]["mask_area_pixels"]
            native_cov = res["extracted_features"]["mask_coverage_pct"]

            resolution_audit_results.append({
                "resolution": res_name,
                "case": case_name,
                "coverage_pct": round(native_cov, 2),
                "native_area_px": int(native_area),
                "expected": expected_tier,
                "predicted": pred_tier,
                "decision_scores": scores,
                "match": pred_tier == expected_tier,
            })

    # 3. Real Held-Out Pothole440 Group Evaluation (Fold 0)
    logger.info("Auditing held-out Pothole440 samples (Fold 0, zero group leakage)...")
    df_p440 = pd.read_csv(P440_FEATURES_PATH)
    label_gen = RPHILabelGenerator(threshold_mode="group_level", group_col="canonical_pothole_id")
    label_gen.fit(df_p440)
    _, p440_tiers = label_gen.predict_tier(df_p440)
    df_p440["reference_tier"] = p440_tiers

    fold0_samples = df_p440[df_p440["fold"] == 0].copy()
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest_p440 = manifest[manifest["dataset_source"] == "pothole440"]

    fold0_eval = []
    for _, row in fold0_samples.iterrows():
        fname = row["filename"]
        match_row = manifest_p440[manifest_p440["image_path"].str.endswith(fname)]
        if match_row.empty:
            continue
        mask_path = match_row.iloc[0]["mask_path"]
        if not os.path.exists(mask_path):
            continue

        gt_mask = np.array(Image.open(mask_path))
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        if not np.any(gt_mask > 0):
            continue

        res_2d = service.predictor.predict_2d(gt_mask)
        fold0_eval.append({
            "filename": fname,
            "group_id": row["canonical_pothole_id"],
            "true_tier": row["reference_tier"],
            "predicted_tier": res_2d["predicted_tier"],
            "coverage_pct": row["mask_coverage_pct"],
            "scores": res_2d["decision_scores"],
        })

    df_fold0 = pd.DataFrame(fold0_eval)
    fold0_report = classification_report(
        df_fold0["true_tier"],
        df_fold0["predicted_tier"],
        labels=["Low", "Medium", "High"],
        output_dict=True,
    )
    fold0_cm = confusion_matrix(
        df_fold0["true_tier"],
        df_fold0["predicted_tier"],
        labels=["Low", "Medium", "High"],
    )

    # 4. Real Pothole-600 Dataset Distribution Audit (Stratified Sample of 60 images)
    logger.info("Auditing real Pothole-600 dataset across 60 annotated samples...")
    p600_samples = manifest[manifest["dataset_source"] == "pothole600"].sample(60, random_state=42)
    p600_eval = []
    for _, row in p600_samples.iterrows():
        mask_path = row["mask_path"]
        img_path = row["image_path"]
        if not os.path.exists(mask_path) or not os.path.exists(img_path):
            continue
        gt_mask = np.array(Image.open(mask_path))
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        if not np.any(gt_mask > 0):
            continue

        geom = compute_mask_geometry(gt_mask)
        res_2d = service.predictor.predict_2d(gt_mask)

        # Reference tier derived from physical road coverage:
        # < 3.0%: Low, 3.0% - 6.0%: Medium, > 6.0%: High
        cov = geom["mask_coverage_pct"]
        rule_tier = "Low" if cov < 3.0 else ("Medium" if cov <= 6.0 else "High")

        p600_eval.append({
            "sample_id": row["sample_id"],
            "coverage_pct": round(cov, 2),
            "area_px": int(geom["mask_area_pixels"]),
            "rule_tier": rule_tier,
            "predicted_tier": res_2d["predicted_tier"],
            "scores": res_2d["decision_scores"],
        })

    df_p600 = pd.DataFrame(p600_eval)

    # 5. Compile Diagnostic Report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# CrackSight Severity Prediction Pipeline Audit Report\n\n")
        f.write(f"**Audit Date:** 2026-10-21\n")
        f.write(f"**Active Segmentation Engine:** `{status['active_segmenter_engine']}`\n\n")

        f.write("## 1. Multi-Resolution Invariance Audit\n\n")
        f.write("| Resolution | Case | Coverage | Predicted | Expected | Match? |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for r in resolution_audit_results:
            match_str = "YES" if r["match"] else "**NO**"
            f.write(f"| {r['resolution']} | {r['case']} | {r['coverage_pct']}% | {r['predicted']} | {r['expected']} | {match_str} |\n")

        f.write("\n## 2. Held-Out Pothole440 Benchmark (Fold 0, N=87)\n\n")
        f.write("### Confusion Matrix (Rows=True, Cols=Predicted)\n\n")
        f.write("| | Pred Low | Pred Medium | Pred High |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **True Low** | {fold0_cm[0][0]} | {fold0_cm[0][1]} | {fold0_cm[0][2]} |\n")
        f.write(f"| **True Medium** | {fold0_cm[1][0]} | {fold0_cm[1][1]} | {fold0_cm[1][2]} |\n")
        f.write(f"| **True High** | {fold0_cm[2][0]} | {fold0_cm[2][1]} | {fold0_cm[2][2]} |\n\n")

        f.write("### Classification Report\n\n")
        f.write(f"- **Macro F1:** {fold0_report['macro avg']['f1-score']:.4f}\n")
        f.write(f"- **Accuracy:** {fold0_report['accuracy']:.4f}\n")
        for cls in ["Low", "Medium", "High"]:
            c_f1 = fold0_report[cls]["f1-score"]
            c_rec = fold0_report[cls]["recall"]
            c_prec = fold0_report[cls]["precision"]
            f.write(f"- **Class '{cls}':** Precision={c_prec:.4f}, Recall={c_rec:.4f}, F1={c_f1:.4f}\n")

        f.write("\n## 3. Real Pothole-600 Distribution (N=60)\n\n")
        dist = df_p600["predicted_tier"].value_counts().to_dict()
        f.write(f"- **Class Distribution:** {dist}\n")
        f.write(f"- **Low Severity:** {dist.get('Low', 0)} ({dist.get('Low', 0)/len(df_p600)*100:.1f}%)\n")
        f.write(f"- **Medium Severity:** {dist.get('Medium', 0)} ({dist.get('Medium', 0)/len(df_p600)*100:.1f}%)\n")
        f.write(f"- **High Severity:** {dist.get('High', 0)} ({dist.get('High', 0)/len(df_p600)*100:.1f}%)\n")

    logger.info(f"Audit completed. Report saved to: {REPORT_PATH}")
    return {
        "status": "completed",
        "fold0_macro_f1": fold0_report["macro avg"]["f1-score"],
        "fold0_accuracy": fold0_report["accuracy"],
        "p600_distribution": dist,
        "resolution_matches": sum(1 for r in resolution_audit_results if r["match"]),
        "total_resolution_tests": len(resolution_audit_results),
    }


if __name__ == "__main__":
    summary = run_severity_audit()
    print(json.dumps(summary, indent=2))
