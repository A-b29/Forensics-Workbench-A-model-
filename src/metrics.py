"""
metrics.py - Aggregate and disaggregated metrics engine for Model Forensics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    brier_score_loss
)


def compute_aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall classification performance and confidence dynamics."""
    y_true = df["y_true"].astype(str).values
    y_pred = df["y_pred"].astype(str).values
    classes = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    conf_mat = confusion_matrix(y_true, y_pred, labels=classes)
    
    is_correct = (y_true == y_pred)
    total_samples = len(df)
    total_errors = int((~is_correct).sum())
    error_rate = float(total_errors / total_samples) if total_samples > 0 else 0.0

    res: Dict[str, Any] = {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "overall_accuracy": acc,
        "overall_error_rate": error_rate,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "classes": classes,
        "confusion_matrix": conf_mat.tolist(),
    }

    # Confidence dynamics if available
    if "confidence" in df.columns and pd.to_numeric(df["confidence"], errors="coerce").notnull().all():
        confs = df["confidence"].astype(float).values
        correct_confs = confs[is_correct]
        error_confs = confs[~is_correct]

        res["mean_conf_correct"] = float(np.mean(correct_confs)) if len(correct_confs) > 0 else None
        res["mean_conf_incorrect"] = float(np.mean(error_confs)) if len(error_confs) > 0 else None
        
        # High confidence errors (> 0.80)
        high_conf_errors = int((~is_correct & (confs >= 0.80)).sum())
        res["high_confidence_errors"] = high_conf_errors
        res["high_confidence_error_rate"] = float(high_conf_errors / total_errors) if total_errors > 0 else 0.0

        # Calibration: ECE with 10 bins
        res["ece"] = compute_ece(is_correct.astype(int), confs, n_bins=10)

    return res


def compute_per_class_metrics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute precision, recall, specificity, and error counts for each class."""
    y_true = df["y_true"].astype(str).values
    y_pred = df["y_pred"].astype(str).values
    classes = sorted(list(set(y_true) | set(y_pred)))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class = []
    for i, cls_name in enumerate(classes):
        cls_mask_true = (y_true == cls_name)
        cls_mask_pred = (y_pred == cls_name)

        tp = int((cls_mask_true & cls_mask_pred).sum())
        fn = int((cls_mask_true & ~cls_mask_pred).sum())
        fp = int((~cls_mask_true & cls_mask_pred).sum())
        tn = int((~cls_mask_true & ~cls_mask_pred).sum())

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        class_error_rate = float(fn / support[i]) if support[i] > 0 else 0.0

        per_class.append({
            "class_name": cls_name,
            "support": int(support[i]),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "precision": float(prec[i]),
            "recall_sensitivity": float(rec[i]),
            "specificity": specificity,
            "f1_score": float(f1[i]),
            "class_error_rate": class_error_rate,
        })

    return per_class


def compute_ece(accuracy_arr: np.ndarray, confidence_arr: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) via equal-width confidence binning."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidence_arr)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidence_arr > bin_lower) & (confidence_arr <= bin_upper) if i > 0 else (confidence_arr >= bin_lower) & (confidence_arr <= bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(accuracy_arr[in_bin]))
            bin_conf = float(np.mean(confidence_arr[in_bin]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)
