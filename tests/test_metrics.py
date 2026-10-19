"""
test_metrics.py - Unit tests for metrics calculation.
"""

import pandas as pd
import pytest
from src.metrics import compute_aggregate_metrics, compute_per_class_metrics, compute_ece


def test_aggregate_metrics_hand_checked():
    # 4 samples: 3 correct, 1 incorrect -> accuracy = 0.75, error_rate = 0.25
    df = pd.DataFrame({
        "sample_id": ["1", "2", "3", "4"],
        "y_true": ["Cat", "Dog", "Cat", "Dog"],
        "y_pred": ["Cat", "Dog", "Dog", "Dog"],  # 3rd is wrong
        "confidence": [0.90, 0.80, 0.85, 0.70]
    })
    agg = compute_aggregate_metrics(df)
    assert agg["overall_accuracy"] == 0.75
    assert agg["overall_error_rate"] == 0.25
    assert agg["total_samples"] == 4
    assert agg["total_errors"] == 1
    assert agg["high_confidence_errors"] == 1  # 3rd sample has conf 0.85


def test_per_class_metrics():
    df = pd.DataFrame({
        "sample_id": ["1", "2", "3", "4"],
        "y_true": ["Cat", "Dog", "Cat", "Dog"],
        "y_pred": ["Cat", "Dog", "Dog", "Dog"],
    })
    per_class = compute_per_class_metrics(df)
    cat_row = next(r for r in per_class if r["class_name"] == "Cat")
    dog_row = next(r for r in per_class if r["class_name"] == "Dog")

    # Cat: 2 true samples, 1 TP, 1 FN -> recall = 0.50
    assert cat_row["support"] == 2
    assert cat_row["recall_sensitivity"] == 0.50
    assert cat_row["precision"] == 1.0  # Only predicted Cat once, and it was correct

    # Dog: 2 true samples, 2 TP, 0 FN -> recall = 1.0
    assert dog_row["support"] == 2
    assert dog_row["recall_sensitivity"] == 1.0


def test_ece_perfect_calibration():
    # When accuracy matches confidence exactly
    acc = pd.Series([1, 1, 0, 0]).values
    conf = pd.Series([1.0, 1.0, 0.0, 0.0]).values
    ece = compute_ece(acc, conf, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)
