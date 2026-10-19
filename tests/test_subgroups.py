"""
test_subgroups.py - Unit tests for subgroup analysis and Wilson score bounds.
"""

import pandas as pd
import pytest
from src.subgroup_analysis import wilson_score_interval, analyze_subgroups


def test_wilson_interval_bounds():
    # 0 errors out of 10 samples
    ci_zero = wilson_score_interval(0, 10, confidence=0.95)
    assert ci_zero["lower"] == 0.0
    assert 0.0 < ci_zero["upper"] < 0.35  # Wilson correctly reflects uncertainty even at p=0

    # 10 errors out of 10 samples
    ci_full = wilson_score_interval(10, 10, confidence=0.95)
    assert ci_full["upper"] == pytest.approx(1.0, abs=1e-5)
    assert ci_full["lower"] > 0.65


def test_analyze_subgroups_ranking_and_small_cohort_flag():
    df = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(25)],
        "y_true": ["A"] * 25,
        "y_pred": ["A"] * 10 + ["B"] * 5 + ["A"] * 10,  # 5 errors total
        "device": ["Hospital_Main"] * 20 + ["Clinic_Rural"] * 5
    })
    # Hospital_Main: 20 samples, 2 errors (10% error)
    # Clinic_Rural: 5 samples, 3 errors (60% error)
    df.loc[:19, "y_pred"] = ["A"] * 18 + ["B"] * 2
    df.loc[20:, "y_pred"] = ["A"] * 2 + ["B"] * 3

    subgroups = analyze_subgroups(df, "device", min_subgroup_size=10)
    rural = next(g for g in subgroups if g["subgroup_value"] == "Clinic_Rural")
    main = next(g for g in subgroups if g["subgroup_value"] == "Hospital_Main")

    assert rural["sample_count"] == 5
    assert rural["error_count"] == 3
    assert rural["is_unstable_small_cohort"] is True  # 5 < 10

    assert main["sample_count"] == 20
    assert main["error_count"] == 2
    assert main["is_unstable_small_cohort"] is False
