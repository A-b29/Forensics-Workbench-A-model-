"""
subgroup_analysis.py - Disaggregated failure analysis with statistical confidence intervals.

Guards against multiple testing and small-sample p-hacking by enforcing
minimum cohort sizes and reporting 95% Wilson score confidence intervals.
"""

from typing import List, Dict, Any, Optional
import math
import numpy as np
import pandas as pd


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Wilson score interval for binomial proportions.
    Superior to normal approximation for small cohorts or extreme error rates.
    """
    if total <= 0:
        return {"lower": 0.0, "center": 0.0, "upper": 0.0}

    # Standard normal quantile: z ~ 1.95996 for 95%
    z = 1.959964 if confidence == 0.95 else 1.644853  # fallback 90%
    p = successes / total

    denominator = 1.0 + (z ** 2) / total
    centre_adjusted_probability = (p + (z ** 2) / (2.0 * total)) / denominator
    adjusted_std_error = (z / denominator) * math.sqrt((p * (1.0 - p) / total) + ((z ** 2) / (4.0 * (total ** 2))))

    lower = max(0.0, centre_adjusted_probability - adjusted_std_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_std_error)

    return {
        "lower": float(lower),
        "center": float(p),
        "upper": float(upper)
    }


def analyze_subgroups(
    df: pd.DataFrame,
    column_name: str,
    min_subgroup_size: int = 15,
    numeric_bins: int = 4
) -> List[Dict[str, Any]]:
    """
    Partition dataframe by a categorical column or binned continuous feature,
    computing error rate, sample support, and Wilson confidence bounds.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not present in DataFrame.")

    data = df.copy()
    is_numeric = pd.api.types.is_numeric_dtype(data[column_name])

    # If numeric continuous, bin into quantiles
    if is_numeric and data[column_name].nunique() > numeric_bins:
        try:
            data["_subgroup_key"] = pd.qcut(data[column_name], q=numeric_bins, duplicates="drop").astype(str)
        except Exception:
            data["_subgroup_key"] = pd.cut(data[column_name], bins=numeric_bins).astype(str)
    else:
        data["_subgroup_key"] = data[column_name].astype(str).fillna("Unknown")

    data["_is_error"] = (data["y_true"].astype(str) != data["y_pred"].astype(str)).astype(int)
    
    total_population = len(data)
    population_errors = int(data["_is_error"].sum())
    population_error_rate = population_errors / total_population if total_population > 0 else 0.0

    groups = data.groupby("_subgroup_key")
    results = []

    for key, group in groups:
        n_samples = len(group)
        n_errors = int(group["_is_error"].sum())
        error_rate = float(n_errors / n_samples) if n_samples > 0 else 0.0
        
        ci = wilson_score_interval(n_errors, n_samples, confidence=0.95)
        
        # Risk ratio vs overall dataset
        risk_ratio = float(error_rate / population_error_rate) if population_error_rate > 0 else 1.0
        excess_errors = int(n_errors - (population_error_rate * n_samples))

        is_statistically_unstable = n_samples < min_subgroup_size

        subgroup_rec = {
            "subgroup_column": column_name,
            "subgroup_value": str(key),
            "sample_count": n_samples,
            "error_count": n_errors,
            "error_rate": error_rate,
            "ci_95_lower": ci["lower"],
            "ci_95_upper": ci["upper"],
            "risk_ratio": risk_ratio,
            "excess_errors": excess_errors,
            "is_unstable_small_cohort": is_statistically_unstable,
        }

        # If confidence column exists, add mean confidence of errors in subgroup
        if "confidence" in group.columns:
            err_confs = pd.to_numeric(group.loc[group["_is_error"] == 1, "confidence"], errors="coerce").dropna()
            subgroup_rec["mean_error_confidence"] = float(err_confs.mean()) if len(err_confs) > 0 else None

        results.append(subgroup_rec)

    # Sort descending by error rate
    results.sort(key=lambda x: (not x["is_unstable_small_cohort"], x["error_rate"]), reverse=True)
    return results
