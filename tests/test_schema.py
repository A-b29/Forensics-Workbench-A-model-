"""
test_schema.py - Unit tests for input dataset schema validation.
"""

import pandas as pd
import pytest
from src.schema import validate_dataframe


def test_valid_dataframe():
    df = pd.DataFrame({
        "sample_id": ["S1", "S2", "S3"],
        "y_true": ["A", "B", "A"],
        "y_pred": ["A", "B", "B"],
        "confidence": [0.95, 0.88, 0.72],
        "scanner": ["GE", "Siemens", "GE"]
    })
    res = validate_dataframe(df)
    assert res.is_valid is True
    assert res.total_samples == 3
    assert res.num_classes == 2
    assert res.has_confidence is True
    assert "scanner" in res.metadata_columns
    assert len(res.error_messages) == 0


def test_missing_required_column():
    df = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "y_true": ["A", "B"]
        # missing y_pred
    })
    res = validate_dataframe(df)
    assert res.is_valid is False
    assert any("y_pred" in err for err in res.error_messages)


def test_duplicate_sample_ids():
    df = pd.DataFrame({
        "sample_id": ["S1", "S1"],
        "y_true": ["A", "B"],
        "y_pred": ["A", "A"]
    })
    res = validate_dataframe(df)
    assert res.is_valid is False
    assert any("duplicate" in err for err in res.error_messages)


def test_invalid_confidence_range():
    df = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "y_true": ["A", "B"],
        "y_pred": ["A", "B"],
        "confidence": [0.5, 1.5]  # > 1.0
    })
    res = validate_dataframe(df)
    assert res.is_valid is False
    assert any("bounded between 0.0 and 1.0" in err for err in res.error_messages)
