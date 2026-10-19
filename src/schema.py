"""
schema.py - Input validation schema for Model Forensics Workbench.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class PredictionRecord(BaseModel):
    sample_id: str
    y_true: str
    y_pred: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetValidationResult(BaseModel):
    is_valid: bool
    total_samples: int
    num_classes: int
    classes: List[str]
    has_confidence: bool
    has_image_paths: bool
    metadata_columns: List[str]
    error_messages: List[str] = Field(default_factory=list)


def validate_dataframe(df: pd.DataFrame) -> DatasetValidationResult:
    """
    Validate an ingested DataFrame for Model Forensics.
    Required columns: sample_id, y_true, y_pred.
    Optional columns: confidence, image_path.
    All other columns are captured as metadata attributes.
    """
    errors: List[str] = []

    if df.empty:
        return DatasetValidationResult(
            is_valid=False,
            total_samples=0,
            num_classes=0,
            classes=[],
            has_confidence=False,
            has_image_paths=False,
            metadata_columns=[],
            error_messages=["The uploaded file is empty."]
        )

    # Check required columns
    required = ["sample_id", "y_true", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
        return DatasetValidationResult(
            is_valid=False,
            total_samples=len(df),
            num_classes=0,
            classes=[],
            has_confidence=False,
            has_image_paths=False,
            metadata_columns=[],
            error_messages=errors
        )

    # Validate uniqueness of sample_id
    if df["sample_id"].duplicated().any():
        dup_count = int(df["sample_id"].duplicated().sum())
        errors.append(f"sample_id column contains {dup_count} duplicate values; sample IDs must be unique.")

    # Validate non-null truth and prediction
    if df["y_true"].isnull().any() or df["y_pred"].isnull().any():
        errors.append("Columns 'y_true' and 'y_pred' must not contain missing/null values.")

    # Check confidence validity if present
    has_conf = "confidence" in df.columns
    if has_conf:
        # Check non-numeric or out-of-range
        numeric_conf = pd.to_numeric(df["confidence"], errors="coerce")
        if numeric_conf.isnull().any():
            errors.append("Confidence column contains non-numeric or NaN values.")
        elif (numeric_conf < 0.0).any() or (numeric_conf > 1.0).any():
            errors.append("Confidence values must be bounded between 0.0 and 1.0.")

    # Collect metadata columns
    reserved = {"sample_id", "y_true", "y_pred", "confidence", "image_path"}
    metadata_cols = [c for c in df.columns if c not in reserved]

    # Collect class taxonomy
    all_classes = sorted(list(set(df["y_true"].astype(str).unique()) | set(df["y_pred"].astype(str).unique())))

    is_valid = len(errors) == 0
    return DatasetValidationResult(
        is_valid=is_valid,
        total_samples=len(df),
        num_classes=len(all_classes),
        classes=all_classes,
        has_confidence=has_conf,
        has_image_paths="image_path" in df.columns,
        metadata_columns=metadata_cols,
        error_messages=errors
    )
