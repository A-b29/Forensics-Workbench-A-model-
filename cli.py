"""
cli.py - Headless evaluation CLI for Model Forensics Workbench.
"""

import argparse
from pathlib import Path
import pandas as pd

from src.schema import validate_dataframe
from src.metrics import compute_aggregate_metrics, compute_per_class_metrics
from src.subgroup_analysis import analyze_subgroups
from src.report import generate_forensics_markdown, export_forensics_json


def run_forensics(csv_path: Path, output_dir: Path, min_size: int = 15):
    print(f"=== Running Model Forensics Audit on {csv_path.name} ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    val = validate_dataframe(df)

    if not val.is_valid:
        print("Dataset Schema Validation FAILED:")
        for err in val.error_messages:
            print(f"  - {err}")
        raise ValueError("Invalid schema")

    print(f"Validated {val.total_samples} samples across {val.num_classes} classes.")
    print(f"Available metadata columns: {val.metadata_columns}")

    # 1. Compute aggregate and per-class metrics
    agg = compute_aggregate_metrics(df)
    per_class = compute_per_class_metrics(df)
    print(f"Overall Accuracy: {agg['overall_accuracy']:.2%}, Error Rate: {agg['overall_error_rate']:.2%}")

    # 2. Compute subgroup analysis across all metadata columns
    subgroups = {}
    for col in val.metadata_columns:
        print(f"Analyzing subgroup disparities along '{col}'...")
        subgroups[col] = analyze_subgroups(df, col, min_subgroup_size=min_size)

    # 3. Export reports
    md_file = generate_forensics_markdown(agg, per_class, subgroups, output_dir / "forensics_report.md")
    json_payload = {
        "dataset_name": csv_path.name,
        "aggregate_metrics": agg,
        "per_class_metrics": per_class,
        "subgroup_analysis": subgroups,
    }
    json_file = export_forensics_json(json_payload, output_dir / "forensics_findings.json")

    print(f"\nAudit complete!")
    print(f"-> Markdown report: {md_file}")
    print(f"-> JSON findings:   {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Model Forensics Workbench CLI")
    parser.add_argument("--csv", type=str, default=None, help="Path to predictions CSV.")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports.")
    parser.add_argument("--min-size", type=int, default=15, help="Minimum cohort size for statistical stability.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    if args.csv is None:
        csv_path = project_root / "examples" / "sample_predictions.csv"
    else:
        csv_path = Path(args.csv)

    run_forensics(csv_path, Path(args.output_dir), min_size=args.min_size)


if __name__ == "__main__":
    main()
