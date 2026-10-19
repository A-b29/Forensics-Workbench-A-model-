# Model Forensics Workbench

An evaluation platform that uncovers hidden failure modes, demographic disparities, and uncalibrated errors rather than relying solely on aggregate accuracy metrics.

> **Research Prototype Disclaimer:** Identified correlations between metadata and error rates are observational hypotheses for developer investigation. They do not constitute causal proof or clinical certification.

---

## Features
- **Pydantic Schema Validation:** Enforces uniqueness of `sample_id`, validity of `y_true`/`y_pred`, and bounded confidence.
- **Wilson Score 95% Confidence Intervals:** Robust error rate intervals that remain mathematically sound even on tiny cohorts or zero-error classes.
- **Small-Cohort Stability Guard:** Flags small bins ($N < N_{min}$) to prevent p-hacking and over-interpretation of random noise.
- **High-Confidence Error Audit:** Isolates severe silent failures where the model was misclassified with certainty $\ge 80\%$.
- **Image Proxy Quality Extraction:** Computes Laplacian variance (blur proxy), RMS contrast, and spatial entropy.
- **Interactive UI & Headless CLI:** Streamlit web lab with Plotly visualizations and batch evaluation CLI.

---

## Quickstart

### 1. Run Unit Tests
```bash
pytest tests/ -v
```

### 2. Run CLI Audit
```bash
python cli.py --csv examples/sample_predictions.csv --output-dir reports/
```

### 3. Launch Streamlit Visual Workbench
```bash
streamlit run app.py
```

---

## File Structure
```text
model_forensics_workbench/
├── app.py                      # Streamlit interactive visualizer
├── cli.py                      # Headless batch evaluation CLI
├── DECISIONS.md                # Cunning and architectural decisions
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── src/
│   ├── schema.py               # Pydantic schema validation
│   ├── metrics.py              # Aggregate & per-class classification metrics
│   ├── image_features.py       # Deterministic image proxy quality features
│   ├── subgroup_analysis.py    # Disaggregation, binning, Wilson CIs, ranking
│   ├── sample_data_generator.py# Generates synthetic test fixtures
│   └── report.py               # JSON & Markdown report generator
├── examples/
│   ├── sample_predictions.csv  # 400 realistic cases across subgroups
│   └── README.md
└── tests/
    ├── conftest.py
    ├── test_schema.py
    ├── test_metrics.py
    └── test_subgroups.py
```
