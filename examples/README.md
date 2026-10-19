# Example Dataset: Model Forensics Workbench

- **Filename:** `sample_predictions.csv`
- **Rows:** 400 cases
- **Task:** Chest Radiograph 3-Class Categorization (Normal, Pneumonia, Pleural Effusion)
- **Included Metadata Columns:**
  - `sample_id`: Unique exam identifier
  - `y_true`: Expert ground-truth label
  - `y_pred`: Algorithmic prediction
  - `confidence`: Model output confidence $[0.50, 0.99]$
  - `patient_age_group`: Demographic stratification
  - `scanner_model`: Hardware acquisition device
  - `acquisition_view`: Patient orientation (AP Portable vs. PA Erect)
  - `motion_artifact_present`: Binary flag for motion blur
- **License / Privacy:** 100% synthetically simulated data. Contains no real patient data (CC0).
