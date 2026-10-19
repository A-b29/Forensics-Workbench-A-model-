"""
generate_sample_predictions.py - Creates realistic synthetic prediction dataset for forensics demo.
"""

from pathlib import Path
import numpy as np
import pandas as pd

def generate_predictions_csv(output_path: Path, n_samples: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    
    classes = ["Normal", "Pneumonia", "Pleural Effusion"]
    age_groups = ["Pediatric (<18)", "Adult (18-64)", "Geriatric (65+)"]
    scanners = ["Siemens_Multix", "GE_Optima", "Apex_XR400", "Portable_Carestream"]
    views = ["PA_Erect", "AP_Portable"]
    
    records = []
    
    for i in range(1, n_samples + 1):
        sample_id = f"CXR_{i:04d}"
        
        # Ground truth
        y_true = rng.choice(classes, p=[0.50, 0.30, 0.20])
        
        # Metadata factors
        age = rng.choice(age_groups, p=[0.15, 0.55, 0.30])
        scanner = rng.choice(scanners, p=[0.35, 0.30, 0.20, 0.15])
        view = "AP_Portable" if scanner == "Portable_Carestream" else rng.choice(views, p=[0.75, 0.25])
        
        # Synthetic flaw: portable carestream or motion artifact triggers higher error rate!
        has_motion = bool(rng.random() < 0.25 if view == "AP_Portable" else rng.random() < 0.05)
        
        # Base accuracy is 88%, but drops to 60% if AP_Portable + Geriatric + motion artifact
        p_correct = 0.90
        if scanner == "Portable_Carestream":
            p_correct -= 0.18
        if age == "Geriatric (65+)":
            p_correct -= 0.08
        if has_motion:
            p_correct -= 0.20
            
        p_correct = max(0.40, p_correct)
        
        if rng.random() < p_correct:
            y_pred = y_true
            confidence = float(np.clip(rng.beta(8, 2), 0.60, 0.99))
        else:
            wrong_classes = [c for c in classes if c != y_true]
            y_pred = rng.choice(wrong_classes)
            # Some errors are overconfident
            confidence = float(np.clip(rng.beta(5, 3), 0.51, 0.92))
            
        records.append({
            "sample_id": sample_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": round(confidence, 3),
            "patient_age_group": age,
            "scanner_model": scanner,
            "acquisition_view": view,
            "motion_artifact_present": has_motion,
        })
        
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sample records to {output_path}")

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "examples" / "sample_predictions.csv"
    generate_predictions_csv(out)
