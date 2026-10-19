# Architectural & Cunning Decisions: Model Forensics Workbench

## 1. Replacing Standard Normal Approximation with Wilson Score Confidence Intervals
**Decision:** We rejected the standard Wald confidence interval ($\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$) in favor of the **Wilson Score Interval**.
**Cunning Rationale:**
- In clinical evaluation, subgroups often have high accuracy ($\hat{p} \approx 0$) or tiny sample sizes ($n < 20$).
- Wald intervals collapse to zero width when errors = 0, giving an illusion of infallible performance, or produce impossible intervals extending below 0% or above 100%.
- The Wilson interval adjusts the center probability outward ($p_{adj} = \frac{p + z^2/(2n)}{1 + z^2/n}$), guaranteeing valid bounds in $[0, 1]$ and accurately communicating that 0 errors in 10 patients still carries an upper error risk bound of up to 28%.

## 2. Minimum Cohort Size Enforcement Against Multiple Hypothesis Testing (P-Hacking)
**Decision:** Subgroups with fewer than $N_{min}$ observations (default: 15) are explicitly marked with an alert flag (`⚠️ Small Sample`) and separated from reliable rankings.
**Cunning Rationale:**
- Slicing data across many metadata combinations inevitably yields small cohorts with 100% or 0% error simply by chance (the multiple comparisons problem).
- Forcing sample size visibility prevents engineers and clinicians from over-interpreting random anomalies.

## 3. High-Confidence Error Forensics
**Decision:** Created a dedicated forensic metric for *Uncalibrated Severe Errors* (misclassifications occurring with confidence $\ge 0.80$).
**Cunning Rationale:**
- Low-confidence errors can be deferred or flagged by human oversight.
- High-confidence errors represent silent catastrophic failures where the model asserts an incorrect diagnosis with high certainty. Quantifying this subset directly pinpoints automation risk.

## 4. Strict Non-Causal Framing
**Decision:** Explicit disclaimers and schema validations enforce that metadata associations are labeled *diagnostic hypotheses*, never causal explanations.
