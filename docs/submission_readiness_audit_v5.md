# Submission Readiness Audit v5

Date: 2026-06-22

Paper: 92 Safety Filter Novelty Boundaries

Terminal decision: KILL_ARCHIVE

## Commands Run

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
python -m py_compile scripts\generate_manuscript.py scripts\validate_submission_artifacts.py
python scripts\generate_manuscript.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
python scripts\validate_submission_artifacts.py
```

All required commands completed.

## Evidence Coverage

- Main rollout rows: 215,040.
- Dataset-summary rows: 15,360.
- Main seed-metric rows: 1,120.
- Main aggregate metric rows: 1,456.
- Main pairwise rows: 1,248.
- Hard aggregate seed rows: 140.
- Hard aggregate metric rows: 182.
- Hard aggregate pairwise rows: 156.
- Ablation rollout rows: 76,800.
- Ablation seed rows: 100.
- Ablation metric rows: 130.
- Stress raw rows: 302,400.
- Stress seed rows: 840.
- Stress metric rows: 1,092.
- Fixed-risk raw rows: 69,120.
- Fixed-risk seed rows: 480.
- Fixed-risk metric rows: 288.
- Fixed-risk pairwise rows: 240.
- Negative cases: 24.

## Frozen Gate

- `success_gate=False`
- `deployment_gate=False`
- `mechanism_gate=False`
- `calibration_gate=True`
- `utility_gate=False`
- `ablation_gate=True`
- `stress_gate=False`
- `fixed_risk_gate=False`
- `scope_gate=False`

## Key Results

- Best success reference: `robust_mpc_shield`.
- Safest deployed reference: `robust_mpc_shield`.
- Best calibration reference: `counterfactual_boundary_learning_filter_v5`.
- Best unshielded non-oracle reference: `counterfactual_boundary_learning_filter_v5`.
- v5 hard success: `0.22982`.
- best hard success: `0.47122`.
- v5 hard deployed violation: `0.51888`.
- safest hard deployed violation: `0.29701`.
- v5 hard unshielded violation: `0.89726`.
- v5 hard boundary ECE: `0.10024`.
- v5 hard robust utility: `-0.24424`.
- best hard robust utility: `-0.00351`.
- fixed-risk budget `0.05` v5 coverage on `low_signal_high_risk_shift`: `0.00000`.
- fixed-risk budget `0.05` v5 coverage on `combined_safety_stress`: `0.00000`.

## Readiness Judgment

The v5 evidence is much stronger than v4. The method is not a trivial clipping rule; its ablations show real mechanism effects. However, it is not submission-ready for ICLR main because the deployed robotics objectives are worse than robust MPC, strict fixed-risk deployment has zero accepted coverage on the hard splits, and the scope lacks robot or accepted high-fidelity validation.

## Terminal Action

Keep `KILL_ARCHIVE`. Do not submit this paper to ICLR main in the current form.
