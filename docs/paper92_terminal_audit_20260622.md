# Paper 92 Terminal Audit

Date: 2026-06-22

Paper: `92_safety_filter_novelty_boundaries`

Decision: `KILL_ARCHIVE`

## Reproduction

- `python -m py_compile src\run_experiment.py`: passed.
- `python src\run_experiment.py`: passed.
- `python -m py_compile scripts\generate_manuscript.py scripts\validate_submission_artifacts.py`: passed.
- `python scripts\generate_manuscript.py`: passed.
- LaTeX sequence `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`: passed.
- `python scripts\validate_submission_artifacts.py`: passed.
- PDF target: `C:/Users/wangz/Downloads/92.pdf`.
- Visible Desktop copy: absent.

## Evidence Files

- `results/rollouts.csv`: 215,040 rows.
- `results/dataset_summary.csv`: 15,360 rows.
- `results/raw_seed_metrics.csv`: 1,120 rows.
- `results/metrics.csv`: 1,456 rows.
- `results/pairwise_stats.csv`: 1,248 rows.
- `results/hard_aggregate_seed_metrics.csv`: 140 rows.
- `results/hard_aggregate_metrics.csv`: 182 rows.
- `results/hard_aggregate_pairwise_stats.csv`: 156 rows.
- `results/ablation_rollouts.csv`: 76,800 rows.
- `results/ablation_seed_metrics.csv`: 100 rows.
- `results/ablation_metrics.csv`: 130 rows.
- `results/stress_sweep_raw.csv`: 302,400 rows.
- `results/stress_sweep_seed_metrics.csv`: 840 rows.
- `results/stress_sweep.csv`: 1,092 rows.
- `results/fixed_risk_raw.csv`: 69,120 rows.
- `results/fixed_risk_seed_metrics.csv`: 480 rows.
- `results/fixed_risk_metrics.csv`: 288 rows.
- `results/fixed_risk_pairwise.csv`: 240 rows.
- `results/negative_cases.csv`: 24 rows.

## Key Results

Hard aggregate:

- `counterfactual_boundary_learning_filter_v5`: success `0.22982`, deployed violation `0.51888`, unshielded violation `0.89726`, filter dependence `0.25775`, boundary ECE `0.10024`, robust utility `-0.24424`, mechanism utility `0.00049`.
- `boundary_learning_filter_v4`: success `0.16654`, deployed violation `0.60807`, unshielded violation `0.90221`, filter dependence `0.31556`, boundary ECE `0.11770`, robust utility `-0.37505`, mechanism utility `-0.04046`.
- `robust_mpc_shield`: success `0.47122`, deployed violation `0.29701`, unshielded violation `0.92018`, filter dependence `0.84285`, boundary ECE `0.15436`, robust utility `-0.00351`, mechanism utility `-0.29104`.

Paired hard aggregate:

- v5 minus robust MPC task success: `-0.241406 +/- 0.016092`.
- v5 minus robust MPC deployed violation: `0.221875 +/- 0.018198`.
- v5 minus robust MPC robust utility: `-0.240730 +/- 0.024943`.
- v5 minus v4 task success: `0.063281 +/- 0.008842`.
- v5 minus v4 deployed violation: `-0.089193 +/- 0.014185`.
- v5 minus v4 robust utility: `0.130813 +/- 0.016447`.

Fixed-risk:

- Budget `0.05` v5 coverage on `low_signal_high_risk_shift`: `0.00000`.
- Budget `0.05` v5 coverage on `combined_safety_stress`: `0.00000`.

## PDF Verification

- Canonical PDF: `C:/Users/wangz/Downloads/92.pdf`.
- PDF pages: 30.
- PDF SHA256: `36406245AA243E208417A0824557E5BBEE7AE221E712B467D0D87EB32687D45A`.
- Bibliography entries: 180.
- Citation links: validator confirmed at least 120 PDF annotations.
- Bright citation boxes: visually confirmed on rendered PNG pages.
- Desktop copy: absent.

## Terminal Reason

The v5 method contains a real mechanism-improvement signal, especially relative to v4 and internal ablations. That is not enough. Under the frozen expanded audit, robust MPC remains the deployed success, deployed safety, and robust-utility reference; strict fixed-risk deployment has zero coverage on the hard splits; maximum stress remains dominated by robust MPC; and there is no real robot or accepted high-fidelity benchmark evidence. The only honest terminal decision is `KILL_ARCHIVE`.
