# Reproducibility Checklist

## What Reproduces

- [x] `python -m py_compile src\run_experiment.py`
- [x] `python src\run_experiment.py`
- [x] `results/rollouts.csv` with 215,040 rows.
- [x] `results/dataset_summary.csv` with 15,360 rows.
- [x] `results/raw_seed_metrics.csv` with 1,120 rows.
- [x] `results/metrics.csv` with 1,456 rows.
- [x] `results/pairwise_stats.csv` with 1,248 rows.
- [x] `results/hard_aggregate_metrics.csv` with 182 rows.
- [x] `results/hard_aggregate_pairwise_stats.csv` with 156 rows.
- [x] `results/ablation_rollouts.csv` with 76,800 rows.
- [x] `results/stress_sweep_raw.csv` with 302,400 rows.
- [x] `results/fixed_risk_raw.csv` with 69,120 rows.
- [x] `results/negative_cases.csv` with 24 rows.
- [x] `paper/main.tex`.
- [x] `paper/references.bib` with 180 entries.
- [x] Canonical PDF: `C:/Users/wangz/Downloads/92.pdf`.
- [x] Validator: `python scripts\validate_submission_artifacts.py`.

## What Does Not Reproduce

- [ ] Real robot results.
- [ ] Accepted high-fidelity safety benchmark runs.
- [ ] Physical timing/latency measurements.
- [ ] Independent external reproduction.

This is reproducible as a local negative evidence audit and 30-page ICLR-style archive manuscript, not as an ICLR-main-ready robotics system paper.
