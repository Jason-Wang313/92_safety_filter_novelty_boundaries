# Paper 92 Terminal Audit

Date: 2026-06-15

Paper: `92_safety_filter_novelty_boundaries`

Decision: `KILL_ARCHIVE`

## Reproduction

- `python -m py_compile src\run_experiment.py`: passed.
- `python src\run_experiment.py`: passed; log at `logs/92_safety_filter_novelty_boundaries_continuation_rerun_20260615.log`.
- PDF target: `C:/Users/wangz/Downloads/92.pdf`.
- Visible Desktop copy: not allowed.

## Evidence Files

- `results/rollouts.csv`: 90,720 rows.
- `results/raw_seed_metrics.csv`: 1,260 rows.
- `results/metrics.csv`: 45 rows.
- `results/pairwise_stats.csv`: 1 row.
- `results/ablation_rollouts.csv`: 14,112 rows.
- `results/ablation_seed_metrics.csv`: 196 rows.
- `results/ablation_metrics.csv`: 7 rows.
- `results/stress_sweep_raw.csv`: 44,352 rows.
- `results/stress_sweep.csv`: 36 rows.
- `results/negative_cases.csv`: 3 rows.

## Key Results

Combined safety stress:

- `proposed_boundary_learning_filter`: success `0.45635 +/- 0.02071`, deployed violation `0.39236`, unshielded violation `0.45486`, filter dependence `0.46726`, transfer violation `0.43651`.
- `recovery_policy_shield`: success `0.54712 +/- 0.02281`, deployed violation `0.28671`, unshielded violation `0.54712`, filter dependence `0.76935`.
- `robust_mpc_shield`: success `0.64633 +/- 0.02185`, deployed violation `0.15526`, unshielded violation `0.58383`, filter dependence `0.94097`.
- Paired unshielded-safety improvement versus `recovery_policy_shield`: `0.09226 +/- 0.02766`.
- Paired task-success difference versus `recovery_policy_shield`: `-0.09077 +/- 0.03549`.
- Paired deployment-violation reduction versus `recovery_policy_shield`: `-0.10565 +/- 0.03373`.

Ablation:

- Full boundary-learning filter: success `0.46429`, deployed violation `0.39137`, unshielded violation `0.45238`, filter dependence `0.45734`.
- `cbf_feedback_only`: success `0.56696`, deployed violation `0.25050`, unshielded violation `0.50496`.
- `minus_boundary_margin_loss`: success `0.54712`, deployed violation `0.28323`, unshielded violation `0.53571`.

Maximum stress:

- `proposed_boundary_learning_filter`: success `0.48701`, deployed violation `0.38718`, unshielded violation `0.46266`.
- `recovery_policy_shield`: success `0.54951`, deployed violation `0.27110`, unshielded violation `0.52841`.
- `robust_mpc_shield`: success `0.63799`, deployed violation `0.15828`, unshielded violation `0.57305`.

## Terminal Reason

The rerun verifies a real mechanism-change signal: the proposed filter lowers unshielded post-training violations and filter dependence. The paper still fails ICLR-main readiness because it loses deployed task success and deployed safety to standard shields, and because the evidence is local simulated evidence rather than robot or accepted high-fidelity benchmark validation. The only honest terminal decision is `KILL_ARCHIVE`.

## PDF Verification

- Build command: two-pass `pdflatex -interaction=nonstopmode -halt-on-error main.tex`.
- Canonical PDF: `C:/Users/wangz/Downloads/92.pdf`.
- PDF SHA256: `6EFA95033A3493D7459544D40D8D9ABB2C48965EA8755CD286F4BB09BEB0E526`.
- PDF size: 472,682 bytes.
- LaTeX log scan: no document warnings/errors requiring action after the second pass.
- Desktop copy: absent.
