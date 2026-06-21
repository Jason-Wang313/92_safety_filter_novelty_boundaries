# Paper 92 Expanded Submission Plan

Date frozen: 2026-06-22

Paper: `92_safety_filter_novelty_boundaries`

Target: ICLR-main readiness audit, not cosmetic expansion.

Terminal policy: report `STRONG_REVISE` only if the frozen gates clear after the experiment is run. Otherwise report `KILL_ARCHIVE` honestly.

## Objective

Rebuild Paper 92 into a 25+ page submission-style artifact that tests whether a safety filter is novel only when it changes the learned policy mechanism, rather than merely clipping actions at deployment time.

The v5 method under test is `counterfactual_boundary_learning_filter_v5`: a boundary-aware training procedure that uses counterfactual unsafe labels, unshielded replay, intervention gradients, boundary-margin loss, and recovery-feasibility targets to reduce post-training unshielded violations while preserving deployed task success and safety.

## Frozen Main Experiment

CPU-only and RAM-light execution:

- Seeds: 10.
- Tasks: 6.
- Splits: 8.
- Methods: 14 total, including oracle.
- Episodes per task/split/method/seed: 32.
- Main rollout rows: 215,040.
- Dataset-summary rows: 15,360.

Tasks:

- `narrow_gap_navigation`
- `human_zone_reaching`
- `unstable_stack_insertion`
- `contact_rich_door_push`
- `slippery_payload_transfer`
- `deformable_obstacle_threading`

Splits:

- `nominal_safety`
- `constraint_shift`
- `novel_obstacle_shift`
- `actuator_lag_shift`
- `human_motion_shift`
- `contact_mode_shift`
- `low_signal_high_risk_shift`
- `combined_safety_stress`

Methods:

- `unfiltered_policy`
- `action_clipping_filter`
- `geometric_projection_filter`
- `cbf_safety_filter`
- `robust_mpc_shield`
- `conformal_uncertainty_shield`
- `recovery_policy_shield`
- `shielded_behavior_cloning`
- `lagrangian_safe_rl`
- `adversarial_safety_critic`
- `safety_filter_distillation`
- `boundary_learning_filter_v4`
- `counterfactual_boundary_learning_filter_v5`
- `oracle_boundary_teacher`

## Metrics

Deployment metrics:

- Task success.
- Deployed violation rate.
- Intervention rate.
- Intervention severity.
- Recovery success.
- Robust utility.

Mechanism metrics:

- Unshielded violation rate after training.
- Filter dependence.
- Boundary F1.
- Boundary calibration ECE.
- Transfer violation rate.
- Mechanism-change utility.

## Frozen Gates

The paper is not submission-ready unless all of these clear:

- Main success gate: v5 must beat the strongest non-oracle baseline on hard-aggregate deployed task success, with positive paired lower95.
- Deployed safety gate: v5 must not be worse than the safest non-oracle baseline on deployed violations, with nonpositive paired upper95.
- Mechanism gate: v5 must beat the best non-oracle baseline on unshielded violation reduction and filter-dependence reduction, with paired support.
- Calibration gate: v5 must beat the best calibrated baseline on boundary ECE or fixed-risk coverage.
- Utility gate: v5 must beat the strongest non-oracle baseline on robust utility.
- Ablation gate: no removed-component ablation may beat full v5 on mechanism-change utility.
- Stress gate: v5 must not be dominated at maximum combined safety stress.
- Fixed-risk gate: v5 must retain nonzero accepted coverage at budget 0.05 on `low_signal_high_risk_shift` and `combined_safety_stress`.
- Scope gate: no ICLR-main claim without real robot, accepted high-fidelity safety benchmark, or external safe-control benchmark validation.

## Additional Experiments

Ablations:

- `full_counterfactual_boundary_v5`
- `minus_counterfactual_labels`
- `minus_unshielded_replay`
- `minus_intervention_gradient`
- `minus_boundary_margin_loss`
- `minus_recovery_feasibility`
- `minus_calibration_layer`
- `distill_only_boundary`
- `cbf_feedback_only`
- `clipping_feedback_only`

Stress sweep:

- Six stress levels across constraint drift, obstacle novelty, actuator lag, human-motion uncertainty, contact-mode shift, and sensor noise.
- Report deployed success/safety, unshielded safety, filter dependence, calibration, and utility curves.

Fixed-risk deployment:

- Budgets: 0.00, 0.05, 0.10, 0.15.
- Hard splits: `low_signal_high_risk_shift`, `combined_safety_stress`.
- Report accepted coverage, accepted success, accepted deployed violation, accepted unshielded violation, accepted transfer violation, and accepted robust utility.

Negative cases:

- At least 24 generated failure cases across tasks and hard splits.
- Include cases where v5 reduces unshielded violation but deploys worse than CBF/MPC, cases where v5 overfits the boundary and loses task success, and cases where all non-oracle methods fail.

## Theory To Add

- A decomposition separating deployment safety from learned-policy safety.
- A theorem showing why action clipping can produce high deployed safety without changing the underlying policy mechanism.
- A calibration lemma linking boundary-score ECE to fixed-risk deployment coverage.
- A negative identifiability result: without counterfactual labels or unshielded replay, deployment-filter interventions cannot identify whether the policy or the shield owns safety.

## Manuscript Requirements

- Minimum 25 pages, but no filler.
- Bright boxed clickable citations using `hyperref` citation borders.
- Full generated tables for main metrics, hard aggregate, paired tests, ablations, stress, fixed-risk, and negative cases.
- Generated figures for deployed-vs-unshielded safety, intervention dependence, ablation utility, stress curves, fixed-risk coverage, and Pareto front.
- Explicit terminal decision in abstract, introduction, results, and final audit.
- Canonical PDF must be `C:/Users/wangz/Downloads/92.pdf`.
- Do not copy PDFs to the visible Desktop.

## Verification

Before commit/push:

- `python -m py_compile src\run_experiment.py`
- `python src\run_experiment.py`
- `python scripts\generate_manuscript.py`
- LaTeX/BibTeX compile to `paper/main.pdf`
- Copy final numbered PDF to Downloads only.
- `python scripts\validate_submission_artifacts.py`
- Render PDF pages and visually inspect representative pages.
- Verify no `C:/Users/wangz/Desktop/92.pdf`.
- Commit and push to public GitHub.
- Update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, `MASTER_SUBMISSION_REPORT.md`, and `SUBMISSION_AUDIT_MATRIX.csv`.
