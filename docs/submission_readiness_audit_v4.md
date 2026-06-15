# Submission Readiness Audit v4.1

Date: 2026-06-15

Paper: 92 Safety Filter Novelty Boundaries

Terminal decision: KILL_ARCHIVE

## Commands Run

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
```

Both commands completed. The full experiment output was redirected to `logs/92_safety_filter_novelty_boundaries_continuation_rerun_20260615.log`.

## Evidence Coverage

- Main rollouts: 90,720 rows.
- Main seed metrics: 1,260 rows.
- Main aggregate metrics: 45 rows.
- Pairwise gate rows: 1 row.
- Ablation rollouts: 14,112 rows.
- Ablation seed metrics: 196 rows.
- Ablation aggregate metrics: 7 rows.
- Stress rollouts: 44,352 rows.
- Stress aggregates: 36 rows.
- Negative cases: 3 rows.
- Seeds: 0, 1, 2, 3, 4, 5, 6.
- Tasks: `narrow_corridor_navigation`, `human_workspace_reaching`, `fragile_object_pushing`, `dynamic_obstacle_crossing`.
- Splits: `nominal_boundary`, `tightened_safe_set`, `delayed_dynamics`, `boundary_shift`, `combined_safety_stress`.
- Methods: `unfiltered_policy`, `action_clipping_filter`, `geometric_projection_filter`, `cbf_safety_filter`, `robust_mpc_shield`, `conformal_uncertainty_shield`, `recovery_policy_shield`, `proposed_boundary_learning_filter`, `oracle_boundary_teacher`.

## Main Gate

On combined safety stress, `proposed_boundary_learning_filter` improves unshielded post-training safety relative to `recovery_policy_shield`: unshielded violation is `0.45486` versus `0.54712`, with paired safety improvement `0.09226 +/- 0.02766`. It also reduces filter dependence from `0.76935` to `0.46726`.

Those mechanism-change gains are not enough for ICLR main readiness. The proposed method reaches deployed task success `0.45635`, while `robust_mpc_shield` reaches `0.64633`. It also has deployed violation `0.39236`, while `robust_mpc_shield` reaches `0.15526` and `conformal_uncertainty_shield` reaches `0.16865`.

## Contradictory Evidence

- Paired task-success difference versus `recovery_policy_shield` is `-0.09077 +/- 0.03549`.
- Paired deployment-violation reduction is negative: `-0.10565 +/- 0.03373`.
- `robust_mpc_shield` dominates deployed success and deployed safety.
- `cbf_feedback_only` ablation reaches deployed success `0.56696`, above the full method's `0.46429`.
- The evidence remains local/simulated and lacks robot hardware or accepted high-fidelity safety-control benchmark validation.

## Readiness Judgment

The paper is reproducible as a local negative evidence audit. It contains a useful diagnostic: counterfactual boundary feedback can reduce later unshielded violations and filter dependence. However, it is not submission-ready for ICLR main because the proposed method sacrifices the deployed robotics objectives that safety filters must satisfy.

## Terminal Action

Keep `KILL_ARCHIVE`. Do not submit this paper to ICLR main in the current form.
