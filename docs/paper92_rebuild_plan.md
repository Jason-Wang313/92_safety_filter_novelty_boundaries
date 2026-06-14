# Paper 92 Rebuild Plan: Safety Filter Novelty Boundaries

Timestamp: 2026-06-14 15:08:00 +01:00

## Starting Point

Paper 92 is currently a v3 archive. The original bet is:

> Identify when safety filters change the learned mechanism rather than merely clipping actions.

The hostile prior-work pressure is severe. Control barrier functions, safety filters, safe RL, robust MPC shielding, probabilistic learned CBFs, safety attention mechanisms, passive compliance, and safe joints already cover much of the safety-filter space. A rebuilt paper cannot claim novelty from "we add a filter." It must test whether the filter changes what the policy learns, not merely whether the deployed shield prevents collisions.

## Rebuilt Claim Under Test

The strongest defensible claim is:

> A safety filter becomes a novel learning mechanism only when its interventions expose counterfactual safe-set boundary information that changes the unshielded policy's later behavior, reduces filter dependence, and transfers across safety-boundary shifts.

This is a local evidence audit. It is not hardware validation.

## Benchmark Design

I will replace the template success-rate generator with a deterministic safe-control learning benchmark. Each run simulates a policy learning over repeated episodes while navigating or manipulating near safety boundaries. The simulator tracks the nominal policy action, filtered action, safety margin, intervention reason, policy adaptation, and later unshielded behavior.

Tasks:

1. `narrow_corridor_navigation`
2. `human_workspace_reaching`
3. `fragile_object_pushing`
4. `dynamic_obstacle_crossing`

Splits:

1. `nominal_boundary`
2. `tightened_safe_set`
3. `delayed_dynamics`
4. `boundary_shift`
5. `combined_safety_stress`

## Methods To Compare

Strong baselines:

1. `unfiltered_policy`
2. `action_clipping_filter`
3. `geometric_projection_filter`
4. `cbf_safety_filter`
5. `robust_mpc_shield`
6. `conformal_uncertainty_shield`
7. `recovery_policy_shield`
8. `proposed_boundary_learning_filter`
9. `oracle_boundary_teacher`

The proposed method receives counterfactual boundary labels and intervention-gradient feedback during learning. The strongest baselines may be safer during deployment but should not necessarily improve unshielded policy behavior.

## Metrics

Primary closed-loop metrics:

1. Task success.
2. Safety violation rate.
3. Intervention rate.
4. Control distortion from nominal policy action.
5. Recovery success after near-boundary events.

Mechanism-change metrics:

1. Unshielded post-training safety: violation rate after removing the filter.
2. Filter dependence: how often the learned policy still needs interventions.
3. Boundary classification F1: whether the policy learned the unsafe boundary.
4. Transfer safety under shifted safe sets.
5. Regret to oracle safe teacher.

Statistics:

1. Seven deterministic seeds.
2. Per-task and per-split means with 95 percent confidence intervals.
3. Paired seed/task comparison against the strongest non-oracle baseline.
4. Explicit terminal decision in `results/summary.txt`.

## Ablations

The full method must beat stripped variants:

1. `full_boundary_learning_filter`
2. `minus_counterfactual_labels`
3. `minus_intervention_gradient`
4. `minus_unshielded_replay`
5. `minus_boundary_margin_loss`
6. `clipping_feedback_only`
7. `cbf_feedback_only`

If stripped variants match or beat the full method on unshielded post-training safety or task success without a clear tradeoff, the mechanism is not supported.

## Stress Tests

Stress axes:

1. Safe-set tightening.
2. Dynamics delay.
3. Sensor noise.
4. Human/obstacle unpredictability.
5. Boundary-shape shift.
6. Combined maximum stress.

Stress curves must include both shielded deployment safety and unshielded post-training safety, because the claim is about learned mechanism change.

## Paper Rewrite Requirements

After experiments:

1. Rewrite `paper/main.tex` as either a strong-revise evidence report or a negative evidence audit.
2. Replace template claims with measured claims only.
3. Include tables for combined stress, mechanism-change metrics, ablations, and failure cases.
4. Include figures for deployment safety/success, unshielded mechanism change, ablations, and stress curves.
5. Update README, child status, claims, final audit, and submission-readiness docs.
6. Build only `C:/Users/wangz/Downloads/92.pdf`; do not copy anything to Desktop.
7. Commit and push to `https://github.com/Jason-Wang313/92_safety_filter_novelty_boundaries`.

## Terminal Gate

Mark `STRONG_REVISE` only if all of the following are true:

1. `proposed_boundary_learning_filter` beats the strongest non-oracle baseline on unshielded post-training safety under combined safety stress.
2. It also matches or improves shielded task success without materially increasing deployment violations.
3. It reduces filter dependence relative to CBF/MPC/conformal shields.
4. Core ablations degrade in expected directions.
5. Maximum-stress curves do not reverse in favor of CBF, robust MPC, conformal shielding, or recovery shielding.
6. The paper honestly states evidence is local/simulated and not real robot validation.

Otherwise mark `KILL_ARCHIVE`. A filter that is safe only because it clips actions during deployment is not a novel ICLR-main mechanism.
