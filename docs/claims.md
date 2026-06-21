# Claims

- Mechanism claim tested: a safety filter is novel only if its interventions teach policy-level safe-boundary structure that persists after the filter is removed.
- v5 method claim tested: `counterfactual_boundary_learning_filter_v5` uses counterfactual unsafe labels, unshielded replay, intervention gradients, boundary-margin loss, calibration, and recovery-feasibility targets to reduce post-training unshielded violations.
- Supported local finding: v5 improves over v4 on deployed safety, task success, calibration, transfer violation, filter dependence, robust utility, and mechanism utility.
- Supported local finding: v5 has the best non-oracle unshielded violation and boundary ECE in the hard aggregate.
- Unsupported main claim: v5 is not ICLR-main ready because `robust_mpc_shield` dominates hard-aggregate deployed success, deployed safety, and robust utility.
- Unsupported deployment claim: fixed-risk coverage at budget 0.05 is zero on both hard deployment splits.
- Unsupported scope claim explicitly avoided: no claim of real robot deployment, state-of-the-art safe robot learning, or accepted high-fidelity safety benchmark validation.
- Terminal claim: `KILL_ARCHIVE` is the only honest current action.
