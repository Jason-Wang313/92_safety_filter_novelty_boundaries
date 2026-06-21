# Final Audit

1. Chosen thesis: `Safety Filter Novelty Boundaries` tests whether a safety filter is novel only when it changes the learned policy mechanism rather than merely clipping deployment actions.
2. Rebuild version: v5 expanded CPU-only ICLR-main audit on 2026-06-22.
3. Frozen plan: `docs/paper92_expanded_submission_plan_20260622.md`.
4. ICLR-main decision: `KILL_ARCHIVE`.
5. Reason: v5 improves some mechanism/calibration evidence over v4, but robust MPC dominates hard-aggregate deployed success, deployed safety, and robust utility; fixed-risk coverage at budget 0.05 is zero on both hard deployment splits.
6. Evidence scale: 215,040 main rollouts; 15,360 dataset rows; 76,800 ablation rollouts; 302,400 stress rows; 69,120 fixed-risk rows; 24 negative cases.
7. Strong baselines: clipping, projection, CBF, robust MPC, conformal uncertainty, recovery-policy shielding, shielded behavior cloning, safe RL, adversarial safety critic, distillation, v4, v5, and oracle.
8. Reproducibility: `python src\run_experiment.py` regenerates results and figures; `python scripts\generate_manuscript.py` regenerates TeX and references; `python scripts\validate_submission_artifacts.py` validates the final artifacts.
9. PDF verification: `C:/Users/wangz/Downloads/92.pdf`, 30 pages, SHA256 `36406245AA243E208417A0824557E5BBEE7AE221E712B467D0D87EB32687D45A`.
10. GitHub URL: https://github.com/Jason-Wang313/92_safety_filter_novelty_boundaries
11. Confirmation: no visible Desktop copy was requested or made.
