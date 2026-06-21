# Submission Version Log

## v1 - Generated Draft
- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening
- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed metrics, stronger baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Recompiled canonical PDF at `C:/Users/wangz/Downloads/92.pdf`.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive
- Applied the stricter ICLR-main-conference standard.
- Re-read local paper, docs, experiments, prior-work artifacts, PDF state, and repo state.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats are not recoverable from local artifacts.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Paper-Specific Safety-Filter Mechanism Rebuild
- Replaced the generic archive framing with a deterministic safe-control learning benchmark.
- Added four tasks, five boundary shifts, nine filters/controllers, seven seeds, ablations, stress sweeps, negative cases, and figures.
- Reported 90,720 main rollouts, 14,112 ablation rollouts, and 44,352 stress rollouts.
- Found that counterfactual boundary learning improves unshielded post-training safety and filter dependence but fails deployed task-success and deployed-safety gates.
- Terminal decision: KILL_ARCHIVE.

## v4.1 - 2026-06-15 Rerun Audit
- Re-ran `python -m py_compile src\run_experiment.py` and the full `python src\run_experiment.py`.
- Confirmed paired unshielded-safety improvement versus `recovery_policy_shield` is `0.09226 +/- 0.02766`.
- Confirmed paired task-success loss is `-0.09077 +/- 0.03549`.
- Confirmed paired deployment-violation reduction is negative at `-0.10565 +/- 0.03373`.
- Updated child docs and paper source to keep the v4 KILL_ARCHIVE decision evidence-bound.

## v5 - 2026-06-22 Expanded ICLR-Main Audit
- Froze `docs/paper92_expanded_submission_plan_20260622.md` before executing the expanded run.
- Replaced the v4 runner with a 10-seed, 6-task, 8-split, 14-method audit.
- Generated 215,040 main rollouts, 15,360 dataset rows, 76,800 ablation rollouts, 302,400 stress rows, 69,120 fixed-risk rows, and 24 negative cases.
- Added hard-aggregate paired tests, stress-sweep metrics, fixed-risk metrics, generated figures, a manuscript generator, and an artifact validator.
- Generated a 30-page ICLR-style PDF with bright boxed clickable citations and 180 bibliography entries.
- Validated `C:/Users/wangz/Downloads/92.pdf`; SHA256 `36406245AA243E208417A0824557E5BBEE7AE221E712B467D0D87EB32687D45A`.
- Terminal decision remains `KILL_ARCHIVE`: v5 has real mechanism/calibration evidence but fails deployed success, deployed safety, robust utility, fixed-risk, stress, and scope gates.
