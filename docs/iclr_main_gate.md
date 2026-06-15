# ICLR Main Gate

Paper: 92 safety_filter_novelty_boundaries

Existing v4 decision: KILL_ARCHIVE

Gate verdict: KILL_ARCHIVE

Latest rerun: 2026-06-15

Evidence digest: v4-local-safety-filter-mechanism

Fatal blockers:
- Local synthetic evidence only.
- Deployed task-success gate fails against robust MPC and recovery-policy shields.
- Deployed safety gate fails against robust MPC, CBF, conformal, and recovery-policy shields.
- The mechanism-change signal is partial: unshielded safety and filter dependence improve, but boundary F1 and transfer evidence remain weak.
- `cbf_feedback_only` ablation has higher deployed success and lower deployed violation than the full method.
- No real robot or accepted high-fidelity safety-control benchmark.
- No trained neural policy checkpoint or external validation.
- No manual exhaustive related-work synthesis.

The only honest main-conference-safe decision is to archive rather than overclaim.
