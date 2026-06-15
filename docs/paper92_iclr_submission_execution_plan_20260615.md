# Paper 92 ICLR-Main Submission Execution Plan

Date: 2026-06-15
Paper: `92_safety_filter_novelty_boundaries`
Repository: `https://github.com/Jason-Wang313/92_safety_filter_novelty_boundaries`

## Goal

Rebuild and audit Paper 92 as if it were being considered for an ICLR main submission, but only accept a submission-ready outcome if the rerun evidence clears the same standards a hostile reviewer would use. The paper's thesis is that a safety filter is novel only when it changes the learned policy mechanism, not merely when it clips unsafe actions at deployment.

## Current Starting State

- The repository is on `main` at commit `e70a3a3719e0b0a43d9b0149f1c7155945000eb9`.
- The existing v4 audit is terminally negative: `KILL_ARCHIVE`.
- Existing evidence reports a mechanism-change benefit for the proposed boundary-learning filter on unshielded post-training safety and filter dependence.
- Existing evidence also reports worse deployed task success and deployed safety than recovery, CBF, and robust MPC shields.
- `C:/Users/wangz/Downloads/92.pdf` exists.
- `C:/Users/wangz/Desktop/92.pdf` does not exist and must not be created.

## Execution Steps

1. Re-run `python -m py_compile src/run_experiment.py`.
2. Re-run `python src/run_experiment.py` from a clean terminal invocation and save the full console transcript to the batch log directory.
3. Verify that the rerun regenerates the main rollouts, seed metrics, aggregate metrics, paired statistics, ablations, stress sweeps, negative cases, and figures.
4. Independently audit the resulting CSV files with a small pandas check rather than trusting the prose summary.
5. Compare the proposed method against the strongest non-oracle baselines on:
   - deployed task success;
   - deployed safety violation;
   - unshielded post-training safety;
   - filter dependence;
   - boundary classification F1;
   - transfer violation;
   - stress-sweep reversal behavior.
6. Check that the full method beats stripped ablations in the dimensions needed to support the mechanism claim.
7. Update the paper and documentation with measured claims only.
8. Rebuild `paper/main.pdf` with `pdflatex` and copy the final artifact only to `C:/Users/wangz/Downloads/92.pdf`.
9. Scan the LaTeX log for real warnings or errors and fix recoverable typesetting issues.
10. Update the root batch ledgers after the child repo is correct.
11. Commit, push, and verify the public GitHub repository.
12. Confirm the child git tree is clean, `origin/main` matches local `HEAD`, the numbered PDF exists in Downloads, and no Desktop PDF exists.

## Submission-Readiness Gates

Paper 92 may be marked `STRONG_REVISE` only if all gates pass:

1. The proposed boundary-learning filter beats the strongest non-oracle baseline on unshielded post-training safety under combined safety stress.
2. It does not lose deployed task success by a practically meaningful margin.
3. It does not worsen deployed safety violation relative to the strongest shield baselines.
4. It reduces filter dependence relative to deployment-only shields.
5. Core ablations degrade in the expected direction for the mechanism metrics.
6. Stress sweeps do not reverse the claimed advantage at maximum stress.
7. The paper clearly labels the evidence as local simulated evidence and does not imply robot hardware validation.

If any deployed-safety or deployed-success gate fails, the terminal decision remains `KILL_ARCHIVE` even if the mechanism-change metric improves.

## Expected Honest Outcome

The prior v4 evidence already suggests a likely `KILL_ARCHIVE` decision because the proposed method improves unshielded post-training safety while sacrificing deployed task success and deployed safety. The continuation rerun must therefore treat the earlier result as a hypothesis to verify, not a conclusion to preserve.

## Deliverables

- Updated rerun log in `C:/Users/wangz/robotics_massive_pool_paper_factory/logs/`.
- Updated Paper 92 result CSVs and figures if the rerun changes timestamps or content.
- Updated child documentation and paper source with the rerun audit.
- Final numbered PDF at `C:/Users/wangz/Downloads/92.pdf` only.
- Updated root ledgers through Paper 92.
- Public GitHub repo pushed and verified clean.
