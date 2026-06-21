# 92 Safety Filter Novelty Boundaries

Submission-hardening version: v5 expanded ICLR-main audit

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository contains a deterministic safe-control learning evidence audit for the claim that a safety filter is novel only when it changes the learned policy mechanism rather than merely clipping unsafe deployment actions.

The 2026-06-22 expanded rebuild raises the evidence standard to a 25+ page ICLR-style audit: 10 seeds, 6 tasks, 8 safety/distribution splits, 14 methods including oracle, 10 ablations, a stress sweep, fixed-risk deployment budgets, paired seed statistics, 24 retained negative cases, bright boxed clickable citations, and a generated 30-page PDF.

## Key Result

The v5 method, `counterfactual_boundary_learning_filter_v5`, improves over `boundary_learning_filter_v4` on several mechanism and calibration metrics. It still fails the submission gate:

- Hard-aggregate v5 success: `0.22982`.
- Best hard-aggregate success: `0.47122` from `robust_mpc_shield`.
- Hard-aggregate v5 deployed violation: `0.51888`.
- Safest hard-aggregate deployed violation: `0.29701` from `robust_mpc_shield`.
- Hard-aggregate v5 unshielded violation: `0.89726`, best among non-oracle methods.
- Hard-aggregate v5 boundary ECE: `0.10024`, best among non-oracle methods.
- Hard-aggregate v5 robust utility: `-0.24424`.
- Best hard-aggregate robust utility: `-0.00351` from `robust_mpc_shield`.
- Fixed-risk budget `0.05` accepted coverage: `0.00000` on both hard deployment splits.

The mechanism signal is real, but deployed success, deployed safety, fixed-risk coverage, stress robustness, and scope evidence are not enough for ICLR main.

## Canonical Artifacts

- PDF: `C:/Users/wangz/Downloads/92.pdf`
- PDF pages: 30
- PDF SHA256: `36406245AA243E208417A0824557E5BBEE7AE221E712B467D0D87EB32687D45A`
- GitHub: https://github.com/Jason-Wang313/92_safety_filter_novelty_boundaries
- Terminal action: `KILL_ARCHIVE`

No PDF should be copied to the visible Desktop.

## Reproduce Evidence

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
```

## Rebuild Manuscript

```powershell
python scripts\generate_manuscript.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
copy main.pdf C:\Users\wangz\Downloads\92.pdf
cd ..
python scripts\validate_submission_artifacts.py
```
