# 92 Safety Filter Novelty Boundaries

Submission-hardening version: v4.1 rerun audit

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository now contains a deterministic safe-control learning evidence audit for the claim that safety filters are novel only when they change the learned policy mechanism rather than merely clip actions. The rebuilt benchmark includes four tasks, five safety-boundary shifts, seven seeds, nine filters/controllers, seven ablations, and stress sweeps.

The 2026-06-15 continuation rerun reproduced the same terminal decision: the mechanism-change signal is real, but deployed safety and success fail the ICLR-main gate.

## Key Result

On combined safety stress:

- Proposed boundary-learning filter: task success 0.456, deployed violation 0.392, unshielded violation 0.455, filter dependence 0.467.
- Recovery-policy shield: task success 0.547, deployed violation 0.287, unshielded violation 0.547, filter dependence 0.769.
- Robust MPC shield: task success 0.646, deployed violation 0.155, unshielded violation 0.584, filter dependence 0.941.
- Paired unshielded-safety gain versus recovery shield: 0.092 +/- 0.028.
- Paired task-success loss versus recovery shield: -0.091 +/- 0.035.

The proposed method improves the mechanism-change metric but loses deployed success and deployed safety to standard shields. It is not submission-ready.

## Reproduce Evidence

```powershell
python src\run_experiment.py
```

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/92.pdf`

No PDF should be copied to the visible Desktop.
