# Experiment Rigor Checklist

## v5 Expanded Audit

- [x] Frozen plan before editing or running experiments.
- [x] 10 seeds.
- [x] 6 tasks.
- [x] 8 safety/distribution splits.
- [x] 14 methods including strong baselines and oracle.
- [x] Paired seed confidence intervals.
- [x] Hard aggregate over the two hardest deployment splits.
- [x] Ablations for 10 v5 mechanism variants.
- [x] Stress sweep over 302,400 stress rows.
- [x] Fixed-risk deployment budgets over 69,120 rows.
- [x] 24 retained negative cases.
- [x] Bright boxed clickable citations in the generated PDF.
- [x] Downloads-only numbered PDF.

## ICLR Main Bar

- [ ] Real-robot validation.
- [ ] Accepted high-fidelity simulator benchmark.
- [ ] Independent reproduction.
- [ ] Trained neural robot-policy checkpoint.
- [x] Strong local safe-control baselines.
- [x] Mechanism-change metrics.
- [x] Stress and fixed-risk tests.
- [x] Honest terminal decision.

Decision: v5 expanded audit still fails ICLR-main empirical-rigor and deployment gates; archive.
