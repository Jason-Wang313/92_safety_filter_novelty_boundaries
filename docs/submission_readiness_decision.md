# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Paper 92 was rebuilt as a v5 expanded CPU-only safety-filter novelty audit on 2026-06-22. The resulting manuscript is 30 pages and substantially stronger than the prior short audit, but the evidence still fails the ICLR-main gate.

Reasons:

- Hard-aggregate v5 success is `0.22982`, while `robust_mpc_shield` reaches `0.47122`.
- Hard-aggregate v5 deployed violation is `0.51888`, while `robust_mpc_shield` reaches `0.29701`.
- Hard-aggregate v5 robust utility is `-0.24424`, while `robust_mpc_shield` reaches `-0.00351`.
- Fixed-risk coverage at budget `0.05` is `0.00000` on both `low_signal_high_risk_shift` and `combined_safety_stress`.
- The mechanism/calibration signal is real but not enough to compensate for deployed safety and utility losses.
- The evidence is local/simulated and lacks robot hardware or accepted high-fidelity safety benchmark validation.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in the current form.

Revival condition: real or accepted high-fidelity safety-control evidence showing that the method changes unshielded policy behavior while preserving deployed success, deployed safety, strict fixed-risk coverage, and robust utility against CBF/MPC/recovery baselines.
