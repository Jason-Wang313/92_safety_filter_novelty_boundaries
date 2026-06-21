import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 920617
SEEDS = list(range(10))
EPISODES = 32
STRESS_EPISODES = 60
FIXED_RISK_EPISODES = 24

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


TASKS = {
    "narrow_gap_navigation": {
        "constraint_density": 0.62,
        "dynamics_stiffness": 0.42,
        "contact_risk": 0.28,
        "human_uncertainty": 0.10,
        "recoverability": 0.58,
        "reward_pressure": 0.50,
    },
    "human_zone_reaching": {
        "constraint_density": 0.48,
        "dynamics_stiffness": 0.34,
        "contact_risk": 0.32,
        "human_uncertainty": 0.62,
        "recoverability": 0.54,
        "reward_pressure": 0.52,
    },
    "unstable_stack_insertion": {
        "constraint_density": 0.44,
        "dynamics_stiffness": 0.72,
        "contact_risk": 0.66,
        "human_uncertainty": 0.12,
        "recoverability": 0.40,
        "reward_pressure": 0.60,
    },
    "contact_rich_door_push": {
        "constraint_density": 0.38,
        "dynamics_stiffness": 0.58,
        "contact_risk": 0.74,
        "human_uncertainty": 0.22,
        "recoverability": 0.48,
        "reward_pressure": 0.56,
    },
    "slippery_payload_transfer": {
        "constraint_density": 0.50,
        "dynamics_stiffness": 0.46,
        "contact_risk": 0.64,
        "human_uncertainty": 0.26,
        "recoverability": 0.36,
        "reward_pressure": 0.66,
    },
    "deformable_obstacle_threading": {
        "constraint_density": 0.56,
        "dynamics_stiffness": 0.40,
        "contact_risk": 0.52,
        "human_uncertainty": 0.38,
        "recoverability": 0.42,
        "reward_pressure": 0.58,
    },
}

SPLITS = {
    "nominal_safety": {
        "constraint_shift": 0.00,
        "novelty": 0.00,
        "lag": 0.00,
        "human_motion": 0.00,
        "contact_mode": 0.00,
        "sensor_noise": 0.00,
    },
    "constraint_shift": {
        "constraint_shift": 0.24,
        "novelty": 0.05,
        "lag": 0.05,
        "human_motion": 0.04,
        "contact_mode": 0.05,
        "sensor_noise": 0.04,
    },
    "novel_obstacle_shift": {
        "constraint_shift": 0.08,
        "novelty": 0.28,
        "lag": 0.05,
        "human_motion": 0.08,
        "contact_mode": 0.08,
        "sensor_noise": 0.06,
    },
    "actuator_lag_shift": {
        "constraint_shift": 0.08,
        "novelty": 0.08,
        "lag": 0.28,
        "human_motion": 0.08,
        "contact_mode": 0.06,
        "sensor_noise": 0.08,
    },
    "human_motion_shift": {
        "constraint_shift": 0.06,
        "novelty": 0.08,
        "lag": 0.08,
        "human_motion": 0.30,
        "contact_mode": 0.06,
        "sensor_noise": 0.08,
    },
    "contact_mode_shift": {
        "constraint_shift": 0.10,
        "novelty": 0.10,
        "lag": 0.08,
        "human_motion": 0.08,
        "contact_mode": 0.30,
        "sensor_noise": 0.10,
    },
    "low_signal_high_risk_shift": {
        "constraint_shift": 0.24,
        "novelty": 0.24,
        "lag": 0.18,
        "human_motion": 0.22,
        "contact_mode": 0.22,
        "sensor_noise": 0.26,
    },
    "combined_safety_stress": {
        "constraint_shift": 0.22,
        "novelty": 0.24,
        "lag": 0.24,
        "human_motion": 0.22,
        "contact_mode": 0.24,
        "sensor_noise": 0.20,
    },
}

HARD_SPLITS = [
    "actuator_lag_shift",
    "contact_mode_shift",
    "low_signal_high_risk_shift",
    "combined_safety_stress",
]

METHODS = [
    "unfiltered_policy",
    "action_clipping_filter",
    "geometric_projection_filter",
    "cbf_safety_filter",
    "robust_mpc_shield",
    "conformal_uncertainty_shield",
    "recovery_policy_shield",
    "shielded_behavior_cloning",
    "lagrangian_safe_rl",
    "adversarial_safety_critic",
    "safety_filter_distillation",
    "boundary_learning_filter_v4",
    "counterfactual_boundary_learning_filter_v5",
    "oracle_boundary_teacher",
]

PROPOSAL = "counterfactual_boundary_learning_filter_v5"
ORACLE = "oracle_boundary_teacher"
NON_ORACLE = [m for m in METHODS if m != ORACLE]

ABLATIONS = [
    "full_counterfactual_boundary_v5",
    "minus_counterfactual_labels",
    "minus_unshielded_replay",
    "minus_intervention_gradient",
    "minus_boundary_margin_loss",
    "minus_recovery_feasibility",
    "minus_calibration_layer",
    "distill_only_boundary",
    "cbf_feedback_only",
    "clipping_feedback_only",
]

STRESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FIXED_RISK_METHODS = [
    PROPOSAL,
    "cbf_safety_filter",
    "robust_mpc_shield",
    "conformal_uncertainty_shield",
    "recovery_policy_shield",
    "safety_filter_distillation",
]
FIXED_RISK_SPLITS = ["low_signal_high_risk_shift", "combined_safety_stress"]
FIXED_RISK_BUDGETS = [0.0, 0.05, 0.10, 0.15]

METRICS = [
    "task_success",
    "deployed_violation",
    "unshielded_violation",
    "intervention_rate",
    "intervention_severity",
    "filter_dependence",
    "boundary_f1",
    "boundary_ece",
    "transfer_violation",
    "recovery_success",
    "robust_utility",
    "mechanism_utility",
    "action_regret",
]

THRESHOLDS = {
    "unfiltered_policy": 1.10,
    "action_clipping_filter": 0.42,
    "geometric_projection_filter": 0.45,
    "cbf_safety_filter": 0.36,
    "robust_mpc_shield": 0.38,
    "conformal_uncertainty_shield": 0.30,
    "recovery_policy_shield": 0.47,
    "shielded_behavior_cloning": 0.50,
    "lagrangian_safe_rl": 0.48,
    "adversarial_safety_critic": 0.44,
    "safety_filter_distillation": 0.46,
    "boundary_learning_filter_v4": 0.49,
    PROPOSAL: 0.52,
    ORACLE: 0.50,
    "full_counterfactual_boundary_v5": 0.52,
    "minus_counterfactual_labels": 0.48,
    "minus_unshielded_replay": 0.48,
    "minus_intervention_gradient": 0.50,
    "minus_boundary_margin_loss": 0.48,
    "minus_recovery_feasibility": 0.50,
    "minus_calibration_layer": 0.43,
    "distill_only_boundary": 0.48,
    "cbf_feedback_only": 0.38,
    "clipping_feedback_only": 0.43,
}

METHOD_PROFILE = {
    "unfiltered_policy": {"shield": 0.00, "policy": 0.00, "depend": 0.00, "success": 0.08, "cost": 0.00, "severity": 0.00, "recovery": 0.00, "calib": 0.20},
    "action_clipping_filter": {"shield": 0.34, "policy": -0.02, "depend": 0.86, "success": 0.05, "cost": 0.25, "severity": 0.30, "recovery": 0.02, "calib": 0.18},
    "geometric_projection_filter": {"shield": 0.44, "policy": 0.00, "depend": 0.82, "success": 0.13, "cost": 0.23, "severity": 0.26, "recovery": 0.03, "calib": 0.14},
    "cbf_safety_filter": {"shield": 0.62, "policy": 0.01, "depend": 0.94, "success": 0.25, "cost": 0.31, "severity": 0.33, "recovery": 0.05, "calib": 0.10},
    "robust_mpc_shield": {"shield": 0.66, "policy": 0.04, "depend": 0.90, "success": 0.32, "cost": 0.35, "severity": 0.30, "recovery": 0.12, "calib": 0.11},
    "conformal_uncertainty_shield": {"shield": 0.58, "policy": 0.05, "depend": 0.97, "success": 0.22, "cost": 0.40, "severity": 0.42, "recovery": 0.05, "calib": 0.07},
    "recovery_policy_shield": {"shield": 0.50, "policy": 0.10, "depend": 0.74, "success": 0.27, "cost": 0.28, "severity": 0.24, "recovery": 0.25, "calib": 0.13},
    "shielded_behavior_cloning": {"shield": 0.24, "policy": 0.13, "depend": 0.46, "success": 0.12, "cost": 0.16, "severity": 0.18, "recovery": 0.05, "calib": 0.16},
    "lagrangian_safe_rl": {"shield": 0.28, "policy": 0.16, "depend": 0.42, "success": 0.13, "cost": 0.15, "severity": 0.16, "recovery": 0.04, "calib": 0.15},
    "adversarial_safety_critic": {"shield": 0.30, "policy": 0.18, "depend": 0.44, "success": 0.10, "cost": 0.18, "severity": 0.20, "recovery": 0.05, "calib": 0.13},
    "safety_filter_distillation": {"shield": 0.34, "policy": 0.20, "depend": 0.36, "success": 0.15, "cost": 0.17, "severity": 0.17, "recovery": 0.07, "calib": 0.11},
    "boundary_learning_filter_v4": {"shield": 0.31, "policy": 0.25, "depend": 0.35, "success": 0.12, "cost": 0.18, "severity": 0.17, "recovery": 0.07, "calib": 0.12},
    PROPOSAL: {"shield": 0.38, "policy": 0.30, "depend": 0.28, "success": 0.16, "cost": 0.20, "severity": 0.18, "recovery": 0.09, "calib": 0.09},
    ORACLE: {"shield": 0.72, "policy": 0.42, "depend": 0.18, "success": 0.22, "cost": 0.12, "severity": 0.10, "recovery": 0.25, "calib": 0.03},
}

ABLATION_PROFILE = {
    "full_counterfactual_boundary_v5": METHOD_PROFILE[PROPOSAL],
    "minus_counterfactual_labels": {"shield": 0.34, "policy": 0.19, "depend": 0.48, "success": 0.15, "cost": 0.22, "severity": 0.20, "recovery": 0.08, "calib": 0.13},
    "minus_unshielded_replay": {"shield": 0.35, "policy": 0.18, "depend": 0.50, "success": 0.15, "cost": 0.22, "severity": 0.20, "recovery": 0.07, "calib": 0.13},
    "minus_intervention_gradient": {"shield": 0.35, "policy": 0.21, "depend": 0.42, "success": 0.11, "cost": 0.25, "severity": 0.25, "recovery": 0.07, "calib": 0.12},
    "minus_boundary_margin_loss": {"shield": 0.32, "policy": 0.17, "depend": 0.50, "success": 0.16, "cost": 0.22, "severity": 0.21, "recovery": 0.08, "calib": 0.14},
    "minus_recovery_feasibility": {"shield": 0.36, "policy": 0.25, "depend": 0.34, "success": 0.08, "cost": 0.21, "severity": 0.20, "recovery": 0.00, "calib": 0.10},
    "minus_calibration_layer": {"shield": 0.38, "policy": 0.29, "depend": 0.30, "success": 0.14, "cost": 0.29, "severity": 0.29, "recovery": 0.08, "calib": 0.17},
    "distill_only_boundary": METHOD_PROFILE["safety_filter_distillation"],
    "cbf_feedback_only": METHOD_PROFILE["cbf_safety_filter"],
    "clipping_feedback_only": METHOD_PROFILE["action_clipping_filter"],
}


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def stable_offset(*parts):
    total = 0
    for part in parts:
        for ch in str(part):
            total = (total * 131 + ord(ch)) % 1_000_003
    return total


def ci95(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return 1.96 * math.sqrt(var) / math.sqrt(len(values))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def f1_score(labels, preds):
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def ece(labels, scores, bins=10):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    out = 0.0
    for lo in np.linspace(0.0, 0.9, bins):
        hi = lo + 0.1
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return out


def generate_case(task_name, split_name, seed, episode, stress_level=None):
    task = TASKS[task_name]
    split = SPLITS[split_name]
    stress = 1.0 if stress_level is None else stress_level
    rng = np.random.default_rng(BASE_SEED + stable_offset(task_name, split_name, seed, episode, stress_level))

    constraint_density = clamp(task["constraint_density"] + stress * split["constraint_shift"] + rng.normal(0, 0.045), 0.02, 0.98)
    novelty = clamp(stress * split["novelty"] + rng.normal(0, 0.035), 0.0, 0.90)
    lag = clamp(stress * split["lag"] + rng.normal(0, 0.025), 0.0, 0.80)
    human_motion = clamp(task["human_uncertainty"] + stress * split["human_motion"] + rng.normal(0, 0.04), 0.0, 0.95)
    contact_mode = clamp(task["contact_risk"] + stress * split["contact_mode"] + rng.normal(0, 0.05), 0.02, 0.98)
    sensor_noise = clamp(stress * split["sensor_noise"] + rng.normal(0, 0.025), 0.0, 0.85)
    dynamics_stiffness = clamp(task["dynamics_stiffness"] + 0.10 * lag + rng.normal(0, 0.04), 0.02, 0.98)
    recoverability = clamp(task["recoverability"] - 0.22 * novelty - 0.18 * lag - 0.12 * contact_mode + rng.normal(0, 0.04), 0.02, 0.95)
    reward_pressure = clamp(task["reward_pressure"] + 0.16 * novelty + rng.normal(0, 0.04), 0.02, 0.98)

    hazard_logit = (
        -1.10
        + 1.55 * constraint_density
        + 1.10 * novelty
        + 1.00 * contact_mode
        + 0.84 * lag
        + 0.62 * human_motion
        + 0.58 * sensor_noise
        + 0.40 * dynamics_stiffness
        + rng.normal(0, 0.33)
    )
    hazard_probability = sigmoid(hazard_logit)
    boundary_label = int(rng.random() < hazard_probability)
    unsafe_action_pressure = clamp(0.20 + 0.62 * reward_pressure + 0.48 * hazard_probability + rng.normal(0, 0.11), 0.0, 1.0)

    nominal_score = clamp(0.18 + 0.58 * hazard_probability + 0.12 * novelty + 0.10 * sensor_noise + rng.normal(0, 0.10), 0.0, 1.0)
    geometric_score = clamp(0.20 + 0.44 * constraint_density + 0.28 * contact_mode + 0.12 * novelty + rng.normal(0, 0.12), 0.0, 1.0)
    dynamics_score = clamp(0.18 + 0.40 * dynamics_stiffness + 0.32 * lag + 0.18 * contact_mode + rng.normal(0, 0.11), 0.0, 1.0)
    uncertainty_score = clamp(0.14 + 0.42 * novelty + 0.34 * sensor_noise + 0.25 * human_motion + rng.normal(0, 0.08), 0.0, 1.0)
    recovery_score = clamp(0.18 + 0.60 * (1.0 - recoverability) + 0.18 * hazard_probability + 0.10 * contact_mode + rng.normal(0, 0.08), 0.0, 1.0)
    boundary_score_base = clamp(0.20 + 0.46 * hazard_probability + 0.18 * novelty + 0.18 * contact_mode + 0.10 * lag + rng.normal(0, 0.08), 0.0, 1.0)

    return {
        "seed": seed,
        "task": task_name,
        "split": split_name,
        "episode": episode,
        "stress_level": "" if stress_level is None else stress_level,
        "constraint_density": constraint_density,
        "novelty": novelty,
        "lag": lag,
        "human_motion": human_motion,
        "contact_mode": contact_mode,
        "sensor_noise": sensor_noise,
        "dynamics_stiffness": dynamics_stiffness,
        "recoverability": recoverability,
        "reward_pressure": reward_pressure,
        "hazard_probability": hazard_probability,
        "boundary_label": boundary_label,
        "unsafe_action_pressure": unsafe_action_pressure,
        "nominal_score": nominal_score,
        "geometric_score": geometric_score,
        "dynamics_score": dynamics_score,
        "uncertainty_score": uncertainty_score,
        "recovery_score": recovery_score,
        "boundary_score_base": boundary_score_base,
    }


def method_score(case, method):
    rng = np.random.default_rng(
        BASE_SEED + stable_offset("score", method, case["task"], case["split"], case["seed"], case["episode"], case["stress_level"])
    )
    n = case["nominal_score"]
    g = case["geometric_score"]
    d = case["dynamics_score"]
    u = case["uncertainty_score"]
    r = case["recovery_score"]
    b = case["boundary_score_base"]
    h = case["hazard_probability"]

    if method == "unfiltered_policy":
        raw = 0.10 + 0.10 * n
        sigma = 0.18
    elif method == "action_clipping_filter":
        raw = 0.35 * n + 0.40 * case["unsafe_action_pressure"] + 0.10 * d
        sigma = 0.13
    elif method == "geometric_projection_filter":
        raw = 0.55 * g + 0.20 * n + 0.10 * d
        sigma = 0.12
    elif method == "cbf_safety_filter":
        raw = 0.42 * d + 0.32 * g + 0.16 * h + 0.08 * u
        sigma = 0.09
    elif method == "robust_mpc_shield":
        raw = 0.30 * d + 0.25 * g + 0.20 * r + 0.16 * h + 0.08 * u
        sigma = 0.10
    elif method == "conformal_uncertainty_shield":
        raw = 0.34 * u + 0.22 * h + 0.20 * n + 0.14 * g
        sigma = 0.075
    elif method == "recovery_policy_shield":
        raw = 0.35 * r + 0.25 * h + 0.16 * d + 0.12 * g + 0.08 * u
        sigma = 0.11
    elif method == "shielded_behavior_cloning":
        raw = 0.32 * b + 0.22 * n + 0.20 * g + 0.14 * r
        sigma = 0.13
    elif method == "lagrangian_safe_rl":
        raw = 0.28 * b + 0.25 * h + 0.20 * d + 0.12 * g
        sigma = 0.13
    elif method == "adversarial_safety_critic":
        raw = 0.30 * b + 0.24 * u + 0.20 * h + 0.12 * d
        sigma = 0.12
    elif method == "safety_filter_distillation":
        raw = 0.34 * b + 0.22 * h + 0.18 * r + 0.14 * d + 0.08 * u
        sigma = 0.10
    elif method == "boundary_learning_filter_v4":
        raw = 0.38 * b + 0.20 * h + 0.16 * r + 0.14 * d + 0.08 * g
        sigma = 0.11
    elif method == PROPOSAL:
        raw = 0.34 * b + 0.22 * h + 0.18 * r + 0.14 * d + 0.10 * u + 0.06 * g - 0.05 * case["lag"]
        sigma = 0.085
    elif method == ORACLE:
        raw = 0.90 * h + 0.05 * b + 0.05 * r
        sigma = 0.035
    elif method == "full_counterfactual_boundary_v5":
        raw = 0.34 * b + 0.22 * h + 0.18 * r + 0.14 * d + 0.10 * u + 0.06 * g - 0.05 * case["lag"]
        sigma = 0.085
    elif method == "minus_counterfactual_labels":
        raw = 0.34 * b + 0.22 * d + 0.18 * r + 0.12 * g + 0.10 * u
        sigma = 0.115
    elif method == "minus_unshielded_replay":
        raw = 0.36 * b + 0.20 * h + 0.16 * d + 0.14 * r + 0.08 * u
        sigma = 0.120
    elif method == "minus_intervention_gradient":
        raw = 0.34 * b + 0.22 * h + 0.18 * r + 0.15 * d + 0.08 * u
        sigma = 0.105
    elif method == "minus_boundary_margin_loss":
        raw = 0.32 * b + 0.20 * h + 0.18 * r + 0.14 * d + 0.10 * u
        sigma = 0.125
    elif method == "minus_recovery_feasibility":
        raw = 0.40 * b + 0.25 * h + 0.18 * d + 0.12 * u
        sigma = 0.095
    elif method == "minus_calibration_layer":
        raw = 0.42 * b + 0.25 * h + 0.18 * r + 0.14 * d + 0.12 * u
        sigma = 0.145
    elif method == "distill_only_boundary":
        raw = 0.34 * b + 0.22 * h + 0.18 * r + 0.14 * d + 0.08 * u
        sigma = 0.10
    elif method == "cbf_feedback_only":
        raw = 0.42 * d + 0.32 * g + 0.16 * h + 0.08 * u
        sigma = 0.09
    elif method == "clipping_feedback_only":
        raw = 0.35 * n + 0.40 * case["unsafe_action_pressure"] + 0.10 * d
        sigma = 0.13
    else:
        raise KeyError(method)
    return clamp(raw + rng.normal(0, sigma))


def evaluate_method(case, method, fixed_risk_budget=None):
    score = method_score(case, method)
    threshold = THRESHOLDS[method]
    if fixed_risk_budget is None:
        accepted = 1
        intervention = int(score >= threshold)
    else:
        accept_threshold = fixed_risk_budget + {
            "conformal_uncertainty_shield": 0.02,
            "robust_mpc_shield": 0.015,
            "cbf_safety_filter": 0.01,
            "recovery_policy_shield": 0.01,
            "safety_filter_distillation": 0.00,
            PROPOSAL: 0.00,
        }.get(method, 0.0)
        accepted = int(score <= accept_threshold)
        intervention = int(score >= threshold and accepted)

    profile = ABLATION_PROFILE.get(method, METHOD_PROFILE.get(method))
    if profile is None:
        raise KeyError(method)

    rng = np.random.default_rng(
        BASE_SEED + stable_offset("outcome", method, fixed_risk_budget, case["task"], case["split"], case["seed"], case["episode"], case["stress_level"])
    )
    boundary_label = int(case["boundary_label"])
    predicted_boundary = int(score >= threshold)
    base_unshielded_risk = clamp(
        0.14
        + 0.68 * boundary_label
        + 0.30 * case["unsafe_action_pressure"]
        + 0.20 * case["novelty"]
        + 0.18 * case["lag"]
        + 0.16 * case["contact_mode"]
        - 0.52 * profile["policy"]
    )
    unshielded_violation = int(rng.random() < base_unshielded_risk and accepted)
    shield_effect = intervention * profile["shield"] * (0.70 + 0.30 * (1.0 - case["lag"]))
    recovery_effect = profile["recovery"] * (0.35 + 0.65 * intervention) * case["recoverability"]
    deployed_risk = clamp(base_unshielded_risk - shield_effect - recovery_effect + 0.04 * case["sensor_noise"])
    deployed_violation = int(rng.random() < deployed_risk and accepted)

    severity = intervention * profile["severity"] * (0.45 + 0.55 * score)
    intervention_severity = severity + intervention * 0.04 * case["lag"]
    dependence = intervention * profile["depend"] * (0.70 + 0.30 * boundary_label)
    transfer_risk = clamp(base_unshielded_risk - 0.62 * profile["policy"] - 0.22 * profile["recovery"] + 0.10 * case["novelty"] + 0.08 * case["contact_mode"])
    transfer_violation = int(rng.random() < transfer_risk and accepted)

    success_prob = clamp(
        0.74
        - 0.34 * case["constraint_density"]
        - 0.22 * case["novelty"]
        - 0.18 * case["lag"]
        - 0.20 * case["contact_mode"]
        - 0.10 * intervention_severity
        + profile["success"]
        + 0.16 * recovery_effect
        - 0.44 * deployed_violation
    )
    task_success = int(rng.random() < success_prob and accepted)
    recovery_success = int(deployed_violation == 0 and unshielded_violation == 1 and (intervention or profile["policy"] > 0.20))
    action_regret = clamp(
        0.20
        + 0.32 * deployed_violation
        + 0.20 * (1 - task_success)
        + 0.18 * intervention_severity
        + 0.12 * dependence
        - 0.12 * profile["policy"]
        - 0.08 * profile["recovery"]
    )
    robust_utility = (
        task_success
        - 0.58 * deployed_violation
        - 0.26 * intervention_severity
        - 0.18 * dependence
        - 0.16 * action_regret
    )
    mechanism_utility = (
        (1 - unshielded_violation)
        + 0.30 * (1 - transfer_violation)
        - 0.35 * dependence
        - 0.18 * intervention_severity
        - 0.10 * action_regret
    )

    return {
        "seed": case["seed"],
        "task": case["task"],
        "split": case["split"],
        "episode": case["episode"],
        "stress_level": case["stress_level"],
        "method": method,
        "threshold": threshold,
        "boundary_score": score,
        "boundary_label": boundary_label,
        "predicted_boundary": predicted_boundary,
        "accepted": accepted,
        "task_success": task_success,
        "deployed_violation": deployed_violation,
        "unshielded_violation": unshielded_violation,
        "intervention": intervention,
        "intervention_severity": intervention_severity,
        "filter_dependence": dependence,
        "transfer_violation": transfer_violation,
        "recovery_success": recovery_success,
        "robust_utility": robust_utility,
        "mechanism_utility": mechanism_utility,
        "action_regret": action_regret,
        "hazard_probability": case["hazard_probability"],
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataset_row(case):
    return {
        "seed": case["seed"],
        "task": case["task"],
        "split": case["split"],
        "episode": case["episode"],
        "constraint_density": f"{case['constraint_density']:.6f}",
        "novelty": f"{case['novelty']:.6f}",
        "lag": f"{case['lag']:.6f}",
        "human_motion": f"{case['human_motion']:.6f}",
        "contact_mode": f"{case['contact_mode']:.6f}",
        "sensor_noise": f"{case['sensor_noise']:.6f}",
        "recoverability": f"{case['recoverability']:.6f}",
        "hazard_probability": f"{case['hazard_probability']:.6f}",
        "boundary_label": case["boundary_label"],
        "unsafe_action_pressure": f"{case['unsafe_action_pressure']:.6f}",
    }


def rollout_csv_row(row):
    out = dict(row)
    for key in [
        "threshold",
        "boundary_score",
        "intervention_severity",
        "filter_dependence",
        "robust_utility",
        "mechanism_utility",
        "action_regret",
        "hazard_probability",
    ]:
        out[key] = f"{float(out[key]):.6f}"
    return out


def summarize_rollouts(rows):
    labels = [int(r["boundary_label"]) for r in rows]
    preds = [int(r["predicted_boundary"]) for r in rows]
    scores = [float(r["boundary_score"]) for r in rows]
    return {
        "task_success": mean(int(r["task_success"]) for r in rows),
        "deployed_violation": mean(int(r["deployed_violation"]) for r in rows),
        "unshielded_violation": mean(int(r["unshielded_violation"]) for r in rows),
        "intervention_rate": mean(int(r["intervention"]) for r in rows),
        "intervention_severity": mean(float(r["intervention_severity"]) for r in rows),
        "filter_dependence": mean(float(r["filter_dependence"]) for r in rows),
        "boundary_f1": f1_score(labels, preds),
        "boundary_ece": ece(labels, scores),
        "transfer_violation": mean(int(r["transfer_violation"]) for r in rows),
        "recovery_success": mean(int(r["recovery_success"]) for r in rows),
        "robust_utility": mean(float(r["robust_utility"]) for r in rows),
        "mechanism_utility": mean(float(r["mechanism_utility"]) for r in rows),
        "action_regret": mean(float(r["action_regret"]) for r in rows),
    }


def group_by(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def seed_metric_rows(rows, keys):
    out = []
    for key, group in sorted(group_by(rows, keys).items()):
        metrics = summarize_rollouts(group)
        row = {k: v for k, v in zip(keys, key)}
        row.update({m: f"{metrics[m]:.6f}" for m in METRICS})
        out.append(row)
    return out


def metric_long_rows(seed_rows, group_keys):
    grouped = defaultdict(list)
    for row in seed_rows:
        key = tuple(row[k] for k in group_keys)
        for metric in METRICS:
            grouped[(key, metric)].append(float(row[metric]))
    out = []
    for (key, metric), values in sorted(grouped.items()):
        row = {k: v for k, v in zip(group_keys, key)}
        row.update({"metric": metric, "mean": f"{mean(values):.6f}", "ci95": f"{ci95(values):.6f}", "n": len(values)})
        out.append(row)
    return out


def pairwise_rows(seed_rows, group_keys, baseline_methods, reference_method=PROPOSAL):
    index = {}
    for row in seed_rows:
        key = tuple(row[k] for k in group_keys) + (row["seed"], row["method"])
        index[key] = row
    out = []
    group_values = sorted(set(tuple(row[k] for k in group_keys) for row in seed_rows))
    for group_key in group_values:
        for baseline in baseline_methods:
            for metric in METRICS:
                diffs = []
                for seed in SEEDS:
                    ref = index.get(group_key + (seed, reference_method))
                    base = index.get(group_key + (seed, baseline))
                    if ref is None or base is None:
                        continue
                    diffs.append(float(ref[metric]) - float(base[metric]))
                if diffs:
                    row = {k: v for k, v in zip(group_keys, group_key)}
                    row.update(
                        {
                            "comparison": f"{reference_method}_minus_{baseline}",
                            "metric": metric,
                            "mean": f"{mean(diffs):.6f}",
                            "ci95": f"{ci95(diffs):.6f}",
                            "lower95": f"{mean(diffs) - ci95(diffs):.6f}",
                            "upper95": f"{mean(diffs) + ci95(diffs):.6f}",
                            "better_seeds": sum(1 for d in diffs if d > 0),
                            "n": len(diffs),
                        }
                    )
                    out.append(row)
    return out


def make_main_rollouts():
    dataset_rows = []
    rollout_rows = []
    for seed in SEEDS:
        for task in TASKS:
            for split in SPLITS:
                for episode in range(EPISODES):
                    case = generate_case(task, split, seed, episode)
                    dataset_rows.append(dataset_row(case))
                    for method in METHODS:
                        rollout_rows.append(evaluate_method(case, method))
    return dataset_rows, rollout_rows


def make_ablation_rollouts():
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            for split in HARD_SPLITS:
                for episode in range(EPISODES):
                    case = generate_case(task, split, seed, episode)
                    for method in ABLATIONS:
                        rows.append(evaluate_method(case, method))
    return rows


def make_stress_rollouts():
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            for stress_level in STRESS_LEVELS:
                for method in METHODS:
                    for episode in range(STRESS_EPISODES):
                        case = generate_case(task, "combined_safety_stress", seed, episode, stress_level=stress_level)
                        row = evaluate_method(case, method)
                        row["stress_level"] = stress_level
                        rows.append(row)
    return rows


def make_fixed_risk_rollouts():
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            for split in FIXED_RISK_SPLITS:
                for budget in FIXED_RISK_BUDGETS:
                    for method in FIXED_RISK_METHODS:
                        for episode in range(FIXED_RISK_EPISODES):
                            case = generate_case(task, split, seed, episode)
                            row = evaluate_method(case, method, fixed_risk_budget=budget)
                            row["budget"] = budget
                            rows.append(row)
    return rows


def summarize_fixed_risk(rows):
    seed_out = []
    for key, group in sorted(group_by(rows, ["seed", "split", "budget", "method"]).items()):
        accepted = [r for r in group if int(r["accepted"])]
        coverage = len(accepted) / len(group) if group else 0.0
        metrics = summarize_rollouts(accepted) if accepted else {m: 0.0 for m in METRICS}
        row = {k: v for k, v in zip(["seed", "split", "budget", "method"], key)}
        row.update(
            {
                "coverage": f"{coverage:.6f}",
                "accepted_success": f"{metrics['task_success']:.6f}",
                "accepted_deployed_violation": f"{metrics['deployed_violation']:.6f}",
                "accepted_unshielded_violation": f"{metrics['unshielded_violation']:.6f}",
                "accepted_transfer_violation": f"{metrics['transfer_violation']:.6f}",
                "accepted_utility": f"{metrics['robust_utility']:.6f}",
            }
        )
        seed_out.append(row)

    metric_out = []
    fixed_metrics = [
        "coverage",
        "accepted_success",
        "accepted_deployed_violation",
        "accepted_unshielded_violation",
        "accepted_transfer_violation",
        "accepted_utility",
    ]
    for key, group in sorted(group_by(seed_out, ["split", "budget", "method"]).items()):
        for metric in fixed_metrics:
            values = [float(r[metric]) for r in group]
            row = {k: v for k, v in zip(["split", "budget", "method"], key)}
            row.update({"metric": metric, "mean": f"{mean(values):.6f}", "ci95": f"{ci95(values):.6f}", "n": len(values)})
            metric_out.append(row)

    pairwise = []
    index = {}
    for row in seed_out:
        index[(row["seed"], row["split"], row["budget"], row["method"])] = row
    for split in FIXED_RISK_SPLITS:
        for budget in FIXED_RISK_BUDGETS:
            for baseline in [m for m in FIXED_RISK_METHODS if m != PROPOSAL]:
                for metric in fixed_metrics:
                    diffs = []
                    for seed in SEEDS:
                        ref = index[(seed, split, budget, PROPOSAL)]
                        base = index[(seed, split, budget, baseline)]
                        diffs.append(float(ref[metric]) - float(base[metric]))
                    pairwise.append(
                        {
                            "split": split,
                            "budget": budget,
                            "comparison": f"{PROPOSAL}_minus_{baseline}",
                            "metric": metric,
                            "mean": f"{mean(diffs):.6f}",
                            "ci95": f"{ci95(diffs):.6f}",
                            "lower95": f"{mean(diffs) - ci95(diffs):.6f}",
                            "upper95": f"{mean(diffs) + ci95(diffs):.6f}",
                            "better_seeds": sum(1 for d in diffs if d > 0),
                            "n": len(diffs),
                        }
                    )
    return seed_out, metric_out, pairwise


def hard_aggregate_rows(rollouts):
    hard = [r for r in rollouts if r["split"] in HARD_SPLITS]
    return seed_metric_rows(hard, ["seed", "method"])


def make_negative_cases(main_rows):
    by_case = defaultdict(dict)
    for row in main_rows:
        if row["split"] in HARD_SPLITS:
            key = (row["seed"], row["task"], row["split"], row["episode"])
            by_case[key][row["method"]] = row
    cases = []
    for key, methods in sorted(by_case.items()):
        if PROPOSAL not in methods:
            continue
        v5 = methods[PROPOSAL]
        candidates = [m for m in ["robust_mpc_shield", "cbf_safety_filter", "recovery_policy_shield", "conformal_uncertainty_shield", "safety_filter_distillation"] if m in methods]
        best = max(candidates, key=lambda m: float(methods[m]["robust_utility"])) if candidates else ""
        best_row = methods.get(best, {})
        failure_mode = None
        if int(v5["unshielded_violation"]) == 0 and int(v5["deployed_violation"]) and best and not int(best_row["deployed_violation"]):
            failure_mode = "mechanism_improves_deployment_worse"
        elif not int(v5["task_success"]) and best and int(best_row["task_success"]):
            failure_mode = "baseline_success_v5_fails"
        elif float(v5["filter_dependence"]) > 0.35:
            failure_mode = "dependence_not_removed"
        elif int(v5["deployed_violation"]) and all(int(methods[m]["deployed_violation"]) for m in candidates if m in methods):
            failure_mode = "all_non_oracle_unsafe"
        if failure_mode:
            seed, task, split, episode = key
            cases.append(
                {
                    "case_id": len(cases) + 1,
                    "seed": seed,
                    "task": task,
                    "split": split,
                    "episode": episode,
                    "failure_mode": failure_mode,
                    "v5_score": f"{float(v5['boundary_score']):.6f}",
                    "v5_success": v5["task_success"],
                    "v5_deployed_violation": v5["deployed_violation"],
                    "v5_unshielded_violation": v5["unshielded_violation"],
                    "v5_dependence": f"{float(v5['filter_dependence']):.6f}",
                    "best_baseline": best,
                    "best_baseline_success": best_row.get("task_success", ""),
                    "best_baseline_deployed_violation": best_row.get("deployed_violation", ""),
                    "best_baseline_utility": f"{float(best_row.get('robust_utility', 0.0)):.6f}" if best_row else "",
                }
            )
        if len(cases) >= 24:
            break
    return cases


def metric_lookup(metric_rows, group_keys):
    out = {}
    for row in metric_rows:
        key = tuple(row[k] for k in group_keys) + (row["metric"],)
        out[key] = float(row["mean"])
    return out


def best_reference(hard_metrics, metric, lower_is_better=False, exclude_oracle=True):
    candidates = [r for r in hard_metrics if r["metric"] == metric]
    if exclude_oracle:
        candidates = [r for r in candidates if r["method"] != ORACLE]
    return min(candidates, key=lambda r: float(r["mean"])) if lower_is_better else max(candidates, key=lambda r: float(r["mean"]))


def pairwise_stat(pairwise, baseline, metric):
    comp = f"{PROPOSAL}_minus_{baseline}"
    for row in pairwise:
        if row["comparison"] == comp and row["metric"] == metric:
            return row
    return None


def plot_bar(path, title, labels, series):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + i * width - 0.4 + width / 2, values, width=width, label=name)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(hard_metric_rows, ablation_metric_rows, stress_metric_rows, fixed_metric_rows):
    hard = metric_lookup(hard_metric_rows, ["method"])
    labels = [m for m in METHODS if m != ORACLE]
    plot_bar(
        FIGURES / "safety_boundary_deployed_vs_unshielded_v5.png",
        "Hard-Aggregate Deployed vs Unshielded Safety",
        labels,
        {
            "Deployed violation": [hard[(m, "deployed_violation")] for m in labels],
            "Unshielded violation": [hard[(m, "unshielded_violation")] for m in labels],
            "Filter dependence": [hard[(m, "filter_dependence")] for m in labels],
        },
    )
    plot_bar(
        FIGURES / "safety_boundary_success_utility_v5.png",
        "Hard-Aggregate Success and Utility",
        labels,
        {
            "Success": [hard[(m, "task_success")] for m in labels],
            "Robust utility": [hard[(m, "robust_utility")] for m in labels],
            "Mechanism utility": [hard[(m, "mechanism_utility")] for m in labels],
        },
    )

    ab = metric_lookup(ablation_metric_rows, ["method"])
    plot_bar(
        FIGURES / "safety_boundary_ablation_v5.png",
        "Ablation Mechanism Utility",
        ABLATIONS,
        {
            "Mechanism utility": [ab[(m, "mechanism_utility")] for m in ABLATIONS],
            "Success": [ab[(m, "task_success")] for m in ABLATIONS],
        },
    )

    stress = metric_lookup(stress_metric_rows, ["stress_level", "method"])
    shown = [PROPOSAL, "robust_mpc_shield", "cbf_safety_filter", "recovery_policy_shield", "safety_filter_distillation", "conformal_uncertainty_shield"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in shown:
        ax.plot(STRESS_LEVELS, [stress[(level, method, "robust_utility")] for level in STRESS_LEVELS], marker="o", label=method)
    ax.set_title("Combined Safety Stress Sweep: Robust Utility")
    ax.set_xlabel("Stress level")
    ax.set_ylabel("Robust utility")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "safety_boundary_stress_sweep_v5.png", dpi=180)
    plt.close(fig)

    fixed = metric_lookup(fixed_metric_rows, ["split", "budget", "method"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in FIXED_RISK_METHODS:
        values = []
        for budget in FIXED_RISK_BUDGETS:
            vals = [fixed[(split, budget, method, "coverage")] for split in FIXED_RISK_SPLITS]
            values.append(mean(vals))
        ax.plot(FIXED_RISK_BUDGETS, values, marker="o", label=method)
    ax.set_title("Fixed-Risk Coverage on Hard Safety Splits")
    ax.set_xlabel("Risk budget")
    ax.set_ylabel("Accepted coverage")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "safety_boundary_fixed_risk_v5.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for method in labels:
        ax.scatter(hard[(method, "deployed_violation")], hard[(method, "task_success")], s=60)
        ax.text(hard[(method, "deployed_violation")] + 0.004, hard[(method, "task_success")] + 0.002, method, fontsize=7)
    ax.set_title("Task Success vs Deployed Violation")
    ax.set_xlabel("Deployed violation")
    ax.set_ylabel("Task success")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "safety_boundary_pareto_v5.png", dpi=180)
    plt.close(fig)


def write_summary(
    main_rows,
    dataset_rows,
    seed_rows_main,
    metric_rows_main,
    pairwise_main,
    hard_seed_rows,
    hard_metric_rows,
    hard_pairwise,
    ablation_rows,
    ablation_seed_rows,
    ablation_metric_rows,
    stress_rows,
    stress_seed_rows,
    stress_metric_rows,
    fixed_rows,
    fixed_seed_rows,
    fixed_metric_rows,
    fixed_pairwise,
    negative_cases,
):
    best_success = best_reference(hard_metric_rows, "task_success")
    safest = best_reference(hard_metric_rows, "deployed_violation", lower_is_better=True)
    best_unshielded = best_reference(hard_metric_rows, "unshielded_violation", lower_is_better=True)
    lowest_dependence = best_reference(hard_metric_rows, "filter_dependence", lower_is_better=True)
    best_calibration = best_reference(hard_metric_rows, "boundary_ece", lower_is_better=True)
    best_utility = best_reference(hard_metric_rows, "robust_utility")
    best_mechanism = best_reference(hard_metric_rows, "mechanism_utility")
    hard = metric_lookup(hard_metric_rows, ["method"])
    proposal = {m: hard[(PROPOSAL, m)] for m in METRICS}

    success_pair = pairwise_stat(hard_pairwise, best_success["method"], "task_success")
    safety_pair = pairwise_stat(hard_pairwise, safest["method"], "deployed_violation")
    unshielded_pair = pairwise_stat(hard_pairwise, best_unshielded["method"], "unshielded_violation")
    utility_pair = pairwise_stat(hard_pairwise, best_utility["method"], "robust_utility")
    mechanism_pair = pairwise_stat(hard_pairwise, best_mechanism["method"], "mechanism_utility")

    ab = metric_lookup(ablation_metric_rows, ["method"])
    best_ablation = max(ABLATIONS, key=lambda m: ab[(m, "mechanism_utility")])
    ablation_gate = best_ablation == "full_counterfactual_boundary_v5"

    stress = metric_lookup(stress_metric_rows, ["stress_level", "method"])
    max_level = STRESS_LEVELS[-1]
    stress_candidates = [m for m in NON_ORACLE if m != PROPOSAL]
    stress_best = max(stress_candidates, key=lambda m: stress[(max_level, m, "robust_utility")])
    stress_gate = stress[(max_level, PROPOSAL, "robust_utility")] >= stress[(max_level, stress_best, "robust_utility")]

    fixed = metric_lookup(fixed_metric_rows, ["split", "budget", "method"])
    fixed_gate = all(fixed[(split, 0.05, PROPOSAL, "coverage")] > 0.0 for split in FIXED_RISK_SPLITS)

    success_gate = float(success_pair["lower95"]) > 0.0 if success_pair else False
    deployment_gate = float(safety_pair["upper95"]) <= 0.0 if safety_pair else False
    mechanism_gate = proposal["unshielded_violation"] <= float(best_unshielded["mean"]) and proposal["filter_dependence"] <= float(lowest_dependence["mean"]) + 0.02
    calibration_gate = proposal["boundary_ece"] <= float(best_calibration["mean"])
    utility_gate = float(utility_pair["lower95"]) > 0.0 if utility_pair else False
    scope_gate = False

    lines = [
        "Paper 92 safety_filter_novelty_boundaries v5 expanded audit",
        "Terminal recommendation: KILL_ARCHIVE",
        "ICLR main ready: no",
        "Reason: expanded CPU-only safety-filter audit tests whether filters change the learned policy mechanism, but standard CBF/MPC/recovery shields still dominate deployed success and safety and no real robot or accepted high-fidelity safety benchmark evidence exists.",
        f"Main rollout rows: {len(main_rows)}",
        f"Dataset summary rows: {len(dataset_rows)}",
        f"Main seed-metric rows: {len(seed_rows_main)}",
        f"Main metric rows: {len(metric_rows_main)}",
        f"Main pairwise rows: {len(pairwise_main)}",
        f"Hard aggregate seed rows: {len(hard_seed_rows)}",
        f"Hard aggregate metric rows: {len(hard_metric_rows)}",
        f"Hard aggregate pairwise rows: {len(hard_pairwise)}",
        f"Ablation rollout rows: {len(ablation_rows)}",
        f"Ablation seed rows: {len(ablation_seed_rows)}",
        f"Ablation metric rows: {len(ablation_metric_rows)}",
        f"Stress raw rows: {len(stress_rows)}",
        f"Stress seed rows: {len(stress_seed_rows)}",
        f"Stress metric rows: {len(stress_metric_rows)}",
        f"Fixed-risk raw rows: {len(fixed_rows)}",
        f"Fixed-risk seed rows: {len(fixed_seed_rows)}",
        f"Fixed-risk metric rows: {len(fixed_metric_rows)}",
        f"Fixed-risk pairwise rows: {len(fixed_pairwise)}",
        f"Negative cases: {len(negative_cases)}",
        "",
        "Frozen hard-aggregate gate:",
        f"best_success_reference={best_success['method']}",
        f"safest_reference={safest['method']}",
        f"best_unshielded_reference={best_unshielded['method']}",
        f"lowest_dependence_reference={lowest_dependence['method']}",
        f"best_calibration_reference={best_calibration['method']}",
        f"best_utility_reference={best_utility['method']}",
        f"best_mechanism_reference={best_mechanism['method']}",
        f"proposal_success={proposal['task_success']:.5f}",
        f"best_success={float(best_success['mean']):.5f}",
        f"proposal_deployed_violation={proposal['deployed_violation']:.5f}",
        f"safest_deployed_violation={float(safest['mean']):.5f}",
        f"proposal_unshielded_violation={proposal['unshielded_violation']:.5f}",
        f"best_unshielded_violation={float(best_unshielded['mean']):.5f}",
        f"proposal_dependence={proposal['filter_dependence']:.5f}",
        f"lowest_dependence={float(lowest_dependence['mean']):.5f}",
        f"proposal_boundary_ece={proposal['boundary_ece']:.5f}",
        f"best_boundary_ece={float(best_calibration['mean']):.5f}",
        f"proposal_utility={proposal['robust_utility']:.5f}",
        f"best_utility={float(best_utility['mean']):.5f}",
        f"proposal_mechanism_utility={proposal['mechanism_utility']:.5f}",
        f"best_mechanism_utility={float(best_mechanism['mean']):.5f}",
        f"paired_success_lower95={float(success_pair['lower95']) if success_pair else 0.0:.5f}",
        f"paired_deployed_violation_upper95={float(safety_pair['upper95']) if safety_pair else 0.0:.5f}",
        f"paired_unshielded_violation_upper95={float(unshielded_pair['upper95']) if unshielded_pair else 0.0:.5f}",
        f"paired_utility_lower95={float(utility_pair['lower95']) if utility_pair else 0.0:.5f}",
        f"paired_mechanism_lower95={float(mechanism_pair['lower95']) if mechanism_pair else 0.0:.5f}",
        f"success_gate={success_gate}",
        f"deployment_gate={deployment_gate}",
        f"mechanism_gate={mechanism_gate}",
        f"calibration_gate={calibration_gate}",
        f"utility_gate={utility_gate}",
        f"ablation_gate={ablation_gate}",
        f"mechanism_best_ablation={best_ablation}",
        f"stress_gate={stress_gate}",
        f"stress_dominated_by={stress_best}",
        f"fixed_risk_gate={fixed_gate}",
        f"scope_gate={scope_gate}",
    ]

    for split in FIXED_RISK_SPLITS:
        lines.append(
            f"{split}: v5_coverage={fixed[(split, 0.05, PROPOSAL, 'coverage')]:.5f}, "
            f"v5_success={fixed[(split, 0.05, PROPOSAL, 'accepted_success')]:.5f}, "
            f"v5_deployed_violation={fixed[(split, 0.05, PROPOSAL, 'accepted_deployed_violation')]:.5f}"
        )

    lines.extend(["", "Hard aggregate metrics:"])
    for method in METHODS:
        vals = {metric: hard[(method, metric)] for metric in METRICS}
        lines.append(
            f"{method} success={vals['task_success']:.5f} deployed={vals['deployed_violation']:.5f} "
            f"unshielded={vals['unshielded_violation']:.5f} intervention={vals['intervention_rate']:.5f} "
            f"dependence={vals['filter_dependence']:.5f} boundary_f1={vals['boundary_f1']:.5f} "
            f"ece={vals['boundary_ece']:.5f} transfer={vals['transfer_violation']:.5f} "
            f"utility={vals['robust_utility']:.5f} mechanism={vals['mechanism_utility']:.5f}"
        )

    lines.extend(["", "Key paired hard-aggregate differences:"])
    interesting = {
        f"{PROPOSAL}_minus_{best_success['method']}",
        f"{PROPOSAL}_minus_{safest['method']}",
        f"{PROPOSAL}_minus_{best_unshielded['method']}",
        f"{PROPOSAL}_minus_{best_utility['method']}",
        f"{PROPOSAL}_minus_boundary_learning_filter_v4",
    }
    for row in hard_pairwise:
        if row["comparison"] in interesting:
            lines.append(
                f"{row['comparison']} {row['metric']}: mean={row['mean']} ci95={row['ci95']} "
                f"lower95={row['lower95']} upper95={row['upper95']}"
            )

    lines.extend(["", "Ablation utility:"])
    for method in ABLATIONS:
        lines.append(
            f"{method} success={ab[(method, 'task_success')]:.5f} deployed={ab[(method, 'deployed_violation')]:.5f} "
            f"unshielded={ab[(method, 'unshielded_violation')]:.5f} dependence={ab[(method, 'filter_dependence')]:.5f} "
            f"utility={ab[(method, 'robust_utility')]:.5f} mechanism={ab[(method, 'mechanism_utility')]:.5f}"
        )

    lines.extend(["", "Maximum combined stress:"])
    for method in [m for m in METHODS if m != ORACLE]:
        lines.append(
            f"{method} success={stress[(max_level, method, 'task_success')]:.5f} "
            f"deployed={stress[(max_level, method, 'deployed_violation')]:.5f} "
            f"unshielded={stress[(max_level, method, 'unshielded_violation')]:.5f} "
            f"utility={stress[(max_level, method, 'robust_utility')]:.5f}"
        )

    lines.extend(["", "Fixed-risk budget 0.05:"])
    for split in FIXED_RISK_SPLITS:
        for method in FIXED_RISK_METHODS:
            lines.append(
                f"{split} {method} coverage={fixed[(split, 0.05, method, 'coverage')]:.5f} "
                f"accepted_success={fixed[(split, 0.05, method, 'accepted_success')]:.5f} "
                f"accepted_deployed_violation={fixed[(split, 0.05, method, 'accepted_deployed_violation')]:.5f} "
                f"accepted_unshielded_violation={fixed[(split, 0.05, method, 'accepted_unshielded_violation')]:.5f}"
            )

    lines.extend(["", f"Negative cases: {len(negative_cases)}", "terminal=KILL_ARCHIVE"])
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataset_rows, main_rows = make_main_rollouts()
    ablation_rows = make_ablation_rollouts()
    stress_rows = make_stress_rollouts()
    fixed_rows = make_fixed_risk_rollouts()
    negative_cases = make_negative_cases(main_rows)

    write_csv(RESULTS / "dataset_summary.csv", dataset_rows, list(dataset_rows[0].keys()))
    write_csv(RESULTS / "rollouts.csv", [rollout_csv_row(r) for r in main_rows], list(rollout_csv_row(main_rows[0]).keys()))
    write_csv(RESULTS / "ablation_rollouts.csv", [rollout_csv_row(r) for r in ablation_rows], list(rollout_csv_row(ablation_rows[0]).keys()))
    write_csv(RESULTS / "stress_sweep_raw.csv", [rollout_csv_row(r) for r in stress_rows], list(rollout_csv_row(stress_rows[0]).keys()))
    fixed_fieldnames = list(rollout_csv_row(fixed_rows[0]).keys()) + ["budget"]
    write_csv(RESULTS / "fixed_risk_raw.csv", [rollout_csv_row(r) for r in fixed_rows], fixed_fieldnames)
    write_csv(RESULTS / "negative_cases.csv", negative_cases, list(negative_cases[0].keys()))

    seed_rows_main = seed_metric_rows(main_rows, ["seed", "split", "method"])
    metric_rows_main = metric_long_rows(seed_rows_main, ["split", "method"])
    pairwise_main = pairwise_rows(seed_rows_main, ["split"], [m for m in NON_ORACLE if m != PROPOSAL])
    hard_seed_rows = hard_aggregate_rows(main_rows)
    hard_metric_rows = metric_long_rows(hard_seed_rows, ["method"])
    hard_pairwise = pairwise_rows(hard_seed_rows, [], [m for m in NON_ORACLE if m != PROPOSAL])
    ablation_seed_rows = seed_metric_rows(ablation_rows, ["seed", "method"])
    ablation_metric_rows = metric_long_rows(ablation_seed_rows, ["method"])
    stress_seed_rows = seed_metric_rows(stress_rows, ["seed", "stress_level", "method"])
    stress_metric_rows = metric_long_rows(stress_seed_rows, ["stress_level", "method"])
    fixed_seed_rows, fixed_metric_rows, fixed_pairwise = summarize_fixed_risk(fixed_rows)

    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows_main, list(seed_rows_main[0].keys()))
    write_csv(RESULTS / "metrics.csv", metric_rows_main, list(metric_rows_main[0].keys()))
    write_csv(RESULTS / "pairwise_stats.csv", pairwise_main, list(pairwise_main[0].keys()))
    write_csv(RESULTS / "hard_aggregate_seed_metrics.csv", hard_seed_rows, list(hard_seed_rows[0].keys()))
    write_csv(RESULTS / "hard_aggregate_metrics.csv", hard_metric_rows, list(hard_metric_rows[0].keys()))
    write_csv(RESULTS / "hard_aggregate_pairwise_stats.csv", hard_pairwise, list(hard_pairwise[0].keys()))
    write_csv(RESULTS / "ablation_seed_metrics.csv", ablation_seed_rows, list(ablation_seed_rows[0].keys()))
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metric_rows, list(ablation_metric_rows[0].keys()))
    write_csv(RESULTS / "ablation_metric_long.csv", ablation_metric_rows, list(ablation_metric_rows[0].keys()))
    write_csv(RESULTS / "stress_sweep_seed_metrics.csv", stress_seed_rows, list(stress_seed_rows[0].keys()))
    write_csv(RESULTS / "stress_sweep.csv", stress_metric_rows, list(stress_metric_rows[0].keys()))
    write_csv(RESULTS / "stress_sweep_metric_long.csv", stress_metric_rows, list(stress_metric_rows[0].keys()))
    write_csv(RESULTS / "fixed_risk_seed_metrics.csv", fixed_seed_rows, list(fixed_seed_rows[0].keys()))
    write_csv(RESULTS / "fixed_risk_metrics.csv", fixed_metric_rows, list(fixed_metric_rows[0].keys()))
    write_csv(RESULTS / "fixed_risk_pairwise.csv", fixed_pairwise, list(fixed_pairwise[0].keys()))

    make_figures(hard_metric_rows, ablation_metric_rows, stress_metric_rows, fixed_metric_rows)
    write_summary(
        main_rows,
        dataset_rows,
        seed_rows_main,
        metric_rows_main,
        pairwise_main,
        hard_seed_rows,
        hard_metric_rows,
        hard_pairwise,
        ablation_rows,
        ablation_seed_rows,
        ablation_metric_rows,
        stress_rows,
        stress_seed_rows,
        stress_metric_rows,
        fixed_rows,
        fixed_seed_rows,
        fixed_metric_rows,
        fixed_pairwise,
        negative_cases,
    )
    print("Paper 92 v5 expanded audit complete")
    print((RESULTS / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
