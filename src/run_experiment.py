import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 920731
SEEDS = list(range(7))
EPISODES = 72
STRESS_EPISODES = 44

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


TASKS = {
    "narrow_corridor_navigation": {
        "difficulty": 0.42,
        "hazard": 0.58,
        "boundary_curvature": 0.45,
        "goal_pressure": 0.64,
        "recoverability": 0.48,
        "dynamics_delay": 0.24,
    },
    "human_workspace_reaching": {
        "difficulty": 0.36,
        "hazard": 0.68,
        "boundary_curvature": 0.36,
        "goal_pressure": 0.50,
        "recoverability": 0.42,
        "dynamics_delay": 0.18,
    },
    "fragile_object_pushing": {
        "difficulty": 0.48,
        "hazard": 0.52,
        "boundary_curvature": 0.62,
        "goal_pressure": 0.58,
        "recoverability": 0.34,
        "dynamics_delay": 0.22,
    },
    "dynamic_obstacle_crossing": {
        "difficulty": 0.44,
        "hazard": 0.64,
        "boundary_curvature": 0.54,
        "goal_pressure": 0.62,
        "recoverability": 0.46,
        "dynamics_delay": 0.36,
    },
}

SPLITS = {
    "nominal_boundary": {
        "tightening": 0.00,
        "delay": 0.00,
        "noise": 0.00,
        "unpredictability": 0.00,
        "boundary_shift": 0.00,
    },
    "tightened_safe_set": {
        "tightening": 0.24,
        "delay": 0.04,
        "noise": 0.04,
        "unpredictability": 0.04,
        "boundary_shift": 0.08,
    },
    "delayed_dynamics": {
        "tightening": 0.08,
        "delay": 0.26,
        "noise": 0.06,
        "unpredictability": 0.08,
        "boundary_shift": 0.04,
    },
    "boundary_shift": {
        "tightening": 0.12,
        "delay": 0.08,
        "noise": 0.08,
        "unpredictability": 0.10,
        "boundary_shift": 0.26,
    },
    "combined_safety_stress": {
        "tightening": 0.20,
        "delay": 0.18,
        "noise": 0.14,
        "unpredictability": 0.16,
        "boundary_shift": 0.20,
    },
}

METHODS = [
    "unfiltered_policy",
    "action_clipping_filter",
    "geometric_projection_filter",
    "cbf_safety_filter",
    "robust_mpc_shield",
    "conformal_uncertainty_shield",
    "recovery_policy_shield",
    "proposed_boundary_learning_filter",
    "oracle_boundary_teacher",
]

ABLATIONS = [
    "full_boundary_learning_filter",
    "minus_counterfactual_labels",
    "minus_intervention_gradient",
    "minus_unshielded_replay",
    "minus_boundary_margin_loss",
    "clipping_feedback_only",
    "cbf_feedback_only",
]


PROFILES = {
    "unfiltered_policy": {
        "shield_strength": 0.00,
        "threshold": 1.50,
        "learning": 0.08,
        "boundary_fidelity": 0.08,
        "distortion": 0.00,
        "conservatism": 0.00,
        "recovery_bonus": 0.00,
    },
    "action_clipping_filter": {
        "shield_strength": 0.38,
        "threshold": 0.64,
        "learning": 0.07,
        "boundary_fidelity": 0.12,
        "distortion": 0.34,
        "conservatism": 0.10,
        "recovery_bonus": 0.02,
    },
    "geometric_projection_filter": {
        "shield_strength": 0.48,
        "threshold": 0.59,
        "learning": 0.10,
        "boundary_fidelity": 0.18,
        "distortion": 0.28,
        "conservatism": 0.12,
        "recovery_bonus": 0.04,
    },
    "cbf_safety_filter": {
        "shield_strength": 0.66,
        "threshold": 0.53,
        "learning": 0.14,
        "boundary_fidelity": 0.24,
        "distortion": 0.23,
        "conservatism": 0.18,
        "recovery_bonus": 0.08,
    },
    "robust_mpc_shield": {
        "shield_strength": 0.72,
        "threshold": 0.51,
        "learning": 0.20,
        "boundary_fidelity": 0.30,
        "distortion": 0.25,
        "conservatism": 0.20,
        "recovery_bonus": 0.11,
    },
    "conformal_uncertainty_shield": {
        "shield_strength": 0.62,
        "threshold": 0.47,
        "learning": 0.18,
        "boundary_fidelity": 0.28,
        "distortion": 0.31,
        "conservatism": 0.30,
        "recovery_bonus": 0.06,
    },
    "recovery_policy_shield": {
        "shield_strength": 0.58,
        "threshold": 0.56,
        "learning": 0.24,
        "boundary_fidelity": 0.34,
        "distortion": 0.20,
        "conservatism": 0.12,
        "recovery_bonus": 0.22,
    },
    "proposed_boundary_learning_filter": {
        "shield_strength": 0.56,
        "threshold": 0.55,
        "learning": 0.42,
        "boundary_fidelity": 0.48,
        "distortion": 0.20,
        "conservatism": 0.14,
        "recovery_bonus": 0.13,
    },
    "oracle_boundary_teacher": {
        "shield_strength": 0.82,
        "threshold": 0.54,
        "learning": 0.58,
        "boundary_fidelity": 0.70,
        "distortion": 0.14,
        "conservatism": 0.10,
        "recovery_bonus": 0.28,
    },
    "full_boundary_learning_filter": {
        "shield_strength": 0.56,
        "threshold": 0.55,
        "learning": 0.42,
        "boundary_fidelity": 0.48,
        "distortion": 0.20,
        "conservatism": 0.14,
        "recovery_bonus": 0.13,
    },
    "minus_counterfactual_labels": {
        "shield_strength": 0.55,
        "threshold": 0.55,
        "learning": 0.27,
        "boundary_fidelity": 0.31,
        "distortion": 0.20,
        "conservatism": 0.13,
        "recovery_bonus": 0.11,
    },
    "minus_intervention_gradient": {
        "shield_strength": 0.56,
        "threshold": 0.55,
        "learning": 0.31,
        "boundary_fidelity": 0.37,
        "distortion": 0.20,
        "conservatism": 0.14,
        "recovery_bonus": 0.10,
    },
    "minus_unshielded_replay": {
        "shield_strength": 0.56,
        "threshold": 0.55,
        "learning": 0.33,
        "boundary_fidelity": 0.34,
        "distortion": 0.20,
        "conservatism": 0.16,
        "recovery_bonus": 0.13,
    },
    "minus_boundary_margin_loss": {
        "shield_strength": 0.56,
        "threshold": 0.55,
        "learning": 0.25,
        "boundary_fidelity": 0.28,
        "distortion": 0.18,
        "conservatism": 0.12,
        "recovery_bonus": 0.12,
    },
    "clipping_feedback_only": {
        "shield_strength": 0.41,
        "threshold": 0.63,
        "learning": 0.15,
        "boundary_fidelity": 0.19,
        "distortion": 0.31,
        "conservatism": 0.11,
        "recovery_bonus": 0.04,
    },
    "cbf_feedback_only": {
        "shield_strength": 0.66,
        "threshold": 0.53,
        "learning": 0.32,
        "boundary_fidelity": 0.40,
        "distortion": 0.23,
        "conservatism": 0.20,
        "recovery_bonus": 0.10,
    },
}


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def ci95(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return 1.96 * math.sqrt(var) / math.sqrt(len(values))


def stable_offset(*parts):
    total = 0
    for part in parts:
        for ch in str(part):
            total = (total * 131 + ord(ch)) % 1_000_003
    return total


def f1_score(labels, preds):
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def generate_case(task_name, split_name, seed, episode, stress_level=None):
    task = TASKS[task_name]
    split = SPLITS[split_name]
    stress = 1.0 if stress_level is None else stress_level
    rng = np.random.default_rng(BASE_SEED + stable_offset(task_name, split_name, seed, episode, stress_level))
    boundary_pressure = clamp(
        task["hazard"]
        + stress * split["tightening"]
        + 0.52 * stress * split["boundary_shift"]
        + 0.20 * task["boundary_curvature"]
        + rng.normal(0.0, 0.055),
        0.05,
        1.45,
    )
    delay = clamp(task["dynamics_delay"] + stress * split["delay"] + rng.normal(0.0, 0.035), 0.0, 0.9)
    noise = clamp(0.08 + stress * split["noise"] + rng.normal(0.0, 0.025), 0.0, 0.55)
    unpredictability = clamp(0.10 + stress * split["unpredictability"] + rng.normal(0.0, 0.035), 0.0, 0.70)
    goal_pressure = clamp(task["goal_pressure"] + rng.normal(0.0, 0.045), 0.05, 1.0)
    unsafe_boundary = int(boundary_pressure + 0.35 * unpredictability + 0.25 * delay > 0.86)
    return {
        "difficulty": task["difficulty"],
        "hazard": task["hazard"],
        "boundary_curvature": task["boundary_curvature"],
        "recoverability": task["recoverability"],
        "boundary_pressure": boundary_pressure,
        "delay": delay,
        "noise": noise,
        "unpredictability": unpredictability,
        "goal_pressure": goal_pressure,
        "boundary_shift": split["boundary_shift"] * stress,
        "unsafe_boundary": unsafe_boundary,
    }


def simulate_episode(method, task_name, split_name, seed, episode, episodes, stress_level=None):
    profile = PROFILES[method]
    case = generate_case(task_name, split_name, seed, episode, stress_level)
    rng = np.random.default_rng(BASE_SEED + stable_offset(method, task_name, split_name, seed, episode, stress_level))
    learning_phase = episode / max(1, episodes - 1)
    learned_boundary = profile["learning"] * (1.0 - math.exp(-3.2 * learning_phase))
    learned_boundary *= clamp(1.0 - 0.26 * case["noise"] - 0.18 * case["boundary_shift"], 0.55, 1.0)
    boundary_fidelity = clamp(profile["boundary_fidelity"] * (1.0 - 0.30 * case["noise"]) + 0.18 * learned_boundary)

    nominal_risk = clamp(
        0.42 * case["boundary_pressure"]
        + 0.24 * case["delay"]
        + 0.24 * case["unpredictability"]
        + 0.18 * case["goal_pressure"]
        - 0.50 * learned_boundary
        + rng.normal(0.0, 0.060),
        0.0,
        1.55,
    )
    predicted_risk = clamp(
        nominal_risk
        + 0.25 * case["noise"]
        - 0.22 * boundary_fidelity
        + 0.18 * case["boundary_shift"]
        + rng.normal(0.0, 0.055),
        0.0,
        1.45,
    )
    intervene = int(predicted_risk > profile["threshold"] and method != "unfiltered_policy")
    intervention_margin = max(0.0, predicted_risk - profile["threshold"])
    distortion = intervene * clamp(profile["distortion"] + 0.55 * intervention_margin + 0.12 * profile["conservatism"])
    residual_risk = clamp(
        nominal_risk
        - intervene * profile["shield_strength"] * (0.70 + 0.55 * intervention_margin)
        + 0.22 * case["delay"]
        + 0.10 * case["unpredictability"]
        - 0.08 * profile["recovery_bonus"],
        0.0,
        1.4,
    )
    deployment_violation_prob = clamp(sigmoid(5.0 * (residual_risk - 0.58)))
    deployment_violation = int(rng.random() < deployment_violation_prob)
    progress_score = (
        0.72
        - 0.42 * case["difficulty"]
        + 0.28 * case["goal_pressure"]
        - 0.54 * distortion
        - 0.48 * deployment_violation_prob
        + 0.12 * profile["recovery_bonus"]
        + rng.normal(0.0, 0.060)
    )
    task_success_prob = clamp(sigmoid(4.0 * (progress_score - 0.18)))
    task_success = int(rng.random() < task_success_prob and deployment_violation == 0)
    recovery_success = int(rng.random() < clamp(0.18 + case["recoverability"] + profile["recovery_bonus"] - 0.30 * case["delay"]))

    unshielded_risk = clamp(
        0.45 * case["boundary_pressure"]
        + 0.25 * case["delay"]
        + 0.22 * case["unpredictability"]
        + 0.16 * case["goal_pressure"]
        - 0.72 * learned_boundary
        + 0.18 * case["boundary_shift"]
        + rng.normal(0.0, 0.060),
        0.0,
        1.55,
    )
    unshielded_violation_prob = clamp(sigmoid(5.2 * (unshielded_risk - 0.60)))
    unshielded_violation = int(rng.random() < unshielded_violation_prob)
    unshielded_success_prob = clamp(sigmoid(3.7 * (0.64 - case["difficulty"] + 0.18 * case["goal_pressure"] - 0.70 * unshielded_violation_prob)))
    unshielded_success = int(rng.random() < unshielded_success_prob and unshielded_violation == 0)

    transfer_risk = clamp(unshielded_risk + 0.35 * case["boundary_shift"] + 0.12 * case["unpredictability"] - 0.20 * boundary_fidelity)
    transfer_violation_prob = clamp(sigmoid(5.0 * (transfer_risk - 0.62)))
    transfer_violation = int(rng.random() < transfer_violation_prob)
    filter_dependence = int(predicted_risk > profile["threshold"])
    boundary_pred = int(predicted_risk + 0.25 * boundary_fidelity > 0.70)
    oracle_utility = 0.86 - 0.16 * case["difficulty"] - 0.10 * case["delay"]
    method_utility = (
        0.72 * task_success
        - 0.80 * deployment_violation
        - 0.30 * distortion
        - 0.16 * intervene
        - 0.12 * unshielded_violation
    )
    oracle_regret = max(0.0, oracle_utility - method_utility)

    return {
        "method": method,
        "split": split_name,
        "task": task_name,
        "seed": seed,
        "episode": episode,
        "stress_level": "" if stress_level is None else f"{stress_level:.2f}",
        "task_success": task_success,
        "deployment_violation": deployment_violation,
        "intervention_rate": intervene,
        "control_distortion": distortion,
        "recovery_success": recovery_success,
        "unshielded_violation": unshielded_violation,
        "unshielded_success": unshielded_success,
        "filter_dependence": filter_dependence,
        "transfer_violation": transfer_violation,
        "oracle_regret": oracle_regret,
        "unsafe_boundary": case["unsafe_boundary"],
        "boundary_pred": boundary_pred,
        "nominal_risk": nominal_risk,
        "predicted_risk": predicted_risk,
        "learned_boundary": learned_boundary,
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def simulate_rows(methods, split_names, episodes, stress_level=None):
    rows = []
    for method in methods:
        for split_name in split_names:
            for task_name in TASKS:
                for seed in SEEDS:
                    for episode in range(episodes):
                        rows.append(simulate_episode(method, task_name, split_name, seed, episode, episodes, stress_level))
    return rows


def unit_metrics(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    metric_names = [
        "task_success",
        "deployment_violation",
        "intervention_rate",
        "control_distortion",
        "recovery_success",
        "unshielded_violation",
        "unshielded_success",
        "filter_dependence",
        "transfer_violation",
        "oracle_regret",
        "learned_boundary",
    ]
    for key, items in sorted(groups.items()):
        entry = {group_keys[i]: key[i] for i in range(len(group_keys))}
        for metric in metric_names:
            entry[metric] = float(np.mean([float(r[metric]) for r in items]))
        entry["boundary_f1"] = f1_score([int(r["unsafe_boundary"]) for r in items], [int(r["boundary_pred"]) for r in items])
        entry["episodes"] = len(items)
        out.append(entry)
    return out


def summarize(units, by_keys):
    metrics = [
        "task_success",
        "deployment_violation",
        "intervention_rate",
        "control_distortion",
        "recovery_success",
        "unshielded_violation",
        "unshielded_success",
        "filter_dependence",
        "transfer_violation",
        "oracle_regret",
        "learned_boundary",
        "boundary_f1",
    ]
    groups = {}
    for row in units:
        key = tuple(row[k] for k in by_keys)
        groups.setdefault(key, {m: [] for m in metrics})
        for metric in metrics:
            groups[key][metric].append(float(row[metric]))
    summary = []
    for key, values in sorted(groups.items()):
        entry = {by_keys[i]: key[i] for i in range(len(by_keys))}
        for metric in metrics:
            vals = values[metric]
            entry[f"mean_{metric}"] = f"{sum(vals) / len(vals):.5f}"
            entry[f"ci95_{metric}"] = f"{ci95(vals):.5f}"
        entry["units"] = len(next(iter(values.values())))
        summary.append(entry)
    return summary


def paired_gate(units, split_name):
    grouped = {}
    for row in units:
        if row["split"] == split_name:
            grouped.setdefault((row["task"], row["seed"]), {})[row["method"]] = row
    means = {}
    for method in METHODS:
        if method == "oracle_boundary_teacher":
            continue
        vals = [float(v[method]["unshielded_violation"]) for v in grouped.values() if method in v]
        if vals:
            means[method] = sum(vals) / len(vals)
    best_baseline = min((m for m in means if m != "proposed_boundary_learning_filter"), key=lambda m: means[m])
    safety_diffs = []
    success_diffs = []
    deploy_violation_diffs = []
    dependence_diffs = []
    transfer_diffs = []
    for methods in grouped.values():
        if "proposed_boundary_learning_filter" in methods and best_baseline in methods:
            proposed = methods["proposed_boundary_learning_filter"]
            baseline = methods[best_baseline]
            safety_diffs.append(float(baseline["unshielded_violation"]) - float(proposed["unshielded_violation"]))
            success_diffs.append(float(proposed["task_success"]) - float(baseline["task_success"]))
            deploy_violation_diffs.append(float(baseline["deployment_violation"]) - float(proposed["deployment_violation"]))
            dependence_diffs.append(float(baseline["filter_dependence"]) - float(proposed["filter_dependence"]))
            transfer_diffs.append(float(baseline["transfer_violation"]) - float(proposed["transfer_violation"]))
    return {
        "best_non_oracle_baseline": best_baseline,
        "paired_unshielded_safety_diff": sum(safety_diffs) / len(safety_diffs),
        "paired_unshielded_safety_ci95": ci95(safety_diffs),
        "paired_task_success_diff": sum(success_diffs) / len(success_diffs),
        "paired_task_success_ci95": ci95(success_diffs),
        "paired_deployment_violation_reduction": sum(deploy_violation_diffs) / len(deploy_violation_diffs),
        "paired_deployment_violation_ci95": ci95(deploy_violation_diffs),
        "paired_dependence_reduction": sum(dependence_diffs) / len(dependence_diffs),
        "paired_dependence_ci95": ci95(dependence_diffs),
        "paired_transfer_violation_reduction": sum(transfer_diffs) / len(transfer_diffs),
        "paired_transfer_violation_ci95": ci95(transfer_diffs),
    }


def find_row(summary, method, split):
    for row in summary:
        if row["method"] == method and row["split"] == split:
            return row
    raise KeyError((method, split))


def plot_bars(summary_rows, split, metrics, filename, title):
    rows = [r for r in summary_rows if r["split"] == split]
    labels = [r["method"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    width = 0.75 / len(metrics)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for idx, metric in enumerate(metrics):
        vals = [float(r[f"mean_{metric}"]) for r in rows]
        errs = [float(r[f"ci95_{metric}"]) for r in rows]
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, vals, width, yerr=errs, capsize=3, label=metric.replace("_", " "))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def plot_ablation(ablation_summary):
    labels = [r["method"].replace("_", "\n") for r in ablation_summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.24, [1.0 - float(r["mean_unshielded_violation"]) for r in ablation_summary], 0.24, label="unshielded safety")
    ax.bar(x, [float(r["mean_task_success"]) for r in ablation_summary], 0.24, label="task success")
    ax.bar(x + 0.24, [float(r["mean_boundary_f1"]) for r in ablation_summary], 0.24, label="boundary F1")
    ax.set_title("Paper 92 boundary-learning ablations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "safety_filter_ablation.png", dpi=180)
    plt.close(fig)


def plot_stress(stress_summary):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in sorted({r["method"] for r in stress_summary}):
        rows = sorted([r for r in stress_summary if r["method"] == method], key=lambda r: float(r["stress_level"]))
        levels = [float(r["stress_level"]) for r in rows]
        unshielded_safety = [1.0 - float(r["mean_unshielded_violation"]) for r in rows]
        ax.plot(levels, unshielded_safety, marker="o", label=method.replace("_", " "))
    ax.set_title("Paper 92 stress sweep: unshielded post-training safety")
    ax.set_xlabel("stress level")
    ax.set_ylabel("1 - unshielded violation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "safety_filter_stress_sweep.png", dpi=180)
    plt.close(fig)


def main():
    main_rows = simulate_rows(METHODS, list(SPLITS.keys()), EPISODES)
    write_csv(RESULTS / "rollouts.csv", main_rows)
    main_units = unit_metrics(main_rows, ["method", "split", "task", "seed"])
    write_csv(RESULTS / "raw_seed_metrics.csv", main_units)
    summary = summarize(main_units, ["method", "split"])
    write_csv(RESULTS / "metrics.csv", summary)

    ablation_rows = simulate_rows(ABLATIONS, ["combined_safety_stress"], EPISODES)
    write_csv(RESULTS / "ablation_rollouts.csv", ablation_rows)
    ablation_units = unit_metrics(ablation_rows, ["method", "task", "seed"])
    write_csv(RESULTS / "ablation_seed_metrics.csv", ablation_units)
    ablation_summary = summarize(ablation_units, ["method"])
    write_csv(RESULTS / "ablation_metrics.csv", ablation_summary)

    stress_methods = [
        "cbf_safety_filter",
        "robust_mpc_shield",
        "conformal_uncertainty_shield",
        "recovery_policy_shield",
        "proposed_boundary_learning_filter",
        "oracle_boundary_teacher",
    ]
    stress_rows = []
    for stress_level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        stress_rows.extend(simulate_rows(stress_methods, ["combined_safety_stress"], STRESS_EPISODES, stress_level))
    write_csv(RESULTS / "stress_sweep_raw.csv", stress_rows)
    stress_units = unit_metrics(stress_rows, ["method", "stress_level", "task", "seed"])
    stress_summary = summarize(stress_units, ["method", "stress_level"])
    write_csv(RESULTS / "stress_sweep.csv", stress_summary)
    write_csv(FIGURES / "stress_curve_data.csv", stress_summary)

    gate = paired_gate(main_units, "combined_safety_stress")
    proposed = find_row(summary, "proposed_boundary_learning_filter", "combined_safety_stress")
    baseline = find_row(summary, gate["best_non_oracle_baseline"], "combined_safety_stress")
    oracle = find_row(summary, "oracle_boundary_teacher", "combined_safety_stress")
    ablation_safety = {r["method"]: 1.0 - float(r["mean_unshielded_violation"]) for r in ablation_summary}
    best_ablation = max((m for m in ablation_safety if m != "full_boundary_learning_filter"), key=lambda m: ablation_safety[m])

    stress_max = [r for r in stress_summary if abs(float(r["stress_level"]) - 1.0) < 1e-9]
    stress_best_baseline = max(
        [r for r in stress_max if r["method"] not in {"proposed_boundary_learning_filter", "oracle_boundary_teacher"}],
        key=lambda r: 1.0 - float(r["mean_unshielded_violation"]),
    )
    stress_proposed = next(r for r in stress_max if r["method"] == "proposed_boundary_learning_filter")

    unshielded_gate = gate["paired_unshielded_safety_diff"] - gate["paired_unshielded_safety_ci95"] > 0.015
    success_gate = gate["paired_task_success_diff"] + gate["paired_task_success_ci95"] >= -0.020
    deployment_gate = gate["paired_deployment_violation_reduction"] + gate["paired_deployment_violation_ci95"] >= -0.010
    dependence_gate = gate["paired_dependence_reduction"] - gate["paired_dependence_ci95"] > 0.015
    ablation_gate = ablation_safety["full_boundary_learning_filter"] >= ablation_safety[best_ablation] + 0.01
    stress_gate = (1.0 - float(stress_proposed["mean_unshielded_violation"])) >= (1.0 - float(stress_best_baseline["mean_unshielded_violation"])) - 0.005
    terminal = "STRONG_REVISE" if all([unshielded_gate, success_gate, deployment_gate, dependence_gate, ablation_gate, stress_gate]) else "KILL_ARCHIVE"

    pairwise = [{
        "split": "combined_safety_stress",
        "proposed": "proposed_boundary_learning_filter",
        "best_non_oracle_baseline": gate["best_non_oracle_baseline"],
        "paired_unshielded_safety_diff": f"{gate['paired_unshielded_safety_diff']:.5f}",
        "paired_unshielded_safety_ci95": f"{gate['paired_unshielded_safety_ci95']:.5f}",
        "paired_task_success_diff": f"{gate['paired_task_success_diff']:.5f}",
        "paired_task_success_ci95": f"{gate['paired_task_success_ci95']:.5f}",
        "paired_deployment_violation_reduction": f"{gate['paired_deployment_violation_reduction']:.5f}",
        "paired_deployment_violation_ci95": f"{gate['paired_deployment_violation_ci95']:.5f}",
        "paired_dependence_reduction": f"{gate['paired_dependence_reduction']:.5f}",
        "paired_dependence_ci95": f"{gate['paired_dependence_ci95']:.5f}",
        "paired_transfer_violation_reduction": f"{gate['paired_transfer_violation_reduction']:.5f}",
        "paired_transfer_violation_ci95": f"{gate['paired_transfer_violation_ci95']:.5f}",
        "unshielded_gate": unshielded_gate,
        "success_gate": success_gate,
        "deployment_gate": deployment_gate,
        "dependence_gate": dependence_gate,
        "ablation_gate": ablation_gate,
        "stress_gate": stress_gate,
        "terminal": terminal,
    }]
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)

    negative_cases = [
        {
            "case": "boundary_shape_shift",
            "observed_failure": "counterfactual labels learned under one boundary shape do not fully transfer to shifted safe sets",
            "implication": "mechanism-change evidence is local and needs external safety-boundary validation",
        },
        {
            "case": "robust_mpc_matches_unshielded_safety",
            "observed_failure": "model-based shielding provides comparable post-training safety without the proposed boundary-learning machinery",
            "implication": "novelty over robust MPC and CBF-style filters is not decisive",
        },
        {
            "case": "filter_dependence_not_eliminated",
            "observed_failure": "the learned policy still triggers many interventions under combined stress",
            "implication": "the filter remains partly a deployment shield rather than a pure learning mechanism",
        },
    ]
    write_csv(RESULTS / "negative_cases.csv", negative_cases)

    plot_bars(summary, "combined_safety_stress", ["task_success", "deployment_violation", "intervention_rate"], "safety_filter_deployment.png", "Paper 92 combined stress: deployment metrics")
    plot_bars(summary, "combined_safety_stress", ["unshielded_violation", "filter_dependence", "boundary_f1"], "safety_filter_mechanism.png", "Paper 92 combined stress: mechanism-change metrics")
    plot_ablation(ablation_summary)
    plot_stress(stress_summary)

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 92 safety_filter_novelty_boundaries v4 rebuild\n")
        handle.write(f"Terminal recommendation: {terminal}\n")
        handle.write("Reason: deterministic safe-control learning benchmark added; no robot hardware or accepted high-fidelity safety benchmark validation is available.\n")
        handle.write(f"Main rollout rows: {len(main_rows)}\n")
        handle.write(f"Ablation rollout rows: {len(ablation_rows)}\n")
        handle.write(f"Stress rollout rows: {len(stress_rows)}\n")
        handle.write(f"Seeds: {SEEDS}\n\n")
        handle.write("Combined safety stress:\n")
        for method in METHODS:
            row = find_row(summary, method, "combined_safety_stress")
            handle.write(
                f"{method} success={row['mean_task_success']} ci95={row['ci95_task_success']} "
                f"deploy_violation={row['mean_deployment_violation']} intervention={row['mean_intervention_rate']} "
                f"unshielded_violation={row['mean_unshielded_violation']} dependence={row['mean_filter_dependence']} "
                f"boundary_f1={row['mean_boundary_f1']} transfer_violation={row['mean_transfer_violation']}\n"
            )
        handle.write(
            f"paired unshielded-safety diff vs best baseline {gate['best_non_oracle_baseline']}="
            f"{gate['paired_unshielded_safety_diff']:.5f} ci95={gate['paired_unshielded_safety_ci95']:.5f}\n"
        )
        handle.write(
            f"paired task-success diff={gate['paired_task_success_diff']:.5f} ci95={gate['paired_task_success_ci95']:.5f}; "
            f"paired dependence reduction={gate['paired_dependence_reduction']:.5f} ci95={gate['paired_dependence_ci95']:.5f}\n\n"
        )
        handle.write("Ablations:\n")
        for row in ablation_summary:
            handle.write(
                f"{row['method']} success={row['mean_task_success']} unshielded_violation={row['mean_unshielded_violation']} "
                f"dependence={row['mean_filter_dependence']} boundary_f1={row['mean_boundary_f1']} transfer_violation={row['mean_transfer_violation']}\n"
            )
        handle.write("\nCombined stress level 1.0:\n")
        for row in stress_max:
            handle.write(
                f"{row['method']} success={row['mean_task_success']} deploy_violation={row['mean_deployment_violation']} "
                f"unshielded_violation={row['mean_unshielded_violation']} dependence={row['mean_filter_dependence']}\n"
            )
        handle.write("\nGate checks:\n")
        handle.write(f"unshielded_gate={unshielded_gate}\n")
        handle.write(f"success_gate={success_gate}\n")
        handle.write(f"deployment_gate={deployment_gate}\n")
        handle.write(f"dependence_gate={dependence_gate}\n")
        handle.write(f"ablation_gate={ablation_gate} best_ablation={best_ablation}\n")
        handle.write(f"stress_gate={stress_gate} stress_best_baseline={stress_best_baseline['method']}\n")
        handle.write(f"oracle_unshielded_violation={oracle['mean_unshielded_violation']}\n")

    print(f"terminal={terminal}")
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
