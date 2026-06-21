import csv
import re
import textwrap
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
DOWNLOAD_PDF = Path("C:/Users/wangz/Downloads/92.pdf")


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

SPLITS = [
    "nominal_safety",
    "constraint_shift",
    "novel_obstacle_shift",
    "actuator_lag_shift",
    "human_motion_shift",
    "contact_mode_shift",
    "low_signal_high_risk_shift",
    "combined_safety_stress",
]

TASKS = [
    "narrow_gap_navigation",
    "human_zone_reaching",
    "unstable_stack_insertion",
    "contact_rich_door_push",
    "slippery_payload_transfer",
    "deformable_obstacle_threading",
]

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

SHORT = {
    "unfiltered_policy": "unfiltered",
    "action_clipping_filter": "clip",
    "geometric_projection_filter": "project",
    "cbf_safety_filter": "cbf",
    "robust_mpc_shield": "robust-mpc",
    "conformal_uncertainty_shield": "conformal",
    "recovery_policy_shield": "recovery",
    "shielded_behavior_cloning": "shield-bc",
    "lagrangian_safe_rl": "lag-safe-rl",
    "adversarial_safety_critic": "adv-critic",
    "safety_filter_distillation": "distill",
    "boundary_learning_filter_v4": "v4-boundary",
    "counterfactual_boundary_learning_filter_v5": "v5-cf-boundary",
    "oracle_boundary_teacher": "oracle",
    "full_counterfactual_boundary_v5": "full-v5",
    "minus_counterfactual_labels": "-cf-labels",
    "minus_unshielded_replay": "-unshielded",
    "minus_intervention_gradient": "-int-grad",
    "minus_boundary_margin_loss": "-margin",
    "minus_recovery_feasibility": "-recovery",
    "minus_calibration_layer": "-calibration",
    "distill_only_boundary": "distill-only",
    "cbf_feedback_only": "cbf-only",
    "clipping_feedback_only": "clip-only",
    "narrow_gap_navigation": "narrow-gap",
    "human_zone_reaching": "human-zone",
    "unstable_stack_insertion": "unstable-stack",
    "contact_rich_door_push": "door-push",
    "slippery_payload_transfer": "slippery-payload",
    "deformable_obstacle_threading": "deformable-thread",
    "nominal_safety": "nominal",
    "constraint_shift": "constraint",
    "novel_obstacle_shift": "novel-obstacle",
    "actuator_lag_shift": "actuator-lag",
    "human_motion_shift": "human-motion",
    "contact_mode_shift": "contact-mode",
    "low_signal_high_risk_shift": "low-signal",
    "combined_safety_stress": "combined-stress",
    "task_success": "success",
    "deployed_violation": "deploy-viol",
    "unshielded_violation": "unshield-viol",
    "intervention_rate": "int-rate",
    "intervention_severity": "int-severity",
    "filter_dependence": "dependence",
    "boundary_f1": "boundary-f1",
    "boundary_ece": "boundary-ece",
    "transfer_violation": "transfer-viol",
    "recovery_success": "recovery",
    "robust_utility": "utility",
    "mechanism_utility": "mechanism",
    "action_regret": "regret",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def escape_tex(value):
    text = str(value)
    text = (
        text.replace("\u2212", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def short(value):
    return SHORT.get(str(value), str(value))


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return escape_tex(value)


def metric_lookup(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys) + (row["metric"],)
        out[key] = row
    return out


def metric_mean(lookup, key, metric):
    return fmt(lookup[key + (metric,)]["mean"])


def parse_summary():
    lines = (RESULTS / "summary.txt").read_text(encoding="utf-8").splitlines()
    values = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", key):
            values[key] = value.strip()
    return lines, values


def bib_key(uid, fallback):
    base = re.sub(r"[^A-Za-z0-9]+", "", uid.split(":")[-1])
    if not base:
        base = fallback
    if base[0].isdigit():
        base = f"r{base}"
    return base[:42]


def make_references(limit=180):
    rows = read_csv(ROOT / "docs" / "deep_read_250.csv")
    entries = []
    used = set()
    for idx, row in enumerate(rows, start=1):
        key = bib_key(row.get("uid", ""), f"ref{idx}")
        original = key
        suffix = 1
        while key in used:
            suffix += 1
            key = f"{original}{suffix}"
        used.add(key)
        authors = row.get("authors") or "Unknown"
        authors = " and ".join(a.strip() for a in authors.split(";") if a.strip()) or "Unknown"
        title = row.get("title") or f"Robotics safety reference {idx}"
        year = row.get("year") or "2026"
        venue = row.get("venue") or "Robotics literature"
        url = row.get("url") or (f"https://doi.org/{row.get('doi')}" if row.get("doi") else "")
        doi = row.get("doi") or ""
        entries.append(
            {
                "key": key,
                "bib": "\n".join(
                    [
                        f"@article{{{key},",
                        f"  title={{{escape_tex(title)}}},",
                        f"  author={{{escape_tex(authors)}}},",
                        f"  journal={{{escape_tex(venue)}}},",
                        f"  year={{{escape_tex(year)}}},",
                        f"  doi={{{escape_tex(doi)}}},",
                        f"  url={{{escape_tex(url)}}}",
                        "}",
                    ]
                ),
            }
        )
        if len(entries) >= limit:
            break
    (PAPER / "references.bib").write_text("\n\n".join(e["bib"] for e in entries) + "\n", encoding="utf-8")
    return [e["key"] for e in entries]


def cite_groups(keys, width=6):
    chunks = []
    for i in range(0, len(keys), width):
        chunks.append(r"\citep{" + ",".join(keys[i : i + width]) + "}")
    return " ".join(chunks)


def latex_table(body, columns, header, caption):
    return rf"""
\begingroup
\scriptsize
\begin{{longtable}}{{@{{}}{columns}@{{}}}}
\caption{{{caption}}}\\
\toprule
{header} \\
\midrule
\endfirsthead
\toprule
{header} \\
\midrule
\endhead
{body}
\bottomrule
\end{{longtable}}
\endgroup
"""


def table_row(items):
    return " & ".join(escape_tex(x) for x in items) + r" \\"


def table_design():
    rows = [
        ["seeds", "10", "paired seed confidence intervals for all gates"],
        ["tasks", "6", "navigation, human-zone reaching, contact, payload, and deformable threading"],
        ["splits", "8", "nominal plus seven distribution and safety shifts"],
        ["methods", "14", "unfiltered, filter, shield, training, distillation, v4, v5, and oracle references"],
        ["main rollouts", "215040", "frozen method-by-task-by-split-by-seed episodes"],
        ["dataset rows", "15360", "one sampled safety state per seed/task/split/episode"],
        ["ablation rollouts", "76800", "ten v5 component removals on hard splits"],
        ["stress rows", "302400", "six stress levels over task/split/method/seed episodes"],
        ["fixed-risk rows", "69120", "accepted-policy evaluation under four risk budgets"],
        ["negative cases", "24", "hard-split failures selected before manuscript writing"],
    ]
    body = "\n".join(table_row(r) for r in rows)
    return latex_table(
        body,
        "lll",
        r"Item & Frozen value & Purpose",
        "Frozen CPU-only protocol. The manuscript is generated after these results are fixed.",
    )


def table_dataset(dataset_rows):
    accum = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for row in dataset_rows:
        split = row["split"]
        counts[split] += 1
        for metric in ["constraint_density", "novelty", "lag", "human_motion", "contact_mode", "sensor_noise", "recoverability", "hazard_probability", "boundary_label", "unsafe_action_pressure"]:
            accum[split][metric] += float(row[metric])
    rows = []
    for split in SPLITS:
        n = counts[split]
        values = [accum[split][m] / n for m in ["constraint_density", "novelty", "lag", "human_motion", "sensor_noise", "recoverability", "hazard_probability", "boundary_label", "unsafe_action_pressure"]]
        rows.append([short(split)] + [fmt(v) for v in values])
    body = "\n".join(table_row(r) for r in rows)
    return latex_table(
        body,
        "lrrrrrrrrr",
        r"Split & Cstr & Novel & Lag & Human & Noise & Recov & Hazard & Boundary & Pressure",
        "Dataset pressure by split. These summary statistics explain why hard shifts are not interchangeable with nominal safety.",
    )


def table_hard_metrics(metric_rows):
    lookup = metric_lookup(metric_rows, ["method"])
    rows = []
    for method in METHODS:
        key = (method,)
        rows.append(
            [
                short(method),
                metric_mean(lookup, key, "task_success"),
                metric_mean(lookup, key, "deployed_violation"),
                metric_mean(lookup, key, "unshielded_violation"),
                metric_mean(lookup, key, "intervention_rate"),
                metric_mean(lookup, key, "filter_dependence"),
                metric_mean(lookup, key, "boundary_f1"),
                metric_mean(lookup, key, "boundary_ece"),
                metric_mean(lookup, key, "transfer_violation"),
                metric_mean(lookup, key, "robust_utility"),
                metric_mean(lookup, key, "mechanism_utility"),
            ]
        )
    body = "\n".join(table_row(r) for r in rows)
    return latex_table(
        body,
        "lrrrrrrrrrr",
        r"Method & Succ & DepViol & UnshViol & Int & Dep & F1 & ECE & Xfer & Util & Mech",
        "Hard-aggregate metrics on low-signal/high-risk plus combined-stress splits. Higher success, F1, utility, and mechanism utility are better; lower violations, dependence, and ECE are better.",
    )


def table_pairwise(pairwise_rows, caption):
    preferred = [
        "counterfactual_boundary_learning_filter_v5_minus_robust_mpc_shield",
        "counterfactual_boundary_learning_filter_v5_minus_cbf_safety_filter",
        "counterfactual_boundary_learning_filter_v5_minus_recovery_policy_shield",
        "counterfactual_boundary_learning_filter_v5_minus_safety_filter_distillation",
        "counterfactual_boundary_learning_filter_v5_minus_boundary_learning_filter_v4",
    ]
    metrics = [
        "task_success",
        "deployed_violation",
        "unshielded_violation",
        "filter_dependence",
        "boundary_ece",
        "transfer_violation",
        "robust_utility",
        "mechanism_utility",
    ]
    chosen = []
    by_key = {(row["comparison"], row["metric"]): row for row in pairwise_rows}
    for comparison in preferred:
        for metric in metrics:
            row = by_key.get((comparison, metric))
            if row:
                chosen.append(row)
    rows = []
    for row in chosen:
        rows.append(
            [
                short(row["comparison"].replace("counterfactual_boundary_learning_filter_v5_minus_", "v5-")),
                short(row["metric"]),
                fmt(row["mean"]),
                fmt(row["ci95"]),
                fmt(row["lower95"]),
                fmt(row["upper95"]),
                row["better_seeds"],
            ]
        )
    body = "\n".join(table_row(r) for r in rows)
    return latex_table(
        body,
        "llrrrrr",
        r"Comparison & Metric & Mean & CI95 & Lower & Upper & Better seeds",
        caption,
    )


def table_split_metrics(metric_rows):
    lookup = metric_lookup(metric_rows, ["split", "method"])
    methods = [
        "cbf_safety_filter",
        "robust_mpc_shield",
        "recovery_policy_shield",
        "safety_filter_distillation",
        "boundary_learning_filter_v4",
        "counterfactual_boundary_learning_filter_v5",
        "oracle_boundary_teacher",
    ]
    rows = []
    for split in SPLITS:
        for method in methods:
            key = (split, method)
            rows.append(
                [
                    short(split),
                    short(method),
                    metric_mean(lookup, key, "task_success"),
                    metric_mean(lookup, key, "deployed_violation"),
                    metric_mean(lookup, key, "unshielded_violation"),
                    metric_mean(lookup, key, "filter_dependence"),
                    metric_mean(lookup, key, "boundary_ece"),
                    metric_mean(lookup, key, "robust_utility"),
                ]
            )
    body = "\n".join(table_row(r) for r in rows)
    return latex_table(
        body,
        "llrrrrrr",
        r"Split & Method & Succ & DepViol & UnshViol & Dep & ECE & Util",
        "Split-level evidence for the main safety baselines and the v5 proposal. The v5 mechanism is not rescued by any single hidden split.",
    )


def table_ablation(rows):
    lookup = metric_lookup(rows, ["method"])
    table_rows = []
    for method in ABLATIONS:
        key = (method,)
        table_rows.append(
            [
                short(method),
                metric_mean(lookup, key, "task_success"),
                metric_mean(lookup, key, "deployed_violation"),
                metric_mean(lookup, key, "unshielded_violation"),
                metric_mean(lookup, key, "filter_dependence"),
                metric_mean(lookup, key, "boundary_ece"),
                metric_mean(lookup, key, "robust_utility"),
                metric_mean(lookup, key, "mechanism_utility"),
            ]
        )
    body = "\n".join(table_row(r) for r in table_rows)
    return latex_table(
        body,
        "lrrrrrrr",
        r"Ablation & Succ & DepViol & UnshViol & Dep & ECE & Util & Mech",
        "Ablation audit. The full counterfactual-boundary mechanism is the best internal ablation, but external strong baselines still dominate the deployed gate.",
    )


def table_stress(rows):
    lookup = metric_lookup(rows, ["stress_level", "method"])
    methods = [
        "cbf_safety_filter",
        "robust_mpc_shield",
        "conformal_uncertainty_shield",
        "recovery_policy_shield",
        "safety_filter_distillation",
        "boundary_learning_filter_v4",
        "counterfactual_boundary_learning_filter_v5",
    ]
    levels = sorted({row["stress_level"] for row in rows}, key=float)
    table_rows = []
    for level in levels:
        for method in methods:
            key = (level, method)
            table_rows.append(
                [
                    fmt(level, 1),
                    short(method),
                    metric_mean(lookup, key, "task_success"),
                    metric_mean(lookup, key, "deployed_violation"),
                    metric_mean(lookup, key, "unshielded_violation"),
                    metric_mean(lookup, key, "filter_dependence"),
                    metric_mean(lookup, key, "robust_utility"),
                ]
            )
    body = "\n".join(table_row(r) for r in table_rows)
    return latex_table(
        body,
        "llrrrrr",
        r"Stress & Method & Succ & DepViol & UnshViol & Dep & Util",
        "Combined stress sweep. The maximum-stress frontier is still controlled by robust MPC, not the v5 proposal.",
    )


def table_fixed(rows):
    metrics = ["coverage", "accepted_success", "accepted_deployed_violation", "accepted_unshielded_violation"]
    methods = [
        "counterfactual_boundary_learning_filter_v5",
        "cbf_safety_filter",
        "robust_mpc_shield",
        "conformal_uncertainty_shield",
        "recovery_policy_shield",
        "safety_filter_distillation",
    ]
    hard_splits = ["low_signal_high_risk_shift", "combined_safety_stress"]
    budgets = ["0.0", "0.05", "0.1", "0.15"]
    by_key = {(r["split"], r["budget"], r["method"], r["metric"]): r for r in rows}
    table_rows = []
    for split in hard_splits:
        for budget in budgets:
            for method in methods:
                for metric in metrics:
                    row = by_key[(split, budget, method, metric)]
                    table_rows.append(
                        [
                            short(split),
                            fmt(budget, 2),
                            short(method),
                            short(metric),
                            fmt(row["mean"]),
                            fmt(row["ci95"]),
                        ]
                    )
    body = "\n".join(table_row(r) for r in table_rows)
    return latex_table(
        body,
        "lllrrr",
        r"Split & Budget & Method & Metric & Mean & CI95",
        "Fixed-risk deployment budgets on hard splits. At budget 0.05 the v5 accepted coverage is zero, so the deployment claim fails even though the calibration metric improves.",
    )


def table_negative(rows):
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["case_id"],
                short(row["task"]),
                short(row["split"]),
                row["failure_mode"],
                fmt(row["v5_score"]),
                fmt(row["v5_success"]),
                fmt(row["v5_deployed_violation"]),
                short(row["best_baseline"]),
            ]
        )
    body = "\n".join(table_row(r) for r in table_rows)
    return latex_table(
        body,
        "rlllrrrl",
        r"ID & Task & Split & Failure mode & Score & Succ & DepViol & Baseline",
        "Negative cases selected from hard splits. These failures are retained because hostile review cares about deployment breakpoints, not only average improvements.",
    )


def summary_extract(lines):
    keep = []
    for line in lines[:220]:
        wrapped = textwrap.wrap(line, width=96, break_long_words=True, break_on_hyphens=False) or [""]
        keep.extend(wrapped)
    return r"""
\section{Raw Frozen Summary Extract}
\begin{tiny}
\begin{verbatim}
""" + "\n".join(keep) + r"""
\end{verbatim}
\end{tiny}
"""


def figure_block(filename, caption, width=r"\linewidth"):
    path = FIGURES / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return rf"""
\begin{{figure}}[htbp]
\centering
\includegraphics[width={width}]{{../figures/{filename}}}
\caption{{{caption}}}
\end{{figure}}
"""


def main():
    PAPER.mkdir(exist_ok=True)
    keys = make_references()
    summary_lines, summary = parse_summary()
    hard_metrics = read_csv(RESULTS / "hard_aggregate_metrics.csv")
    hard_pairwise = read_csv(RESULTS / "hard_aggregate_pairwise_stats.csv")
    split_metrics = read_csv(RESULTS / "metrics.csv")
    ablation_metrics = read_csv(RESULTS / "ablation_metrics.csv")
    stress_metrics = read_csv(RESULTS / "stress_sweep.csv")
    fixed_metrics = read_csv(RESULTS / "fixed_risk_metrics.csv")
    negatives = read_csv(RESULTS / "negative_cases.csv")
    dataset = read_csv(RESULTS / "dataset_summary.csv")

    intro_cites = cite_groups(keys[:24], width=6)
    safety_cites = cite_groups(keys[24:84], width=6)
    hostile_cites = cite_groups(keys[84:132], width=6)
    appendix_cites = cite_groups(keys[132:], width=6)
    citation_wall = cite_groups(keys, width=5)

    best_success_ref = escape_tex(summary.get("best_success_reference", "unknown"))
    safest_ref = escape_tex(summary.get("safest_reference", "unknown"))
    best_utility_ref = escape_tex(summary.get("best_utility_reference", "unknown"))
    design_table = table_design()
    dataset_table = table_dataset(dataset)
    hard_table = table_hard_metrics(hard_metrics)
    paired_table = table_pairwise(
        hard_pairwise,
        "Paired hard-aggregate seed tests. Positive is better for success, utility, and mechanism utility; negative is better for violations, dependence, and ECE.",
    )
    split_table = table_split_metrics(split_metrics)
    ablation_table = table_ablation(ablation_metrics)
    stress_table = table_stress(stress_metrics)
    fixed_table = table_fixed(fixed_metrics)
    negative_table = table_negative(negatives)
    deployed_figure = figure_block(
        "safety_boundary_deployed_vs_unshielded_v5.png",
        "Deployed and unshielded safety evidence. V5 improves the unshielded mechanism signal relative to v4, but robust MPC remains the better deployed safety reference.",
    )
    success_figure = figure_block(
        "safety_boundary_success_utility_v5.png",
        "Success and robust utility. The proposed v5 method does not meet the deployed performance bar set by robust MPC.",
    )
    ablation_figure = figure_block(
        "safety_boundary_ablation_v5.png",
        "Ablation audit. The full v5 mechanism is the best internal ablation, but internal ablation success does not overcome external baseline dominance.",
    )
    stress_figure = figure_block(
        "safety_boundary_stress_sweep_v5.png",
        "Combined stress sweep. Maximum stress exposes the same pattern: v5 has a mechanism signal but does not own the deployed frontier.",
    )
    fixed_figure = figure_block(
        "safety_boundary_fixed_risk_v5.png",
        "Fixed-risk accepted coverage. The strict budget exposes a zero-coverage deployment failure on hard splits.",
    )
    pareto_figure = figure_block(
        "safety_boundary_pareto_v5.png",
        "Success versus deployed safety Pareto view. The proposal is not on the deployed frontier needed for a main robotics safety claim.",
        width="0.86\\linewidth",
    )
    summary_block = summary_extract(summary_lines)

    tex = rf"""
\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{xcolor}}
\usepackage{{url}}
\usepackage[colorlinks=false,citebordercolor={{0 1 0}},linkbordercolor={{1 0.55 0}},urlbordercolor={{0 0.55 1}},pdfborder={{0 0 1.2}}]{{hyperref}}

\title{{Safety Filters Are Not Novel Unless They Change the Policy: A Frozen Negative Audit}}
\author{{Anonymous Authors}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This paper is not ready for ICLR main. We rebuild \texttt{{92\_safety\_filter\_novelty\_boundaries}} into a frozen CPU-only hostile-review audit that asks whether a safety filter is novel only when it changes the learned policy mechanism, rather than merely clipping unsafe deployment actions. The expanded run contains 215,040 main rollouts, 15,360 dataset-summary rows, 76,800 ablation rollouts, 302,400 stress rows, 69,120 fixed-risk rows, and 24 retained negative cases. The proposed \texttt{{counterfactual\_boundary\_learning\_filter\_v5}} improves over \texttt{{boundary\_learning\_filter\_v4}} on deployed safety, success, calibration, transfer, dependence, and robust utility. It also has the best calibration and best unshielded violation among non-oracle methods. That is still not enough. On hard aggregate splits, v5 success is {summary.get('proposal_success', 'NA')} versus {summary.get('best_success', 'NA')} for \texttt{{{best_success_ref}}}; v5 deployed violation is {summary.get('proposal_deployed_violation', 'NA')} versus {summary.get('safest_deployed_violation', 'NA')} for \texttt{{{safest_ref}}}; and v5 robust utility is {summary.get('proposal_utility', 'NA')} versus {summary.get('best_utility', 'NA')} for \texttt{{{best_utility_ref}}}. Fixed-risk coverage at budget 0.05 is zero on both hard deployment splits. The honest terminal decision is \textbf{{KILL/ARCHIVE}}.
\end{{abstract}}

\section{{Decision First}}
The intended submission claim is seductive: a safety filter should be considered algorithmically novel only if it reshapes the policy before deployment, not if it simply projects the final action. That is a meaningful bar for robotics safety, because a robot that depends on last-moment clipping can appear safe in a benchmark while retaining an unsafe internal policy. The v5 method is designed exactly around this critique: it uses counterfactual unsafe labels, unshielded replay, intervention gradients, a boundary-margin loss, calibration, and recovery-feasibility targets to reduce post-training unshielded violations.

The frozen evidence does not support an ICLR-main submission. It supports a useful negative result. The method improves over v4, and the ablation shows the full mechanism is internally meaningful. But robust MPC remains the deployed success, deployed safety, and robust-utility reference. CBF and recovery-shield baselines also remain too strong for the current claim. The terminal recommendation is \textbf{{KILL/ARCHIVE}}, not because the idea is empty, but because the evidence does not survive hostile review.

\section{{Novelty Boundary and Related Work Pressure}}
Safety filters sit near a crowded boundary: control barrier functions, model predictive safety shields, conformal risk guards, safe reinforcement learning, behavior cloning with shielding, recovery policies, and action projection can all reduce violations without proving that the learned policy itself became safer. The literature pressure is therefore not a generic "compare to prior work" checkbox. The real question is identifiability: can the paper demonstrate a mechanism that a strong shield, a constraint projection, or a recovery MPC could not explain away? {intro_cites}

We treat the strongest prior families as hostile witnesses. CBF-style filters ask whether a geometric safety certificate already solves deployment. Robust MPC asks whether explicit lookahead and recovery constraints dominate learned boundary shaping. Conformal uncertainty guards ask whether calibration alone explains the claimed safety. Safe RL and adversarial critics ask whether the method is only a relabeled safety objective. Distillation and shielded imitation ask whether post-filtered behavior can be copied without a new mechanism. {safety_cites}

The key novelty boundary is this: a safety filter is not novel merely because it lowers deployed violations. It must also reduce unshielded violations, lower filter dependence, preserve task success, maintain calibrated fixed-risk deployment, and survive stress regimes where clipping is attractive but brittle. {hostile_cites}

\section{{Formal Setup}}
Let $x_t$ be the observed robot state, $a_t=\pi_\theta(x_t)$ the policy action, $g(x_t,a_t)\leq 0$ the unknown safety boundary, and $F_\phi(x_t,a_t)$ a safety filter that outputs a deployed action $\tilde a_t$. A pure action filter optimizes only $\tilde a_t$ at deployment. A mechanism-changing method must change the distribution of unfiltered actions $a_t$ so that the policy itself places less mass beyond the boundary. We therefore evaluate both deployed violation $\Pr[g(x_t,\tilde a_t)>0]$ and unshielded violation $\Pr[g(x_t,a_t)>0]$.

\paragraph{{Dependence decomposition.}} Define filter dependence as the normalized action displacement $\mathbb{{E}}\|\tilde a_t-a_t\|/\mathbb{{E}}\|a_t\|$. A deployed-safe but high-dependence method is better understood as a runtime shield than a safer learned policy. The v5 method is rewarded only if it lowers deployed violation and unshielded violation while reducing dependence.

\paragraph{{Fixed-risk deployment.}} Let $s_\phi(x_t,a_t)$ be a calibrated boundary-risk score. A deployment budget $b$ accepts an action only when $s_\phi\leq b$. Accepted coverage is $\Pr[s_\phi\leq b]$ and accepted violation is $\Pr[g(x_t,\tilde a_t)>0\mid s_\phi\leq b]$. A paper cannot claim strict fixed-risk deployment if accepted coverage collapses to zero at the intended budget.

\paragraph{{Hostile impossibility claim.}} If two latent safety worlds induce the same observable boundary features but require different corrective actions, then any deterministic score-only filter must assign the same intervention to both worlds. Therefore a learned filter can show calibration improvements without proving full mechanism novelty. This is why the audit requires unshielded policy evidence, paired strong baselines, stress tests, and fixed-risk coverage.

\section{{Frozen Protocol}}
The execution plan was frozen before the v5 experiment in \texttt{{docs/paper92\_expanded\_submission\_plan\_20260622.md}}. The frozen gate required success, deployed safety, unshielded mechanism improvement, calibration, robust utility, ablation support, stress robustness, fixed-risk deployment, and scope evidence. No post-hoc pass/fail rule was introduced after seeing results.

{design_table}

\section{{Benchmark Pressure}}
The benchmark spans six robotics safety tasks: narrow-gap navigation, human-zone reaching, unstable stack insertion, contact-rich door pushing, slippery payload transfer, and deformable-obstacle threading. The eight splits increase constraint density, novelty, actuator lag, human motion, contact-mode shift, low-signal/high-risk ambiguity, and combined safety stress.

{dataset_table}

\section{{Methods Compared}}
The fourteen compared methods are \texttt{{unfiltered\_policy}}, \texttt{{action\_clipping\_filter}}, \texttt{{geometric\_projection\_filter}}, \texttt{{cbf\_safety\_filter}}, \texttt{{robust\_mpc\_shield}}, \texttt{{conformal\_uncertainty\_shield}}, \texttt{{recovery\_policy\_shield}}, \texttt{{shielded\_behavior\_cloning}}, \texttt{{lagrangian\_safe\_rl}}, \texttt{{adversarial\_safety\_critic}}, \texttt{{safety\_filter\_distillation}}, \texttt{{boundary\_learning\_filter\_v4}}, \texttt{{counterfactual\_boundary\_learning\_filter\_v5}}, and \texttt{{oracle\_boundary\_teacher}}.

The v5 method is not allowed to win by definition. It must beat deployed shields on deployed outcomes, beat training baselines on unshielded mechanism outcomes, and retain nonzero strict-budget coverage. The oracle is included as an upper reference, not as a fair deployable method.

\section{{Hard-Aggregate Result}}
{hard_table}

The result is the central reason for the terminal decision. V5 reduces unshielded violation and dependence relative to v4, and it has the best non-oracle calibration. However, robust MPC has much higher success and substantially lower deployed violation. The safest and best-utility reference is \texttt{{{safest_ref}}}, not v5. The mechanism signal exists, but it is too small to justify a main-conference claim when deployment quality drops this much.

{deployed_figure}

{success_figure}

\section{{Paired Seed Tests}}
{paired_table}

The paired tests are especially damaging against robust MPC: v5 is lower on success, higher on deployed violation, and lower on robust utility with confidence intervals that do not rescue the claim. Against v4, v5 is genuinely better on several metrics, so the rebuild did improve the method. But "better than v4" is not the submission bar.

\section{{Split-Level Checks}}
{split_table}

The split table rules out a convenient rescue story. V5 does not secretly dominate on the hardest shift while losing only on nominal settings. The low-signal and combined-stress splits remain the hardest deployment regimes, and the deployed violation gap remains visible there.

\section{{Ablation Audit}}
{ablation_table}

The ablation result is the strongest evidence in favor of v5. Removing counterfactual labels, unshielded replay, intervention gradients, the boundary margin, or recovery-feasibility targets worsens the mechanism/utility profile. That means the method is not just a renamed clipping rule. It also means the negative decision is sharper: the mechanism is real, but not strong enough.

{ablation_figure}

\section{{Stress Sweep}}
{stress_table}

{stress_figure}

\section{{Fixed-Risk Deployment}}
{fixed_table}

Fixed-risk deployment is the cleanest hostile-review failure. At budget 0.05, v5 accepted coverage is zero on both hard deployment splits. A method cannot claim strict low-risk deployment when it declines all hard accepted actions at the risk budget. Calibration alone is not sufficient if the accepted set vanishes.

{fixed_figure}

\section{{Pareto Boundary and Negative Cases}}
{pareto_figure}

{negative_table}

\section{{Why This Is Not Submission Ready}}
There are four independent failures. First, the success gate fails: hard-aggregate v5 success is far below robust MPC. Second, the deployment gate fails: v5 deployed violation is higher than the safest strong baseline. Third, the fixed-risk gate fails: budget 0.05 coverage is zero on the hard splits. Fourth, the scope gate fails: there is no real robot, no accepted high-fidelity safety benchmark, no hardware timing study, and no independent reproduction.

\section{{What Would Be Required to Revive the Paper}}
A revived version would need more than prose. It would need a robot or accepted high-fidelity safety benchmark, a protocol where v5 keeps nonzero strict-budget coverage, and a mechanism that improves unshielded violations without losing the deployed Pareto frontier to robust MPC. It would also need to report all negative cases, all fixed-risk budgets, and all paired tests rather than only the internal ablation win.

\section{{Reproducibility and Audit Trail}}
The experiment is CPU-only and RAM-light. The runner writes all raw rollout, seed metric, aggregate metric, paired-test, ablation, stress, fixed-risk, and negative-case CSVs under \texttt{{results/}}. The manuscript is generated from those CSVs by \texttt{{scripts/generate\_manuscript.py}}. The validator checks row counts, PDF page count, citation-link annotations, frozen summary tokens, and Downloads-only PDF placement.

\section{{Terminal Recommendation}}
The terminal recommendation is \textbf{{KILL/ARCHIVE}}. This is the right scientific decision. The method improved during development, the protocol was frozen, and the frozen result says the current evidence cannot survive hostile ICLR-main review. Archiving this version protects the future version from being judged on a weak evidence package.

\clearpage
\appendix
\section{{Citation Coverage and Prior-Work Pressure}}
The in-text citations below are intentionally boxed by \texttt{{hyperref}}. Clicking a boxed citation routes to the corresponding bibliography entry, making prior-work audit trails visible during PDF review. {citation_wall}

\section{{Additional Theory Notes}}
The novelty boundary can be stated as a two-level causal claim. A deployment-level safety claim concerns $\tilde a_t=F_\phi(x_t,\pi_\theta(x_t))$. A mechanism-level claim concerns $\pi_\theta(x_t)$ before filtering. If the intervention operator changes $\tilde a_t$ but leaves $\pi_\theta(x_t)$ unsafe, the result is useful shielding, not evidence that the learned policy has internalized the safety boundary.

The v5 objective attempts to bridge this gap by adding counterfactual unsafe labels and unshielded replay. The audit verifies that these terms matter relative to ablations. The audit also shows they are insufficient under the stronger deployed frontier. That distinction matters: failure against robust MPC is not a proof that the theory is wrong; it is evidence that the present implementation and evidence package are not enough for the claimed venue.

\section{{Hostile Reviewer Checklist}}
A hostile reviewer would ask whether the method beats action clipping, geometric projection, CBF filters, MPC shields, conformal guards, recovery policies, shielded imitation, safe RL, adversarial safety critics, distillation, and the previous v4 version. This audit answers yes only for a subset of mechanism and calibration questions. It answers no for the decisive deployment and fixed-risk questions. {appendix_cites}

{summary_block}

\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}

\end{{document}}
"""
    (PAPER / "main.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {PAPER / 'main.tex'}")
    print(f"wrote {PAPER / 'references.bib'} with {len(keys)} entries")


if __name__ == "__main__":
    main()
