from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


PAPER_MARS_GENERAL = {
    "boolean_expressions": 93.17,
    "disambiguation_qa": 71.89,
    "formal_fallacies": 74.70,
    "geometric_shapes": 59.44,
    "ruin_names": 90.36,
    "sports_understanding": 87.95,
    "college_biology": 97.90,
    "college_medicine": 86.05,
    "electrical_engineering": 84.03,
    "high_school_world_history": 93.22,
    "human_aging": 85.59,
    "marketing": 97.00,
}

PAPER_MARS_DOMAIN = {
    "art_studies": 81.25,
    "urban_and_rural_planner": 84.44,
    "clinical_medicine": 85.71,
    "gsm8k": 89.22,
    "lsat_ar": 38.42,
}

TASK_DISPLAY_NAMES = {
    "boolean_expressions": "B.E.",
    "disambiguation_qa": "D.QA",
    "formal_fallacies": "F.F.",
    "geometric_shapes": "G.S.",
    "ruin_names": "R.N.",
    "sports_understanding": "S.U.",
    "college_biology": "C.B.",
    "college_medicine": "C.M.",
    "electrical_engineering": "E.E.",
    "high_school_world_history": "W.H.",
    "human_aging": "H.A.",
    "marketing": "M.T.",
    "art_studies": "A.S.",
    "urban_and_rural_planner": "U.R.P.",
    "clinical_medicine": "CL.M.",
    "gsm8k": "GSM",
    "lsat_ar": "L.A.",
}

GROUP_ORDER = ["BBH", "MMLU", "Domain"]

GROUP_TASKS = {
    "BBH": [
        "boolean_expressions",
        "disambiguation_qa",
        "formal_fallacies",
        "geometric_shapes",
        "ruin_names",
        "sports_understanding",
    ],
    "MMLU": [
        "college_biology",
        "college_medicine",
        "electrical_engineering",
        "high_school_world_history",
        "human_aging",
        "marketing",
    ],
    "Domain": [
        "art_studies",
        "urban_and_rural_planner",
        "clinical_medicine",
        "gsm8k",
        "lsat_ar",
    ],
}


def normalize_accuracy(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if value <= 1.0:
        value *= 100.0
    return round(value, 2)


def find_latest_run(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    runs = [path for path in results_root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def load_summary(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv in {run_dir}")
    return pd.read_csv(summary_path)


def _extract_histories_from_directory(task_dir: Path) -> Optional[pd.DataFrame]:
    history_path = task_dir / "prompt_accuracy_history.csv"
    if not history_path.exists():
        return None
    try:
        df = pd.read_csv(history_path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def load_histories(run_dir: Path) -> Dict[str, pd.DataFrame]:
    task_root = run_dir / "tasks"
    histories: Dict[str, pd.DataFrame] = {}
    if not task_root.exists():
        return histories
    for task_dir in task_root.iterdir():
        if not task_dir.is_dir():
            continue
        df = _extract_histories_from_directory(task_dir)
        if df is not None:
            histories[task_dir.name] = df
    return histories


def _paper_lookup(task_id: str) -> Optional[float]:
    if task_id in PAPER_MARS_GENERAL:
        return PAPER_MARS_GENERAL[task_id]
    if task_id in PAPER_MARS_DOMAIN:
        return PAPER_MARS_DOMAIN[task_id]
    return None


def attach_paper_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    df["paper_mars_accuracy"] = df["task_id"].map(_paper_lookup)
    df["our_best_accuracy"] = df["best_accuracy"].apply(normalize_accuracy)
    df["our_final_accuracy"] = df["final_accuracy"].apply(normalize_accuracy)
    df["delta_best_minus_paper"] = df.apply(
        lambda row: None
        if row["paper_mars_accuracy"] is None or row["our_best_accuracy"] is None
        else round(row["our_best_accuracy"] - row["paper_mars_accuracy"], 2),
        axis=1,
    )
    df["delta_final_minus_paper"] = df.apply(
        lambda row: None
        if row["paper_mars_accuracy"] is None or row["our_final_accuracy"] is None
        else round(row["our_final_accuracy"] - row["paper_mars_accuracy"], 2),
        axis=1,
    )
    return df


def _prep_axes(ax, title: str, ylabel: str):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _save(fig, out_path: Path, dpi: int, fmt: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)


def plot_summary_overview(df: pd.DataFrame, out_path: Path, dpi: int = 300, fmt: str = "png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    total = len(df)
    success = int((df["status"] == "success").sum())
    failed = total - success
    avg_best = df["our_best_accuracy"].dropna().mean()
    avg_final = df["our_final_accuracy"].dropna().mean()
    total_runtime = df["total_runtime_seconds"].fillna(0).sum()

    ax = axes[0]
    labels = ["Tasks", "Success", "Failed"]
    values = [total, success, failed]
    colors = ["#4c78a8", "#54a24b", "#e45756"]
    bars = ax.bar(labels, values, color=colors)
    _prep_axes(ax, "MARS run overview", "Count")
    ax.bar_label(bars, padding=3)

    axes[1].axis("off")
    text_lines = [
        f"Average best accuracy: {avg_best:.2f}%" if pd.notna(avg_best) else "Average best accuracy: unknown",
        f"Average final accuracy: {avg_final:.2f}%" if pd.notna(avg_final) else "Average final accuracy: unknown",
        f"Total runtime: {total_runtime:.2f}s",
    ]
    axes[1].text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=12)
    axes[1].set_title("Run stats")
    _save(fig, out_path, dpi, fmt)


def plot_task_accuracy_vs_paper(df: pd.DataFrame, out_path: Path, dpi: int = 300, fmt: str = "png") -> None:
    ordered = df.copy()
    ordered["task_display"] = ordered["task_id"].map(lambda x: TASK_DISPLAY_NAMES.get(x, x))
    ordered = ordered.sort_values(["group", "task_id"])
    x = range(len(ordered))
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.36
    paper = ordered["paper_mars_accuracy"].fillna(0)
    our = ordered["our_best_accuracy"].fillna(0)
    ax.bar([i - width / 2 for i in x], paper, width=width, label="Paper MARS", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], our, width=width, label="Our run MARS", color="#f58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["task_display"], rotation=45, ha="right")
    _prep_axes(ax, "Task accuracy vs paper", "Accuracy (%)")
    ax.legend()
    _save(fig, out_path, dpi, fmt)


def plot_task_delta_vs_paper(df: pd.DataFrame, out_path: Path, dpi: int = 300, fmt: str = "png") -> None:
    ordered = df.copy()
    ordered["task_display"] = ordered["task_id"].map(lambda x: TASK_DISPLAY_NAMES.get(x, x))
    ordered = ordered.sort_values(["group", "task_id"])
    deltas = ordered["delta_best_minus_paper"].fillna(0)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in deltas]
    positions = range(len(ordered))
    ax.bar(list(positions), deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(ordered["task_display"], rotation=45, ha="right")
    _prep_axes(ax, "Delta between our best accuracy and paper MARS", "Delta (%)")
    _save(fig, out_path, dpi, fmt)


def plot_group_average_vs_paper(df: pd.DataFrame, out_path: Path, dpi: int = 300, fmt: str = "png") -> None:
    records = []
    for group in GROUP_ORDER:
        tasks = GROUP_TASKS.get(group, [])
        group_df = df[df["task_id"].isin(tasks)]
        if group == "Domain":
            group_df = group_df[group_df["paper_mars_accuracy"].notna() & group_df["our_best_accuracy"].notna()]
        if group_df.empty:
            continue
        records.append({
            "group": group,
            "paper": group_df["paper_mars_accuracy"].mean(),
            "our_best": group_df["our_best_accuracy"].mean(),
            "our_final": group_df["our_final_accuracy"].mean(),
        })
    if not records:
        return
    group_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(group_df))
    width = 0.25
    ax.bar([i - width for i in x], group_df["paper"], width=width, label="Paper MARS", color="#4c78a8")
    ax.bar(x, group_df["our_best"], width=width, label="Our best", color="#f58518")
    ax.bar([i + width for i in x], group_df["our_final"], width=width, label="Our final", color="#54a24b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_df["group"])
    _prep_axes(ax, "Group average comparison", "Accuracy (%)")
    ax.legend()
    _save(fig, out_path, dpi, fmt)


def _history_to_xy(history: pd.DataFrame) -> Tuple[List[int], List[float]]:
    if history is None or history.empty:
        return [], []
    columns = {col.lower(): col for col in history.columns}
    iteration_col = None
    for candidate in ["iteration", "iter", "step", "round"]:
        if candidate in columns:
            iteration_col = columns[candidate]
            break
    accuracy_col = None
    for candidate in ["accuracy", "acc", "best_accuracy", "final_accuracy"]:
        if candidate in columns:
            accuracy_col = columns[candidate]
            break
    if iteration_col is None or accuracy_col is None:
        return [], []
    x = pd.to_numeric(history[iteration_col], errors="coerce").fillna(0).astype(int).tolist()
    y = []
    for value in history[accuracy_col].tolist():
        normalized = normalize_accuracy(value)
        y.append(0.0 if normalized is None else normalized)
    return x, y


def plot_convergence_curves(
    histories: Dict[str, pd.DataFrame],
    task_ids: Sequence[str],
    out_path: Path,
    title: str,
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    if not task_ids:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for task_id in task_ids:
        history = histories.get(task_id)
        if history is None or history.empty:
            continue
        x, y = _history_to_xy(history)
        if not x or not y:
            continue
        ax.plot(x, y, marker="o", linewidth=1.5, label=TASK_DISPLAY_NAMES.get(task_id, task_id))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    if plotted <= 17:
        ax.legend(loc="best", fontsize=8, ncol=2)
    _save(fig, out_path, dpi, fmt)


def plot_best_iteration_distribution(
    df: pd.DataFrame,
    histories: Dict[str, pd.DataFrame],
    out_path: Path,
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    values = []
    for _, row in df.iterrows():
        best_iteration = row.get("best_iteration")
        if pd.notna(best_iteration) and str(best_iteration) != "":
            try:
                values.append(int(best_iteration))
                continue
            except Exception:
                pass
        history = histories.get(row["task_id"])
        if history is None or history.empty:
            continue
        x, y = _history_to_xy(history)
        if y:
            best_idx = max(range(len(y)), key=lambda i: y[i])
            if x:
                values.append(int(x[best_idx]))
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(1, max(values) + 2)
    ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Task count")
    ax.set_title("Distribution of best-performing iteration")
    _save(fig, out_path, dpi, fmt)


def plot_status_overview(df: pd.DataFrame, out_path: Path, dpi: int = 300, fmt: str = "png") -> None:
    counts = df["status"].fillna("missing").value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Task status overview")
    _save(fig, out_path, dpi, fmt)


def write_paper_comparison_table(df: pd.DataFrame, out_path: Path) -> None:
    columns = [
        "task_id",
        "display_name",
        "group",
        "paper_mars_accuracy",
        "our_best_accuracy",
        "our_final_accuracy",
        "delta_best_minus_paper",
        "delta_final_minus_paper",
        "best_iteration",
        "num_samples",
        "num_success",
        "num_failed",
        "status",
        "stopped_reason",
        "total_runtime_seconds",
    ]
    table = df.copy()
    table["display_name"] = table["task_id"].map(lambda x: TASK_DISPLAY_NAMES.get(x, x))
    table = table[columns]
    table.to_csv(out_path, index=False)


def write_report(run_dir: Path, df: pd.DataFrame, histories: Dict[str, pd.DataFrame], out_dir: Path, generated_files: Sequence[Path]) -> None:
    config_path = run_dir / "config.yaml"
    config = {}
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            config = {}
    total_tasks = len(df)
    success_tasks = int((df["status"] == "success").sum()) if "status" in df else 0
    failed_tasks = total_tasks - success_tasks
    avg_best = df["our_best_accuracy"].dropna().mean()
    avg_final = df["our_final_accuracy"].dropna().mean()
    deltas = df["delta_best_minus_paper"].dropna()
    avg_delta = deltas.mean() if not deltas.empty else None
    total_runtime = df["total_runtime_seconds"].fillna(0).sum()
    lines = [
        "# MARS Visualization Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Output directory: `{out_dir}`",
        "",
        "## Run metadata",
        f"- model: {config.get('model', 'unknown')}",
        f"- temperature: {config.get('temperature', 'unknown')}",
        f"- max_iterations: {config.get('max_iterations', 'unknown')}",
        f"- early_stop_delta: {config.get('early_stop_delta', 'unknown')}",
        f"- concurrency: {config.get('concurrency', 'unknown')}",
        f"- dry_run: {config.get('dry_run', 'unknown')}",
        "",
        "## Overall summary",
        f"- number of tasks: {total_tasks}",
        f"- number of successful tasks: {success_tasks}",
        f"- number of failed tasks: {failed_tasks}",
        f"- average best accuracy: {avg_best:.2f}%" if pd.notna(avg_best) else "- average best accuracy: unknown",
        f"- average final accuracy: {avg_final:.2f}%" if pd.notna(avg_final) else "- average final accuracy: unknown",
        f"- average delta vs paper: {avg_delta:.2f}%" if avg_delta is not None and pd.notna(avg_delta) else "- average delta vs paper: unknown",
        f"- total runtime: {total_runtime:.2f}s",
        "",
        "## Caveats",
        "This visualization compares the local MARS-only run against paper-reported MARS results. It does not reproduce the paper's baseline comparisons, ablation studies, efficiency analysis, or cross-model transfer experiments.",
        "The local run used deepseek-chat, while the paper reports main results with deepseek-v2.5-1210. Therefore, numerical differences may reflect model-version differences, API behavior, sampling variance, runnable-task filtering, and implementation differences.",
        "",
        "## Required note",
        "This run uses deepseek-chat, while the paper reports results with deepseek-v2.5-1210. The comparison is for reference only and should not be interpreted as an exact reproduction of the paper's numerical results.",
        "",
        "## Generated figures",
    ]
    for file_path in generated_files:
        lines.append(f"- {file_path.relative_to(run_dir)}")
    lines.append("")
    lines.append("## Comparison table")
    table_path = out_dir / "paper_comparison_table.csv"
    if table_path.exists():
        lines.append(f"- `{table_path.relative_to(run_dir)}`")
    lines.append("")
    lines.append("## Missing inputs")
    missing_histories = sorted(set(df["task_id"]) - set(histories.keys()))
    if missing_histories:
        lines.append("- missing prompt history for: " + ", ".join(missing_histories))
    else:
        lines.append("- none")
    (out_dir / "visualization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
