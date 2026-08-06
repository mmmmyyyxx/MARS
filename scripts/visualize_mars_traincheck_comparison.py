from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


DEFAULT_TASK_ORDER = [
    "boolean_expressions",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "ruin_names",
    "sports_understanding",
]

SERIES_ORDER = [
    "Origin",
    "CoT-ZS",
    "CoT-FS",
    "MARS",
    "TrainCheck baseline vote",
    "TrainCheck baseline oracle",
    "TrainCheck guarded beam vote",
    "TrainCheck guarded beam oracle",
]

SERIES_COLORS = {
    "Origin": "#7b8794",
    "CoT-ZS": "#4c78a8",
    "CoT-FS": "#54a24b",
    "MARS": "#2f6f8f",
    "TrainCheck baseline vote": "#e27b7b",
    "TrainCheck baseline oracle": "#8b1e1e",
    "TrainCheck guarded beam vote": "#f0a0a0",
    "TrainCheck guarded beam oracle": "#5c0f0f",
}

BASELINE_LABELS = {
    "origin": "Origin",
    "cot_zs": "CoT-ZS",
    "cot_fs": "CoT-FS",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _latest_run(results_root: Path) -> Path:
    runs = sorted(path for path in results_root.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {results_root}")
    return runs[-1]


def _resolve_run_dir(value: str | None, results_root: Path) -> Path:
    if not value or value == "latest":
        return _latest_run(results_root)
    return Path(value)


def _task_display_names(config_path: Path = Path("configs/tasks.yaml")) -> dict[str, str]:
    data = _load_yaml(config_path)
    names: dict[str, str] = {}
    for task_id, config in data.items():
        if isinstance(config, dict):
            names[str(task_id)] = str(config.get("paper_display_name") or task_id)
    return names


def load_mars_results(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics_root = run_dir / "methods" / "mars_official"
    if metrics_root.exists():
        for metrics_path in sorted(metrics_root.glob("*/metrics.json")):
            metrics = _load_json(metrics_path)
            if not metrics:
                continue
            task_id = str(metrics.get("task_id") or metrics_path.parent.name)
            rows.append(
                {
                    "task_id": task_id,
                    "series": "MARS",
                    "source": "results_full",
                    "setting": "mars_official",
                    "accuracy_metric": "accuracy",
                    "accuracy": float(metrics["accuracy"]),
                    "num_samples": int(metrics.get("num_samples") or 0),
                }
            )

    if rows:
        return pd.DataFrame(rows)

    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find MARS metrics or summary.csv under {run_dir}"
        )
    summary = pd.read_csv(summary_path)
    if "paper_method_id" in summary:
        summary = summary[summary["paper_method_id"].fillna("") == "mars"]
    elif "method_id" in summary:
        summary = summary[summary["method_id"].fillna("").isin(["mars", "mars_official"])]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "task_id": row["task_id"],
                "series": "MARS",
                "source": "results_full",
                "setting": str(row.get("method_id") or "mars"),
                "accuracy_metric": "accuracy",
                "accuracy": float(row["accuracy"]),
                "num_samples": int(row.get("num_samples") or row.get("num_test_samples") or 0),
            }
        )
    return pd.DataFrame(rows)


def load_results_full_baselines(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)
    if "method_id" not in summary:
        return pd.DataFrame()
    summary = summary[summary["method_id"].fillna("").isin(BASELINE_LABELS)]
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        method_id = str(row["method_id"])
        rows.append(
            {
                "task_id": str(row["task_id"]),
                "series": BASELINE_LABELS[method_id],
                "source": "results_full",
                "setting": method_id,
                "accuracy_metric": "accuracy",
                "accuracy": float(row["accuracy"]),
                "num_samples": int(row.get("num_samples") or row.get("num_test_samples") or 0),
            }
        )
    return pd.DataFrame(rows)


def load_traincheck_results(traincheck_dir: Path) -> pd.DataFrame:
    summary_path = traincheck_dir / "accuracy_summary.csv"
    detail_path = traincheck_dir / "accuracy_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        detail_agg = (
            detail.groupby(["task_id", "setting"], as_index=False)
            .agg(
                num_samples=("num_test_samples", "mean"),
                leakage_warning=("leakage_warning", "max"),
                split_protocol=("split_protocol", "first"),
                total_llm_calls=("total_llm_calls", "mean"),
                total_tokens=("total_tokens", "mean"),
            )
        )
        summary = summary.merge(detail_agg, on=["task_id", "setting"], how="left")

    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        setting = str(row["setting"])
        setting_label = {
            "shared_baseline": "TrainCheck baseline",
            "shared_guarded_beam": "TrainCheck guarded beam",
        }.get(setting, f"TrainCheck {setting}")
        metric_specs = [
            ("vote_acc_mean", "vote_acc", f"{setting_label} vote"),
            ("oracle_acc_mean", "oracle_acc", f"{setting_label} oracle"),
        ]
        for column, metric_name, series in metric_specs:
            value = row.get(column)
            if pd.isna(value):
                continue
            rows.append(
                {
                "task_id": str(row["task_id"]),
                "series": series,
                "source": "runs_task_level_bbh_traincheck_x5",
                "setting": setting,
                "accuracy_metric": metric_name,
                "accuracy": float(value),
                "num_samples": int(row.get("num_samples") or 0),
                "vote_accuracy": float(row.get("vote_acc_mean", float("nan"))),
                "oracle_accuracy": float(row.get("oracle_acc_mean", float("nan"))),
                "mean_individual_accuracy": float(
                    row.get("mean_individual_acc_mean", float("nan"))
                ),
                "best_individual_accuracy": float(
                    row.get("best_individual_acc_mean", float("nan"))
                ),
                "total_llm_calls": row.get("total_llm_calls", ""),
                "total_tokens": row.get("total_tokens", ""),
                "leakage_warning": row.get("leakage_warning", ""),
                "split_protocol": row.get("split_protocol", ""),
                }
            )
    return pd.DataFrame(rows)


def _ordered_tasks(df: pd.DataFrame) -> list[str]:
    present = set(df["task_id"])
    ordered = [task_id for task_id in DEFAULT_TASK_ORDER if task_id in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _ordered_series(df: pd.DataFrame) -> list[str]:
    present = set(df["series"])
    ordered = [series for series in SERIES_ORDER if series in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _series_colors(columns: list[str]) -> list[str]:
    return [SERIES_COLORS.get(column, "#9aa0a6") for column in columns]


def _plot_accuracy(
    df: pd.DataFrame,
    path: Path,
    display_names: dict[str, str],
    *,
    dpi: int,
) -> None:
    tasks = _ordered_tasks(df)
    plot_df = df.copy()
    plot_df["task_label"] = plot_df["task_id"].map(display_names).fillna(plot_df["task_id"])
    plot_df["task_label"] = pd.Categorical(
        plot_df["task_label"],
        [display_names.get(task_id, task_id) for task_id in tasks],
        ordered=True,
    )
    pivot = plot_df.pivot_table(
        index="task_label",
        columns="series",
        values="accuracy",
        aggfunc="mean",
        observed=False,
    )
    columns = [column for column in _ordered_series(plot_df) if column in pivot.columns]
    pivot = pivot[columns] * 100.0

    ax = pivot.plot(
        kind="bar",
        figsize=(15.5, 6.6),
        width=0.84,
        color=_series_colors(columns),
    )
    ax.set_title("Results Full Baselines/MARS vs TrainCheck x5 on Overlapping BBH Tasks")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=5, padding=2)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close()


def _plot_delta(
    df: pd.DataFrame,
    path: Path,
    display_names: dict[str, str],
    *,
    dpi: int,
) -> pd.DataFrame:
    tasks = _ordered_tasks(df)
    wide = df.pivot_table(
        index="task_id", columns="series", values="accuracy", aggfunc="mean"
    )
    if "MARS" not in wide:
        raise ValueError("MARS series is required for delta plot")
    columns = [column for column in _ordered_series(df) if column in wide.columns]
    delta = wide.drop(columns=["MARS"], errors="ignore").subtract(wide["MARS"], axis=0)
    delta = delta[[column for column in columns if column in delta.columns]]
    delta = delta.reindex(tasks) * 100.0
    delta.index = [display_names.get(task_id, task_id) for task_id in delta.index]
    delta_columns = list(delta.columns)
    ax = delta.plot(
        kind="bar",
        figsize=(15.0, 6.0),
        width=0.82,
        color=_series_colors(delta_columns),
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Accuracy Delta vs MARS on Overlapping BBH Tasks")
    ax.set_ylabel("Delta (percentage points)")
    ax.set_xlabel("")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close()
    return delta.reset_index(names="task")


def _write_report(
    path: Path,
    run_dir: Path,
    traincheck_dir: Path,
    df: pd.DataFrame,
    delta: pd.DataFrame,
) -> None:
    mean_rows = (
        df.groupby("series", as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), mean_num_samples=("num_samples", "mean"))
        .sort_values("series")
    )
    lines = [
        "# MARS vs TrainCheck x5 Comparison",
        "",
        f"- mars_run_dir: {run_dir}",
        f"- traincheck_dir: {traincheck_dir}",
        f"- overlapping_tasks: {df['task_id'].nunique()}",
        "- traincheck_accuracy_metrics: vote_acc_mean, oracle_acc_mean",
        "- delta_definition: each non-MARS series accuracy minus MARS accuracy",
        "",
        "## Mean Accuracy",
        "",
        "| series | mean_accuracy | mean_num_samples |",
        "|---|---:|---:|",
    ]
    for _, row in mean_rows.iterrows():
        lines.append(
            f"| {row['series']} | {row['mean_accuracy']:.4f} | "
            f"{row['mean_num_samples']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Comparability Notes",
            "",
            "- MARS and reproduction baselines are loaded from `results_full`.",
            "- TrainCheck x5 results are loaded from `accuracy_summary.csv`; both `vote_acc_mean` and `oracle_acc_mean` are plotted.",
            "- The overlapping TrainCheck rows use 100 test samples, while the MARS BBH rows use 250 samples in this run.",
            "- The TrainCheck detail file marks `leakage_warning=True`; interpret direct accuracy deltas as diagnostic rather than final paper-style claims.",
            "",
            "## Files",
            "",
            "- `mars_vs_traincheck_x5_accuracy.png`",
            "- `mars_vs_traincheck_x5_delta.png`",
            "- `mars_vs_traincheck_x5_comparison.csv`",
            "- `mars_vs_traincheck_x5_delta.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize MARS results against TrainCheck x5 task-level BBH results."
    )
    parser.add_argument("--results-root", default="results_full")
    parser.add_argument("--mars-run-dir", default="latest")
    parser.add_argument("--traincheck-dir", default="runs_task_level_bbh_traincheck_x5")
    parser.add_argument("--out-dir")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.mars_run_dir, Path(args.results_root))
    traincheck_dir = Path(args.traincheck_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else run_dir / "figures" / "traincheck_x5_comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = load_results_full_baselines(run_dir)
    mars = load_mars_results(run_dir)
    traincheck = load_traincheck_results(traincheck_dir)
    results_full_tasks = set(mars["task_id"])
    if not baselines.empty:
        results_full_tasks.update(set(baselines["task_id"]))
    overlap = sorted(results_full_tasks.intersection(set(traincheck["task_id"])))
    if not overlap:
        raise ValueError("No overlapping task_id values between results_full and TrainCheck")
    df = pd.concat(
        [
            baselines[baselines["task_id"].isin(overlap)],
            mars[mars["task_id"].isin(overlap)],
            traincheck[traincheck["task_id"].isin(overlap)],
        ],
        ignore_index=True,
        sort=False,
    )
    df["accuracy_percent"] = df["accuracy"] * 100.0
    df.to_csv(out_dir / "mars_vs_traincheck_x5_comparison.csv", index=False)

    display_names = _task_display_names()
    _plot_accuracy(
        df,
        out_dir / "mars_vs_traincheck_x5_accuracy.png",
        display_names,
        dpi=args.dpi,
    )
    delta = _plot_delta(
        df,
        out_dir / "mars_vs_traincheck_x5_delta.png",
        display_names,
        dpi=args.dpi,
    )
    delta.to_csv(out_dir / "mars_vs_traincheck_x5_delta.csv", index=False)
    _write_report(out_dir / "README.md", run_dir, traincheck_dir, df, delta)

    print(f"Wrote comparison figures to {out_dir}")
    print(f"Rows: {len(df)}; overlapping tasks: {len(overlap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
