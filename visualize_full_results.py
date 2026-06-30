from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def find_latest(results_root: Path) -> Path:
    runs = sorted(path for path in results_root.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {results_root}")
    return runs[-1]


def resolve_run(args) -> Path:
    if args.latest:
        return find_latest(Path(args.results_root))
    if args.run_dir:
        if args.run_dir == "latest":
            return find_latest(Path(args.results_root))
        return Path(args.run_dir)
    return find_latest(Path(args.results_root))


def task_group_lookup(config_path: Path = Path("configs/tasks.yaml")) -> dict[str, str]:
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {task_id: str(config.get("group", "")) for task_id, config in data.items()}


def _save_bar(
    df: pd.DataFrame,
    path: Path,
    title: str,
    group_col: str = "method",
    dpi: int = 200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("No data available for this figure.\n", encoding="utf-8")
        return
    pivot = df.pivot_table(
        index="task_id", columns=group_col, values="accuracy", aggfunc="mean"
    )
    ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 0.8), 5))
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _save_scatter(
    df: pd.DataFrame, path: Path, title: str, x: str, y: str, dpi: int = 200
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or x not in df or y not in df:
        path.write_text("No data available for this figure.\n", encoding="utf-8")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in df.groupby("method"):
        ax.scatter(group[x], group[y], label=method)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _save_convergence(run_dir: Path, path: Path, dpi: int = 200) -> None:
    curve_files = list((run_dir / "convergence").glob("*_curves.csv"))
    frames = [pd.read_csv(file) for file in curve_files if file.stat().st_size > 0]
    if not frames:
        path.write_text("No convergence data available.\n", encoding="utf-8")
        return
    df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (task_id, method), group in df.groupby(["task_id", "method"]):
        ax.plot(
            group["iteration"],
            group["accuracy"],
            marker="o",
            label=f"{task_id}/{method}",
        )
    ax.set_title("Convergence Curves")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def write_report(run_dir: Path, summary: pd.DataFrame, report_path: Path) -> None:
    api_errors = (
        int(summary.get("api_errors", pd.Series(dtype=int)).fillna(0).sum())
        if not summary.empty
        else 0
    )
    parse_errors = (
        int(summary.get("parse_errors", pd.Series(dtype=int)).fillna(0).sum())
        if not summary.empty
        else 0
    )
    suites = ", ".join(
        sorted(
            str(item)
            for item in summary.get("suite", pd.Series(dtype=str)).dropna().unique()
        )
    )
    exactness = (
        summary.get("exactness", pd.Series(dtype=str))
        .fillna("")
        .value_counts()
        .to_dict()
    )
    lines = [
        "# Full Reproduction Visualization Report",
        "",
        f"- run_dir: {run_dir}",
        f"- suites: {suites or 'none'}",
        f"- rows: {len(summary)}",
        f"- api_errors: {api_errors}",
        f"- parse_errors: {parse_errors}",
        "",
        "## Reproduction Categories",
        "",
    ]
    for name, count in exactness.items():
        lines.append(f"- {name or 'unspecified'}: {count}")
    lines.extend(
        [
            "",
            "## Trust Notes",
            "",
            "Paper results are not substituted for local results. "
            "Rows marked as reimplementation are best-effort local implementations when official templates/code were not available.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize full MARS reproduction results."
    )
    parser.add_argument("--results-root", default="results_full")
    parser.add_argument("--run-dir")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--out-dir-name", default="figures")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()
    run_dir = resolve_run(args)
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    task_groups = task_group_lookup()
    if task_groups and "task_id" in summary:
        summary = summary.copy()
        summary["task_group"] = summary["task_id"].map(task_groups).fillna("")
    figures = run_dir / args.out_dir_name
    figures.mkdir(parents=True, exist_ok=True)

    main = (
        summary[summary.get("suite") == "main"] if "suite" in summary else summary
    )
    general = (
        main[main.get("task_group") == "BBH"]
        if "task_group" in main
        else main
    )
    domain = (
        main[main.get("task_group") != "BBH"]
        if "task_group" in main
        else main.iloc[0:0]
    )
    suffix = args.format
    _save_bar(
        general,
        figures / f"table1_general_bar.{suffix}",
        "Table 1 General Tasks",
        dpi=args.dpi,
    )
    _save_bar(
        domain,
        figures / f"table2_domain_bar.{suffix}",
        "Table 2 Domain Tasks",
        dpi=args.dpi,
    )
    ablation = (
        summary[summary.get("suite") == "ablation"]
        if "suite" in summary
        else pd.DataFrame()
    )
    if not ablation.empty and "delta" in ablation:
        _save_scatter(
            ablation,
            figures / f"table3_ablation_delta.{suffix}",
            "Ablation Delta",
            "delta",
            "accuracy",
            dpi=args.dpi,
        )
    else:
        _save_bar(
            ablation,
            figures / f"table3_ablation_delta.{suffix}",
            "Table 3 Ablation",
            dpi=args.dpi,
        )
    efficiency = (
        summary[summary.get("suite") == "efficiency"]
        if "suite" in summary
        else pd.DataFrame()
    )
    _save_scatter(
        efficiency,
        figures / f"inference_time_scaling.{suffix}",
        "Inference Time Scaling",
        "runtime_seconds",
        "accuracy",
        dpi=args.dpi,
    )
    _save_scatter(
        efficiency,
        figures / f"efficiency_accuracy_vs_tokens.{suffix}",
        "Accuracy vs Tokens",
        "tokens_total",
        "accuracy",
        dpi=args.dpi,
    )
    _save_scatter(
        efficiency,
        figures / f"cost_accuracy_tradeoff.{suffix}",
        "Cost Accuracy Tradeoff",
        "cost_estimate",
        "accuracy",
        dpi=args.dpi,
    )
    _save_convergence(run_dir, figures / f"convergence_all.{suffix}", dpi=args.dpi)
    transfer = (
        summary[summary.get("suite") == "transfer"]
        if "suite" in summary
        else pd.DataFrame()
    )
    _save_bar(
        transfer,
        figures / f"transfer_model_comparison.{suffix}",
        "Transfer Model Comparison",
        group_col="model",
        dpi=args.dpi,
    )
    _save_bar(
        transfer,
        figures / f"transfer_models.{suffix}",
        "Transfer Models",
        group_col="model",
        dpi=args.dpi,
    )
    report_path = figures / "visualization_report.md"
    write_report(run_dir, summary, report_path)
    print(f"Visualization complete: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
