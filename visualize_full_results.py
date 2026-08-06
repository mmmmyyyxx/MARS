from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_csv_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if not text or text.lower() == "all":
            return None
        items = [item.strip() for item in text.split(",")]
    return {item for item in items if item}


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_expected_counts(config_path: Path = Path("configs/tasks.yaml")) -> dict[str, int]:
    data = _load_yaml(config_path)
    counts: dict[str, int] = {}
    for task_id, config in data.items():
        if not isinstance(config, dict):
            continue
        dataset_path = config.get("test_path") or config.get("dataset_path")
        if not dataset_path:
            continue
        path = Path(str(dataset_path))
        if not path.exists():
            continue
        try:
            counts[str(task_id)] = len(pd.read_csv(path))
        except Exception:
            continue
    return counts


def _selected_expected_count(
    task_id: str, task_counts: dict[str, int], max_samples: Any
) -> int | None:
    full_count = task_counts.get(task_id)
    max_count = _to_int(max_samples)
    if full_count is None:
        return max_count
    if max_count is None:
        return full_count
    return min(full_count, max_count)


def _summary_key(row: pd.Series) -> tuple[str, str, str]:
    suite = str(row.get("suite") or "main")
    method_id = str(row.get("method_id") or "")
    task_id = str(row.get("task_id") or "")
    return suite, method_id, task_id


def augment_with_completed_method_metrics(
    summary: pd.DataFrame,
    run_dir: Path,
    *,
    methods_config_path: Path = Path("configs/methods.yaml"),
) -> pd.DataFrame:
    methods_root = run_dir / "methods"
    if not methods_root.exists():
        return summary

    run_config = _load_yaml(run_dir / "run_config.yaml")
    selected_tasks = _parse_csv_set(run_config.get("tasks"))
    selected_methods = _parse_csv_set(run_config.get("methods"))
    summary_methods = (
        set(summary["method_id"].dropna().astype(str))
        if "method_id" in summary
        else set()
    )
    include_methods = set(summary_methods)
    if selected_methods is not None:
        include_methods.update(selected_methods)
    include_methods.add("mars_official")

    methods_config = _load_yaml(methods_config_path)
    task_config = _load_yaml(Path("configs/tasks.yaml"))
    task_counts = _task_expected_counts()
    existing = {_summary_key(row) for _, row in summary.iterrows()}
    rows: list[dict[str, Any]] = []

    for metrics_path in sorted(methods_root.glob("*/*/metrics.json")):
        task_id = metrics_path.parent.name
        method_id = metrics_path.parent.parent.name
        if method_id not in include_methods:
            continue
        if selected_tasks is not None and task_id not in selected_tasks:
            continue

        metrics = _load_json(metrics_path)
        if not metrics:
            continue
        state = _load_json(metrics_path.parent / "run_state.json")
        if state and state.get("status") not in {None, "", "completed"}:
            continue

        num_samples = _to_int(metrics.get("num_samples"))
        expected_count = _selected_expected_count(
            task_id, task_counts, run_config.get("max_samples")
        )
        if expected_count is not None and num_samples != expected_count:
            continue

        expected_ids = state.get("expected_sample_ids") if state else None
        if isinstance(expected_ids, list) and num_samples != len(expected_ids):
            continue

        suite = str(state.get("suite") or run_config.get("suite") or "main")
        key = (suite, method_id, task_id)
        if key in existing:
            continue

        method_config = methods_config.get(method_id) or {}
        task_meta = task_config.get(task_id) or {}
        split_hashes = state.get("split_hashes") if isinstance(state, dict) else {}
        if not isinstance(split_hashes, dict):
            split_hashes = {}
        rows.append(
            {
                "accuracy": metrics.get("accuracy"),
                "api_calls": metrics.get("api_calls", ""),
                "api_errors": metrics.get("api_errors", 0),
                "cache_hits": metrics.get("cache_hits", ""),
                "cost_estimate": metrics.get("cost_estimate", ""),
                "display_name": task_meta.get("paper_display_name", task_id),
                "eval_protocol": run_config.get("eval_protocol", ""),
                "exactness_level": metrics.get(
                    "exactness_level", method_config.get("exactness_level", "")
                ),
                "exactness_note": method_config.get("exactness_note", ""),
                "exactness_notes": method_config.get("exactness_note", ""),
                "method": metrics.get(
                    "method_display_name",
                    method_config.get("display_name", method_id),
                ),
                "method_id": method_id,
                "model": state.get("model") or run_config.get("model", ""),
                "num_correct": metrics.get("num_correct", ""),
                "num_failed": metrics.get("num_failed", ""),
                "num_iterations": metrics.get("num_iterations", ""),
                "num_opt_samples": metrics.get("num_opt_samples", num_samples),
                "num_samples": num_samples,
                "num_test_samples": metrics.get("num_test_samples", num_samples),
                "num_val_samples": metrics.get("num_val_samples", num_samples),
                "opt_hash": split_hashes.get("opt", ""),
                "paper_method_id": method_config.get("paper_method_id", method_id),
                "parse_errors": metrics.get("parse_errors", 0),
                "runtime_seconds": metrics.get("runtime_seconds", ""),
                "status": "completed",
                "suite": suite,
                "task_id": task_id,
                "test_hash": split_hashes.get("test", ""),
                "tokens_completion": metrics.get("tokens_completion", ""),
                "tokens_prompt": metrics.get("tokens_prompt", ""),
                "tokens_total": metrics.get("tokens_total", ""),
                "val_hash": split_hashes.get("val", ""),
            }
        )
        existing.add(key)

    if not rows:
        return summary
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)


def load_paper_results(
    config_path: Path = Path("configs/paper_results.yaml"),
) -> pd.DataFrame:
    if not config_path.exists():
        return pd.DataFrame()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rows = []
    for table_name in ["table1", "table2", "table3"]:
        suite = "ablation" if table_name == "table3" else "main"
        for method_id, task_scores in (data.get(table_name) or {}).items():
            for task_id, accuracy in (task_scores or {}).items():
                rows.append(
                    {
                        "suite": suite,
                        "paper_table": table_name,
                        "task_id": task_id,
                        "method_key": method_id,
                        "method": method_id,
                        "source": "paper",
                        "accuracy": float(accuracy) / 100.0,
                    }
                )
    return pd.DataFrame(rows)


def _method_key(summary: pd.DataFrame) -> pd.Series:
    if "paper_method_id" in summary:
        keys = summary["paper_method_id"].fillna("").astype(str)
    elif "method_id" in summary:
        keys = summary["method_id"].fillna("").astype(str)
    else:
        keys = pd.Series([""] * len(summary), index=summary.index)
    if "method_id" in summary:
        fallback = summary["method_id"].fillna("").astype(str)
        keys = keys.mask(keys == "", fallback)
    return keys.replace({"mars_official": "mars"})


def paper_comparison_frame(
    local: pd.DataFrame,
    paper: pd.DataFrame,
    *,
    suite: str,
    task_ids: set[str] | None = None,
) -> pd.DataFrame:
    if local.empty or paper.empty or "accuracy" not in local:
        return pd.DataFrame()
    local_slice = local.copy()
    if "suite" in local_slice:
        local_slice = local_slice[local_slice["suite"] == suite]
    if task_ids is not None:
        local_slice = local_slice[local_slice["task_id"].isin(task_ids)]
    if local_slice.empty:
        return pd.DataFrame()
    local_slice["method_key"] = _method_key(local_slice)
    local_slice["source"] = "local"
    local_slice["series"] = "local:" + local_slice["method_key"].astype(str)

    paper_slice = paper[paper["suite"] == suite].copy()
    if task_ids is not None:
        paper_slice = paper_slice[paper_slice["task_id"].isin(task_ids)]
    paper_slice = paper_slice[
        paper_slice["method_key"].isin(set(local_slice["method_key"]))
    ].copy()
    paper_slice["series"] = "paper:" + paper_slice["method_key"].astype(str)
    columns = ["task_id", "method_key", "series", "source", "accuracy"]
    return pd.concat(
        [local_slice[columns], paper_slice[columns]],
        ignore_index=True,
    )


def write_paper_comparison_csv(
    figures: Path, local: pd.DataFrame, paper: pd.DataFrame
) -> pd.DataFrame:
    if local.empty or paper.empty or "accuracy" not in local:
        comparison = pd.DataFrame()
    else:
        local_rows = local.copy()
        local_rows["method_key"] = _method_key(local_rows)
        local_rows = local_rows[
            ["suite", "task_id", "method_key", "accuracy", "num_samples"]
        ].rename(
            columns={
                "accuracy": "local_accuracy",
                "num_samples": "local_num_samples",
            }
        )
        paper_rows = paper[["suite", "task_id", "method_key", "accuracy"]].rename(
            columns={"accuracy": "paper_accuracy"}
        )
        comparison = local_rows.merge(
            paper_rows, on=["suite", "task_id", "method_key"], how="inner"
        )
        comparison["delta_local_minus_paper"] = (
            comparison["local_accuracy"] - comparison["paper_accuracy"]
        )
    figures.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(figures / "paper_local_comparison.csv", index=False)
    return comparison


def _save_placeholder(path: Path, title: str, message: str, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _save_bar(
    df: pd.DataFrame,
    path: Path,
    title: str,
    group_col: str = "method",
    dpi: int = 200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        _save_placeholder(path, title, "No data available for this figure.", dpi=dpi)
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
        _save_placeholder(path, title, "No data available for this figure.", dpi=dpi)
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


def _save_comparison_bar(
    df: pd.DataFrame,
    path: Path,
    title: str,
    dpi: int = 200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        _save_placeholder(path, title, "No paper/local overlap available.", dpi=dpi)
        return
    pivot = df.pivot_table(
        index="task_id", columns="series", values="accuracy", aggfunc="mean"
    )
    columns = sorted(pivot.columns)
    pivot = pivot[columns]
    ax = pivot.plot(kind="bar", figsize=(max(10, len(pivot) * 0.9), 5.5))
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _save_delta_bar(
    comparison: pd.DataFrame,
    path: Path,
    title: str,
    task_ids: set[str] | None = None,
    dpi: int = 200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if comparison.empty:
        _save_placeholder(path, title, "No paper/local overlap available.", dpi=dpi)
        return
    df = comparison.copy()
    if task_ids is not None:
        df = df[df["task_id"].isin(task_ids)]
    if df.empty:
        _save_placeholder(path, title, "No paper/local overlap available.", dpi=dpi)
        return
    pivot = df.pivot_table(
        index="task_id",
        columns="method_key",
        values="delta_local_minus_paper",
        aggfunc="mean",
    )
    ax = pivot.plot(kind="bar", figsize=(max(10, len(pivot) * 0.9), 5.5))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Local - Paper Accuracy")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _save_convergence(run_dir: Path, path: Path, dpi: int = 200) -> None:
    curve_files = list((run_dir / "convergence").glob("*_curves.csv"))
    frames = [pd.read_csv(file) for file in curve_files if file.stat().st_size > 0]
    if not frames:
        _save_placeholder(path, "Convergence Curves", "No convergence data available.", dpi=dpi)
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


def write_report(
    run_dir: Path,
    summary: pd.DataFrame,
    report_path: Path,
    comparison: pd.DataFrame,
) -> None:
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
        summary.get("exactness_level", pd.Series(dtype=str))
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
        f"- paper_local_overlap_rows: {len(comparison)}",
        "",
        "## Reproduction Categories",
        "",
    ]
    for name, count in exactness.items():
        lines.append(f"- {name or 'unspecified'}: {count}")
    lines.extend(
        [
            "",
            "## Paper Comparison",
            "",
            "- `paper_local_comparison.csv` contains overlapping local and paper accuracy rows.",
            "- `paper_vs_local_*` figures compare local reproduction results against stored paper table values.",
            "- `local_minus_paper_*` figures show local accuracy minus paper accuracy.",
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
    summary = augment_with_completed_method_metrics(summary, run_dir)
    task_groups = task_group_lookup()
    if task_groups and "task_id" in summary:
        summary = summary.copy()
        summary["task_group"] = summary["task_id"].map(task_groups).fillna("")
    paper = load_paper_results()
    if task_groups and not paper.empty:
        paper = paper.copy()
        paper["task_group"] = paper["task_id"].map(task_groups).fillna("")
    figures = run_dir / args.out_dir_name
    figures.mkdir(parents=True, exist_ok=True)
    comparison = write_paper_comparison_csv(figures, summary, paper)

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
    general_tasks = set(general["task_id"]) if "task_id" in general else set()
    domain_tasks = set(domain["task_id"]) if "task_id" in domain else set()
    _save_comparison_bar(
        paper_comparison_frame(main, paper, suite="main", task_ids=general_tasks),
        figures / f"paper_vs_local_general.{suffix}",
        "Paper vs Local: General Tasks",
        dpi=args.dpi,
    )
    _save_comparison_bar(
        paper_comparison_frame(main, paper, suite="main", task_ids=domain_tasks),
        figures / f"paper_vs_local_domain.{suffix}",
        "Paper vs Local: Domain Tasks",
        dpi=args.dpi,
    )
    _save_delta_bar(
        comparison,
        figures / f"local_minus_paper_general.{suffix}",
        "Local Minus Paper: General Tasks",
        task_ids=general_tasks,
        dpi=args.dpi,
    )
    _save_delta_bar(
        comparison,
        figures / f"local_minus_paper_domain.{suffix}",
        "Local Minus Paper: Domain Tasks",
        task_ids=domain_tasks,
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
    write_report(run_dir, summary, report_path, comparison)
    print(f"Visualization complete: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
