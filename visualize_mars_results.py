from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from mars_utils.visualization import (
    GROUP_TASKS,
    attach_paper_results,
    find_latest_run,
    load_histories,
    load_summary,
    plot_best_iteration_distribution,
    plot_convergence_curves,
    plot_group_average_vs_paper,
    plot_status_overview,
    plot_summary_overview,
    plot_task_accuracy_vs_paper,
    plot_task_delta_vs_paper,
    write_paper_comparison_table,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MARS reproduction results.")
    parser.add_argument(
        "--run-dir", type=str, help="Specific run directory under results_mars."
    )
    parser.add_argument("--results-root", type=str, default="results_mars")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically select the latest run_* directory.",
    )
    parser.add_argument("--out-dir-name", type=str, default="visualizations")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--format", type=str, default="png", choices=["png", "pdf", "svg"]
    )
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if run_dir.name == "latest":
            latest = find_latest_run(Path(args.results_root).resolve())
            if latest is None:
                raise FileNotFoundError(
                    f"No run_* directory found under {args.results_root}"
                )
            return latest
        return run_dir.resolve()
    results_root = Path(args.results_root).resolve()
    if args.latest or not args.run_dir:
        latest = find_latest_run(results_root)
        if latest is None:
            raise FileNotFoundError(f"No run_* directory found under {results_root}")
        return latest
    raise FileNotFoundError("No run directory specified.")


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    summary_df = load_summary(run_dir)
    histories = load_histories(run_dir)
    merged_df = attach_paper_results(summary_df)

    out_dir = run_dir / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    plot_summary_overview(
        merged_df,
        out_dir / f"summary_overview.{args.format}",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"summary_overview.{args.format}")
    plot_task_accuracy_vs_paper(
        merged_df,
        out_dir / f"task_accuracy_vs_paper.{args.format}",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"task_accuracy_vs_paper.{args.format}")
    plot_task_delta_vs_paper(
        merged_df,
        out_dir / f"task_delta_vs_paper.{args.format}",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"task_delta_vs_paper.{args.format}")
    plot_group_average_vs_paper(
        merged_df,
        out_dir / f"group_average_vs_paper.{args.format}",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"group_average_vs_paper.{args.format}")
    plot_status_overview(
        merged_df,
        out_dir / f"status_overview.{args.format}",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"status_overview.{args.format}")
    plot_best_iteration_distribution(
        merged_df,
        histories,
        out_dir / f"best_iteration_distribution.{args.format}",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"best_iteration_distribution.{args.format}")

    all_tasks = merged_df["task_id"].tolist()
    bbh_tasks = GROUP_TASKS["BBH"]
    mmlu_tasks = GROUP_TASKS["MMLU"]
    domain_tasks = GROUP_TASKS["Domain"]
    plot_convergence_curves(
        histories,
        all_tasks,
        out_dir / f"convergence_curves_all.{args.format}",
        "MARS optimization trajectories across runnable tasks",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"convergence_curves_all.{args.format}")
    plot_convergence_curves(
        histories,
        bbh_tasks,
        out_dir / f"convergence_curves_bbh.{args.format}",
        "BBH convergence curves",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"convergence_curves_bbh.{args.format}")
    plot_convergence_curves(
        histories,
        mmlu_tasks,
        out_dir / f"convergence_curves_mmlu.{args.format}",
        "MMLU convergence curves",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"convergence_curves_mmlu.{args.format}")
    plot_convergence_curves(
        histories,
        domain_tasks,
        out_dir / f"convergence_curves_domain.{args.format}",
        "Domain convergence curves",
        dpi=args.dpi,
        fmt=args.format,
    )
    generated.append(out_dir / f"convergence_curves_domain.{args.format}")

    write_paper_comparison_table(merged_df, out_dir / "paper_comparison_table.csv")
    write_report(run_dir, merged_df, histories, out_dir, generated)

    print("Visualization complete.")
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Generated figures: {len([path for path in generated if path.exists()])}")
    print(f"Report: {out_dir / 'visualization_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
