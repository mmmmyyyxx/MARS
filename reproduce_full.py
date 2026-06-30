from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.ablation_experiment import run_ablation_suite
from experiments.convergence_experiment import run_convergence_suite
from experiments.efficiency_experiment import run_efficiency_suite
from experiments.main_experiment import run_main_suite
from experiments.transfer_experiment import run_transfer_suite
from mars_core.logging_utils import (
    collect_environment,
    collect_git_info,
    timestamp,
    write_csv,
    write_json,
    write_text,
    write_yaml,
)
from mars_core.mars_runner import RunSettings, load_task_specs, load_yaml
from mars_core.prompt_loader import PromptLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full MARS reproduction suites.")
    parser.add_argument(
        "--suite",
        default="main",
        choices=["main", "ablation", "efficiency", "convergence", "transfer", "all"],
    )
    parser.add_argument(
        "--methods",
        help="Comma-separated method ids for suites that support filtering.",
    )
    parser.add_argument(
        "--tasks", help="Comma-separated task ids for smoke/debug runs."
    )
    parser.add_argument("--model", default="deepseek-v2.5-1210")
    parser.add_argument("--source-model")
    parser.add_argument("--target-models", default="deepseek-r1,gpt-3.5,gpt-4,gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=0.01)
    parser.add_argument("--max-critic-revisions", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument(
        "--eval-protocol", choices=["paper_mode", "strict_mode"], default="paper_mode"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--results-root", default="results_full")
    parser.add_argument("--cache-enabled", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_method_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def override_suite_tasks(
    suites: dict[str, Any], task_ids: list[str] | None
) -> dict[str, Any]:
    if not task_ids:
        return suites
    updated_suites = dict(suites)
    for suite_name, suite_config in suites.items():
        if isinstance(suite_config, dict) and "tasks" in suite_config:
            updated_config = dict(suite_config)
            updated_config["tasks"] = task_ids
            updated_suites[suite_name] = updated_config
    return updated_suites


def make_run_dir(results_root: str, resume: bool) -> Path:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        runs = sorted(path for path in root.glob("run_*") if path.is_dir())
        if runs:
            return runs[-1]
    base = root / f"run_{timestamp()}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    for dirname in [
        "tables",
        "figures",
        "methods",
        "tasks",
        "logs",
        "efficiency",
        "convergence",
    ]:
        (run_dir / dirname).mkdir(parents=True, exist_ok=True)
    return run_dir


def preflight_rows(tasks, methods, prompt_loader: PromptLoader) -> list[dict[str, Any]]:
    rows = []
    for task in tasks.values():
        prompt_status = prompt_loader.status(task.task_id)
        dataset_exists = Path(task.dataset_path).exists()
        missing = []
        if not dataset_exists:
            missing.append("missing_dataset")
        missing.extend(name for name, exists in prompt_status.items() if not exists)
        rows.append(
            {
                "task_id": task.task_id,
                "dataset_exists": dataset_exists,
                "origin_prompt_exists": prompt_status.get("origin_exists", False),
                "user_prompt_exists": prompt_status.get("user_proxy_exists", False),
                "planner_prompt_exists": prompt_status.get("planner_exists", False),
                "few_shot_exists": prompt_status.get("few_shot_exists", False),
                "answer_format": task.answer_format,
                "methods_supported": ",".join(methods.keys()),
                "runnable": not missing,
                "missing_reason": ",".join(missing),
            }
        )
    return rows


def write_preflight(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "task_id",
        "dataset_exists",
        "origin_prompt_exists",
        "user_prompt_exists",
        "planner_prompt_exists",
        "few_shot_exists",
        "answer_format",
        "methods_supported",
        "runnable",
        "missing_reason",
    ]
    write_csv(run_dir / "preflight_report.csv", rows, columns)
    lines = ["# Preflight Report", ""]
    for row in rows:
        mark = "OK" if row["runnable"] else "FAIL"
        lines.append(f"- {mark} `{row['task_id']}`: {row['missing_reason'] or 'ready'}")
    write_text(run_dir / "preflight_report.md", "\n".join(lines) + "\n")


def suite_order(suite: str, suites_config: dict[str, Any]) -> list[str]:
    if suite == "all":
        return suites_config["all"]["includes"]
    return [suite]


def main() -> int:
    args = parse_args()
    tasks = load_task_specs("configs/tasks.yaml")
    methods = load_yaml("configs/methods.yaml")
    suites = load_yaml("configs/suites.yaml")
    model_config = load_yaml("configs/models.yaml")
    model_config.setdefault("default", {})["request_timeout"] = args.request_timeout
    model_config.setdefault("default", {})["concurrency"] = args.concurrency
    suites = override_suite_tasks(suites, selected_method_ids(args.tasks))

    run_dir = make_run_dir(args.results_root, args.resume)
    settings = RunSettings(
        model=args.model,
        temperature=args.temperature,
        max_samples=args.max_samples,
        max_iterations=args.max_iterations,
        early_stop_delta=args.early_stop_delta,
        max_critic_revisions=args.max_critic_revisions,
        eval_protocol=args.eval_protocol,
        split_seed=args.split_seed,
        output_dir=run_dir,
        cache_enabled=args.cache_enabled,
        resume=args.resume,
        force_rerun=args.force_rerun,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    prompt_loader = PromptLoader()
    run_config = vars(args)
    run_config["run_dir"] = str(run_dir)
    write_yaml(run_dir / "run_config.yaml", run_config)
    write_json(run_dir / "environment.json", collect_environment())
    write_json(run_dir / "git_info.json", collect_git_info())

    preflight = preflight_rows(tasks, methods, prompt_loader)
    write_preflight(run_dir, preflight)
    if any(not row["runnable"] for row in preflight):
        print(f"Preflight failed. See {run_dir / 'preflight_report.md'}")
        return 2

    all_rows = []
    method_filter = selected_method_ids(args.methods)
    for suite_name in suite_order(args.suite, suites):
        if suite_name == "main":
            all_rows.extend(
                run_main_suite(
                    tasks=tasks,
                    methods=methods,
                    suite_config=suites["main"],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                    selected_methods=method_filter,
                )
            )
        elif suite_name == "ablation":
            all_rows.extend(
                run_ablation_suite(
                    tasks=tasks,
                    methods=methods,
                    suite_config=suites["ablation"],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                )
            )
        elif suite_name == "efficiency":
            all_rows.extend(
                run_efficiency_suite(
                    tasks=tasks,
                    methods=methods,
                    suite_config=suites["efficiency"],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                )
            )
        elif suite_name == "convergence":
            all_rows.extend(
                run_convergence_suite(
                    tasks=tasks,
                    methods=methods,
                    suite_config=suites["convergence"],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                )
            )
        elif suite_name == "transfer":
            all_rows.extend(
                run_transfer_suite(
                    tasks=tasks,
                    methods=methods,
                    suite_config=suites["transfer"],
                    settings=settings,
                    model_config=model_config,
                    run_dir=run_dir,
                    prompt_loader=prompt_loader,
                    source_model=args.source_model,
                    target_models=selected_method_ids(args.target_models),
                )
            )

    summary_columns = sorted({key for row in all_rows for key in row.keys()})
    write_csv(run_dir / "summary.csv", all_rows, summary_columns)
    write_json(run_dir / "summary.json", all_rows)
    write_csv(
        run_dir / "paper_comparison.csv",
        [],
        ["table", "task_id", "method", "local_accuracy", "paper_accuracy", "delta"],
    )
    report = [
        "# Full MARS Reproduction Report",
        "",
        f"- suite: {args.suite}",
        f"- model: {args.model}",
        f"- temperature: {args.temperature}",
        f"- max_iterations: {args.max_iterations}",
        f"- eval_protocol: {args.eval_protocol}",
        f"- rows: {len(all_rows)}",
        "",
        "## Reproduction Status",
        "",
        "Origin and MARS are local implementations based on the checked-in repository. "
        "APE, ProTeGi, OPRO, PE2 and unavailable exact templates are marked as best-effort reimplementation in method_config.yaml.",
        "",
        "## API And Parse Errors",
        "",
        f"- api_errors: {sum(int(row.get('api_errors', 0) or 0) for row in all_rows)}",
        f"- parse_errors: {sum(int(row.get('parse_errors', 0) or 0) for row in all_rows)}",
    ]
    write_text(run_dir / "final_report.md", "\n".join(report) + "\n")
    print(f"Full reproduction run complete: {run_dir}")
    print(f"Summary: {run_dir / 'summary.csv'}")
    print(f"Report: {run_dir / 'final_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
