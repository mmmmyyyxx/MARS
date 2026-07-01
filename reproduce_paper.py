from __future__ import annotations

import argparse
from copy import deepcopy
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
from mars_core.reporting import write_full_reproduction_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full paper-level MARS reproduction presets."
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=["smoke", "mars_full", "paper_full"],
        help="Preset reproduction matrix to run.",
    )
    parser.add_argument(
        "--suite",
        choices=["main", "ablation", "efficiency", "convergence", "transfer", "all"],
        help="Override the preset suite selection.",
    )
    parser.add_argument("--model")
    parser.add_argument("--source-model")
    parser.add_argument("--target-models")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--early-stop-delta", type=float, default=0.01)
    parser.add_argument("--max-critic-revisions", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument(
        "--eval-protocol", choices=["paper_mode", "strict_mode"], default="paper_mode"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--results-root", default="results_full")
    parser.add_argument("--tasks", help="Override preset tasks with comma-separated ids.")
    parser.add_argument(
        "--methods", help="Override preset main-suite methods with comma-separated ids."
    )
    parser.add_argument("--cache-enabled", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-compatible-cache", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    if value.strip().lower() == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv(value: list[str] | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return None if value == "all" else value
    return ",".join(value)


def build_run_args(args: argparse.Namespace, matrix: dict[str, Any]) -> argparse.Namespace:
    preset = deepcopy(matrix["presets"][args.preset])
    cache_enabled = bool(preset.get("cache_enabled", False))
    if args.cache_enabled:
        cache_enabled = True
    if args.no_cache:
        cache_enabled = False

    return argparse.Namespace(
        preset=args.preset,
        suite=args.suite or preset.get("suite", "main"),
        methods=args.methods or _csv(preset.get("methods")),
        tasks=args.tasks or _csv(preset.get("tasks")),
        model=args.model or "deepseek-chat",
        source_model=args.source_model,
        target_models=args.target_models or "deepseek-r1,gpt-3.5,gpt-4,gpt-4o",
        temperature=args.temperature if args.temperature is not None else 0.6,
        max_samples=args.max_samples
        if args.max_samples is not None
        else preset.get("max_samples"),
        max_iterations=args.max_iterations
        if args.max_iterations is not None
        else int(preset.get("max_iterations", 10)),
        early_stop_delta=args.early_stop_delta,
        max_critic_revisions=args.max_critic_revisions,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        eval_protocol=args.eval_protocol,
        split_seed=args.split_seed,
        results_root=args.results_root,
        cache_enabled=cache_enabled,
        resume=args.resume,
        reuse_compatible_cache=args.reuse_compatible_cache,
        force_rerun=args.force_rerun,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )


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


def override_suite_methods(
    suites: dict[str, Any], method_ids: list[str] | None
) -> dict[str, Any]:
    if not method_ids:
        return suites
    updated_suites = dict(suites)
    if isinstance(updated_suites.get("main"), dict):
        updated_config = dict(updated_suites["main"])
        updated_config["methods"] = method_ids
        updated_suites["main"] = updated_config
    return updated_suites


def _run_dir_matches_dry_run(run_dir: Path, dry_run: bool) -> bool:
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return False
    try:
        config = load_yaml(config_path)
    except Exception:
        return False
    return bool(config.get("dry_run", False)) == bool(dry_run)


def make_run_dir(results_root: str, resume: bool, dry_run: bool = False) -> Path:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        runs = sorted(path for path in root.glob("run_*") if path.is_dir())
        compatible = [path for path in runs if _run_dir_matches_dry_run(path, dry_run)]
        if compatible:
            return compatible[-1]
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


def run_from_args(args: argparse.Namespace) -> int:
    tasks = load_task_specs("configs/tasks.yaml")
    methods = load_yaml("configs/methods.yaml")
    suites = load_yaml("configs/suites.yaml")
    model_config = load_yaml("configs/models.yaml")
    paper_results = load_yaml("configs/paper_results.yaml")
    model_config.setdefault("default", {})["request_timeout"] = args.request_timeout
    model_config.setdefault("default", {})["concurrency"] = args.concurrency
    suites = override_suite_tasks(suites, selected_ids(args.tasks))
    suites = override_suite_methods(suites, selected_ids(args.methods))

    run_dir = make_run_dir(args.results_root, args.resume, args.dry_run)
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
        reuse_compatible_cache=args.reuse_compatible_cache,
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
                    selected_methods=None,
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
                    target_models=selected_ids(args.target_models),
                )
            )

    summary_columns = sorted({key for row in all_rows for key in row.keys()})
    write_csv(run_dir / "summary.csv", all_rows, summary_columns)
    write_json(run_dir / "summary.json", all_rows)
    write_full_reproduction_outputs(
        run_dir=run_dir,
        args=args,
        summary_rows=all_rows,
        tasks=tasks,
        methods=methods,
        suites=suites,
        paper_results=paper_results,
    )
    print(f"Full reproduction run complete: {run_dir}")
    print(f"Summary: {run_dir / 'summary.csv'}")
    print(f"Report: {run_dir / 'final_report.md'}")
    return 0


def main() -> int:
    matrix = load_yaml("configs/reproduction_matrix.yaml")
    return run_from_args(build_run_args(parse_args(), matrix))


if __name__ == "__main__":
    raise SystemExit(main())
