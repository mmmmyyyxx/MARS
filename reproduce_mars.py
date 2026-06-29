import argparse
import csv
from dataclasses import asdict
from copy import copy
import os
import random
import time
from typing import Any, Dict, List, Optional

from mars_utils.config_loader import load_config
from mars_utils.prompt_manager import PromptManager
from mars_utils.result_writer import (
    make_run_dir,
    write_prompt_history,
    write_report,
    write_summary,
    write_text,
    write_yaml,
)
from mars_utils.run_logger import append_error
from mars_utils.task_registry import dataset_exists, load_tasks, resolve_tasks


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-command MARS reproduction tasks.")
    parser.add_argument("--tasks", default="all", help="Task id, comma list, or group: all, bbh, mmlu, domain.")
    parser.add_argument("--config", default="configs/mars_default.yaml")
    parser.add_argument("--task-config", default="configs/mars_tasks.yaml")
    parser.add_argument("--api-key-env")
    parser.add_argument("--base-url-env")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--early-stop-delta", type=float)
    parser.add_argument("--max-critic-revisions", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runnable-only", action="store_true", help="Run only registered tasks whose dataset and prompts are available.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_dataset_samples(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", newline="") as file:
            row_count = sum(1 for _ in csv.reader(file))
        # TargetAgent preserves the legacy behavior of skipping the first data row.
        return max(row_count - 2, 0)
    except OSError:
        return 0


def failed_summary_row(task, status: str, reason: str, runtime: float, num_samples: int = 0) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "group": task.group,
        "dataset_path": task.dataset_path,
        "question_type": task.question_type,
        "status": status,
        "num_samples": num_samples,
        "num_success": 0,
        "num_failed": num_samples,
        "best_accuracy": "",
        "final_accuracy": "",
        "best_iteration": "",
        "stopped_reason": reason,
        "total_runtime_seconds": round(runtime, 4),
    }


def success_summary_row(task, run_result: Dict[str, Any], num_samples: int) -> Dict[str, Any]:
    history = run_result.get("prompt_history", [])
    accuracies = [item[1] for item in history]
    final_accuracy = accuracies[-1] if accuracies else 0.0
    best_accuracy = max(accuracies) if accuracies else 0.0
    best_iteration = accuracies.index(best_accuracy) + 1 if accuracies else ""
    num_success = round(final_accuracy * num_samples)
    return {
        "task_id": task.task_id,
        "group": task.group,
        "dataset_path": task.dataset_path,
        "question_type": task.question_type,
        "status": "success",
        "num_samples": num_samples,
        "num_success": num_success,
        "num_failed": max(num_samples - num_success, 0),
        "best_accuracy": best_accuracy,
        "final_accuracy": final_accuracy,
        "best_iteration": best_iteration,
        "stopped_reason": run_result.get("stopped_reason", "completed"),
        "total_runtime_seconds": round(run_result.get("runtime_seconds", 0.0), 4),
    }


def prepare_task_files(task_dir: str, task, config, user_prompt: Optional[str], planner_prompt: Optional[str]) -> None:
    write_yaml(os.path.join(task_dir, "config.yaml"), {
        "task": asdict(task),
        "model": config.model,
        "temperature": config.temperature,
        "max_iterations": config.max_iterations,
        "early_stop_delta": config.early_stop_delta,
        "concurrency": config.concurrency,
        "dry_run": config.dry_run,
    })
    if user_prompt is not None:
        write_text(os.path.join(task_dir, "user_prompt.txt"), user_prompt)
    if planner_prompt is not None:
        write_text(os.path.join(task_dir, "planner_prompt.txt"), planner_prompt)
    for filename in ("errors.jsonl", "raw_logs.txt", "final_prompt.txt"):
        path = os.path.join(task_dir, filename)
        if not os.path.exists(path):
            write_text(path, "")


def main() -> int:
    args = parse_args()
    overrides = {
        "api_key_env": args.api_key_env,
        "base_url_env": args.base_url_env,
        "model": args.model,
        "temperature": args.temperature,
        "max_iterations": args.max_iterations,
        "early_stop_delta": args.early_stop_delta,
        "max_critic_revisions": args.max_critic_revisions,
        "concurrency": args.concurrency,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }
    config = load_config(os.path.join(ROOT_DIR, args.config), overrides)
    config.concurrency = max(int(config.concurrency), 1)
    random.seed(config.seed)

    all_tasks = load_tasks(os.path.join(ROOT_DIR, args.task_config))
    prompt_manager = PromptManager(os.path.join(ROOT_DIR, "Prompt"))

    def is_runnable_task(task) -> bool:
        if not dataset_exists(task, ROOT_DIR):
            return False
        try:
            prompt_manager.get_user_prompt(task.user_prompt_key)
            prompt_manager.get_planner_prompt(task.planner_prompt_key)
            return True
        except KeyError:
            return False

    selected_tasks = resolve_tasks(args.tasks, all_tasks, is_runnable=is_runnable_task)
    if args.runnable_only:
        selected_tasks = [task for task in selected_tasks if is_runnable_task(task)]
    if not selected_tasks:
        print(f"No tasks matched selector: {args.tasks}")
        return 2

    run_dir = make_run_dir(os.path.join(ROOT_DIR, config.output_dir))
    config_to_write = asdict(config)
    config_to_write["api_key_value"] = "<from environment>" if config.api_key else None
    config_to_write["base_url_value"] = "<from environment>" if config.base_url else None
    write_yaml(os.path.join(run_dir, "config.yaml"), config_to_write)

    summary_rows: List[Dict[str, Any]] = []

    for task in selected_tasks:
        task_start = time.time()
        task_dir = os.path.join(run_dir, "tasks", task.task_id)
        os.makedirs(task_dir, exist_ok=True)
        dataset_path = os.path.join(ROOT_DIR, task.dataset_path)
        num_samples = count_dataset_samples(dataset_path)
        user_prompt = None
        planner_prompt = None

        if not dataset_exists(task, ROOT_DIR):
            append_error(task_dir, "missing_dataset", f"Dataset not found: {task.dataset_path}")
            prepare_task_files(task_dir, task, config, user_prompt, planner_prompt)
            summary_rows.append(failed_summary_row(task, "missing_dataset", "missing_dataset", time.time() - task_start))
            continue

        try:
            user_prompt = prompt_manager.get_user_prompt(task.user_prompt_key)
            planner_prompt = prompt_manager.get_planner_prompt(task.planner_prompt_key)
        except KeyError as exc:
            append_error(task_dir, "missing_prompt", str(exc), {
                "user_prompt_key": task.user_prompt_key,
                "planner_prompt_key": task.planner_prompt_key,
            })
            prepare_task_files(task_dir, task, config, user_prompt, planner_prompt)
            summary_rows.append(failed_summary_row(task, "failed", "missing_prompt", time.time() - task_start, num_samples))
            continue

        prepare_task_files(task_dir, task, config, user_prompt, planner_prompt)

        if not config.dry_run and not config.api_key:
            append_error(task_dir, "missing_api_key", f"Environment variable {config.api_key_env} is not set.")
            summary_rows.append(failed_summary_row(task, "failed", "missing_api_key", time.time() - task_start, num_samples))
            continue

        try:
            from main_MARS import run_mars_task
            task_config = copy(config)
            task_config.answer_format = task.answer_format

            run_result = run_mars_task(
                task_id=task.task_id,
                dataset_path=dataset_path,
                question_type=task.question_type,
                user_prompt=user_prompt,
                planner_prompt=planner_prompt,
                config=task_config,
                output_dir=task_dir,
            )
            history = run_result.get("prompt_history", [])
            stopped_reason = run_result.get("stopped_reason", "completed")
            write_prompt_history(os.path.join(task_dir, "prompt_accuracy_history.csv"), history, stopped_reason)
            final_prompt = history[-1][0] if history else ""
            write_text(os.path.join(task_dir, "final_prompt.txt"), final_prompt)
            summary_rows.append(success_summary_row(task, run_result, num_samples))
        except Exception as exc:
            append_error(task_dir, "task_failed", str(exc), {"exception_class": exc.__class__.__name__})
            write_text(os.path.join(task_dir, "final_prompt.txt"), "")
            summary_rows.append(failed_summary_row(task, "failed", "task_failed", time.time() - task_start, num_samples))
            continue

    write_summary(run_dir, summary_rows)
    write_report(run_dir, summary_rows, config.dry_run)
    print(f"MARS reproduction run complete: {run_dir}")
    print(f"Summary: {os.path.join(run_dir, 'summary.csv')}")
    print(f"Report: {os.path.join(run_dir, 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
