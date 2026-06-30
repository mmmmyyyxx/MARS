from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .api_client import ApiCallError, LLMClient
from .evaluator import (
    PREDICTION_FIELDS,
    answer_instruction,
    compute_accuracy,
    compute_final_metrics_from_predictions,
    diagnostics_markdown,
    prediction_row,
    truthy,
)
from .logging_utils import write_csv, write_json, write_jsonl, write_text, write_yaml
from .prompt_loader import PromptLoader, TaskPrompts


@dataclass
class TaskSpec:
    task_id: str
    group: str
    paper_table: str
    dataset_path: str
    test_path: str
    question_type: str
    answer_format: str
    metric: str
    user_prompt_key: str
    planner_prompt_key: str
    few_shot_key: str
    paper_display_name: str
    train_path: str | None = None
    val_path: str | None = None


@dataclass
class RunSettings:
    model: str
    temperature: float
    max_samples: int | None
    max_iterations: int
    early_stop_delta: float
    max_critic_revisions: int
    eval_protocol: str
    split_seed: int
    output_dir: Path
    cache_enabled: bool
    resume: bool
    force_rerun: bool
    skip_existing: bool
    reuse_compatible_cache: bool
    dry_run: bool


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_task_specs(path: str | Path = "configs/tasks.yaml") -> dict[str, TaskSpec]:
    data = load_yaml(path)
    return {
        task_id: TaskSpec(task_id=task_id, **config) for task_id, config in data.items()
    }


def load_dataset(
    path: str | Path, max_samples: int | None = None
) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if max_samples is not None and len(rows) >= max_samples:
                break
            rows.append(
                {
                    "sample_id": index,
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                }
            )
    return rows


def hash_rows(rows: list[dict[str, Any]]) -> str:
    serialized = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def split_dataset(
    rows: list[dict[str, Any]], protocol: str, seed: int
) -> dict[str, list[dict[str, Any]]]:
    if protocol == "paper_mode" or len(rows) < 3:
        return {"opt": rows, "val": rows, "test": rows}
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    opt_end = max(1, int(n * 0.3))
    val_end = max(opt_end + 1, int(n * 0.5))
    return {
        "opt": shuffled[:opt_end],
        "val": shuffled[opt_end:val_end],
        "test": shuffled[val_end:] or shuffled[-1:],
    }


def split_info(
    splits: dict[str, list[dict[str, Any]]], protocol: str, seed: int
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "split_seed": seed,
        "splits": {
            name: {"num_samples": len(rows), "hash": hash_rows(rows)}
            for name, rows in splits.items()
        },
    }


def build_question_prompt(base_prompt: str, question: str, instruction: str) -> str:
    return f"{base_prompt.strip()}\n\nQuestion:\n{question}\n\n{instruction.strip()}"


def evaluate_prompt(
    *,
    client: LLMClient,
    task: TaskSpec,
    rows: list[dict[str, Any]],
    prompt: str,
    method: str,
    iteration: int,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    predictions = []
    instruction = answer_instruction(task.answer_format)
    for row in rows:
        user_prompt = build_question_prompt(prompt, row["question"], instruction)
        error_type = ""
        raw_output = ""
        try:
            raw_output = client.complete_text(
                system="You are a careful evaluation assistant.",
                user=user_prompt,
                method=method,
                task_id=task.task_id,
                iteration=iteration,
                question=row["question"],
                agent_name="Target",
                sample_id=row.get("sample_id", ""),
            )
        except ApiCallError as exc:
            error_type = exc.error_type
            raw_output = ""
        predictions.append(
            prediction_row(
                sample_id=row["sample_id"],
                question=row["question"],
                gold=row["answer"],
                raw_output=raw_output,
                answer_format=task.answer_format,
                method=method,
                task_id=task.task_id,
                iteration=iteration,
                error_type=error_type,
            )
        )
    if out_dir is not None:
        write_csv(out_dir / "predictions.csv", predictions, PREDICTION_FIELDS)
    return predictions


def write_method_outputs(
    *,
    out_dir: Path,
    task: TaskSpec,
    method: str,
    method_config: dict[str, Any],
    predictions: list[dict[str, Any]],
    history: list[dict[str, Any]],
    best_prompt: str,
    final_prompt: str,
    raw_logs: str = "",
    run_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_final_metrics_from_predictions(predictions)
    metrics["num_iterations"] = len(history)
    write_yaml(out_dir / "method_config.yaml", method_config)
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "predictions.csv", predictions, PREDICTION_FIELDS)
    write_csv(
        out_dir / "prompt_accuracy_history.csv",
        history,
        ["iteration", "prompt", "accuracy", "num_samples", "num_correct", "num_failed"],
    )
    write_text(out_dir / "best_prompt.txt", best_prompt)
    write_text(out_dir / "final_prompt.txt", final_prompt)
    write_text(
        out_dir / "diagnostics.md",
        diagnostics_markdown(task.task_id, method, predictions),
    )
    write_text(out_dir / "raw_logs.txt", raw_logs)
    if run_state is not None:
        write_json(out_dir / "run_state.json", run_state)
    return metrics


def run_direct_method(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    test_rows: list[dict[str, Any]],
    method: str,
    method_config: dict[str, Any],
    out_dir: Path,
    few_shot_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if method == "origin":
        prompt = prompts.origin
    elif method == "cot_zs":
        prompt = prompts.cot_zero_shot
    elif method == "cot_fs":
        num_shots = int(method_config.get("num_shots", 3))
        examples = (few_shot_rows or prompts.few_shot_examples)[:num_shots]
        example_text = "\n\n".join(
            f"Example question:\n{item['question']}\nExample answer:\nFinal answer: {item.get('answer', item.get('gold', ''))}"
            for item in examples
        )
        write_jsonl(out_dir / "used_few_shot_examples.jsonl", examples)
        prompt = f"{prompts.cot_few_shot}\n\n{example_text}"
    else:
        raise ValueError(f"Unknown direct method: {method}")

    start = time.time()
    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=prompt,
        method=method,
        iteration=1,
    )
    accuracy = compute_accuracy(predictions)
    history = [
        {
            "iteration": 1,
            "prompt": prompt,
            "accuracy": accuracy,
            "num_samples": len(predictions),
            "num_correct": sum(truthy(row["correct"]) for row in predictions),
            "num_failed": len(predictions)
            - sum(truthy(row["correct"]) for row in predictions),
        }
    ]
    metrics = write_method_outputs(
        out_dir=out_dir,
        task=task,
        method=method,
        method_config=method_config,
        predictions=predictions,
        history=history,
        best_prompt=prompt,
        final_prompt=prompt,
        raw_logs=f"runtime_seconds: {time.time() - start:.4f}\n",
    )
    metrics["runtime_seconds"] = time.time() - start
    return metrics


def evaluate_fixed_prompt_method(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompt: str,
    test_rows: list[dict[str, Any]],
    method: str,
    method_config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    start = time.time()
    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=prompt,
        method=method,
        iteration=1,
    )
    accuracy = compute_accuracy(predictions)
    history = [
        {
            "iteration": 1,
            "prompt": prompt,
            "accuracy": accuracy,
            "num_samples": len(predictions),
            "num_correct": sum(truthy(row["correct"]) for row in predictions),
            "num_failed": len(predictions)
            - sum(truthy(row["correct"]) for row in predictions),
        }
    ]
    metrics = write_method_outputs(
        out_dir=out_dir,
        task=task,
        method=method,
        method_config=method_config,
        predictions=predictions,
        history=history,
        best_prompt=prompt,
        final_prompt=prompt,
        raw_logs="transfer_target_evaluation_only: true\n",
    )
    metrics["runtime_seconds"] = time.time() - start
    metrics["num_iterations"] = 0
    return metrics


def generate_prompt_candidate(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    method: str,
    iteration: int,
    context: str,
) -> str:
    user = (
        f"Task: {task.paper_display_name}\n"
        f"Task description:\n{prompts.user_proxy}\n\n"
        f"Current context:\n{context}\n\n"
        "Generate one instruction prompt for this task. Output only the prompt."
    )
    return client.complete_text(
        system="You are a prompt engineering researcher.",
        user=user,
        method=method,
        task_id=task.task_id,
        iteration=iteration,
        max_tokens=500,
        agent_name="Student",
    ).strip()


def run_candidate_method(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    opt_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    method: str,
    method_config: dict[str, Any],
    out_dir: Path,
    max_iterations: int,
) -> dict[str, Any]:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    history = []
    best_prompt = prompts.origin
    best_accuracy = -1.0

    if method == "ape":
        total = int(method_config.get("num_candidates", 20))
        iterations = min(total, max_iterations if max_iterations else total)
        context_template = (
            "Generate candidate instruction {iteration} for APE-style validation."
        )
    elif method == "pe2":
        total = int(method_config.get("num_candidates", 10))
        iterations = min(total, max_iterations if max_iterations else total)
        context_template = (
            "Use a prompt-engineer meta-prompt to refine candidate {iteration}."
        )
    elif method == "opro":
        iterations = min(int(method_config.get("num_iterations", 10)), max_iterations)
        context_template = (
            "Use previous prompt-score history to propose candidate {iteration}."
        )
    elif method == "protegi":
        iterations = min(int(method_config.get("num_iterations", 10)), max_iterations)
        context_template = "Generate textual-gradient-inspired prompt edit {iteration}."
    else:
        raise ValueError(f"Unknown candidate method: {method}")

    for iteration in range(1, max(iterations, 1) + 1):
        context = context_template.format(iteration=iteration)
        if history:
            context += "\nHistory:\n" + "\n".join(
                f"- score={item['accuracy']:.4f}: {item['prompt'][:200]}"
                for item in history[-5:]
            )
        candidate = generate_prompt_candidate(
            client=client,
            task=task,
            prompts=prompts,
            method=method,
            iteration=iteration,
            context=context,
        )
        if not candidate:
            candidate = prompts.origin
        eval_rows = val_rows or opt_rows or test_rows
        eval_predictions = evaluate_prompt(
            client=client,
            task=task,
            rows=eval_rows,
            prompt=candidate,
            method=method,
            iteration=iteration,
        )
        accuracy = compute_accuracy(eval_predictions)
        record = {
            "iteration": iteration,
            "prompt": candidate,
            "accuracy": accuracy,
            "num_samples": len(eval_predictions),
            "num_correct": sum(truthy(row["correct"]) for row in eval_predictions),
            "num_failed": len(eval_predictions)
            - sum(truthy(row["correct"]) for row in eval_predictions),
        }
        history.append(record)
        candidates.append(
            {"iteration": iteration, "prompt": candidate, "accuracy": accuracy}
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt = candidate

    final_predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=best_prompt,
        method=method,
        iteration=len(history) or 1,
    )

    write_jsonl(out_dir / "candidate_prompts.jsonl", candidates)
    write_csv(
        out_dir / "candidate_scores.csv",
        candidates,
        ["iteration", "prompt", "accuracy"],
    )
    if method == "protegi":
        write_jsonl(out_dir / "textual_gradients.jsonl", candidates)
        write_jsonl(out_dir / "beam_candidates.jsonl", candidates)
    if method == "opro":
        write_jsonl(out_dir / "opro_history.jsonl", history)

    metrics = write_method_outputs(
        out_dir=out_dir,
        task=task,
        method=method,
        method_config=method_config,
        predictions=final_predictions,
        history=history,
        best_prompt=best_prompt,
        final_prompt=best_prompt,
        raw_logs=f"{method_config.get('exactness_level', 'best_effort_reimplementation')}\nruntime_seconds: {time.time() - start:.4f}\n",
    )
    metrics["runtime_seconds"] = time.time() - start
    return metrics


def method_table_row(
    *,
    task: TaskSpec,
    method: str,
    method_config: dict[str, Any],
    metrics: dict[str, Any],
    client: LLMClient,
    runtime_seconds: float,
) -> dict[str, Any]:
    pricing = method_config.get("pricing", {})
    tokens_prompt = client.stats.tokens_prompt
    tokens_completion = client.stats.tokens_completion
    cost_estimate = tokens_prompt / 1000 * float(
        pricing.get("prompt_per_1k", 0) or 0
    ) + tokens_completion / 1000 * float(pricing.get("completion_per_1k", 0) or 0)
    return {
        "task_id": task.task_id,
        "display_name": task.paper_display_name,
        "method": method_config.get("display_name", method),
        "method_id": method,
        "accuracy": metrics.get("accuracy", 0.0),
        "num_samples": metrics.get("num_samples", 0),
        "num_correct": metrics.get("num_correct", 0),
        "num_failed": metrics.get("num_failed", 0),
        "api_errors": metrics.get("api_errors", 0) + client.stats.api_errors,
        "parse_errors": metrics.get("parse_errors", 0),
        "runtime_seconds": runtime_seconds,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "tokens_total": client.stats.tokens_total,
        "cost_estimate": cost_estimate,
        "api_calls": client.stats.api_calls,
        "cache_hits": client.stats.cache_hits,
        "num_iterations": metrics.get("num_iterations", ""),
        "exactness_level": method_config.get(
            "exactness_level", method_config.get("exactness", "")
        ),
        "exactness_note": method_config.get("exactness_note", ""),
    }
