from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
    is_valid_prediction,
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
    concurrency: int = 1
    initial_prompt_source: str = "legacy_student_generated"
    legacy_skip_first_data_row: bool = False
    max_answer_retries: int = 1
    planner_strict_mode: bool = False
    legacy_target_prompt_mode: bool = True
    target_call_count_limit: int = 10


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_task_specs(path: str | Path = "configs/tasks.yaml") -> dict[str, TaskSpec]:
    data = load_yaml(path)
    return {
        task_id: TaskSpec(task_id=task_id, **config) for task_id, config in data.items()
    }


def load_dataset(
    path: str | Path,
    max_samples: int | None = None,
    *,
    skip_first_data_row: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if skip_first_data_row and index == 0:
                continue
            if max_samples is not None and len(rows) >= max_samples:
                break
            rows.append(
                {
                    "sample_id": len(rows),
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


LEGACY_TARGET_SYSTEM = "You are a helpful assistant"
LEGACY_CHOICE_INSTRUCTION = (
    "Please don't output the process of doing the question, only the content of "
    "the answer. The answer should be a parenthesis containing the capital letter "
    "of the chosen answer. Please do not add any other spaces or symbols."
)
LEGACY_SHORT_ANSWER_INSTRUCTION = (
    "Please don't output the process of doing the question, only the content of "
    "the answer."
)


def build_legacy_target_prompt(task: TaskSpec, base_prompt: str, question: str) -> str:
    instruction = (
        LEGACY_CHOICE_INSTRUCTION
        if (task.answer_format or "").lower() == "option_letter"
        else LEGACY_SHORT_ANSWER_INSTRUCTION
    )
    return f"{base_prompt}\nQuestion: {question}\n {instruction}"


LEGACY_INITIAL_PROMPT_SEED = "Think step by step and solve the question."
LEGACY_INITIAL_PROMPT_SYSTEM = (
    "You are a prompt generator, please proceed to iterate over the existing "
    "prompts as required.\n"
    "Note that you should only output the new prompt you generated."
)


def generate_initial_prompt_legacy(
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    seed_prompt: str = LEGACY_INITIAL_PROMPT_SEED,
) -> tuple[str, dict[str, Any]]:
    user = (
        f"Here is the task definition:\n{prompts.user_proxy}\n\n"
        "Please generate a more appropriate prompt based on the following prompt "
        f"and task definition:\n{seed_prompt}"
    )
    generated = client.complete_text(
        system=LEGACY_INITIAL_PROMPT_SYSTEM,
        user=user,
        method="initial_prompt",
        task_id=task.task_id,
        iteration=0,
        max_tokens=700,
        agent_name="Student",
    ).strip()
    initial_prompt = generated or seed_prompt
    metadata = {
        "label": "p0",
        "source": "legacy_student_generated",
        "seed_prompt": seed_prompt,
        "system_message": LEGACY_INITIAL_PROMPT_SYSTEM,
        "user_message": user,
        "task_id": task.task_id,
        "prompt_hash": hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest(),
    }
    return initial_prompt, metadata


def manual_origin_initial_prompt(
    task: TaskSpec, prompts: TaskPrompts
) -> tuple[str, dict[str, Any]]:
    initial_prompt = prompts.origin
    metadata = {
        "label": "p0",
        "source": "manual_origin",
        "task_id": task.task_id,
        "prompt_hash": hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest(),
    }
    return initial_prompt, metadata


def write_initial_prompt_artifacts(
    *,
    task_dir: Path,
    method_dir: Path,
    initial_prompt: str,
    metadata: dict[str, Any],
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    write_text(task_dir / "initial_prompt.txt", initial_prompt)
    write_json(task_dir / "initial_prompt_metadata.json", metadata)
    write_text(method_dir / "initial_prompt.txt", initial_prompt)
    write_json(method_dir / "initial_prompt_metadata.json", metadata)


def history_record(
    *,
    iteration: int,
    prompt: str,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    num_correct = sum(truthy(row["correct"]) for row in predictions)
    return {
        "iteration": iteration,
        "prompt": prompt,
        "accuracy": compute_accuracy(predictions),
        "num_samples": len(predictions),
        "num_correct": num_correct,
        "num_failed": len(predictions) - num_correct,
    }


def _valid_target_format(raw_output: str, parsed_prediction: str, answer_format: str) -> bool:
    answer_format = (answer_format or "free_text").lower()
    if answer_format == "option_letter":
        return bool(re.search(r"\([A-Z]\)", raw_output or ""))
    return is_valid_prediction(parsed_prediction, answer_format)


def _mark_invalid_target_format(
    prediction: dict[str, Any], answer_format: str
) -> dict[str, Any]:
    answer_format = (answer_format or "free_text").lower()
    if answer_format == "option_letter" and not re.search(
        r"\([A-Z]\)", str(prediction.get("raw_output") or "")
    ):
        prediction = dict(prediction)
        prediction["error_type"] = "invalid_answer_format"
        prediction["correct"] = False
    return prediction


def evaluate_prompt(
    *,
    client: LLMClient,
    task: TaskSpec,
    rows: list[dict[str, Any]],
    prompt: str,
    method: str,
    iteration: int,
    out_dir: Path | None = None,
    max_answer_retries: int = 1,
    legacy_target_prompt_mode: bool = False,
) -> list[dict[str, Any]]:
    instruction = answer_instruction(task.answer_format)
    max_attempts = max(1, int(max_answer_retries or 1))

    def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
        user_prompt = (
            build_legacy_target_prompt(task, prompt, row["question"])
            if legacy_target_prompt_mode
            else build_question_prompt(prompt, row["question"], instruction)
        )
        error_type = ""
        raw_output = ""
        last_prediction: dict[str, Any] | None = None
        for attempt in range(max_attempts):
            try:
                question_key = (
                    row["question"]
                    if attempt == 0
                    else f"{row['question']}\n[answer_retry={attempt}]"
                )
                raw_output = client.complete_text(
                    system=LEGACY_TARGET_SYSTEM
                    if legacy_target_prompt_mode
                    else "You are a careful evaluation assistant.",
                    user=user_prompt,
                    method=method,
                    task_id=task.task_id,
                    iteration=iteration,
                    question=question_key,
                    agent_name="Target",
                    sample_id=row.get("sample_id", ""),
                )
                error_type = ""
            except ApiCallError as exc:
                error_type = exc.error_type
                raw_output = ""
            last_prediction = prediction_row(
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
            if error_type:
                return last_prediction
            if _valid_target_format(
                raw_output,
                str(last_prediction.get("parsed_prediction", "")),
                task.answer_format,
            ):
                return last_prediction
        if last_prediction is not None:
            return _mark_invalid_target_format(last_prediction, task.answer_format)
        return prediction_row(
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

    concurrency = max(1, int(getattr(client, "concurrency", 1) or 1))
    if concurrency == 1 or len(rows) <= 1:
        predictions = [evaluate_row(row) for row in rows]
    else:
        workers = min(concurrency, len(rows))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            predictions = list(executor.map(evaluate_row, rows))
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
    best_history = max(
        history,
        key=lambda item: float(item.get("accuracy") or -1),
        default=None,
    )
    final_history = history[-1] if history else None
    metrics.update(
        {
            "task_id": task.task_id,
            "method_id": method,
            "method_display_name": method_config.get("display_name", method),
            "answer_format": task.answer_format,
            "metric": task.metric,
            "best_prompt_path": "best_prompt.txt",
            "final_prompt_path": "final_prompt.txt",
            "num_history_rows": len(history),
            "best_validation_accuracy": None
            if best_history is None
            else best_history.get("accuracy"),
            "best_iteration": None if best_history is None else best_history.get("iteration"),
            "final_iteration": None if final_history is None else final_history.get("iteration"),
            "num_iterations": len(history),
            "exactness": method_config.get("exactness", method_config.get("exactness_level", "")),
            "exactness_level": method_config.get(
                "exactness_level", method_config.get("exactness", "")
            ),
        }
    )
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
    manifest_required = [
        "method_config.yaml",
        "metrics.json",
        "output_manifest.json",
        "run_state.json",
        "predictions.csv",
        "prompt_accuracy_history.csv",
        "best_prompt.txt",
        "final_prompt.txt",
        "diagnostics.md",
        "api_calls.csv",
        "raw_logs.txt",
    ]
    write_json(
        out_dir / "output_manifest.json",
        {
            "required_files": manifest_required,
            "created_files": sorted(
                {
                    *(path.name for path in out_dir.iterdir() if path.is_file()),
                    "output_manifest.json",
                }
            ),
        },
    )
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
    initial_prompt: str | None = None,
    max_answer_retries: int = 1,
    legacy_target_prompt_mode: bool = False,
) -> dict[str, Any]:
    def example_answer(item: dict[str, Any]) -> str:
        return str(item.get("answer", item.get("gold", "")))

    def example_block(item: dict[str, Any]) -> str:
        return (
            f"Example question:\n{item['question']}\n"
            f"Example answer:\nFinal answer: {example_answer(item)}"
        )

    def fixed_non_test_examples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        num_shots = int(method_config.get("num_shots", 3))
        test_ids = {
            str(row.get("sample_id"))
            for row in test_rows
            if row.get("sample_id") is not None
        }
        selected: list[dict[str, Any]] = []
        skipped_test_overlap: list[dict[str, Any]] = []
        for item in prompts.few_shot_examples:
            sample_id = item.get("sample_id")
            if sample_id is not None and str(sample_id) in test_ids:
                skipped_test_overlap.append(item)
                continue
            selected.append(item)
            if len(selected) >= num_shots:
                break
        if len(selected) < num_shots:
            raise ValueError(
                f"cot_fs_fixed requires {num_shots} non-test examples for "
                f"{task.task_id}; found {len(selected)} after filtering test overlap."
            )
        return selected, {
            "selection_strategy": "fixed_prompt_file_non_test",
            "num_shots": num_shots,
            "prompt_file": f"Prompt/task_prompts/{task.task_id}/few_shot.jsonl",
            "num_prompt_examples": len(prompts.few_shot_examples),
            "num_test_overlap_filtered": len(skipped_test_overlap),
            "selected_sample_ids": [
                item.get("sample_id", "") for item in selected
            ],
            "filtered_test_overlap_sample_ids": [
                item.get("sample_id", "") for item in skipped_test_overlap
            ],
        }

    if method == "origin":
        prompt = initial_prompt or prompts.origin
        iteration = 0 if initial_prompt else 1
    elif method == "cot_zs":
        prompt = prompts.cot_zero_shot
        iteration = 1
    elif method == "cot_fs":
        num_shots = int(method_config.get("num_shots", 3))
        examples = (few_shot_rows or prompts.few_shot_examples)[:num_shots]
        example_text = "\n\n".join(example_block(item) for item in examples)
        write_jsonl(out_dir / "used_few_shot_examples.jsonl", examples)
        prompt = f"{prompts.cot_few_shot}\n\n{example_text}"
        iteration = 1
    elif method == "cot_fs_fixed":
        examples, selection_metadata = fixed_non_test_examples()
        example_text = "\n\n".join(example_block(item) for item in examples)
        write_jsonl(out_dir / "used_few_shot_examples.jsonl", examples)
        write_json(out_dir / "few_shot_selection.json", selection_metadata)
        prompt = f"{prompts.cot_few_shot}\n\n{example_text}"
        iteration = 1
    else:
        raise ValueError(f"Unknown direct method: {method}")

    start = time.time()
    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=prompt,
        method=method,
        iteration=iteration,
        max_answer_retries=max_answer_retries,
        legacy_target_prompt_mode=legacy_target_prompt_mode,
    )
    history = [history_record(iteration=iteration, prompt=prompt, predictions=predictions)]
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
    max_answer_retries: int = 1,
    legacy_target_prompt_mode: bool = False,
) -> dict[str, Any]:
    start = time.time()
    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=prompt,
        method=method,
        iteration=1,
        max_answer_retries=max_answer_retries,
        legacy_target_prompt_mode=legacy_target_prompt_mode,
    )
    history = [history_record(iteration=1, prompt=prompt, predictions=predictions)]
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
    initial_prompt: str | None = None,
    max_answer_retries: int = 1,
    legacy_target_prompt_mode: bool = False,
) -> dict[str, Any]:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    history = []
    best_prompt = initial_prompt or prompts.origin
    best_accuracy = -1.0
    best_iteration = 0
    eval_rows = val_rows or opt_rows or test_rows
    initial_predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=eval_rows,
        prompt=best_prompt,
        method=method,
        iteration=0,
        max_answer_retries=max_answer_retries,
        legacy_target_prompt_mode=legacy_target_prompt_mode,
    )
    initial_record = history_record(
        iteration=0,
        prompt=best_prompt,
        predictions=initial_predictions,
    )
    history.append(initial_record)
    best_accuracy = float(initial_record["accuracy"])
    candidates.append(
        {"iteration": 0, "prompt": best_prompt, "accuracy": best_accuracy}
    )

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
            candidate = initial_prompt or prompts.origin
        eval_predictions = evaluate_prompt(
            client=client,
            task=task,
            rows=eval_rows,
            prompt=candidate,
            method=method,
            iteration=iteration,
            max_answer_retries=max_answer_retries,
            legacy_target_prompt_mode=legacy_target_prompt_mode,
        )
        record = history_record(
            iteration=iteration,
            prompt=candidate,
            predictions=eval_predictions,
        )
        accuracy = float(record["accuracy"])
        history.append(record)
        candidates.append(
            {"iteration": iteration, "prompt": candidate, "accuracy": accuracy}
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt = candidate
            best_iteration = iteration

    final_predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=best_prompt,
        method=method,
        iteration=best_iteration,
        max_answer_retries=max_answer_retries,
        legacy_target_prompt_mode=legacy_target_prompt_mode,
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
    if method == "pe2":
        write_jsonl(out_dir / "pe2_candidates.jsonl", candidates)
    write_json(
        out_dir / "selection_trace.json",
        {
            "method": method,
            "num_candidates": len(candidates),
            "best_accuracy": best_accuracy,
            "best_prompt": best_prompt,
            "exactness_level": method_config.get("exactness_level", ""),
        },
    )

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
        "exactness_notes": method_config.get("exactness_note", ""),
    }
