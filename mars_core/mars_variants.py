from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .api_client import LLMClient
from .evaluator import compute_accuracy
from .logging_utils import write_json, write_jsonl, write_text
from .mars_runner import (
    TaskSpec,
    evaluate_prompt,
    generate_prompt_candidate,
    write_method_outputs,
)
from .prompt_loader import TaskPrompts


def _generate_subgoals(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    method: str,
    enabled: bool,
) -> list[str]:
    if not enabled:
        return [
            "Clarify the required answer format.",
            "Solve the input question carefully before returning the final answer.",
        ]
    raw = client.complete_text(
        system="You are a planner for prompt optimization.",
        user=prompts.planner.format(task_description=prompts.user_proxy),
        method=method,
        task_id=task.task_id,
        iteration=0,
        agent_name="Planner",
    )
    subgoals = []
    for line in raw.splitlines():
        line = line.strip(" -\t")
        if line:
            subgoals.append(line)
    return subgoals or ["Clarify answer format.", "Improve task-specific reasoning."]


def run_mars_variant(
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
    planner_enabled = bool(method_config.get("planner_enabled", True))
    socratic_enabled = bool(method_config.get("socratic_enabled", True))
    critic_enabled = bool(method_config.get("critic_enabled", True))
    subgoals = _generate_subgoals(
        client=client,
        task=task,
        prompts=prompts,
        method=method,
        enabled=planner_enabled,
    )
    write_json(
        out_dir / "subgoals.json",
        {"subgoals": subgoals, "planner_enabled": planner_enabled},
    )

    if not socratic_enabled:
        prompt = prompts.origin + "\n\nOptimization sub-goals:\n" + "\n".join(subgoals)
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
                "num_correct": sum(bool(row["correct"]) for row in predictions),
                "num_failed": len(predictions)
                - sum(bool(row["correct"]) for row in predictions),
            }
        ]
        write_jsonl(out_dir / "teacher_questions.jsonl", [])
        write_jsonl(out_dir / "critic_feedback.jsonl", [])
        write_jsonl(
            out_dir / "student_prompts.jsonl", [{"iteration": 1, "prompt": prompt}]
        )
        write_text(
            out_dir / "diagnostics_note.md",
            "This variant removes the Socratic Teacher-Critic-Student refinement module.\n",
        )
        metrics = write_method_outputs(
            out_dir=out_dir,
            task=task,
            method=method,
            method_config=method_config,
            predictions=predictions,
            history=history,
            best_prompt=prompt,
            final_prompt=prompt,
            raw_logs="This variant removes the Socratic Teacher-Critic-Student refinement module.\n",
        )
        metrics["runtime_seconds"] = time.time() - start
        metrics["num_iterations"] = 1
        return metrics

    current_prompt = prompts.origin
    best_prompt = current_prompt
    best_accuracy = -1.0
    history = []
    teacher_questions = []
    critic_feedback = []
    student_prompts = []
    eval_rows = val_rows or opt_rows or test_rows

    iterations = max(1, max_iterations)
    for iteration in range(1, iterations + 1):
        subgoal = subgoals[(iteration - 1) % len(subgoals)]
        question = client.complete_text(
            system="You are a Socratic teacher for prompt improvement.",
            user=(
                f"Task:\n{prompts.user_proxy}\n\n"
                f"Current prompt:\n{current_prompt}\n\n"
                f"Sub-goal:\n{subgoal}\n\nAsk one useful heuristic question."
            ),
            method=method,
            task_id=task.task_id,
            iteration=iteration,
            agent_name="Teacher",
        )
        teacher_questions.append(
            {"iteration": iteration, "subgoal": subgoal, "question": question}
        )

        feedback = ""
        if critic_enabled:
            feedback = client.complete_text(
                system="You critique whether a teacher question is useful and Socratic.",
                user=f"Question:\n{question}\n\nReturn concise feedback.",
                method=method,
                task_id=task.task_id,
                iteration=iteration,
                agent_name="Critic",
            )
        critic_feedback.append(
            {
                "iteration": iteration,
                "feedback": feedback,
                "critic_enabled": critic_enabled,
            }
        )

        context = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"Teacher question:\n{question}\n\n"
            f"Critic feedback:\n{feedback}\n\n"
            "Revise the prompt."
        )
        current_prompt = generate_prompt_candidate(
            client=client,
            task=task,
            prompts=prompts,
            method=method,
            iteration=iteration,
            context=context,
        )
        student_prompts.append({"iteration": iteration, "prompt": current_prompt})

        eval_predictions = evaluate_prompt(
            client=client,
            task=task,
            rows=eval_rows,
            prompt=current_prompt,
            method=method,
            iteration=iteration,
        )
        accuracy = compute_accuracy(eval_predictions)
        record = {
            "iteration": iteration,
            "prompt": current_prompt,
            "accuracy": accuracy,
            "num_samples": len(eval_predictions),
            "num_correct": sum(bool(row["correct"]) for row in eval_predictions),
            "num_failed": len(eval_predictions)
            - sum(bool(row["correct"]) for row in eval_predictions),
        }
        history.append(record)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt = current_prompt

    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=best_prompt,
        method=method,
        iteration=len(history) or 1,
    )
    write_jsonl(out_dir / "teacher_questions.jsonl", teacher_questions)
    write_jsonl(out_dir / "critic_feedback.jsonl", critic_feedback)
    write_jsonl(out_dir / "student_prompts.jsonl", student_prompts)
    metrics = write_method_outputs(
        out_dir=out_dir,
        task=task,
        method=method,
        method_config=method_config,
        predictions=predictions,
        history=history,
        best_prompt=best_prompt,
        final_prompt=current_prompt,
        raw_logs=(
            f"planner_enabled: {planner_enabled}\n"
            f"socratic_enabled: {socratic_enabled}\n"
            f"critic_enabled: {critic_enabled}\n"
            f"runtime_seconds: {time.time() - start:.4f}\n"
        ),
    )
    metrics["runtime_seconds"] = time.time() - start
    metrics["num_iterations"] = len(history)
    return metrics
