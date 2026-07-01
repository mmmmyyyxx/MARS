from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .api_client import API_CALL_COLUMNS, LLMClient
from .evaluator import compute_accuracy, truthy
from .logging_utils import write_csv, write_json, write_jsonl
from .mars_runner import TaskSpec, evaluate_prompt, hash_rows, write_method_outputs
from .prompt_loader import TaskPrompts
from .run_state import REQUIRED_METHOD_FILES, build_run_state, prompt_hash, save_run_state


def _parse_steps(raw: str) -> list[str]:
    steps = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line:
            steps.append(line)
    return steps or [
        "Clarify the answer format.",
        "Improve task-specific reasoning before giving the final answer.",
    ]


def _planner_steps(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    method: str,
    enabled: bool,
) -> list[str]:
    if not enabled:
        return [
            "Use the original prompt without Planner-generated decomposition.",
        ]
    planner_prompt = prompts.planner.format(task_description=prompts.user_proxy)
    raw = client.complete_text(
        system="You are the Planner agent in the MARS prompt optimization workflow.",
        user=(
            f"UserProxy task description:\n{prompts.user_proxy}\n\n"
            f"Planner template:\n{planner_prompt}\n\n"
            f"Task name: {task.paper_display_name}\n\n"
            "Return a concise numbered list of prompt-optimization steps."
        ),
        method=method,
        task_id=task.task_id,
        iteration=0,
        agent_name="Planner",
    )
    return _parse_steps(raw)


def _critic_accepts(feedback: str) -> bool:
    lowered = feedback.lower()
    if any(word in lowered for word in ["reject", "revise", "not useful", "unclear"]):
        return False
    return True


def _teacher_question(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    method: str,
    iteration: int,
    step_index: int,
    step: str,
    current_prompt: str,
    revision: int,
    previous_feedback: str = "",
) -> str:
    return client.complete_text(
        system="You are the Teacher agent. Ask one Socratic question that helps improve the prompt.",
        user=(
            f"UserProxy task description:\n{prompts.user_proxy}\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"Planner step {step_index + 1}:\n{step}\n\n"
            f"Previous Critic feedback:\n{previous_feedback}\n\n"
            f"Revision attempt: {revision}\n\n"
            "Return only one useful Socratic question."
        ),
        method=method,
        task_id=task.task_id,
        iteration=iteration,
        agent_name="Teacher",
    )


def _critic_feedback(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    method: str,
    iteration: int,
    step: str,
    question: str,
) -> str:
    return client.complete_text(
        system="You are the Critic agent. Decide whether the Teacher question is useful and Socratic.",
        user=(
            f"Task description:\n{prompts.user_proxy}\n\n"
            f"Planner step:\n{step}\n\n"
            f"Teacher question:\n{question}\n\n"
            "Reply with ACCEPT or REVISE, followed by concise feedback."
        ),
        method=method,
        task_id=task.task_id,
        iteration=iteration,
        agent_name="Critic",
    )


def _student_update(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    method: str,
    iteration: int,
    step: str,
    current_prompt: str,
    question: str,
    feedback: str,
) -> str:
    updated = client.complete_text(
        system="You are the Student agent. Update the task prompt using the Socratic guidance.",
        user=(
            f"Task: {task.paper_display_name}\n"
            f"Task description:\n{prompts.user_proxy}\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"Planner step:\n{step}\n\n"
            f"Teacher question:\n{question}\n\n"
            f"Critic feedback:\n{feedback}\n\n"
            "Return only the revised prompt."
        ),
        method=method,
        task_id=task.task_id,
        iteration=iteration,
        max_tokens=700,
        agent_name="Student",
    ).strip()
    return updated or current_prompt


def run_mars_official(
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
    early_stop_delta: float | None = None,
    max_critic_revisions: int = 1,
) -> dict[str, Any]:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    planner_enabled = bool(method_config.get("planner_enabled", True))
    socratic_enabled = bool(method_config.get("socratic_enabled", True))
    critic_enabled = bool(method_config.get("critic_enabled", True))
    eval_rows = val_rows or opt_rows or test_rows

    steps = _planner_steps(
        client=client,
        task=task,
        prompts=prompts,
        method=method,
        enabled=planner_enabled,
    )
    write_json(
        out_dir / "planner_steps.json",
        {
            "agent": "Planner",
            "planner_enabled": planner_enabled,
            "steps": steps,
        },
    )

    current_prompt = prompts.origin
    best_prompt = current_prompt
    best_accuracy = -1.0
    history = []
    teacher_questions = []
    critic_feedback_rows = []
    student_prompts = []
    target_scores = []
    iterations = max(1, int(max_iterations or 1))

    if not socratic_enabled:
        current_prompt = (
            prompts.origin
            + "\n\nPlanner steps:\n"
            + "\n".join(f"- {step}" for step in steps)
        )

    for iteration in range(1, iterations + 1):
        previous_best = best_accuracy
        if socratic_enabled:
            for step_index, step in enumerate(steps):
                question = ""
                feedback = ""
                accepted = not critic_enabled
                for revision in range(0, max(0, max_critic_revisions) + 1):
                    question = _teacher_question(
                        client=client,
                        task=task,
                        prompts=prompts,
                        method=method,
                        iteration=iteration,
                        step_index=step_index,
                        step=step,
                        current_prompt=current_prompt,
                        revision=revision,
                        previous_feedback=feedback,
                    )
                    teacher_questions.append(
                        {
                            "iteration": iteration,
                            "step_index": step_index,
                            "step": step,
                            "revision": revision,
                            "question": question,
                        }
                    )
                    if critic_enabled:
                        feedback = _critic_feedback(
                            client=client,
                            task=task,
                            prompts=prompts,
                            method=method,
                            iteration=iteration,
                            step=step,
                            question=question,
                        )
                        accepted = _critic_accepts(feedback)
                    critic_feedback_rows.append(
                        {
                            "iteration": iteration,
                            "step_index": step_index,
                            "step": step,
                            "revision": revision,
                            "critic_enabled": critic_enabled,
                            "accepted": accepted,
                            "feedback": feedback,
                        }
                    )
                    if accepted:
                        break
                current_prompt = _student_update(
                    client=client,
                    task=task,
                    prompts=prompts,
                    method=method,
                    iteration=iteration,
                    step=step,
                    current_prompt=current_prompt,
                    question=question,
                    feedback=feedback,
                )
                student_prompts.append(
                    {
                        "iteration": iteration,
                        "step_index": step_index,
                        "step": step,
                        "prompt": current_prompt,
                    }
                )
        else:
            student_prompts.append(
                {
                    "iteration": iteration,
                    "step_index": "",
                    "step": "socratic_disabled",
                    "prompt": current_prompt,
                }
            )

        eval_predictions = evaluate_prompt(
            client=client,
            task=task,
            rows=eval_rows,
            prompt=current_prompt,
            method=method,
            iteration=iteration,
        )
        accuracy = compute_accuracy(eval_predictions)
        score_row = {
            "iteration": iteration,
            "prompt": current_prompt,
            "accuracy": accuracy,
            "num_samples": len(eval_predictions),
            "num_correct": sum(truthy(row["correct"]) for row in eval_predictions),
            "num_failed": len(eval_predictions)
            - sum(truthy(row["correct"]) for row in eval_predictions),
        }
        history.append(score_row)
        target_scores.append(score_row)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt = current_prompt
        accuracy_gain = best_accuracy - previous_best if previous_best >= 0 else None
        if (
            early_stop_delta is not None
            and previous_best >= 0
            and accuracy_gain is not None
            and accuracy_gain < early_stop_delta
        ):
            break

    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=best_prompt,
        method=method,
        iteration=len(history) or 1,
    )

    write_jsonl(out_dir / "teacher_questions.jsonl", teacher_questions)
    write_jsonl(out_dir / "critic_feedback.jsonl", critic_feedback_rows)
    write_jsonl(out_dir / "student_prompts.jsonl", student_prompts)
    write_jsonl(out_dir / "target_scores.jsonl", target_scores)
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
            "official_compatible: true\n"
            "agents: Manager,UserProxy,Planner,Teacher,Critic,Student,Target\n"
            f"planner_enabled: {planner_enabled}\n"
            f"socratic_enabled: {socratic_enabled}\n"
            f"critic_enabled: {critic_enabled}\n"
            f"max_critic_revisions: {max_critic_revisions}\n"
            f"early_stop_delta: {early_stop_delta}\n"
            f"runtime_seconds: {time.time() - start:.4f}\n"
        ),
    )
    metrics["runtime_seconds"] = time.time() - start
    metrics["num_iterations"] = len(history)
    if not (out_dir / "run_state.json").exists():
        state = build_run_state(
            run_id=getattr(client, "run_id", ""),
            suite=getattr(client, "suite", ""),
            method_id=method,
            task_id=task.task_id,
            model=getattr(client, "model", ""),
            temperature=float(getattr(client, "temperature", 0) or 0),
            max_samples=len(test_rows),
            dataset_path=task.dataset_path,
            dataset_hash=hash_rows(test_rows),
            split_hashes={
                "opt": hash_rows(opt_rows),
                "val": hash_rows(val_rows),
                "test": hash_rows(test_rows),
            },
            prompt_hash_value=prompt_hash(prompts.origin),
            config_hash=str(method_config.get("config_hash", "")),
            expected_ids=[row.get("sample_id") for row in test_rows],
            predictions_path=out_dir / "predictions.csv",
            status="completed",
        )
        save_run_state(out_dir / "run_state.json", state)
    if not (out_dir / "api_calls.csv").exists():
        write_csv(out_dir / "api_calls.csv", client.stats.call_records, API_CALL_COLUMNS)
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "output_manifest.json",
        {
            "required_files": REQUIRED_METHOD_FILES,
            "created_files": sorted(
                {
                    *(path.name for path in out_dir.iterdir() if path.is_file()),
                    "output_manifest.json",
                }
            ),
        },
    )
    return metrics


def run_official_mars(**kwargs) -> dict[str, Any]:
    return run_mars_official(**kwargs)
