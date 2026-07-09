from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .api_client import API_CALL_COLUMNS, LLMClient
from .evaluator import truthy
from .logging_utils import write_csv, write_json, write_jsonl
from .mars_runner import (
    TaskSpec,
    LEGACY_INITIAL_PROMPT_SYSTEM,
    evaluate_prompt,
    hash_rows,
    history_record,
    write_method_outputs,
)
from .prompt_loader import TaskPrompts
from .run_state import REQUIRED_METHOD_FILES, build_run_state, prompt_hash, save_run_state


def _read_prompt_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


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


def parse_steps_strict(raw: str) -> list[str]:
    total_match = re.search(r"^\s*Total steps:\s*(\d+)\s*$", raw, re.MULTILINE)
    if not total_match:
        raise ValueError("Planner output missing `Total steps: <number>`.")
    total = int(total_match.group(1))
    steps: dict[int, str] = {}
    for match in re.finditer(r"^\s*Step\s+(\d+):\s*(.+?)\s*$", raw, re.MULTILINE):
        steps[int(match.group(1))] = match.group(2).strip()
    expected = list(range(1, total + 1))
    missing = [index for index in expected if index not in steps]
    if missing:
        raise ValueError(f"Planner output missing steps: {missing}")
    return [steps[index] for index in expected]


def _planner_steps(
    *,
    client: LLMClient,
    task: TaskSpec,
    prompts: TaskPrompts,
    method: str,
    enabled: bool,
    strict: bool = False,
) -> list[str]:
    if not enabled:
        return [
            "Use the original prompt without Planner-generated decomposition.",
        ]
    planner_prompt = prompts.planner.format(task_description=prompts.user_proxy)
    raw = client.complete_text(
        system=_read_prompt_file("Prompt/system_prompt_planner.txt"),
        user=planner_prompt,
        method=method,
        task_id=task.task_id,
        iteration=0,
        agent_name="Planner",
    )
    if strict:
        return parse_steps_strict(raw)
    return _parse_steps(raw)


def _critic_accepts(feedback: str) -> bool:
    lowered = feedback.lower()
    if "[false]" in lowered or re.search(r"\bfalse\b", lowered):
        return False
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
    if previous_feedback:
        user = (
            "Here is feedback on whether your output matches the Socratic "
            "questioning, please refer to the suggestion to regenerate the "
            f"questioning:\n{previous_feedback}\n"
            f"Here is the task definition:\n{prompts.user_proxy}\n"
            "Here is the prompt given by the student from the previous round:\n"
            f"{current_prompt}\n"
            "Ask heuristic questions based on the students' historical responses "
            f"and the current step: {step}\n"
        )
    else:
        user = (
            f"Here is the task definition:\n{prompts.user_proxy}\n"
            "Here is the prompt given by the student from the previous round:\n"
            f"{current_prompt}\n"
            "Ask heuristic questions based on the students' historical responses "
            f"and the current step: {step}\n"
        )
    return client.complete_text(
        system=_read_prompt_file("Prompt/system_prompt_teacher.txt"),
        user=user,
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
    user = (
        "Read the following questions posed by the teacher and judge whether "
        "the teacher's questioning follows the Socratic style of questioning.\n"
        f"questions:\n{question}"
    )
    return client.complete_text(
        system=_read_prompt_file("Prompt/system_prompt_critic.txt"),
        user=user,
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
    user = (
        f"Here is the task definition:\n{prompts.user_proxy}\n"
        f"Here is your last prompt:\n{current_prompt}\n"
        "Please base on the following question update your prompt:\n"
        f"{question}"
    )
    updated = client.complete_text(
        system=LEGACY_INITIAL_PROMPT_SYSTEM,
        user=user,
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
    initial_prompt: str | None = None,
    max_answer_retries: int = 1,
    planner_strict_mode: bool = False,
    legacy_target_prompt_mode: bool = False,
    target_call_count_limit: int = 10,
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
        strict=planner_strict_mode,
    )
    write_json(
        out_dir / "planner_steps.json",
        {
            "agent": "Planner",
            "planner_enabled": planner_enabled,
            "steps": steps,
        },
    )

    current_prompt = initial_prompt or prompts.origin
    best_prompt = current_prompt
    best_accuracy = -1.0
    best_iteration = 0
    history = []
    teacher_questions = []
    critic_feedback_rows = []
    student_prompts = []
    target_scores = []
    iterations = max(1, int(max_iterations or 1))
    target_call_count = 0

    initial_predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=eval_rows,
        prompt=current_prompt,
        method=method,
        iteration=0,
        max_answer_retries=max_answer_retries,
        legacy_target_prompt_mode=legacy_target_prompt_mode,
    )
    target_call_count += 1
    initial_score = history_record(
        iteration=0,
        prompt=current_prompt,
        predictions=initial_predictions,
    )
    history.append(initial_score)
    target_scores.append(initial_score)
    best_accuracy = float(initial_score["accuracy"])
    best_prompt = current_prompt
    best_iteration = 0

    if not socratic_enabled:
        current_prompt = (
            current_prompt
            + "\n\nPlanner steps:\n"
            + "\n".join(f"- {step}" for step in steps)
        )

    for iteration in range(1, iterations + 1):
        if target_call_count >= target_call_count_limit:
            break
        if socratic_enabled:
            for step_index, step in enumerate(steps):
                question = ""
                feedback = ""
                accepted = not critic_enabled
                question = _teacher_question(
                    client=client,
                    task=task,
                    prompts=prompts,
                    method=method,
                    iteration=iteration,
                    step_index=step_index,
                    step=step,
                    current_prompt=current_prompt,
                    revision=0,
                    previous_feedback="",
                )
                teacher_questions.append(
                    {
                        "iteration": iteration,
                        "step_index": step_index,
                        "step": step,
                        "revision": 0,
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
                        "revision": 0,
                        "critic_enabled": critic_enabled,
                        "accepted": accepted,
                        "feedback": feedback,
                    }
                )
                if (
                    critic_enabled
                    and not accepted
                    and max(0, max_critic_revisions) >= 1
                ):
                    question = _teacher_question(
                        client=client,
                        task=task,
                        prompts=prompts,
                        method=method,
                        iteration=iteration,
                        step_index=step_index,
                        step=step,
                        current_prompt=current_prompt,
                        revision=1,
                        previous_feedback=feedback,
                    )
                    teacher_questions.append(
                        {
                            "iteration": iteration,
                            "step_index": step_index,
                            "step": step,
                            "revision": 1,
                            "question": question,
                        }
                    )
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
            max_answer_retries=max_answer_retries,
            legacy_target_prompt_mode=legacy_target_prompt_mode,
        )
        target_call_count += 1
        score_row = history_record(
            iteration=iteration,
            prompt=current_prompt,
            predictions=eval_predictions,
        )
        accuracy = float(score_row["accuracy"])
        history.append(score_row)
        target_scores.append(score_row)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt = current_prompt
            best_iteration = iteration
        if (
            early_stop_delta is not None
            and len(history) >= 2
            and abs(
                float(history[-1].get("accuracy") or 0.0)
                - float(history[-2].get("accuracy") or 0.0)
            )
            < early_stop_delta
        ):
            break

    predictions = evaluate_prompt(
        client=client,
        task=task,
        rows=test_rows,
        prompt=best_prompt,
        method=method,
        iteration=best_iteration,
        max_answer_retries=max_answer_retries,
        legacy_target_prompt_mode=legacy_target_prompt_mode,
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
            f"initial_prompt_source: {'provided' if initial_prompt is not None else 'manual_origin'}\n"
            f"planner_strict_mode: {planner_strict_mode}\n"
            f"max_answer_retries: {max_answer_retries}\n"
            f"legacy_target_prompt_mode: {legacy_target_prompt_mode}\n"
            f"target_call_count_limit: {target_call_count_limit}\n"
            f"target_call_count: {target_call_count}\n"
            f"runtime_seconds: {time.time() - start:.4f}\n"
        ),
    )
    metrics["runtime_seconds"] = time.time() - start
    metrics["num_iterations"] = len(history)
    metrics["target_call_count"] = target_call_count
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
            prompt_hash_value=prompt_hash(initial_prompt or prompts.origin),
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
