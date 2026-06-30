from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

PREDICTION_FIELDS = [
    "sample_id",
    "question",
    "gold",
    "raw_output",
    "parsed_prediction",
    "canonical_gold",
    "correct",
    "answer_format",
    "error_type",
    "method",
    "task_id",
    "iteration",
]


def normalize_free_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip().lower())


def _final_answer_segment(raw_output: Any) -> str:
    text = "" if raw_output is None else str(raw_output).strip()
    if not text:
        return ""
    patterns = [
        r"(?:final\s+answer|answer|therefore|thus|so)\s*(?:is|:|-)?\s*(.+)$",
        r"####\s*(.+)$",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _extract_option(text: str) -> str:
    segment = _final_answer_segment(text)
    for source in (segment, text):
        match = re.search(r"\(([A-Z])\)", source, flags=re.IGNORECASE)
        if match:
            return f"({match.group(1).upper()})"
        match = re.search(
            r"\b(?:option|answer|choice|letter)\s*(?:is|:|-)?\s*([A-Z])\b",
            source,
            flags=re.IGNORECASE,
        )
        if match:
            return f"({match.group(1).upper()})"
        match = re.fullmatch(r"\s*([A-Z])\s*", source, flags=re.IGNORECASE)
        if match:
            return f"({match.group(1).upper()})"
    return ""


def _extract_numeric(text: str) -> str:
    segment = _final_answer_segment(text)
    number_pattern = r"[-+]?\d[\d,]*(?:\.\d+)?"
    for source in (segment, text):
        matches = re.findall(number_pattern, source)
        if matches:
            return matches[-1].replace(",", "")
    return ""


def canonical_answer(value: Any, answer_format: str) -> str:
    text = "" if value is None else str(value).strip()
    lowered = text.lower()
    answer_format = (answer_format or "free_text").lower()

    if answer_format == "option_letter":
        return _extract_option(text)

    if answer_format == "valid_invalid":
        segment = _final_answer_segment(text).lower()
        for source in (segment, lowered):
            if re.search(r"\binvalid\b", source):
                return "invalid"
            if re.search(r"\bvalid\b", source):
                return "valid"
        return ""

    if answer_format == "yes_no":
        segment = _final_answer_segment(text).lower()
        for source in (segment, lowered):
            yes_match = re.search(r"\byes\b", source)
            no_match = re.search(r"\bno\b", source)
            if yes_match and (not no_match or yes_match.start() < no_match.start()):
                return "yes"
            if no_match:
                return "no"
        return ""

    if answer_format == "boolean":
        segment = _final_answer_segment(text).lower()
        for source in (segment, lowered):
            true_match = re.search(r"\btrue\b", source)
            false_match = re.search(r"\bfalse\b", source)
            if true_match and (
                not false_match or true_match.start() < false_match.start()
            ):
                return "true"
            if false_match:
                return "false"
        return ""

    if answer_format == "numeric":
        return _extract_numeric(text)

    return normalize_free_text(text)


def extract_prediction(raw_output: Any, answer_format: str) -> str:
    return canonical_answer(raw_output, answer_format)


def is_valid_prediction(value: str, answer_format: str) -> bool:
    answer_format = (answer_format or "free_text").lower()
    if not value:
        return False
    if answer_format == "option_letter":
        return bool(re.fullmatch(r"\([A-Z]\)", value))
    if answer_format == "yes_no":
        return value in {"yes", "no"}
    if answer_format == "valid_invalid":
        return value in {"valid", "invalid"}
    if answer_format == "boolean":
        return value in {"true", "false"}
    if answer_format == "numeric":
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value))
    return True


def is_correct_prediction(raw_output: Any, gold: Any, answer_format: str) -> bool:
    prediction = extract_prediction(raw_output, answer_format)
    canonical_gold = canonical_answer(gold, answer_format)
    if not is_valid_prediction(prediction, answer_format):
        return False
    if (answer_format or "").lower() == "numeric":
        try:
            return math.isclose(
                float(prediction), float(canonical_gold), rel_tol=1e-6, abs_tol=1e-6
            )
        except (TypeError, ValueError):
            return False
    return prediction == canonical_gold


def answer_instruction(answer_format: str) -> str:
    answer_format = (answer_format or "free_text").lower()
    if answer_format == "option_letter":
        return "End with `Final answer: (X)` where X is the single best option letter."
    if answer_format == "yes_no":
        return "End with `Final answer: yes` or `Final answer: no`."
    if answer_format == "valid_invalid":
        return "End with `Final answer: valid` or `Final answer: invalid`."
    if answer_format == "boolean":
        return "End with `Final answer: true` or `Final answer: false`."
    if answer_format == "numeric":
        return "End with `Final answer: <number>` and do not include units."
    return "End with `Final answer: <answer>`."


def prediction_row(
    *,
    sample_id: Any,
    question: Any,
    gold: Any,
    raw_output: Any,
    answer_format: str,
    method: str,
    task_id: str,
    iteration: int,
    error_type: str = "",
) -> dict[str, Any]:
    parsed_prediction = extract_prediction(raw_output, answer_format)
    canonical_gold = canonical_answer(gold, answer_format)
    if not error_type and not is_valid_prediction(parsed_prediction, answer_format):
        error_type = "invalid_answer_format" if raw_output else "empty_output"
    correct = bool(
        not error_type and is_correct_prediction(raw_output, gold, answer_format)
    )
    return {
        "sample_id": sample_id,
        "question": question,
        "gold": gold,
        "raw_output": raw_output,
        "parsed_prediction": parsed_prediction,
        "canonical_gold": canonical_gold,
        "correct": correct,
        "answer_format": answer_format,
        "error_type": error_type,
        "method": method,
        "task_id": task_id,
        "iteration": iteration,
    }


def compute_accuracy(predictions: Iterable[dict[str, Any]]) -> float:
    rows = list(predictions)
    return sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else 0.0


def compute_final_metrics_from_predictions(
    predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(predictions)
    num_samples = len(rows)
    num_correct = sum(bool(row.get("correct")) for row in rows)
    api_errors = sum(str(row.get("error_type", "")).startswith("api_") for row in rows)
    parse_errors = sum(
        row.get("error_type")
        in {"parse_error", "empty_output", "invalid_answer_format"}
        for row in rows
    )
    return {
        "accuracy": num_correct / num_samples if num_samples else 0.0,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "num_failed": num_samples - num_correct,
        "api_errors": api_errors,
        "parse_errors": parse_errors,
    }


def diagnostics_markdown(
    task_id: str, method: str, predictions: Iterable[dict[str, Any]]
) -> str:
    rows = list(predictions)
    metrics = compute_final_metrics_from_predictions(rows)
    errors = Counter(row.get("error_type", "") or "none" for row in rows)
    lines = [
        f"# Diagnostics: {method} / {task_id}",
        "",
        f"- accuracy: {metrics['accuracy']}",
        f"- num_samples: {metrics['num_samples']}",
        f"- num_correct: {metrics['num_correct']}",
        f"- num_failed: {metrics['num_failed']}",
        f"- api_errors: {metrics['api_errors']}",
        f"- parse_errors: {metrics['parse_errors']}",
        "",
        "## Error Types",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in errors.most_common())
    return "\n".join(lines) + "\n"
