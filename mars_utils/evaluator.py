import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


TRUTHY_VALUES = {"true", "1", "yes"}


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).replace(" ", "").strip().lower()


def canonical_answer(value: Any, answer_format: str) -> str:
    text = "" if value is None else str(value).strip()
    lowered = text.lower()
    compact = normalize_text(text)
    answer_format = (answer_format or "auto").lower()

    if answer_format == "option_letter":
        match = re.search(r"\(([A-Z])\)", text, flags=re.IGNORECASE)
        if match:
            return f"({match.group(1).upper()})"
        match = re.search(r"\b([A-Z])\b", text, flags=re.IGNORECASE)
        if match:
            return f"({match.group(1).upper()})"
        return compact

    if answer_format == "boolean":
        if re.search(r"\btrue\b", lowered):
            return "true"
        if re.search(r"\bfalse\b", lowered):
            return "false"
        return compact

    if answer_format == "valid_invalid":
        if re.search(r"\binvalid\b", lowered):
            return "invalid"
        if re.search(r"\bvalid\b", lowered):
            return "valid"
        return compact

    if answer_format == "yes_no":
        if re.search(r"\byes\b", lowered):
            return "yes"
        if re.search(r"\bno\b", lowered):
            return "no"
        return compact

    if answer_format == "short_answer":
        return text.strip().lower()

    return compact


def fallback_extract_from_raw(raw: Any, answer_format: str) -> str:
    return canonical_answer(raw, answer_format)


def answer_instruction(answer_format: str) -> str:
    answer_format = (answer_format or "auto").lower()
    if answer_format == "boolean":
        return "Output exactly one word: True or False. Do not output an option letter or explanation."
    if answer_format == "valid_invalid":
        return "Output exactly one word: valid or invalid. Do not output an option letter or explanation."
    if answer_format == "yes_no":
        return "Output exactly one word: yes or no. Do not output an option letter or explanation."
    if answer_format == "option_letter":
        return "Output only the answer option as a parenthesized capital letter, such as (A). Do not add any other text."
    return "Output only the final answer in the same format as the gold label. Do not add any explanation."


def is_valid_canonical(value: str, answer_format: str) -> bool:
    answer_format = (answer_format or "auto").lower()
    if not value:
        return False
    if answer_format == "option_letter":
        return bool(re.fullmatch(r"\([A-Z]\)", value))
    if answer_format == "boolean":
        return value in {"true", "false"}
    if answer_format == "valid_invalid":
        return value in {"valid", "invalid"}
    if answer_format == "yes_no":
        return value in {"yes", "no"}
    return True


def is_correct_prediction(prediction: Any, gold: Any, answer_format: str) -> Tuple[bool, str, str]:
    canonical_prediction = canonical_answer(prediction, answer_format)
    canonical_gold = canonical_answer(gold, answer_format)
    return canonical_prediction == canonical_gold, canonical_prediction, canonical_gold


def row_is_correct(row: Dict[str, Any]) -> bool:
    return str(row.get("correct", "")).strip().lower() in TRUTHY_VALUES


def final_prediction_rows(predictions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(predictions or [])
    if not rows:
        return []

    def iteration_value(row: Dict[str, Any]) -> int:
        try:
            return int(row.get("iteration", 0))
        except (TypeError, ValueError):
            return 0

    last_iteration = max(iteration_value(row) for row in rows)
    return [row for row in rows if iteration_value(row) == last_iteration]


def compute_final_metrics_from_predictions(predictions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    final_rows = final_prediction_rows(predictions)
    num_samples = len(final_rows)
    num_success = sum(1 for row in final_rows if row_is_correct(row))
    num_failed = num_samples - num_success
    final_accuracy = num_success / num_samples if num_samples else 0.0
    return {
        "num_samples": num_samples,
        "num_success": num_success,
        "num_failed": num_failed,
        "final_accuracy": final_accuracy,
    }


def _counter_lines(title: str, values: Iterable[Any], limit: int = 20) -> List[str]:
    counter = Counter("" if value is None else str(value) for value in values)
    lines = [f"## {title}", ""]
    if not counter:
        lines.append("- None")
    else:
        for value, count in counter.most_common(limit):
            display = value if value else "<empty>"
            lines.append(f"- `{display}`: {count}")
    lines.append("")
    return lines


def build_diagnostics_markdown(task_id: str, predictions: Iterable[Dict[str, Any]]) -> str:
    rows = final_prediction_rows(predictions)
    answer_formats = sorted({str(row.get("answer_format", "")) for row in rows if row.get("answer_format")})
    metrics = compute_final_metrics_from_predictions(rows)
    parse_failed = sum(
        1
        for row in rows
        if not str(row.get("prediction", "")).strip()
        or str(row.get("error", "")).strip() == "answer_parse_failed"
    )

    lines = [
        f"# Evaluation Diagnostics: {task_id}",
        "",
        f"- answer_format: {', '.join(answer_formats) if answer_formats else 'unknown'}",
        f"- final_samples: {metrics['num_samples']}",
        f"- num_success: {metrics['num_success']}",
        f"- num_failed: {metrics['num_failed']}",
        f"- final_accuracy: {metrics['final_accuracy']}",
        f"- parse_failed: {parse_failed}",
        "",
    ]
    lines.extend(_counter_lines("Gold Label Distribution", (row.get("answer") for row in rows)))
    lines.extend(_counter_lines("Canonical Answer Distribution", (row.get("canonical_answer") for row in rows)))
    lines.extend(_counter_lines("Canonical Prediction Distribution", (row.get("prediction") for row in rows)))

    lines.extend(["## Raw Prediction Examples", ""])
    if rows:
        for row in rows[:20]:
            raw = str(row.get("raw_prediction", "")).replace("\n", "\\n")
            lines.append(f"- `{raw if raw else '<empty>'}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.extend(["## Canonical Answer Examples", ""])
    if rows:
        for row in rows[:20]:
            answer = str(row.get("canonical_answer", ""))
            lines.append(f"- `{answer if answer else '<empty>'}`")
    else:
        lines.append("- None")
    lines.append("")

    if metrics["num_samples"] and metrics["final_accuracy"] == 0:
        lines.extend([
            "## Warning",
            "",
            "Final accuracy is 0. Inspect whether predictions are empty, parsed into the wrong answer format, or genuinely incorrect.",
            "",
        ])

    return "\n".join(lines)
