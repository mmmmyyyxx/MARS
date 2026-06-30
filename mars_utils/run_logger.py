import json
import os
from typing import Any, Dict, Optional


def append_error(
    task_dir: str,
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(task_dir, exist_ok=True)
    path = os.path.join(task_dir, "errors.jsonl")
    payload = {
        "error_type": error_type,
        "message": message,
        "details": details or {},
    }
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
