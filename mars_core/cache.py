from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, cache_dir: str | Path, enabled: bool = False):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(self.make_key(payload))
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def set(self, payload: dict[str, Any], value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(self.make_key(payload))
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
