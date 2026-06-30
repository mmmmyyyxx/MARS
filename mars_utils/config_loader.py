from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

import yaml


@dataclass
class MarsConfig:
    api_key_env: str
    base_url_env: str
    model: str
    temperature: float
    max_iterations: int
    early_stop_delta: float
    max_critic_revisions: int
    concurrency: int
    max_samples: Optional[int]
    max_answer_retries: int
    request_timeout: float
    output_dir: str
    cache_enabled: bool
    seed: int
    dry_run: bool = False

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)

    @property
    def base_url(self) -> Optional[str]:
        return os.getenv(self.base_url_env)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config(path: str, overrides: Dict[str, Any]) -> MarsConfig:
    data = load_yaml(path)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return MarsConfig(**data)
