from __future__ import annotations

import os
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError
except ImportError:  # pragma: no cover
    APIConnectionError = APITimeoutError = RateLimitError = None

from .cache import DiskCache
from .token_counter import estimate_message_tokens, estimate_tokens


class ApiCallError(RuntimeError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class ApiStats:
    api_calls: int = 0
    cache_hits: int = 0
    api_errors: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    retries: int = 0
    error_records: list[dict[str, Any]] = field(default_factory=list)
    call_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion


def classify_api_error(exc: Exception) -> str:
    if APITimeoutError is not None and isinstance(exc, APITimeoutError):
        return "api_timeout"
    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return "api_connection_error"
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return "rate_limit"
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return "rate_limit"
    if status_code in {408, 504}:
        return "api_timeout"
    if status_code in {500, 502, 503}:
        return "api_connection_error"
    return "unknown_error"


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        request_timeout: float = 120,
        max_api_retries: int = 5,
        retry_backoff: str = "exponential",
        retry_delay: float = 5,
        cache: DiskCache | None = None,
        dry_run: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_api_retries = max_api_retries
        self.retry_backoff = retry_backoff
        self.retry_delay = retry_delay
        self.cache = cache or DiskCache(".cache/mars_full", enabled=False)
        self.dry_run = dry_run
        self.stats = ApiStats()
        self.client = None
        if not dry_run:
            api_key = os.getenv(api_key_env)
            base_url = os.getenv(base_url_env)
            if not api_key:
                raise ApiCallError(
                    "unknown_error",
                    f"Missing API key environment variable: {api_key_env}",
                )
            self.client = OpenAI(
                api_key=api_key, base_url=base_url, timeout=request_timeout
            )

    def _sleep_seconds(self, attempt: int) -> float:
        if self.retry_backoff == "exponential":
            return min(self.retry_delay * (2 ** max(attempt - 1, 0)), 60)
        return self.retry_delay

    def _cache_payload(
        self, messages: list[dict[str, str]], extra: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            **extra,
        }

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        method: str,
        task_id: str,
        iteration: int,
        question: str = "",
        max_tokens: int | None = None,
    ) -> str:
        started_at = time.time()
        estimated_prompt_tokens = estimate_message_tokens(messages)
        extra = {
            "method": method,
            "task_id": task_id,
            "iteration": iteration,
            "question": question,
        }
        payload = self._cache_payload(messages, extra)
        cached = self.cache.get(payload)
        if cached is not None:
            self.stats.cache_hits += 1
            content = str(cached.get("content", ""))
            self._record_call(
                method=method,
                task_id=task_id,
                iteration=iteration,
                question=question,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=time.time() - started_at,
                cache_hit=True,
                error_type="",
                estimated_prompt_tokens=estimated_prompt_tokens,
            )
            return content

        if self.dry_run:
            content = self._mock_response(messages, task_id=task_id)
            self.cache.set(payload, {"content": content})
            self._record_call(
                method=method,
                task_id=task_id,
                iteration=iteration,
                question=question,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=time.time() - started_at,
                cache_hit=False,
                error_type="dry_run",
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_completion_tokens=estimate_tokens(content),
            )
            return content

        attempt = 1
        while True:
            try:
                self.stats.api_calls += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                if usage is not None:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                else:
                    prompt_tokens = estimated_prompt_tokens
                    completion_tokens = estimate_tokens(content)
                self.stats.tokens_prompt += prompt_tokens
                self.stats.tokens_completion += completion_tokens
                self.cache.set(payload, {"content": content})
                self._record_call(
                    method=method,
                    task_id=task_id,
                    iteration=iteration,
                    question=question,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_seconds=time.time() - started_at,
                    cache_hit=False,
                    error_type="",
                    estimated_prompt_tokens=estimated_prompt_tokens,
                )
                return content
            except Exception as exc:
                error_type = classify_api_error(exc)
                self.stats.api_errors += 1
                self.stats.error_records.append(
                    {
                        "error_type": error_type,
                        "message": str(exc),
                        "attempt": attempt,
                        "method": method,
                        "task_id": task_id,
                    }
                )
                if self.max_api_retries >= 0 and attempt >= self.max_api_retries:
                    self._record_call(
                        method=method,
                        task_id=task_id,
                        iteration=iteration,
                        question=question,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_seconds=time.time() - started_at,
                        cache_hit=False,
                        error_type=error_type,
                        estimated_prompt_tokens=estimated_prompt_tokens,
                    )
                    raise ApiCallError(error_type, str(exc)) from exc
                self.stats.retries += 1
                time.sleep(self._sleep_seconds(attempt))
                attempt += 1

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        method: str,
        task_id: str,
        iteration: int,
        question: str = "",
        max_tokens: int | None = None,
    ) -> str:
        return self.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            method=method,
            task_id=task_id,
            iteration=iteration,
            question=question,
            max_tokens=max_tokens,
        )

    def _mock_response(self, messages: list[dict[str, str]], task_id: str) -> str:
        text = "\n".join(message.get("content", "") for message in messages).lower()
        if "final answer: (x)" in text or "single best option letter" in text:
            return "Final answer: (A)"
        if "final answer: yes" in text and "final answer: no" in text:
            return "Final answer: yes"
        if "final answer: valid" in text and "final answer: invalid" in text:
            return "Final answer: valid"
        if "final answer: true" in text and "final answer: false" in text:
            return "Final answer: true"
        if "final answer: <number>" in text or task_id == "gsm8k":
            return "Final answer: 0"
        if "yes" in text and "no" in text:
            return "Final answer: yes"
        if "valid" in text and "invalid" in text:
            return "Final answer: valid"
        if "true" in text and "false" in text:
            return "Final answer: true"
        if "sub-goal" in text or "subgoal" in text:
            return "1. Clarify the answer format.\n2. Solve carefully before giving the final answer."
        if "candidate" in text or "prompt" in text:
            return "Be careful and end with the required final answer format."
        return "Final answer: (A)"

    def _record_call(
        self,
        *,
        method: str,
        task_id: str,
        iteration: int,
        question: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        cache_hit: bool,
        error_type: str,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int | None = None,
    ) -> None:
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        self.stats.call_records.append(
            {
                "model": self.model,
                "method": method,
                "task_id": task_id,
                "iteration": iteration,
                "question_hash": question_hash,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_completion_tokens": estimated_completion_tokens
                if estimated_completion_tokens is not None
                else completion_tokens,
                "latency_seconds": round(latency_seconds, 6),
                "cache_hit": cache_hit,
                "error_type": error_type,
            }
        )
