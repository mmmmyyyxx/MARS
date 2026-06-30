from __future__ import annotations


def estimate_tokens(text: object) -> int:
    """Small dependency-free token estimate used for cost reporting."""
    if text is None:
        return 0
    value = str(text)
    if not value:
        return 0
    return max(1, int(len(value) / 4))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(message.get("content", "")) + 4 for message in messages)
