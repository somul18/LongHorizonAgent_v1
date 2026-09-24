"""Cheap, dependency-free token estimate.

Real tokenizers differ per model; ~4 chars/token is close enough for budgeting
and for comparing architectures against each other with the same yardstick.
"""


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0
