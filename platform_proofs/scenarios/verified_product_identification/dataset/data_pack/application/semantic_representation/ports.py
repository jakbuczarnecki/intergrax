"""Provider-neutral ports for semantic representation budgeting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class TokenEstimatorPort(Protocol):
    """Estimate token count for embedding input without binding to a vendor."""

    def estimate_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class CharacterRatioTokenEstimator:
    """Heuristic estimator (~4 characters per token for multilingual catalog text)."""

    chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        if self.chars_per_token <= 0.0:
            msg = "chars_per_token must be > 0"
            raise ValueError(msg)

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))
