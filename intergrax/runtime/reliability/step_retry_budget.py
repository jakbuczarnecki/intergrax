# © Artur Czarnecki. All rights reserved.

"""Per-step retry budget enforcement (IDEAL-22.6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepRetryBudget:
    max_retries: int
    retries_used: int = 0

    def can_retry(self) -> bool:
        return self.retries_used < self.max_retries

    def consume(self) -> StepRetryBudget:
        if not self.can_retry():
            raise ValueError("step retry budget exhausted")
        return StepRetryBudget(
            max_retries=self.max_retries,
            retries_used=self.retries_used + 1,
        )
