# © Artur Czarnecki. All rights reserved.

"""Injected clock for runtime invariant evaluation."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.runtime_invariants import RuntimeInvariantEvaluationClock


class SystemRuntimeInvariantEvaluationClock:
    """Default wall-clock (timezone-aware UTC)."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


__all__ = ["SystemRuntimeInvariantEvaluationClock"]
