# © Artur Czarnecki. All rights reserved.

"""Evaluation identity issuance for runtime invariants."""

from __future__ import annotations

from itertools import count

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantEvaluationId,
    RuntimeInvariantEvaluationIdFactory,
)

_EVALUATION_ID_PREFIX = "ri-eval"


class MonotonicRuntimeInvariantEvaluationIdFactory:
    """Deterministic opaque IDs for tests and qualification."""

    __slots__ = ("_counter",)

    def __init__(self, *, start: int = 1) -> None:
        self._counter = count(start)

    def mint_evaluation_id(self) -> RuntimeInvariantEvaluationId:
        return f"{_EVALUATION_ID_PREFIX}-{next(self._counter)}"


class DefaultRuntimeInvariantEvaluationIdFactory:
    """Process-scoped monotonic factory (no uuid in rules)."""

    def mint_evaluation_id(self) -> RuntimeInvariantEvaluationId:
        return _default_factory.mint_evaluation_id()


_default_factory = MonotonicRuntimeInvariantEvaluationIdFactory()


__all__ = [
    "DefaultRuntimeInvariantEvaluationIdFactory",
    "MonotonicRuntimeInvariantEvaluationIdFactory",
    "RuntimeInvariantEvaluationIdFactory",
]
