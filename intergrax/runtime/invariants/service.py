# © Artur Czarnecki. All rights reserved.

"""Runtime invariant service — evaluation and aggregation only."""

from __future__ import annotations

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantEvaluationClock,
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantEvaluationIdFactory,
    RuntimeInvariantEvaluationRequest,
    RuntimeInvariantReport,
    RuntimeInvariantRulePack,
    RuntimeInvariantRunner,
)
from intergrax.runtime.invariants.composition import validate_runtime_invariant_rule_packs


class RuntimeInvariantService:
    """Composition-owned invariant evaluation — no global registry."""

    __slots__ = ("_clock", "_id_factory", "_packs", "_runner")

    def __init__(
        self,
        *,
        rule_packs: tuple[RuntimeInvariantRulePack, ...],
        clock: RuntimeInvariantEvaluationClock,
        evaluation_id_factory: RuntimeInvariantEvaluationIdFactory,
        runner: RuntimeInvariantRunner,
    ) -> None:
        validate_runtime_invariant_rule_packs(rule_packs)
        self._packs = rule_packs
        self._clock = clock
        self._id_factory = evaluation_id_factory
        self._runner = runner

    @property
    def rule_packs(self) -> tuple[RuntimeInvariantRulePack, ...]:
        return self._packs

    def evaluate(
        self,
        request: RuntimeInvariantEvaluationRequest | None = None,
    ) -> RuntimeInvariantReport:
        invocation = request or RuntimeInvariantEvaluationRequest()
        evaluation_id = self._id_factory.mint_evaluation_id()
        correlation_id = invocation.correlation_id or evaluation_id
        context = RuntimeInvariantEvaluationContext(
            evaluation_id=evaluation_id,
            correlation_id=correlation_id,
            requested_at=self._clock.now(),
            mode=invocation.mode,
            execution_id=invocation.execution_id,
        )
        return self._runner.evaluate(
            invocation,
            context=context,
            evaluated_at=self._clock.now(),
        )


__all__ = ["RuntimeInvariantService"]
