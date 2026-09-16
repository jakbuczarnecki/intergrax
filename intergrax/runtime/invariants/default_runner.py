# © Artur Czarnecki. All rights reserved.

"""Deterministic sequential runtime invariant runner."""

from __future__ import annotations

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantEvaluationRequest,
    RuntimeInvariantReport,
    RuntimeInvariantResult,
    RuntimeInvariantRule,
    RuntimeInvariantRulePack,
    RuntimeInvariantStatus,
    aggregate_runtime_invariant_overall_status,
    summarize_runtime_invariant_results,
)

_EVALUATION_FAILED_SUMMARY = "invariant evaluation failed"
_EVALUATION_ERROR_CODE = "RI_EVALUATION_ERROR"


class DefaultRuntimeInvariantRunner:
    """Read-only, sequential, deterministic rule execution."""

    __slots__ = ("_packs", "_rules")

    def __init__(
        self,
        *,
        rule_packs: tuple[RuntimeInvariantRulePack, ...],
        rules: tuple[RuntimeInvariantRule, ...],
    ) -> None:
        self._packs = rule_packs
        self._rules = rules

    def evaluate(
        self,
        request: RuntimeInvariantEvaluationRequest,
        *,
        context: RuntimeInvariantEvaluationContext,
        evaluated_at: object,
    ) -> RuntimeInvariantReport:
        from datetime import datetime

        if not isinstance(evaluated_at, datetime):
            raise TypeError("evaluated_at must be datetime")
        selected = [
            rule
            for rule in self._rules
            if request.selection.includes_rule(domain=rule.domain, rule_id=rule.rule_id)
        ]
        results: list[RuntimeInvariantResult] = []
        for rule in selected:
            results.append(_evaluate_rule(rule, context))
        result_tuple = tuple(results)
        pack_ids = tuple(pack.pack_id for pack in self._packs)
        pack_versions = tuple(pack.pack_version for pack in self._packs)
        return RuntimeInvariantReport(
            evaluation_id=context.evaluation_id,
            correlation_id=context.correlation_id,
            evaluated_at=evaluated_at,
            mode=context.mode,
            pack_ids=pack_ids,
            pack_versions=pack_versions,
            results=result_tuple,
            summary=summarize_runtime_invariant_results(result_tuple),
            overall_status=aggregate_runtime_invariant_overall_status(result_tuple),
            execution_id=context.execution_id,
        )


def _evaluate_rule(
    rule: RuntimeInvariantRule,
    context: RuntimeInvariantEvaluationContext,
) -> RuntimeInvariantResult:
    try:
        return rule.evaluate(context)
    except Exception:
        return RuntimeInvariantResult(
            rule_id=rule.rule_id,
            domain=rule.domain,
            rule_version=rule.rule_version,
            severity=rule.severity,
            status=RuntimeInvariantStatus.EVALUATION_ERROR,
            summary=_EVALUATION_FAILED_SUMMARY,
            evaluation_id=context.evaluation_id,
            correlation_id=context.correlation_id,
            diagnostic_code=_EVALUATION_ERROR_CODE,
        )


__all__ = ["DefaultRuntimeInvariantRunner"]
