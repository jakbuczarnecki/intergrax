# © Artur Czarnecki. All rights reserved.

"""Deterministic sequential runtime invariant runner."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.runtime_invariants import (
    RuntimeInvariantEvaluationContext,
    RuntimeInvariantEvaluationRequest,
    RuntimeInvariantPackRef,
    RuntimeInvariantReport,
    RuntimeInvariantResult,
    RuntimeInvariantRule,
    RuntimeInvariantRuleEvaluation,
    RuntimeInvariantRulePack,
    RuntimeInvariantStatus,
    aggregate_runtime_invariant_overall_status,
    runtime_invariant_pack_sort_key,
    summarize_runtime_invariant_results,
)

_EVALUATION_FAILED_SUMMARY = "invariant evaluation failed"
_EVALUATION_ERROR_CODE = "RI_EVALUATION_ERROR"
_RULE_OUTPUT_TYPE_MISMATCH_CODE = "RI_RULE_OUTPUT_TYPE_MISMATCH"
_RESULT_CONTRACT_MISMATCH_CODE = "RI_RESULT_CONTRACT_MISMATCH"
_CONTRACT_MISMATCH_SUMMARY = "invariant result contract mismatch"

_ALLOWED_RULE_STATUSES = frozenset(
    {
        RuntimeInvariantStatus.PASS,
        RuntimeInvariantStatus.VIOLATION,
        RuntimeInvariantStatus.NOT_APPLICABLE,
    },
)


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
        evaluated_at: datetime,
    ) -> RuntimeInvariantReport:
        selected = [
            rule
            for rule in self._rules
            if request.selection.includes_rule(domain=rule.domain, rule_id=rule.rule_id)
        ]
        results: list[RuntimeInvariantResult] = []
        for rule in selected:
            results.append(_evaluate_rule(rule, context))
        result_tuple = tuple(results)
        sorted_packs = tuple(sorted(self._packs, key=runtime_invariant_pack_sort_key))
        pack_refs = tuple(
            RuntimeInvariantPackRef(
                pack_id=pack.pack_id,
                pack_version=pack.pack_version,
                domain=pack.domain,
            )
            for pack in sorted_packs
        )
        return RuntimeInvariantReport(
            evaluation_id=context.evaluation_id,
            correlation_id=context.correlation_id,
            evaluated_at=evaluated_at,
            mode=context.mode,
            packs=pack_refs,
            results=result_tuple,
            summary=summarize_runtime_invariant_results(result_tuple),
            overall_status=aggregate_runtime_invariant_overall_status(result_tuple),
            execution_id=context.execution_id,
        )


def _authoritative_result(
    rule: RuntimeInvariantRule,
    context: RuntimeInvariantEvaluationContext,
    *,
    evaluation: RuntimeInvariantRuleEvaluation,
) -> RuntimeInvariantResult:
    return RuntimeInvariantResult(
        rule_id=rule.rule_id,
        domain=rule.domain,
        rule_version=rule.rule_version,
        severity=rule.severity,
        status=evaluation.status,
        summary=evaluation.summary,
        evaluation_id=context.evaluation_id,
        correlation_id=context.correlation_id,
        diagnostic_code=evaluation.diagnostic_code,
        evidence_refs=evaluation.evidence_refs,
    )


def _evaluation_error_result(
    rule: RuntimeInvariantRule,
    context: RuntimeInvariantEvaluationContext,
    *,
    diagnostic_code: str,
    summary: str,
) -> RuntimeInvariantResult:
    return RuntimeInvariantResult(
        rule_id=rule.rule_id,
        domain=rule.domain,
        rule_version=rule.rule_version,
        severity=rule.severity,
        status=RuntimeInvariantStatus.EVALUATION_ERROR,
        summary=summary,
        evaluation_id=context.evaluation_id,
        correlation_id=context.correlation_id,
        diagnostic_code=diagnostic_code,
    )


def _evaluate_rule(
    rule: RuntimeInvariantRule,
    context: RuntimeInvariantEvaluationContext,
) -> RuntimeInvariantResult:
    try:
        raw = rule.evaluate(context)
    except Exception:
        return _evaluation_error_result(
            rule,
            context,
            diagnostic_code=_EVALUATION_ERROR_CODE,
            summary=_EVALUATION_FAILED_SUMMARY,
        )
    if not isinstance(raw, RuntimeInvariantRuleEvaluation):
        return _evaluation_error_result(
            rule,
            context,
            diagnostic_code=_RULE_OUTPUT_TYPE_MISMATCH_CODE,
            summary=_EVALUATION_FAILED_SUMMARY,
        )
    if raw.status not in _ALLOWED_RULE_STATUSES:
        return _evaluation_error_result(
            rule,
            context,
            diagnostic_code=_RESULT_CONTRACT_MISMATCH_CODE,
            summary=_CONTRACT_MISMATCH_SUMMARY,
        )
    return _authoritative_result(rule, context, evaluation=raw)


__all__ = ["DefaultRuntimeInvariantRunner"]
