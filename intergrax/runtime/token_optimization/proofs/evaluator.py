# © Artur Czarnecki. All rights reserved.

"""Deterministic, evaluation-only TOKEN-10G hard-gate evaluator."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from intergrax.runtime.token_optimization.proofs.contracts import (
    UniversalProofCaseResult,
    UniversalProofRunResult,
)
from intergrax.runtime.token_optimization.proofs.evaluation_contracts import (
    CacheAttribution,
    CacheExpectationMode,
    CaseEvaluation,
    EVALUATION_GATE_IDS,
    CorpusCase,
    EvaluationConfiguration,
    EvaluationConfigurationError,
    EvaluationGateRequirement,
    EvaluationProfile,
    GateResult,
    GateStatus,
    MeasurementRequirement,
    ProofCorpus,
    ProviderCacheEvidence,
    UniversalProofEvaluation,
)

_KNOWN_GATE_IDS = frozenset(EVALUATION_GATE_IDS)
_SAFE_SUMMARY_RE = re.compile(r"^[A-Za-z0-9._:/=, -]{1,512}$")


@dataclass(frozen=True, slots=True)
class _GateContext:
    case: CorpusCase
    result: UniversalProofCaseResult
    run: UniversalProofRunResult
    config: EvaluationConfiguration
    cache: ProviderCacheEvidence | None
    cases_by_id: Mapping[str, UniversalProofCaseResult]


def _requirement(
    gate_id: str, context: _GateContext
) -> EvaluationGateRequirement:
    return context.config.requirement_for(gate_id)


def _profile_not_applicable(
    gate_ids: Iterable[str], context: _GateContext
) -> bool:
    return all(
        _requirement(gate_id, context)
        is EvaluationGateRequirement.NOT_APPLICABLE
        for gate_id in gate_ids
    )


def _summary(value: object) -> str:
    text = str(value)
    if not _SAFE_SUMMARY_RE.fullmatch(text):
        return "<redacted>"
    return text


def _gate(
    gate_id: str,
    status: GateStatus,
    context: _GateContext,
    reason_code: str,
    expected: object,
    actual: object,
) -> GateResult:
    if gate_id not in _KNOWN_GATE_IDS:
        raise EvaluationConfigurationError("UNKNOWN_GATE_ID")
    return GateResult(
        gate_id=gate_id,
        status=status,
        case_id=context.case.case_id,
        reason_code=reason_code,
        expected_safe_summary=_summary(expected),
        actual_safe_summary=_summary(actual),
        required=(
            _requirement(gate_id, context)
            in {
                EvaluationGateRequirement.REQUIRED,
                EvaluationGateRequirement.UNAVAILABLE_ALLOWED,
            }
        ),
    )


def _pass(
    gate_id: str, context: _GateContext, expected: object, actual: object
) -> GateResult:
    return _gate(
        gate_id, GateStatus.PASS, context, "EXPECTATION_SATISFIED", expected, actual
    )


def _fail(
    gate_id: str, context: _GateContext, reason: str, expected: object, actual: object
) -> GateResult:
    return _gate(gate_id, GateStatus.FAIL, context, reason, expected, actual)


def _unavailable(gate_id: str, context: _GateContext, expected: object) -> GateResult:
    return _gate(
        gate_id,
        GateStatus.UNAVAILABLE,
        context,
        "EVIDENCE_UNAVAILABLE",
        expected,
        "unavailable",
    )


def _not_applicable(
    gate_id: str, context: _GateContext, reason: str = "NOT_APPLICABLE"
) -> GateResult:
    return _gate(
        gate_id,
        GateStatus.NOT_APPLICABLE,
        context,
        reason,
        "not_applicable",
        "not_applicable",
    )


def _allowed(
    gate_id: str,
    value: str | None,
    allowed: frozenset[str],
    context: _GateContext,
) -> GateResult:
    if not allowed:
        return _not_applicable(gate_id, context)
    if value is None:
        return _unavailable(gate_id, context, sorted(allowed))
    return (
        _pass(gate_id, context, sorted(allowed), value)
        if value in allowed
        else _fail(gate_id, context, "VALUE_NOT_ALLOWED", sorted(allowed), value)
    )


def _router_gates(context: _GateContext) -> list[GateResult]:
    router_gate_ids = (
        "ROUTER_STATUS",
        "ROUTER_CONFIGURATION",
        "ROUTER_REASON",
        "ROUTER_REVIEW_REQUIREMENT",
        "ROUTER_CONFIDENCE",
        "ROUTER_TRANSPORT",
    )
    if _profile_not_applicable(router_gate_ids, context):
        return [_not_applicable(gate_id, context) for gate_id in router_gate_ids]
    expectation = context.case.router
    evidence = context.result.router_evidence
    gates: list[GateResult] = []
    if not expectation.allowed_statuses:
        gates.append(_not_applicable("ROUTER_STATUS", context))
    elif evidence.status is None:
        gates.append(
            _unavailable("ROUTER_STATUS", context, sorted(expectation.allowed_statuses))
        )
    elif (
        context.result.router_status is not None
        and context.result.router_status != evidence.status
    ):
        gates.append(
            _fail(
                "ROUTER_STATUS",
                context,
                "EVIDENCE_CONTRADICTION",
                evidence.status,
                context.result.router_status,
            )
        )
    else:
        gates.append(
            _allowed(
                "ROUTER_STATUS", evidence.status, expectation.allowed_statuses, context
            )
        )
    gates.append(
        _allowed(
            "ROUTER_CONFIGURATION",
            evidence.configuration_id,
            expectation.allowed_configuration_ids,
            context,
        )
    )
    if (
        evidence.configuration_id is not None
        and context.result.selected_configuration_id is not None
        and evidence.configuration_id != context.result.selected_configuration_id
    ):
        gates[-1] = _fail(
            "ROUTER_CONFIGURATION",
            context,
            "EVIDENCE_CONTRADICTION",
            evidence.configuration_id,
            context.result.selected_configuration_id,
        )
    gates.append(
        _allowed(
            "ROUTER_REASON",
            evidence.reason_code,
            expectation.allowed_reason_codes,
            context,
        )
    )
    if expectation.review_required is None:
        gates.append(_not_applicable("ROUTER_REVIEW_REQUIREMENT", context))
    elif evidence.review_required is None:
        gates.append(
            _unavailable(
                "ROUTER_REVIEW_REQUIREMENT", context, expectation.review_required
            )
        )
    elif evidence.review_required != expectation.review_required:
        gates.append(
            _fail(
                "ROUTER_REVIEW_REQUIREMENT",
                context,
                "REVIEW_REQUIREMENT_MISMATCH",
                expectation.review_required,
                evidence.review_required,
            )
        )
    else:
        gates.append(
            _pass(
                "ROUTER_REVIEW_REQUIREMENT",
                context,
                expectation.review_required,
                evidence.review_required,
            )
        )
    if (
        expectation.confidence_minimum is None
        and expectation.confidence_maximum is None
    ):
        gates.append(_not_applicable("ROUTER_CONFIDENCE", context))
    elif evidence.confidence is None:
        gates.append(_unavailable("ROUTER_CONFIDENCE", context, "bounded confidence"))
    elif (
        expectation.confidence_minimum is not None
        and evidence.confidence < expectation.confidence_minimum
    ) or (
        expectation.confidence_maximum is not None
        and evidence.confidence > expectation.confidence_maximum
    ):
        gates.append(
            _fail(
                "ROUTER_CONFIDENCE",
                context,
                "CONFIDENCE_OUT_OF_RANGE",
                (
                    expectation.confidence_minimum,
                    expectation.confidence_maximum,
                ),
                evidence.confidence,
            )
        )
    else:
        gates.append(
            _pass(
                "ROUTER_CONFIDENCE", context, "bounded confidence", evidence.confidence
            )
        )
    transport_gate = _allowed(
        "ROUTER_TRANSPORT",
        evidence.transport,
        expectation.allowed_transport,
        context,
    )
    if (
        transport_gate.status is GateStatus.PASS
        and expectation.structured_output_fallback is not None
        and evidence.structured_output_fallback_used
        != expectation.structured_output_fallback
    ):
        transport_gate = _fail(
            "ROUTER_TRANSPORT",
            context,
            "STRUCTURED_OUTPUT_FALLBACK_MISMATCH",
            expectation.structured_output_fallback,
            evidence.structured_output_fallback_used,
        )
    gates.append(transport_gate)
    if expectation.allowed_risk:
        risk_gate = _allowed(
            "ROUTER_REASON", evidence.risk, expectation.allowed_risk, context
        )
        if risk_gate.status is GateStatus.FAIL:
            gates[2] = _fail(
                "ROUTER_REASON",
                context,
                "RISK_NOT_ALLOWED",
                sorted(expectation.allowed_risk),
                evidence.risk,
            )
    return gates


def _pipeline_gates(context: _GateContext) -> list[GateResult]:
    pipeline_gate_ids = (
        "PIPELINE_COMPLETION",
        "PIPELINE_REQUIRED_LAYERS",
        "PIPELINE_FORBIDDEN_LAYERS",
        "PIPELINE_FALLBACK",
        "PIPELINE_VALIDATION",
        "PIPELINE_REQUIRED_FAILURE",
    )
    if _profile_not_applicable(pipeline_gate_ids, context):
        return [_not_applicable(gate_id, context) for gate_id in pipeline_gate_ids]
    expected = context.case.pipeline
    evidence = context.result.pipeline_evidence
    gates: list[GateResult] = []
    if expected.expected_completion is None:
        gates.append(_not_applicable("PIPELINE_COMPLETION", context))
    elif evidence.completed is None:
        gates.append(
            _unavailable("PIPELINE_COMPLETION", context, expected.expected_completion)
        )
    elif (
        context.result.pipeline_status
        == ("completed" if evidence.completed else "failed")
    ) is False:
        gates.append(
            _fail(
                "PIPELINE_COMPLETION",
                context,
                "EVIDENCE_CONTRADICTION",
                expected.expected_completion,
                context.result.pipeline_status,
            )
        )
    elif evidence.completed != expected.expected_completion:
        gates.append(
            _fail(
                "PIPELINE_COMPLETION",
                context,
                "COMPLETION_MISMATCH",
                expected.expected_completion,
                evidence.completed,
            )
        )
    else:
        gates.append(
            _pass(
                "PIPELINE_COMPLETION",
                context,
                expected.expected_completion,
                evidence.completed,
            )
        )
    applied = set(context.result.applied_layer_ids)
    missing = sorted(expected.required_layer_ids - applied)
    unexpected = (
        sorted(applied - expected.allowed_layer_ids)
        if expected.allowed_layer_ids
        else []
    )
    if expected.required_layer_ids or expected.allowed_layer_ids:
        gates.append(
            _pass(
                "PIPELINE_REQUIRED_LAYERS",
                context,
                sorted(expected.required_layer_ids or expected.allowed_layer_ids),
                sorted(applied),
            )
            if not missing and not unexpected
            else _fail(
                "PIPELINE_REQUIRED_LAYERS",
                context,
                "REQUIRED_OR_ALLOWED_LAYER_MISMATCH",
                sorted(expected.required_layer_ids or expected.allowed_layer_ids),
                sorted(applied),
            )
        )
    else:
        gates.append(_not_applicable("PIPELINE_REQUIRED_LAYERS", context))
    if expected.forbidden_layer_ids:
        forbidden = sorted(applied & expected.forbidden_layer_ids)
        gates.append(
            _pass(
                "PIPELINE_FORBIDDEN_LAYERS",
                context,
                sorted(expected.forbidden_layer_ids),
                sorted(applied),
            )
            if not forbidden
            else _fail(
                "PIPELINE_FORBIDDEN_LAYERS",
                context,
                "FORBIDDEN_LAYER_APPLIED",
                sorted(expected.forbidden_layer_ids),
                forbidden,
            )
        )
    else:
        gates.append(_not_applicable("PIPELINE_FORBIDDEN_LAYERS", context))
    if expected.expected_fallback is None:
        gates.append(_not_applicable("PIPELINE_FALLBACK", context))
    elif evidence.fallback_applied is None:
        gates.append(
            _unavailable("PIPELINE_FALLBACK", context, expected.expected_fallback)
        )
    elif evidence.fallback_applied != expected.expected_fallback:
        gates.append(
            _fail(
                "PIPELINE_FALLBACK",
                context,
                "FALLBACK_MISMATCH",
                expected.expected_fallback,
                evidence.fallback_applied,
            )
        )
    else:
        gates.append(
            _pass(
                "PIPELINE_FALLBACK",
                context,
                expected.expected_fallback,
                evidence.fallback_applied,
            )
        )
    reason_missing = (
        bool(expected.allowed_validation_reason_codes)
        and evidence.validation_reason_code is None
    )
    reason_mismatch = (
        bool(expected.allowed_validation_reason_codes)
        and evidence.validation_reason_code is not None
        and evidence.validation_reason_code
        not in expected.allowed_validation_reason_codes
    )
    if (
        expected.expected_validation_status is None
        and not expected.allowed_validation_reason_codes
    ):
        gates.append(_not_applicable("PIPELINE_VALIDATION", context))
    elif evidence.validation_status is None or reason_missing:
        gates.append(
            _unavailable(
                "PIPELINE_VALIDATION",
                context,
                expected.expected_validation_status or "validation",
            )
        )
    elif (
        expected.expected_validation_status is not None
        and evidence.validation_status != expected.expected_validation_status
    ):
        gates.append(
            _fail(
                "PIPELINE_VALIDATION",
                context,
                "VALIDATION_MISMATCH",
                expected.expected_validation_status,
                evidence.validation_status,
            )
        )
    elif reason_mismatch:
        gates.append(
            _fail(
                "PIPELINE_VALIDATION",
                context,
                "VALIDATION_REASON_NOT_ALLOWED",
                sorted(expected.allowed_validation_reason_codes),
                evidence.validation_reason_code,
            )
        )
    else:
        gates.append(
            _pass(
                "PIPELINE_VALIDATION",
                context,
                expected.expected_validation_status or "allowed validation",
                evidence.validation_status,
            )
        )
    if expected.required_layer_failure_expected is None:
        gates.append(_not_applicable("PIPELINE_REQUIRED_FAILURE", context))
    elif evidence.required_layer_failure is None:
        gates.append(
            _pass("PIPELINE_REQUIRED_FAILURE", context, False, "absent")
            if expected.required_layer_failure_expected is False
            else _unavailable("PIPELINE_REQUIRED_FAILURE", context, True)
        )
    elif expected.required_layer_failure_expected is False:
        gates.append(
            _fail(
                "PIPELINE_REQUIRED_FAILURE",
                context,
                "REQUIRED_LAYER_FAILURE",
                False,
                evidence.required_layer_failure,
            )
        )
    else:
        gates.append(
            _pass(
                "PIPELINE_REQUIRED_FAILURE",
                context,
                True,
                evidence.required_layer_failure,
            )
        )
    return gates


def _router_integrity_gate(context: _GateContext) -> GateResult:
    result = context.result
    evidence = result.router_evidence
    fields_consistent = (
        result.router_status == evidence.status
        and result.router_reason == evidence.reason_code
        and result.selected_configuration_id == evidence.configuration_id
    )
    typed_fields = (
        evidence.status is not None
        and evidence.configuration_id is not None
        and evidence.reason_code is not None
        and type(evidence.review_required) is bool
        and evidence.confidence is not None
        and evidence.transport is not None
        and type(evidence.structured_output_fallback_used) is bool
    )
    offline_transport = (
        context.config.profile is not EvaluationProfile.OFFLINE_COMPOSITION
        or context.run.run_mode != "offline_smoke"
        or (
            evidence.transport == "structured_output"
            and evidence.structured_output_fallback_used is True
        )
    )
    configured_decision = context.config.configured_offline_decision
    configured = (
        configured_decision is None
        or context.config.profile is not EvaluationProfile.OFFLINE_COMPOSITION
        or context.run.run_mode != "offline_smoke"
        or (
            evidence.configuration_id == configured_decision
            and result.selected_configuration_id == configured_decision
        )
    )
    if fields_consistent and typed_fields and offline_transport and configured:
        return _pass(
            "ROUTER_EVIDENCE_INTEGRITY",
            context,
            "consistent typed router evidence",
            "consistent typed router evidence",
        )
    reason = (
        "EVIDENCE_CONTRADICTION"
        if not fields_consistent
        else "OFFLINE_DECISION_OR_TRANSPORT_MISMATCH"
        if not offline_transport or not configured
        else "ROUTER_EVIDENCE_INCOMPLETE"
    )
    return _fail(
        "ROUTER_EVIDENCE_INTEGRITY",
        context,
        reason,
        "typed consistent evidence",
        "invalid or incomplete evidence",
    )


def _pipeline_integrity_gate(context: _GateContext) -> GateResult:
    result = context.result
    evidence = result.pipeline_evidence
    status_matches = (
        evidence.completed is not None
        and result.pipeline_status
        == ("completed" if evidence.completed else "failed")
    )
    receipt_matches = (
        evidence.receipt_completion_status is None
        or evidence.receipt_completion_status == evidence.completed
    )
    layer_ids_are_safe = bool(
        all(
            isinstance(layer_id, str)
            and bool(re.fullmatch(r"[A-Za-z0-9._-]{1,128}", layer_id))
            for layer_id in result.applied_layer_ids
        )
    ) or not result.applied_layer_ids
    if status_matches and receipt_matches and layer_ids_are_safe:
        return _pass(
            "PIPELINE_EVIDENCE_INTEGRITY",
            context,
            "consistent pipeline execution evidence",
            "consistent pipeline execution evidence",
        )
    return _fail(
        "PIPELINE_EVIDENCE_INTEGRITY",
        context,
        "PIPELINE_EVIDENCE_INCONSISTENT",
        "consistent execution evidence",
        "invalid or contradictory evidence",
    )


def _protected_gates(context: _GateContext) -> list[GateResult]:
    expected = context.case.protected
    evidence = context.result.protected_region_evidence
    gates: list[GateResult] = []
    if expected.expected_input_count is None:
        gates.append(_not_applicable("PROTECTED_REGION_COUNT", context))
    elif evidence.input_protected_region_count != expected.expected_input_count:
        gates.append(
            _fail(
                "PROTECTED_REGION_COUNT",
                context,
                "PROTECTED_COUNT_MISMATCH",
                expected.expected_input_count,
                evidence.input_protected_region_count,
            )
        )
    else:
        gates.append(
            _pass(
                "PROTECTED_REGION_COUNT",
                context,
                expected.expected_input_count,
                evidence.input_protected_region_count,
            )
        )
    if expected.expected_preserved_count is None:
        gates.append(_not_applicable("PROTECTED_REGION_PRESERVATION", context))
    elif evidence.preserved_protected_region_count != expected.expected_preserved_count:
        gates.append(
            _fail(
                "PROTECTED_REGION_PRESERVATION",
                context,
                "PROTECTED_PRESERVATION_MISMATCH",
                expected.expected_preserved_count,
                evidence.preserved_protected_region_count,
            )
        )
    else:
        gates.append(
            _pass(
                "PROTECTED_REGION_PRESERVATION",
                context,
                expected.expected_preserved_count,
                evidence.preserved_protected_region_count,
            )
        )
    if expected.expected_validation_status is None:
        gates.append(_not_applicable("PROTECTED_REGION_VALIDATION", context))
    elif (
        evidence.protected_region_validation_status
        != expected.expected_validation_status
    ):
        gates.append(
            _fail(
                "PROTECTED_REGION_VALIDATION",
                context,
                "PROTECTED_VALIDATION_MISMATCH",
                expected.expected_validation_status,
                evidence.protected_region_validation_status,
            )
        )
    else:
        gates.append(
            _pass(
                "PROTECTED_REGION_VALIDATION",
                context,
                expected.expected_validation_status,
                evidence.protected_region_validation_status,
            )
        )
    if expected.digest_equality_required:
        if (
            evidence.input_identity_digest is None
            or evidence.preserved_identity_digest is None
        ):
            gates[-1] = _unavailable(
                "PROTECTED_REGION_VALIDATION", context, "digest equality"
            )
        elif evidence.input_identity_digest != evidence.preserved_identity_digest:
            gates[-1] = _fail(
                "PROTECTED_REGION_VALIDATION",
                context,
                "PROTECTED_DIGEST_MISMATCH",
                "equal digests",
                "different digests",
            )
    return gates


def _measurement_gate(
    gate_id: str,
    requirement: MeasurementRequirement,
    measurement,
    context: _GateContext,
) -> GateResult:
    if _requirement(gate_id, context) is EvaluationGateRequirement.NOT_APPLICABLE:
        return _not_applicable(gate_id, context)
    if requirement is MeasurementRequirement.NOT_APPLICABLE:
        return _not_applicable(gate_id, context)
    if not measurement.available:
        return _unavailable(gate_id, context, requirement.value)
    return _pass(gate_id, context, requirement.value, measurement.value)


def _measurement_gates(context: _GateContext) -> list[GateResult]:
    expected = context.case.measurement
    gates = [
        _measurement_gate(
            "BASELINE_MEASUREMENT",
            expected.baseline,
            context.result.baseline_measurement,
            context,
        ),
        _measurement_gate(
            "OPTIMIZED_MEASUREMENT",
            expected.optimized,
            context.result.optimized_measurement,
            context,
        ),
    ]
    if not expected.ordering_required:
        gates.append(_not_applicable("MEASUREMENT_ORDERING", context))
    elif not (
        context.result.baseline_measurement.available
        and context.result.optimized_measurement.available
    ):
        gates.append(
            _unavailable("MEASUREMENT_ORDERING", context, "optimized <= baseline")
        )
    elif (
        context.result.optimized_measurement.value
        > context.result.baseline_measurement.value
    ):
        gates.append(
            _fail(
                "MEASUREMENT_ORDERING",
                context,
                "OPTIMIZED_EXCEEDS_BASELINE",
                "optimized <= baseline",
                "optimized > baseline",
            )
        )
    else:
        gates.append(
            _pass("MEASUREMENT_ORDERING", context, "optimized <= baseline", "ordered")
        )
    return gates


def _prefix_gates(context: _GateContext) -> list[GateResult]:
    expected = context.case.prefix
    evidence = context.result.prefix_identity_evidence
    gates: list[GateResult] = []
    if not expected.identity_required:
        gates.append(_not_applicable("PREFIX_IDENTITY_AVAILABLE", context))
    elif not evidence.identity_available:
        gates.append(_unavailable("PREFIX_IDENTITY_AVAILABLE", context, "identity"))
    else:
        gates.append(
            _pass("PREFIX_IDENTITY_AVAILABLE", context, "identity", "available")
        )
    if expected.same_as_case_id is not None:
        other = context.cases_by_id.get(expected.same_as_case_id)
        if (
            other is None
            or not evidence.identity_available
            or not other.prefix_identity_evidence.identity_available
        ):
            gates.append(
                _unavailable("PREFIX_STABILITY", context, expected.same_as_case_id)
            )
        elif (
            evidence.stable_prefix_identity
            != other.prefix_identity_evidence.stable_prefix_identity
        ):
            gates.append(
                _fail(
                    "PREFIX_STABILITY",
                    context,
                    "PREFIX_IDENTITY_CHANGED",
                    expected.same_as_case_id,
                    "different",
                )
            )
        else:
            gates.append(
                _pass("PREFIX_STABILITY", context, expected.same_as_case_id, "same")
            )
    else:
        gates.append(_not_applicable("PREFIX_STABILITY", context))
    if expected.different_from_case_id is not None:
        other = context.cases_by_id.get(expected.different_from_case_id)
        if (
            other is None
            or not evidence.identity_available
            or not other.prefix_identity_evidence.identity_available
        ):
            gates.append(
                _unavailable(
                    "PREFIX_CHANGED_CONTROL", context, expected.different_from_case_id
                )
            )
        elif (
            evidence.stable_prefix_identity
            == other.prefix_identity_evidence.stable_prefix_identity
        ):
            gates.append(
                _fail(
                    "PREFIX_CHANGED_CONTROL",
                    context,
                    "PREFIX_IDENTITY_NOT_CHANGED",
                    expected.different_from_case_id,
                    "same",
                )
            )
        else:
            gates.append(
                _pass(
                    "PREFIX_CHANGED_CONTROL",
                    context,
                    expected.different_from_case_id,
                    "different",
                )
            )
    else:
        gates.append(_not_applicable("PREFIX_CHANGED_CONTROL", context))
    if expected.tool_schema_identity is None:
        gates.append(_not_applicable("TOOL_ENVELOPE_IDENTITY", context))
    elif evidence.tool_schema_hash is None:
        gates.append(
            _unavailable(
                "TOOL_ENVELOPE_IDENTITY", context, expected.tool_schema_identity
            )
        )
    else:
        other_id = expected.same_as_case_id or expected.different_from_case_id
        other = context.cases_by_id.get(other_id) if other_id else None
        if other is None or other.prefix_identity_evidence.tool_schema_hash is None:
            gates.append(
                _unavailable(
                    "TOOL_ENVELOPE_IDENTITY", context, expected.tool_schema_identity
                )
            )
        else:
            same = (
                evidence.tool_schema_hash
                == other.prefix_identity_evidence.tool_schema_hash
            )
            if (expected.tool_schema_identity == "same") == same:
                gates.append(
                    _pass(
                        "TOOL_ENVELOPE_IDENTITY",
                        context,
                        expected.tool_schema_identity,
                        "same" if same else "different",
                    )
                )
            else:
                gates.append(
                    _fail(
                        "TOOL_ENVELOPE_IDENTITY",
                        context,
                        "TOOL_SCHEMA_IDENTITY_MISMATCH",
                        expected.tool_schema_identity,
                        "same" if same else "different",
                    )
                )
    return gates


def _cache_gates(context: _GateContext) -> list[GateResult]:
    expected = context.case.cache
    evidence = context.cache
    if expected.mode in {
        CacheExpectationMode.NOT_APPLICABLE,
        CacheExpectationMode.COLD,
    }:
        return [
            _not_applicable("WARM_CACHE_REUSE", context),
            _not_applicable("CHANGED_PREFIX_NEGATIVE_CONTROL", context),
        ]
    if evidence is None:
        warm = _unavailable("WARM_CACHE_REUSE", context, expected.mode.value)
        negative = _unavailable(
            "CHANGED_PREFIX_NEGATIVE_CONTROL", context, expected.mode.value
        )
        return [warm, negative]
    if evidence.cache_attribution is CacheAttribution.CONFLICTING:
        failed = _fail(
            "WARM_CACHE_REUSE",
            context,
            "CONFLICTING_CACHE_EVIDENCE",
            expected.mode.value,
            "conflicting",
        )
        return [
            failed,
            _fail(
                "CHANGED_PREFIX_NEGATIVE_CONTROL",
                context,
                "CONFLICTING_CACHE_EVIDENCE",
                expected.mode.value,
                "conflicting",
            ),
        ]
    if expected.mode is CacheExpectationMode.WARM_EXPECTED:
        if evidence.cache_attribution is CacheAttribution.LATENCY_ONLY:
            warm = _fail(
                "WARM_CACHE_REUSE",
                context,
                "LATENCY_ONLY_IS_NOT_CACHE_EVIDENCE",
                "reuse_confirmed",
                "latency_only",
            )
        elif evidence.cache_attribution is not CacheAttribution.REUSE_CONFIRMED:
            warm = _fail(
                "WARM_CACHE_REUSE",
                context,
                "CACHE_REUSE_NOT_CONFIRMED",
                "reuse_confirmed",
                evidence.cache_attribution.value,
            )
        elif (
            evidence.cached_prompt_token_count is None
            or evidence.cached_prompt_token_count <= 0
        ):
            warm = _fail(
                "WARM_CACHE_REUSE",
                context,
                "CACHE_TOKEN_REUSE_NOT_REPORTED",
                "positive cached tokens",
                evidence.cached_prompt_token_count,
            )
        elif (
            evidence.stable_prefix_identity
            != context.result.prefix_identity_evidence.stable_prefix_identity
        ):
            warm = _fail(
                "WARM_CACHE_REUSE",
                context,
                "CACHE_PREFIX_IDENTITY_MISMATCH",
                "matching identity",
                "different identity",
            )
        else:
            warm = _pass(
                "WARM_CACHE_REUSE", context, "typed reuse evidence", "reuse_confirmed"
            )
        return [warm, _not_applicable("CHANGED_PREFIX_NEGATIVE_CONTROL", context)]
    if evidence.cache_attribution is CacheAttribution.LATENCY_ONLY:
        negative = _fail(
            "CHANGED_PREFIX_NEGATIVE_CONTROL",
            context,
            "LATENCY_ONLY_IS_NOT_CACHE_EVIDENCE",
            "miss_confirmed",
            "latency_only",
        )
    elif evidence.cache_attribution is not CacheAttribution.MISS_CONFIRMED:
        negative = _fail(
            "CHANGED_PREFIX_NEGATIVE_CONTROL",
            context,
            "CACHE_MISS_NOT_CONFIRMED",
            "miss_confirmed",
            evidence.cache_attribution.value,
        )
    elif evidence.cached_prompt_token_count not in {0, None}:
        negative = _fail(
            "CHANGED_PREFIX_NEGATIVE_CONTROL",
            context,
            "CACHE_REUSE_REPORTED",
            "no cached tokens",
            evidence.cached_prompt_token_count,
        )
    elif expected.same_as_case_id is not None:
        reference = context.cases_by_id.get(expected.same_as_case_id)
        if (
            reference is None
            or not reference.prefix_identity_evidence.identity_available
            or not context.result.prefix_identity_evidence.identity_available
            or evidence.stable_prefix_identity is None
        ):
            negative = _unavailable(
                "CHANGED_PREFIX_NEGATIVE_CONTROL",
                context,
                expected.same_as_case_id,
            )
        elif (
            context.result.prefix_identity_evidence.stable_prefix_identity
            == reference.prefix_identity_evidence.stable_prefix_identity
            or evidence.stable_prefix_identity
            != context.result.prefix_identity_evidence.stable_prefix_identity
        ):
            negative = _fail(
                "CHANGED_PREFIX_NEGATIVE_CONTROL",
                context,
                "PREFIX_CHANGE_NOT_CONFIRMED",
                "changed identity and matching miss evidence",
                "unchanged or conflicting identity",
            )
        else:
            negative = _pass(
                "CHANGED_PREFIX_NEGATIVE_CONTROL",
                context,
                "changed identity and miss evidence",
                "miss_confirmed",
            )
    else:
        negative = _pass(
            "CHANGED_PREFIX_NEGATIVE_CONTROL",
            context,
            "typed miss evidence",
            "miss_confirmed",
        )
    return [_not_applicable("WARM_CACHE_REUSE", context), negative]


def _safety_gates(context: _GateContext) -> list[GateResult]:
    raw_absent = (
        not context.result.raw_content_included
        and not context.run.raw_content_included
        and not context.run.environment.raw_content_included
        and not context.run.artifact_manifest.raw_content_included
    )
    raw_gate = (
        _pass("RAW_CONTENT_ABSENT", context, False, "absent")
        if raw_absent
        else _fail(
            "RAW_CONTENT_ABSENT", context, "RAW_CONTENT_PRESENT", False, "present"
        )
    )
    secret_gate = (
        _pass("SECRET_ABSENT", context, "safe evidence only", "safe evidence only")
        if raw_absent
        else _fail(
            "SECRET_ABSENT", context, "SECRET_OR_RAW_CONTENT_RISK", "absent", "risk"
        )
    )
    refs = context.run.artifact_manifest.files
    paths = {ref.path for ref in refs}
    valid_hashes = all(
        (ref.path == "run.json" and ref.sha256 is None)
        or (ref.sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", ref.sha256))
        for ref in refs
    )
    required_paths = {"run.json", f"cases/{context.case.case_id}.json"}
    manifest_ok = (
        bool(refs)
        and len(paths) == len(refs)
        and valid_hashes
        and required_paths <= paths
    )
    manifest_gate = (
        _pass("MANIFEST_INTEGRITY", context, "unique hashed refs", "valid")
        if manifest_ok
        else _fail(
            "MANIFEST_INTEGRITY",
            context,
            "MANIFEST_INTEGRITY_FAILED",
            "unique hashed refs",
            "invalid",
        )
    )
    return [raw_gate, secret_gate, manifest_gate]


class UniversalProofEvaluator:
    """Consume a completed TOKEN-10F result without executing runtime components."""

    def evaluate(
        self,
        run_result: UniversalProofRunResult,
        corpus: ProofCorpus,
        evaluation_config: EvaluationConfiguration,
        *,
        cache_evidence: Iterable[ProviderCacheEvidence] = (),
        evaluation_id: str = "evaluation",
    ) -> UniversalProofEvaluation:
        if not evaluation_id or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", evaluation_id
        ):
            raise EvaluationConfigurationError("INVALID_EVALUATION_ID")
        if (
            run_result.run_mode == "offline_smoke"
            and evaluation_config.profile is EvaluationProfile.BEHAVIORAL
            and evaluation_config.gate_requirements is not None
        ):
            raise EvaluationConfigurationError("PROFILE_RUN_MODE_MISMATCH")
        unknown = set(evaluation_config.required_gate_ids) - _KNOWN_GATE_IDS
        unknown.update(
            set(evaluation_config.unavailable_allowed_gate_ids) - _KNOWN_GATE_IDS
        )
        if evaluation_config.gate_requirements:
            unknown.update(
                set(evaluation_config.gate_requirements) - _KNOWN_GATE_IDS
            )
        if unknown:
            raise EvaluationConfigurationError("UNKNOWN_GATE_ID")
        required_gate_ids = tuple(
            gate_id
            for gate_id in EVALUATION_GATE_IDS
            if evaluation_config.requirement_for(gate_id)
            in {
                EvaluationGateRequirement.REQUIRED,
                EvaluationGateRequirement.UNAVAILABLE_ALLOWED,
            }
        )
        if {case.case_id for case in corpus.cases} != {
            item.case_id for item in run_result.cases
        }:
            raise EvaluationConfigurationError("CORPUS_RUN_CASE_MISMATCH")
        cache_items = tuple(cache_evidence)
        cache_by_case = {item.case_id: item for item in cache_items}
        if len(cache_by_case) != len(cache_items):
            raise EvaluationConfigurationError("DUPLICATE_CACHE_EVIDENCE_CASE_IDS")
        result_by_id = {item.case_id: item for item in run_result.cases}
        evaluations = []
        for corpus_case in sorted(corpus.cases, key=lambda item: item.case_id):
            context = _GateContext(
                case=corpus_case,
                result=result_by_id[corpus_case.case_id],
                run=run_result,
                config=evaluation_config,
                cache=cache_by_case.get(corpus_case.case_id),
                cases_by_id=result_by_id,
            )
            gates = (
                _router_gates(context)
                + [_router_integrity_gate(context)]
                + _pipeline_gates(context)
                + [_pipeline_integrity_gate(context)]
                + _protected_gates(context)
                + _measurement_gates(context)
                + _prefix_gates(context)
                + _cache_gates(context)
                + _safety_gates(context)
            )
            evaluations.append(
                CaseEvaluation(
                    case_id=corpus_case.case_id,
                    category=corpus_case.category,
                    description=corpus_case.description,
                    gates=tuple(gates),
                )
            )
        all_gates = [gate for case in evaluations for gate in case.gates]
        status_counts = {
            status.value: sum(gate.status is status for gate in all_gates)
            for status in GateStatus
        }
        assessed_required = {gate.gate_id for gate in all_gates if gate.required}
        success = (
            set(required_gate_ids) <= assessed_required
            and not any(
                gate.required and gate.status is GateStatus.FAIL for gate in all_gates
            )
            and not any(
                gate.required
                and gate.status is GateStatus.UNAVAILABLE
                and evaluation_config.requirement_for(gate.gate_id)
                is not EvaluationGateRequirement.UNAVAILABLE_ALLOWED
                for gate in all_gates
            )
        )
        return UniversalProofEvaluation(
            evaluation_id=evaluation_id,
            proof_id=run_result.proof_id,
            run_id=run_result.run_id,
            corpus_version=corpus.schema_version,
            evaluation_version=evaluation_config.evaluation_version,
            run_mode=run_result.run_mode,
            provider=run_result.environment.provider,
            model=run_result.model,
            cases=tuple(evaluations),
            status_counts=status_counts,
            success=success,
            profile=evaluation_config.profile,
        )


__all__ = ["EVALUATION_GATE_IDS", "UniversalProofEvaluator"]
