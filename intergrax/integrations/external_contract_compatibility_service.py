# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic external contract compatibility assessment service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.integrations.contracts.external_contract_compatibility import (
    ExternalContractAssessmentWindow,
    ExternalContractCompatibilityExpectation,
    DimensionCompatibilityStatus,
    ExternalContractCompatibilityAssessment,
    ExternalContractCompatibilityAssessmentRequest,
    ExternalContractCompatibilityDimension,
    ExternalContractCompatibilityEvaluator,
    ExternalContractCompatibilityEvaluationContext,
    ExternalContractCompatibilityEvidence,
    ExternalContractCompatibilityEvidencePolicy,
    ExternalContractCompatibilityFinding,
    ExternalContractCompatibilityOutcome,
    ExternalContractCompatibilityReasonCode,
)

_CANONICAL_DIMENSION_ORDER: tuple[ExternalContractCompatibilityDimension, ...] = (
    ExternalContractCompatibilityDimension.SCHEMA,
    ExternalContractCompatibilityDimension.PROTOCOL,
    ExternalContractCompatibilityDimension.SEMANTIC,
)

_DIMENSION_MISMATCH_REASON: dict[
    ExternalContractCompatibilityDimension, ExternalContractCompatibilityReasonCode
] = {
    ExternalContractCompatibilityDimension.SCHEMA: ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH,
    ExternalContractCompatibilityDimension.PROTOCOL: ExternalContractCompatibilityReasonCode.PROTOCOL_MISMATCH,
    ExternalContractCompatibilityDimension.SEMANTIC: ExternalContractCompatibilityReasonCode.SEMANTIC_MISMATCH,
}

_DIMENSION_INCOMPATIBLE_OUTCOME: dict[
    ExternalContractCompatibilityDimension, ExternalContractCompatibilityOutcome
] = {
    ExternalContractCompatibilityDimension.SCHEMA: ExternalContractCompatibilityOutcome.SCHEMA_INCOMPATIBLE,
    ExternalContractCompatibilityDimension.PROTOCOL: ExternalContractCompatibilityOutcome.PROTOCOL_INCOMPATIBLE,
    ExternalContractCompatibilityDimension.SEMANTIC: ExternalContractCompatibilityOutcome.SEMANTIC_INCOMPATIBLE,
}


def _require_evaluator_id(value: str) -> str:
    if type(value) is not str:
        raise TypeError("evaluator_id must be str")
    if not value or value != value.strip():
        raise ExternalContractCompatibilityServiceError(
            "evaluator_id must be non-empty trimmed text"
        )
    return value


class ExternalContractCompatibilityServiceError(RuntimeError):
    """Service configuration or internal invariant violation."""


class ExternalContractCompatibilityEvaluatorContractError(
    ExternalContractCompatibilityServiceError
):
    """Evaluator plugin violated its purity or output contract."""


@dataclass(frozen=True, slots=True)
class _DimensionEvaluation:
    status: DimensionCompatibilityStatus
    reason: ExternalContractCompatibilityReasonCode
    findings: tuple[ExternalContractCompatibilityFinding, ...]
    material_evidence_refs: tuple[str, ...]


class ExternalContractCompatibilityService:
    def __init__(
        self,
        *,
        evaluators: tuple[ExternalContractCompatibilityEvaluator, ...],
        evidence_policy: ExternalContractCompatibilityEvidencePolicy,
    ) -> None:
        index: dict[str, ExternalContractCompatibilityEvaluator] = {}
        frozen_evaluators: list[ExternalContractCompatibilityEvaluator] = []
        for evaluator in evaluators:
            evaluator_id = _require_evaluator_id(evaluator.evaluator_id)
            if not evaluator.supported_dimensions:
                raise ExternalContractCompatibilityServiceError(
                    f"evaluator {evaluator_id!r} must declare non-empty supported_dimensions"
                )
            if evaluator_id in index:
                raise ExternalContractCompatibilityServiceError(
                    f"duplicate evaluator_id: {evaluator_id!r}"
                )
            index[evaluator_id] = evaluator
            frozen_evaluators.append(evaluator)
        self._evaluators: tuple[ExternalContractCompatibilityEvaluator, ...] = tuple(
            frozen_evaluators
        )
        self._evaluator_index: dict[str, ExternalContractCompatibilityEvaluator] = index
        self._evidence_policy = evidence_policy

    def assess(
        self, request: ExternalContractCompatibilityAssessmentRequest
    ) -> ExternalContractCompatibilityAssessment:
        expectation = request.expectation
        for item in request.evidence:
            if item.subject != expectation.subject:
                return self._build_assessment(
                    request,
                    outcome=ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE,
                    reason=ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH,
                    findings=(),
                    evidence_refs=(),
                )

        if request.explicit_evaluator_ids:
            for explicit_id in request.explicit_evaluator_ids:
                if explicit_id not in self._evaluator_index:
                    return self._build_assessment(
                        request,
                        outcome=ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE,
                        reason=ExternalContractCompatibilityReasonCode.UNSUPPORTED_EVALUATOR,
                        findings=(),
                        evidence_refs=(),
                    )

        dimension_results: dict[
            ExternalContractCompatibilityDimension, _DimensionEvaluation
        ] = {}
        ordered_findings: list[ExternalContractCompatibilityFinding] = []
        material_evidence_refs: list[str] = []

        for dimension in _CANONICAL_DIMENSION_ORDER:
            if dimension not in expectation.required_dimensions:
                continue

            pre_filter = self._authoritative_dimension_evidence(
                request.evidence, dimension
            )
            fresh = self._fresh_authoritative_dimension_evidence(
                request.evidence,
                dimension,
                assessed_at=request.assessed_at,
                assessment_window=request.assessment_window,
            )

            if not fresh:
                insufficient_reason = (
                    ExternalContractCompatibilityReasonCode.STALE_EVIDENCE
                    if pre_filter
                    else ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE
                )
                dimension_results[dimension] = _DimensionEvaluation(
                    status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                    reason=insufficient_reason,
                    findings=(),
                    material_evidence_refs=(),
                )
                continue

            selection = self._select_evaluator(
                expectation=expectation,
                evidence=fresh,
                dimension=dimension,
                explicit_evaluator_ids=request.explicit_evaluator_ids,
            )
            if selection.error_reason is not None:
                dimension_results[dimension] = _DimensionEvaluation(
                    status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                    reason=selection.error_reason,
                    findings=(),
                    material_evidence_refs=(),
                )
                continue

            evaluator = selection.evaluator
            assert evaluator is not None

            context = ExternalContractCompatibilityEvaluationContext(
                assessment_id=request.assessment_id,
                dimension=dimension,
                assessed_at=request.assessed_at,
                assessment_window=request.assessment_window,
            )
            raw_findings = evaluator.evaluate(expectation, fresh, context)
            validated_findings = self._validate_evaluator_findings(
                evaluator,
                dimension,
                raw_findings,
                authoritative_evidence=fresh,
            )
            ordered_findings.extend(validated_findings)
            for ev in fresh:
                for ref in ev.evidence_refs:
                    if ref not in material_evidence_refs:
                        material_evidence_refs.append(ref)

            dimension_results[dimension] = self._normalize_dimension(
                validated_findings, dimension
            )

        outcome, top_reason = self._aggregate_outcome(
            expectation.required_dimensions, dimension_results
        )
        result_evidence_refs = self._collect_result_evidence_refs(
            tuple(ordered_findings), material_evidence_refs
        )
        return self._build_assessment(
            request,
            outcome=outcome,
            reason=top_reason,
            findings=tuple(ordered_findings),
            evidence_refs=result_evidence_refs,
        )

    def _authoritative_dimension_evidence(
        self,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        dimension: ExternalContractCompatibilityDimension,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        return tuple(
            item
            for item in evidence
            if item.dimension is dimension and item.authority.is_authoritative
        )

    def _fresh_authoritative_dimension_evidence(
        self,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        dimension: ExternalContractCompatibilityDimension,
        *,
        assessed_at: datetime,
        assessment_window: ExternalContractAssessmentWindow,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        authoritative = self._authoritative_dimension_evidence(evidence, dimension)
        return tuple(
            item
            for item in authoritative
            if self._evidence_policy.accepts(
                item,
                assessed_at=assessed_at,
                window=assessment_window,
            )
        )

    def _select_evaluator(
        self,
        *,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        dimension: ExternalContractCompatibilityDimension,
        explicit_evaluator_ids: tuple[str, ...],
    ) -> _EvaluatorSelection:
        pool = self._evaluators
        if explicit_evaluator_ids:
            pool = tuple(self._evaluator_index[eid] for eid in explicit_evaluator_ids)
        candidates: list[ExternalContractCompatibilityEvaluator] = []
        for evaluator in pool:
            if dimension not in evaluator.supported_dimensions:
                continue
            if evaluator.can_evaluate(
                expectation,
                evidence,
                dimension=dimension,
            ):
                candidates.append(evaluator)
        if not candidates:
            return _EvaluatorSelection(
                evaluator=None,
                error_reason=ExternalContractCompatibilityReasonCode.UNSUPPORTED_EVALUATOR,
            )
        if len(candidates) > 1:
            return _EvaluatorSelection(
                evaluator=None,
                error_reason=ExternalContractCompatibilityReasonCode.EVALUATOR_AMBIGUITY,
            )
        return _EvaluatorSelection(evaluator=candidates[0], error_reason=None)

    def _validate_evaluator_findings(
        self,
        evaluator: ExternalContractCompatibilityEvaluator,
        dimension: ExternalContractCompatibilityDimension,
        findings: tuple[ExternalContractCompatibilityFinding, ...],
        *,
        authoritative_evidence: tuple[ExternalContractCompatibilityEvidence, ...],
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        evaluator_id = evaluator.evaluator_id
        allowed_refs = frozenset(
            ref
            for evidence_item in authoritative_evidence
            for ref in evidence_item.evidence_refs
        )
        for finding in findings:
            if finding.dimension is not dimension:
                raise ExternalContractCompatibilityEvaluatorContractError(
                    "finding.dimension must match evaluation dimension"
                )
            if finding.evaluator_id != evaluator_id:
                raise ExternalContractCompatibilityEvaluatorContractError(
                    "finding.evaluator_id must match selected evaluator"
                )
            if not frozenset(finding.evidence_refs).issubset(allowed_refs):
                raise ExternalContractCompatibilityEvaluatorContractError(
                    "finding.evidence_refs must reference only evidence supplied to evaluator"
                )
        return findings

    def _normalize_dimension(
        self,
        findings: tuple[ExternalContractCompatibilityFinding, ...],
        dimension: ExternalContractCompatibilityDimension,
    ) -> _DimensionEvaluation:
        material = tuple(f for f in findings if f.source_authority.is_authoritative)
        if not material:
            return _DimensionEvaluation(
                status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                reason=ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE,
                findings=findings,
                material_evidence_refs=(),
            )
        statuses = {f.status for f in material}
        if (
            DimensionCompatibilityStatus.COMPATIBLE in statuses
            and DimensionCompatibilityStatus.INCOMPATIBLE in statuses
        ):
            return _DimensionEvaluation(
                status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                reason=ExternalContractCompatibilityReasonCode.EVIDENCE_CONFLICT,
                findings=findings,
                material_evidence_refs=(),
            )
        if DimensionCompatibilityStatus.INCOMPATIBLE in statuses:
            return _DimensionEvaluation(
                status=DimensionCompatibilityStatus.INCOMPATIBLE,
                reason=_DIMENSION_MISMATCH_REASON[dimension],
                findings=findings,
                material_evidence_refs=(),
            )
        if DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE in statuses:
            reason = ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE
            for finding in material:
                if (
                    finding.reason_code
                    is not ExternalContractCompatibilityReasonCode.NONE
                ):
                    reason = finding.reason_code
                    break
            return _DimensionEvaluation(
                status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                reason=reason,
                findings=findings,
                material_evidence_refs=(),
            )
        if DimensionCompatibilityStatus.COMPATIBLE in statuses:
            return _DimensionEvaluation(
                status=DimensionCompatibilityStatus.COMPATIBLE,
                reason=ExternalContractCompatibilityReasonCode.NONE,
                findings=findings,
                material_evidence_refs=(),
            )
        return _DimensionEvaluation(
            status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
            reason=ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE,
            findings=findings,
            material_evidence_refs=(),
        )

    def _aggregate_outcome(
        self,
        required_dimensions: frozenset[ExternalContractCompatibilityDimension],
        dimension_results: dict[
            ExternalContractCompatibilityDimension, _DimensionEvaluation
        ],
    ) -> tuple[
        ExternalContractCompatibilityOutcome, ExternalContractCompatibilityReasonCode
    ]:
        for dimension in _CANONICAL_DIMENSION_ORDER:
            if dimension not in required_dimensions:
                continue
            result = dimension_results[dimension]
            if result.status is DimensionCompatibilityStatus.INCOMPATIBLE:
                return (
                    _DIMENSION_INCOMPATIBLE_OUTCOME[dimension],
                    _DIMENSION_MISMATCH_REASON[dimension],
                )
            if result.status is DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE:
                return (
                    ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE,
                    result.reason,
                )
        return (
            ExternalContractCompatibilityOutcome.COMPATIBLE,
            ExternalContractCompatibilityReasonCode.NONE,
        )

    def _collect_result_evidence_refs(
        self,
        findings: tuple[ExternalContractCompatibilityFinding, ...],
        evidence_refs: list[str],
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        for ref in evidence_refs:
            if ref not in ordered:
                ordered.append(ref)
        for finding in findings:
            if not finding.source_authority.is_authoritative:
                continue
            for ref in finding.evidence_refs:
                if ref not in ordered:
                    ordered.append(ref)
        return tuple(ordered)

    def _build_assessment(
        self,
        request: ExternalContractCompatibilityAssessmentRequest,
        *,
        outcome: ExternalContractCompatibilityOutcome,
        reason: ExternalContractCompatibilityReasonCode,
        findings: tuple[ExternalContractCompatibilityFinding, ...],
        evidence_refs: tuple[str, ...],
    ) -> ExternalContractCompatibilityAssessment:
        return ExternalContractCompatibilityAssessment(
            assessment_id=request.assessment_id,
            expectation_id=request.expectation.expectation_id,
            subject=request.expectation.subject,
            outcome=outcome,
            findings=findings,
            reason_code=reason,
            evidence_refs=evidence_refs,
            assessed_at=request.assessed_at,
        )


@dataclass(frozen=True, slots=True)
class _EvaluatorSelection:
    evaluator: ExternalContractCompatibilityEvaluator | None
    error_reason: ExternalContractCompatibilityReasonCode | None
