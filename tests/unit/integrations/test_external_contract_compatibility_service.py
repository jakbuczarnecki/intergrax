# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import pytest

from intergrax.integrations.contracts.external_contract_compatibility import (
    DimensionCompatibilityStatus,
    ExternalContractAssessmentWindow,
    ExternalContractCompatibilityAssessmentRequest,
    ExternalContractCompatibilityDimension,
    ExternalContractCompatibilityEvidence,
    ExternalContractCompatibilityEvaluationContext,
    ExternalContractCompatibilityExpectation,
    ExternalContractCompatibilityFinding,
    ExternalContractCompatibilityOutcome,
    ExternalContractCompatibilityReasonCode,
    ExternalContractCompatibilitySubject,
    ExternalContractEvidenceAuthority,
    ExternalContractPin,
    ExternalContractProtocolEvidenceFact,
    ExternalContractSchemaEvidenceFact,
    ExternalContractSemanticEvidenceFact,
    ProtocolValidationStatus,
    SchemaValidationStatus,
)
from intergrax.integrations.external_contract_compatibility_service import (
    ExternalContractCompatibilityService,
    ExternalContractCompatibilityServiceError,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TS = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
_WINDOW = ExternalContractAssessmentWindow(max_age=timedelta(hours=24))


class _AcceptAllPolicy:
    def accepts(
        self,
        evidence: ExternalContractCompatibilityEvidence,
        *,
        assessed_at: datetime,
        window: ExternalContractAssessmentWindow,
    ) -> bool:
        return True


class _RejectAllPolicy:
    def accepts(
        self,
        evidence: ExternalContractCompatibilityEvidence,
        *,
        assessed_at: datetime,
        window: ExternalContractAssessmentWindow,
    ) -> bool:
        return False


def _subject(
    *,
    tenant_id: str = "tenant-a",
    provider_id: str = "prov",
    integration_kind: str = "api",
    integration_id: str = "prov:api",
    external_operation_id: str = "op.read",
    host_binding_ref: str | None = None,
) -> ExternalContractCompatibilitySubject:
    return ExternalContractCompatibilitySubject(
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        integration_id=integration_id,
        external_operation_id=external_operation_id,
        host_binding_ref=host_binding_ref,
    )


def _expectation(
    *,
    required: frozenset[ExternalContractCompatibilityDimension] | None = None,
    subject: ExternalContractCompatibilitySubject | None = None,
) -> ExternalContractCompatibilityExpectation:
    dims = required or frozenset(
        {
            ExternalContractCompatibilityDimension.SCHEMA,
            ExternalContractCompatibilityDimension.PROTOCOL,
            ExternalContractCompatibilityDimension.SEMANTIC,
        }
    )
    kwargs: dict[str, object] = {
        "expectation_id": "exp-1",
        "subject": subject or _subject(),
        "expected_contract": ExternalContractPin("contract/ref", "v1"),
        "required_dimensions": dims,
    }
    if ExternalContractCompatibilityDimension.SCHEMA in dims:
        kwargs["schema_expectation_ref"] = "schema/exp"
    if ExternalContractCompatibilityDimension.PROTOCOL in dims:
        kwargs["protocol_expectation_ref"] = "protocol/exp"
    if ExternalContractCompatibilityDimension.SEMANTIC in dims:
        kwargs["semantic_expectation_refs"] = ("semantic/exp",)
    return ExternalContractCompatibilityExpectation(**kwargs)  # type: ignore[arg-type]


def _schema_evidence(
    subject: ExternalContractCompatibilitySubject,
    *,
    authority: ExternalContractEvidenceAuthority = ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
    observed_version: str | None = "v2",
    evidence_id: str = "ev-schema",
    ref: str = "evidence/schema",
) -> ExternalContractCompatibilityEvidence:
    return ExternalContractCompatibilityEvidence(
        evidence_id=evidence_id,
        subject=subject,
        observed_contract=(
            ExternalContractPin("contract/ref", observed_version)
            if observed_version is not None
            else None
        ),
        dimension=ExternalContractCompatibilityDimension.SCHEMA,
        observed_at=_TS,
        authority=authority,
        evidence_refs=(ref,),
        fact=ExternalContractSchemaEvidenceFact(
            schema_ref="s",
            schema_fingerprint=None,
            validation_status=SchemaValidationStatus.PASS,
        ),
    )


def _finding(
    dimension: ExternalContractCompatibilityDimension,
    status: DimensionCompatibilityStatus,
    *,
    reason: ExternalContractCompatibilityReasonCode,
    evaluator_id: str,
    ref: str = "finding-ref",
) -> ExternalContractCompatibilityFinding:
    return ExternalContractCompatibilityFinding(
        dimension=dimension,
        status=status,
        reason_code=reason,
        evidence_refs=(ref,),
        evaluator_id=evaluator_id,
        source_authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
    )


class _ConfigurableEvaluator:
    def __init__(
        self,
        evaluator_id: str,
        dimensions: frozenset[ExternalContractCompatibilityDimension],
        handler: Callable[
            [
                ExternalContractCompatibilityExpectation,
                tuple[ExternalContractCompatibilityEvidence, ...],
                ExternalContractCompatibilityEvaluationContext,
            ],
            tuple[ExternalContractCompatibilityFinding, ...],
        ],
        *,
        can_evaluate: bool = True,
    ) -> None:
        self._evaluator_id = evaluator_id
        self._dimensions = dimensions
        self._handler = handler
        self._can_evaluate = can_evaluate
        self.evaluate_calls = 0

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    @property
    def supported_dimensions(self) -> frozenset[ExternalContractCompatibilityDimension]:
        return self._dimensions

    def can_evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        *,
        dimension: ExternalContractCompatibilityDimension,
    ) -> bool:
        return self._can_evaluate

    def evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        self.evaluate_calls += 1
        return self._handler(expectation, evidence, context)


def _service(*evaluators: _ConfigurableEvaluator) -> ExternalContractCompatibilityService:
    return ExternalContractCompatibilityService(
        evaluators=evaluators,
        evidence_policy=_AcceptAllPolicy(),
    )


def _request(
    expectation: ExternalContractCompatibilityExpectation,
    evidence: tuple[ExternalContractCompatibilityEvidence, ...],
    *,
    explicit_evaluator_ids: tuple[str, ...] = (),
) -> ExternalContractCompatibilityAssessmentRequest:
    return ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-1",
        expectation=expectation,
        evidence=evidence,
        assessed_at=_TS,
        assessment_window=_WINDOW,
        explicit_evaluator_ids=explicit_evaluator_ids,
    )


def _status_evaluator(
    evaluator_id: str,
    dimension: ExternalContractCompatibilityDimension,
    status: DimensionCompatibilityStatus,
    *,
    reason: ExternalContractCompatibilityReasonCode | None = None,
) -> _ConfigurableEvaluator:
    if status is DimensionCompatibilityStatus.COMPATIBLE:
        finding_reason = ExternalContractCompatibilityReasonCode.NONE
    elif reason is not None:
        finding_reason = reason
    else:
        finding_reason = {
            ExternalContractCompatibilityDimension.SCHEMA: ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH,
            ExternalContractCompatibilityDimension.PROTOCOL: ExternalContractCompatibilityReasonCode.PROTOCOL_MISMATCH,
            ExternalContractCompatibilityDimension.SEMANTIC: ExternalContractCompatibilityReasonCode.SEMANTIC_MISMATCH,
        }[dimension]

    def handler(
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        if status is DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE and not evidence:
            return (
                _finding(
                    dimension,
                    DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                    reason=ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE,
                    evaluator_id=evaluator_id,
                ),
            )
        ref = evidence[0].evidence_refs[0] if evidence else "finding-ref"
        return (
            _finding(
                dimension,
                status,
                reason=finding_reason,
                evaluator_id=evaluator_id,
                ref=ref,
            ),
        )

    return _ConfigurableEvaluator(evaluator_id, frozenset({dimension}), handler)


def test_compatible_all_dimensions() -> None:
    exp = _expectation()
    ev = (
        _schema_evidence(exp.subject),
        ExternalContractCompatibilityEvidence(
            evidence_id="ev-protocol",
            subject=exp.subject,
            observed_contract=ExternalContractPin("contract/ref", "v2"),
            dimension=ExternalContractCompatibilityDimension.PROTOCOL,
            observed_at=_TS,
            authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
            evidence_refs=("evidence/protocol",),
            fact=ExternalContractProtocolEvidenceFact(
                protocol_ref="p",
                protocol_version="1",
                method="GET",
                content_type="application/json",
                validation_status=ProtocolValidationStatus.PASS,
            ),
        ),
        ExternalContractCompatibilityEvidence(
            evidence_id="ev-semantic",
            subject=exp.subject,
            observed_contract=None,
            dimension=ExternalContractCompatibilityDimension.SEMANTIC,
            observed_at=_TS,
            authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
            evidence_refs=("evidence/semantic",),
            fact=ExternalContractSemanticEvidenceFact(assertions=()),
        ),
    )
    service = _service(
        _status_evaluator("schema.v1", ExternalContractCompatibilityDimension.SCHEMA, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator("protocol.v1", ExternalContractCompatibilityDimension.PROTOCOL, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator("semantic.v1", ExternalContractCompatibilityDimension.SEMANTIC, DimensionCompatibilityStatus.COMPATIBLE),
    )
    result = service.assess(_request(exp, ev))
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.NONE


def test_variant_h_semantic_incompatible() -> None:
    exp = _expectation()
    ev = (_schema_evidence(exp.subject),)
    service = _service(
        _status_evaluator("schema.v1", ExternalContractCompatibilityDimension.SCHEMA, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator("protocol.v1", ExternalContractCompatibilityDimension.PROTOCOL, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator(
            "semantic.v1",
            ExternalContractCompatibilityDimension.SEMANTIC,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
    )
    result = service.assess(_request(exp, ev))
    assert result.outcome is ExternalContractCompatibilityOutcome.SEMANTIC_INCOMPATIBLE


def test_schema_insufficient_blocks_semantic_incompatible() -> None:
    exp = _expectation()
    service = _service(
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
        ),
        _status_evaluator(
            "semantic.v1",
            ExternalContractCompatibilityDimension.SEMANTIC,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
    )
    result = service.assess(_request(exp, ()))
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE


def test_protocol_insufficient_blocks_semantic_incompatible() -> None:
    exp = _expectation()
    service = _service(
        _status_evaluator("schema.v1", ExternalContractCompatibilityDimension.SCHEMA, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator(
            "protocol.v1",
            ExternalContractCompatibilityDimension.PROTOCOL,
            DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
        ),
        _status_evaluator(
            "semantic.v1",
            ExternalContractCompatibilityDimension.SEMANTIC,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
    )
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE


def test_schema_precedence_over_protocol() -> None:
    exp = _expectation()
    service = _service(
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
        _status_evaluator(
            "protocol.v1",
            ExternalContractCompatibilityDimension.PROTOCOL,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
    )
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.outcome is ExternalContractCompatibilityOutcome.SCHEMA_INCOMPATIBLE


def test_protocol_precedence_over_semantic() -> None:
    exp = _expectation()
    service = _service(
        _status_evaluator("schema.v1", ExternalContractCompatibilityDimension.SCHEMA, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator(
            "protocol.v1",
            ExternalContractCompatibilityDimension.PROTOCOL,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
        _status_evaluator(
            "semantic.v1",
            ExternalContractCompatibilityDimension.SEMANTIC,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
    )
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.outcome is ExternalContractCompatibilityOutcome.PROTOCOL_INCOMPATIBLE


def test_evidence_conflict() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
    )

    def handler(
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        return (
            _finding(
                ExternalContractCompatibilityDimension.SCHEMA,
                DimensionCompatibilityStatus.COMPATIBLE,
                reason=ExternalContractCompatibilityReasonCode.NONE,
                evaluator_id="schema.v1",
                ref="r1",
            ),
            _finding(
                ExternalContractCompatibilityDimension.SCHEMA,
                DimensionCompatibilityStatus.INCOMPATIBLE,
                reason=ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH,
                evaluator_id="schema.v1",
                ref="r2",
            ),
        )

    service = _service(_ConfigurableEvaluator("schema.v1", frozenset({ExternalContractCompatibilityDimension.SCHEMA}), handler))
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.EVIDENCE_CONFLICT


def test_llm_only_refusal() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    ev = (
        _schema_evidence(
            exp.subject,
            authority=ExternalContractEvidenceAuthority.LLM_ADVISORY,
        ),
    )
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )

    def tracking_handler(
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        assert evidence == ()
        return ()

    evaluator._handler = tracking_handler  # type: ignore[method-assign]
    service = _service(evaluator)
    result = service.assess(_request(exp, ev))
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "other"),
        ("provider_id", "other"),
        ("integration_id", "other:api"),
        ("external_operation_id", "op.other"),
        ("host_binding_ref", "host-b"),
    ],
)
def test_identity_mismatch_wrong_subject(field: str, value: str) -> None:
    exp = _expectation()
    if field == "tenant_id":
        wrong = _subject(tenant_id=value)
    elif field == "provider_id":
        wrong = _subject(provider_id=value, integration_id=f"{value}:api")
    elif field == "integration_id":
        wrong = _subject(provider_id="other", integration_id=value)
    elif field == "external_operation_id":
        wrong = _subject(external_operation_id=value)
    else:
        wrong = _subject(host_binding_ref=value)
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    service = _service(evaluator)
    result = service.assess(_request(exp, (_schema_evidence(wrong),)))
    assert result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    assert evaluator.evaluate_calls == 0


def test_different_contract_version_still_evaluates() -> None:
    exp = _expectation()
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    service = _service(evaluator)
    service.assess(
        _request(
            exp,
            (_schema_evidence(exp.subject, observed_version="v2"),),
        )
    )
    assert evaluator.evaluate_calls == 1


def test_stale_evidence() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    service = ExternalContractCompatibilityService(
        evaluators=(evaluator,),
        evidence_policy=_RejectAllPolicy(),
    )
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.reason_code is ExternalContractCompatibilityReasonCode.STALE_EVIDENCE
    assert evaluator.evaluate_calls == 0


def test_missing_evidence() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    service = _service(
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
        )
    )
    result = service.assess(_request(exp, ()))
    assert result.reason_code is ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE


def test_unsupported_evaluator() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    service = _service()
    result = service.assess(_request(exp, ()))
    assert result.reason_code is ExternalContractCompatibilityReasonCode.UNSUPPORTED_EVALUATOR


def test_evaluator_ambiguity() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    e1 = _status_evaluator(
        "schema.a",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    e2 = _status_evaluator(
        "schema.b",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    service = _service(e1, e2)
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.reason_code is ExternalContractCompatibilityReasonCode.EVALUATOR_AMBIGUITY
    assert e1.evaluate_calls == 0
    assert e2.evaluate_calls == 0


def test_explicit_evaluator_selection() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    chosen = _status_evaluator(
        "custom.schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    other = _status_evaluator(
        "schema.other",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.INCOMPATIBLE,
    )
    service = _service(chosen, other)
    result = service.assess(
        _request(
            exp,
            (_schema_evidence(exp.subject),),
            explicit_evaluator_ids=("custom.schema.v1",),
        )
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE
    assert chosen.evaluate_calls == 1
    assert other.evaluate_calls == 0


def test_unknown_explicit_evaluator() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    service = _service(
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.COMPATIBLE,
        )
    )
    result = service.assess(
        _request(exp, (), explicit_evaluator_ids=("missing.evaluator",))
    )
    assert result.reason_code is ExternalContractCompatibilityReasonCode.UNSUPPORTED_EVALUATOR


def test_duplicate_evaluator_id_at_construction() -> None:
    ev = _status_evaluator(
        "dup",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    with pytest.raises(ExternalContractCompatibilityServiceError):
        ExternalContractCompatibilityService(
            evaluators=(ev, ev),
            evidence_policy=_AcceptAllPolicy(),
        )


class _ExternalPluginEvaluator:
    """Structural Protocol implementation without subclassing defaults."""

    @property
    def evaluator_id(self) -> str:
        return "external.plugin.v1"

    @property
    def supported_dimensions(self) -> frozenset[ExternalContractCompatibilityDimension]:
        return frozenset({ExternalContractCompatibilityDimension.SCHEMA})

    def can_evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        *,
        dimension: ExternalContractCompatibilityDimension,
    ) -> bool:
        return True

    def evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        return (
            _finding(
                ExternalContractCompatibilityDimension.SCHEMA,
                DimensionCompatibilityStatus.COMPATIBLE,
                reason=ExternalContractCompatibilityReasonCode.NONE,
                evaluator_id=self.evaluator_id,
            ),
        )


def test_structural_plugin_evaluator() -> None:
    exp = _expectation(required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}))
    service = ExternalContractCompatibilityService(
        evaluators=(_ExternalPluginEvaluator(),),
        evidence_policy=_AcceptAllPolicy(),
    )
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE


def test_deterministic_finding_and_evidence_ref_order() -> None:
    exp = _expectation()
    service = _service(
        _status_evaluator("schema.v1", ExternalContractCompatibilityDimension.SCHEMA, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator("protocol.v1", ExternalContractCompatibilityDimension.PROTOCOL, DimensionCompatibilityStatus.COMPATIBLE),
        _status_evaluator("semantic.v1", ExternalContractCompatibilityDimension.SEMANTIC, DimensionCompatibilityStatus.COMPATIBLE),
    )
    ev = (
        _schema_evidence(exp.subject, ref="ref-schema"),
        ExternalContractCompatibilityEvidence(
            evidence_id="ev-protocol",
            subject=exp.subject,
            observed_contract=None,
            dimension=ExternalContractCompatibilityDimension.PROTOCOL,
            observed_at=_TS,
            authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
            evidence_refs=("ref-protocol",),
            fact=ExternalContractProtocolEvidenceFact(
                protocol_ref="p",
                protocol_version=None,
                method=None,
                content_type=None,
                validation_status=ProtocolValidationStatus.PASS,
            ),
        ),
        ExternalContractCompatibilityEvidence(
            evidence_id="ev-semantic",
            subject=exp.subject,
            observed_contract=None,
            dimension=ExternalContractCompatibilityDimension.SEMANTIC,
            observed_at=_TS,
            authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
            evidence_refs=("ref-semantic",),
            fact=ExternalContractSemanticEvidenceFact(assertions=()),
        ),
    )
    result = service.assess(_request(exp, ev))
    dims = [f.dimension for f in result.findings]
    assert dims == [
        ExternalContractCompatibilityDimension.SCHEMA,
        ExternalContractCompatibilityDimension.PROTOCOL,
        ExternalContractCompatibilityDimension.SEMANTIC,
    ]
    assert list(result.evidence_refs) == ["ref-schema", "ref-protocol", "ref-semantic"]
