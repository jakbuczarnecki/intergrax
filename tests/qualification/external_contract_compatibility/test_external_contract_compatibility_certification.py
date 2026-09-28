# © Artur Czarnecki. All rights reserved.

"""S24-GAP-04-CERT — adversarial external contract compatibility certification matrix."""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable as TypingCallable

import pytest

from intergrax.contracts.execution_identity import ExecutionId, RunId, TaskId
from intergrax.integrations.contracts.external_contract_compatibility import (
    DimensionCompatibilityStatus,
    ExternalContractAssessmentWindow,
    ExternalContractCompatibilityAssessmentRequest,
    ExternalContractCompatibilityDimension,
    ExternalContractCompatibilityEvidence,
    ExternalContractCompatibilityEvaluationContext,
    ExternalContractCompatibilityEvaluator,
    ExternalContractCompatibilityExpectation,
    ExternalContractCompatibilityFinding,
    ExternalContractCompatibilityOutcome,
    ExternalContractCompatibilityReasonCode,
    ExternalContractCompatibilitySubject,
    ExternalContractEvidenceAuthority,
    ExternalContractEvidenceCollectionRequest,
    ExternalContractExpectationKey,
    ExternalContractPin,
    ExternalContractProtocolEvidenceFact,
    ExternalContractSchemaEvidenceFact,
    ExternalContractSemanticEvidenceFact,
    ProtocolValidationStatus,
    SchemaValidationStatus,
)
from intergrax.integrations.examples.custom_memory_kv.compatibility import (
    CustomMemoryKvContractEvidenceProvider,
    CustomMemoryKvContractObservation,
    CustomMemoryKvSchemaCompatibilityEvaluator,
    custom_memory_kv_compatibility_extensions,
)
from intergrax.integrations.examples.custom_memory_kv.integration import (
    CUSTOM_MEMORY_KV_PROVIDER_ID,
)
from intergrax.integrations.external_contract_compatibility_extensions import (
    ExternalContractCompatibilityExtensionsError,
    assess_external_contract_compatibility,
    external_contract_compatibility_extensions,
)
from intergrax.integrations.external_contract_compatibility_service import (
    ExternalContractCompatibilityEvaluatorContractError,
    ExternalContractCompatibilityService,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_TS = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
_TS_OBSERVED_EARLIER = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
_TS_ASSESSED_LATER = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
_WINDOW = ExternalContractAssessmentWindow(max_age=timedelta(hours=24))
_STALE_WINDOW = ExternalContractAssessmentWindow(max_age=timedelta(hours=1))
_REPO_ROOT = Path(__file__).resolve().parents[3]
_KV_CONTRACT_REF = "integrations/custom_memory_kv/read"
_KV_CONTRACT_VERSION = "v1"


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


class _MaxAgeEvidencePolicy:
    def accepts(
        self,
        evidence: ExternalContractCompatibilityEvidence,
        *,
        assessed_at: datetime,
        window: ExternalContractAssessmentWindow,
    ) -> bool:
        if window.max_age is None:
            return True
        return assessed_at - evidence.observed_at <= window.max_age


def _subject(
    *,
    tenant_id: str = "tenant-a",
    provider_id: str = "prov",
    integration_kind: str = "api",
    integration_id: str = "prov:api",
    external_operation_id: str = "op.read",
    host_binding_ref: str | None = None,
    execution_task_id: TaskId | None = None,
    execution_run_id: RunId | None = None,
    execution_id: ExecutionId | None = None,
) -> ExternalContractCompatibilitySubject:
    return ExternalContractCompatibilitySubject(
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        integration_id=integration_id,
        external_operation_id=external_operation_id,
        host_binding_ref=host_binding_ref,
        execution_task_id=execution_task_id,
        execution_run_id=execution_run_id,
        execution_id=execution_id,
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
    subj = subject or _subject()
    return ExternalContractCompatibilityExpectation(
        expectation_id="exp-cert",
        subject=subj,
        expected_contract=ExternalContractPin("contract/ref", "v1"),
        required_dimensions=dims,
        schema_expectation_ref=(
            "schema/exp"
            if ExternalContractCompatibilityDimension.SCHEMA in dims
            else None
        ),
        protocol_expectation_ref=(
            "protocol/exp"
            if ExternalContractCompatibilityDimension.PROTOCOL in dims
            else None
        ),
        semantic_expectation_refs=(
            ("semantic/exp",)
            if ExternalContractCompatibilityDimension.SEMANTIC in dims
            else ()
        ),
    )


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


def _protocol_evidence(
    subject: ExternalContractCompatibilitySubject,
) -> ExternalContractCompatibilityEvidence:
    return ExternalContractCompatibilityEvidence(
        evidence_id="ev-protocol",
        subject=subject,
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
    )


def _semantic_evidence(
    subject: ExternalContractCompatibilitySubject,
) -> ExternalContractCompatibilityEvidence:
    return ExternalContractCompatibilityEvidence(
        evidence_id="ev-semantic",
        subject=subject,
        observed_contract=None,
        dimension=ExternalContractCompatibilityDimension.SEMANTIC,
        observed_at=_TS,
        authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
        evidence_refs=("evidence/semantic",),
        fact=ExternalContractSemanticEvidenceFact(assertions=()),
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
        handler: TypingCallable[
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
        self.can_evaluate_calls = 0
        self.last_can_evaluate_evidence: (
            tuple[ExternalContractCompatibilityEvidence, ...] | None
        ) = None
        self.last_evaluate_evidence: (
            tuple[ExternalContractCompatibilityEvidence, ...] | None
        ) = None

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
        self.can_evaluate_calls += 1
        self.last_can_evaluate_evidence = evidence
        return self._can_evaluate

    def evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        self.evaluate_calls += 1
        self.last_evaluate_evidence = evidence
        return self._handler(expectation, evidence, context)


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


def _service(
    *evaluators: _ConfigurableEvaluator,
    evidence_policy: _AcceptAllPolicy
    | _RejectAllPolicy
    | _MaxAgeEvidencePolicy
    | None = None,
) -> ExternalContractCompatibilityService:
    policy = evidence_policy if evidence_policy is not None else _AcceptAllPolicy()
    return ExternalContractCompatibilityService(
        evaluators=evaluators,
        evidence_policy=policy,
    )


def _request(
    expectation: ExternalContractCompatibilityExpectation,
    evidence: tuple[ExternalContractCompatibilityEvidence, ...],
    *,
    explicit_evaluator_ids: tuple[str, ...] = (),
) -> ExternalContractCompatibilityAssessmentRequest:
    return ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-cert",
        expectation=expectation,
        evidence=evidence,
        assessed_at=_TS,
        assessment_window=_WINDOW,
        explicit_evaluator_ids=explicit_evaluator_ids,
    )


def _assess(
    exp: ExternalContractCompatibilityExpectation,
    ev: tuple[ExternalContractCompatibilityEvidence, ...],
    *evaluators: _ConfigurableEvaluator,
    evidence_policy: _AcceptAllPolicy
    | _RejectAllPolicy
    | _MaxAgeEvidencePolicy
    | None = None,
):
    return _service(*evaluators, evidence_policy=evidence_policy).assess(
        _request(exp, ev)
    )


def test_cert_01_all_authoritative_dimensions_compatible() -> None:
    exp = _expectation()
    ev = (
        _schema_evidence(exp.subject),
        _protocol_evidence(exp.subject),
        _semantic_evidence(exp.subject),
    )
    service = _service(
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.COMPATIBLE,
        ),
        _status_evaluator(
            "protocol.v1",
            ExternalContractCompatibilityDimension.PROTOCOL,
            DimensionCompatibilityStatus.COMPATIBLE,
        ),
        _status_evaluator(
            "semantic.v1",
            ExternalContractCompatibilityDimension.SEMANTIC,
            DimensionCompatibilityStatus.COMPATIBLE,
        ),
    )
    result = service.assess(_request(exp, ev))
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.NONE
    assert [f.dimension for f in result.findings] == [
        ExternalContractCompatibilityDimension.SCHEMA,
        ExternalContractCompatibilityDimension.PROTOCOL,
        ExternalContractCompatibilityDimension.SEMANTIC,
    ]


def test_cert_02_schema_drift() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    result = _assess(
        exp,
        (_schema_evidence(exp.subject),),
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.SCHEMA_INCOMPATIBLE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH


def test_cert_03_protocol_drift() -> None:
    exp = _expectation()
    result = _assess(
        exp,
        (
            _schema_evidence(exp.subject),
            _protocol_evidence(exp.subject),
            _semantic_evidence(exp.subject),
        ),
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.COMPATIBLE,
        ),
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
    assert result.outcome is ExternalContractCompatibilityOutcome.PROTOCOL_INCOMPATIBLE
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.PROTOCOL_MISMATCH
    )


def test_cert_04_semantic_false_compatibility_variant_h() -> None:
    exp = _expectation()
    result = _assess(
        exp,
        (
            _schema_evidence(exp.subject),
            _protocol_evidence(exp.subject),
            _semantic_evidence(exp.subject),
        ),
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.COMPATIBLE,
        ),
        _status_evaluator(
            "protocol.v1",
            ExternalContractCompatibilityDimension.PROTOCOL,
            DimensionCompatibilityStatus.COMPATIBLE,
        ),
        _status_evaluator(
            "semantic.v1",
            ExternalContractCompatibilityDimension.SEMANTIC,
            DimensionCompatibilityStatus.INCOMPATIBLE,
        ),
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.SEMANTIC_INCOMPATIBLE
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.SEMANTIC_MISMATCH
    )
    assert result.outcome is not ExternalContractCompatibilityOutcome.COMPATIBLE


def test_cert_05_missing_required_evidence() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator).assess(_request(exp, ()))
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert (
        result.reason_code
        is ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE
    )
    assert evaluator.evaluate_calls == 0
    assert evaluator.can_evaluate_calls == 0


def test_cert_06_stale_evidence() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator, evidence_policy=_RejectAllPolicy()).assess(
        _request(exp, (_schema_evidence(exp.subject),))
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.STALE_EVIDENCE
    assert evaluator.evaluate_calls == 0


def test_cert_07_llm_only_compatibility_claim() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    ev = (
        _schema_evidence(
            exp.subject, authority=ExternalContractEvidenceAuthority.LLM_ADVISORY
        ),
    )
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator).assess(_request(exp, ev))
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert evaluator.evaluate_calls == 0


def test_cert_08_contradictory_authoritative_evidence() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
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
                ref=evidence[0].evidence_refs[0],
            ),
            _finding(
                ExternalContractCompatibilityDimension.SCHEMA,
                DimensionCompatibilityStatus.INCOMPATIBLE,
                reason=ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH,
                evaluator_id="schema.v1",
                ref=evidence[1].evidence_refs[0],
            ),
        )

    evaluator = _ConfigurableEvaluator(
        "schema.v1",
        frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        handler,
    )
    result = _service(evaluator).assess(
        _request(
            exp,
            (
                _schema_evidence(exp.subject, ref="r1", evidence_id="ev-1"),
                _schema_evidence(exp.subject, ref="r2", evidence_id="ev-2"),
            ),
        )
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.EVIDENCE_CONFLICT
    )


def test_cert_09_wrong_tenant_evidence() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    wrong = _subject(tenant_id="tenant-b")
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator).assess(_request(exp, (_schema_evidence(wrong),)))
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )
    assert evaluator.evaluate_calls == 0


def test_cert_10_wrong_provider() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    wrong = _subject(provider_id="other", integration_id="other:api")
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator).assess(_request(exp, (_schema_evidence(wrong),)))
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )


def test_cert_11_wrong_operation() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    wrong = _subject(external_operation_id="op.other")
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator).assess(_request(exp, (_schema_evidence(wrong),)))
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )


def test_cert_12_execution_scope_identity_mismatch() -> None:
    task_a = TaskId("task_" + "a" * 32)
    task_b = TaskId("task_" + "b" * 32)
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        subject=_subject(execution_task_id=task_a),
    )
    wrong = _subject(execution_task_id=task_b)
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator).assess(_request(exp, (_schema_evidence(wrong),)))
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )
    assert evaluator.evaluate_calls == 0


def test_cert_13_contract_version_difference_not_identity_mismatch() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(evaluator).assess(
        _request(exp, (_schema_evidence(exp.subject, observed_version="v2"),))
    )
    assert (
        result.reason_code
        is not ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )
    assert evaluator.evaluate_calls == 1


def test_cert_14_unsupported_evaluator() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    result = _service().assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert (
        result.reason_code
        is ExternalContractCompatibilityReasonCode.UNSUPPORTED_EVALUATOR
    )


def test_cert_15_evaluator_ambiguity() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
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
    result = _service(e1, e2).assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert (
        result.reason_code
        is ExternalContractCompatibilityReasonCode.EVALUATOR_AMBIGUITY
    )
    assert e1.evaluate_calls == 0
    assert e2.evaluate_calls == 0


class _ExternalPluginEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "external.plugin.cert"

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
        ref = evidence[0].evidence_refs[0]
        return (
            _finding(
                ExternalContractCompatibilityDimension.SCHEMA,
                DimensionCompatibilityStatus.COMPATIBLE,
                reason=ExternalContractCompatibilityReasonCode.NONE,
                evaluator_id=self.evaluator_id,
                ref=ref,
            ),
        )


def test_cert_16_explicit_external_plugin_evaluator() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    plugin: ExternalContractCompatibilityEvaluator = _ExternalPluginEvaluator()
    service = ExternalContractCompatibilityService(
        evaluators=(plugin,),
        evidence_policy=_AcceptAllPolicy(),
    )
    result = service.assess(_request(exp, (_schema_evidence(exp.subject),)))
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE
    service_source = (
        _REPO_ROOT / "intergrax/integrations/external_contract_compatibility_service.py"
    ).read_text(encoding="utf-8")
    assert "custom_memory_kv" not in service_source
    assert "provider_id ==" not in service_source


def test_cert_17_forged_finding_evidence_ref_lineage() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    fresh = _schema_evidence(exp.subject, ref="real/ref")

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
                ref="forged/ref",
            ),
        )

    evaluator = _ConfigurableEvaluator(
        "schema.v1",
        frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        handler,
    )
    with pytest.raises(ExternalContractCompatibilityEvaluatorContractError):
        _service(evaluator).assess(_request(exp, (fresh,)))


def test_cert_18_advisory_wrong_dimension_evidence_laundering() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    fresh_schema = _schema_evidence(
        exp.subject, ref="fresh/schema", evidence_id="ev-fresh"
    )
    other_dimension = ExternalContractCompatibilityEvidence(
        evidence_id="ev-protocol",
        subject=exp.subject,
        observed_contract=None,
        dimension=ExternalContractCompatibilityDimension.PROTOCOL,
        observed_at=_TS,
        authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
        evidence_refs=("protocol/ref",),
        fact=ExternalContractProtocolEvidenceFact(
            protocol_ref="p",
            protocol_version="1",
            method="GET",
            content_type="application/json",
            validation_status=ProtocolValidationStatus.PASS,
        ),
    )
    advisory_schema = _schema_evidence(
        exp.subject,
        authority=ExternalContractEvidenceAuthority.LLM_ADVISORY,
        ref="advisory/schema",
        evidence_id="ev-advisory",
    )
    evaluator = _status_evaluator(
        "schema.v1",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    _service(evaluator).assess(
        _request(exp, (fresh_schema, other_dimension, advisory_schema))
    )
    assert evaluator.last_can_evaluate_evidence == (fresh_schema,)
    assert evaluator.last_evaluate_evidence == (fresh_schema,)


class _CountingEvidenceProvider:
    def __init__(self) -> None:
        self.collect_calls = 0

    @property
    def evidence_provider_id(self) -> str:
        return "cert.counting.provider"

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        self.collect_calls += 1
        return ()


def test_cert_19_collection_expectation_assessment_expectation_mismatch() -> None:
    provider = _CountingEvidenceProvider()
    base = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SEMANTIC})
    )
    collection_expectation = base
    assessment_expectation = replace(
        base,
        subject=_subject(tenant_id="tenant-b"),
    )
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider,)
    )
    with pytest.raises(
        ExternalContractCompatibilityExtensionsError,
        match="collection expectation does not match assessment expectation",
    ):
        assess_external_contract_compatibility(
            extensions,
            evidence_policy=_AcceptAllPolicy(),
            collection_request=ExternalContractEvidenceCollectionRequest(
                expectation=collection_expectation,
                assessed_at=_TS,
                assessment_window=_WINDOW,
            ),
            assessment_request=ExternalContractCompatibilityAssessmentRequest(
                assessment_id="cert-19",
                expectation=assessment_expectation,
                evidence=(),
                assessed_at=_TS,
                assessment_window=_WINDOW,
            ),
        )
    assert provider.collect_calls == 0


def _kv_subject(tenant_id: str = "tenant-a") -> ExternalContractCompatibilitySubject:
    return ExternalContractCompatibilitySubject(
        tenant_id=tenant_id,
        provider_id=CUSTOM_MEMORY_KV_PROVIDER_ID,
        integration_kind="key_value_cache",
        integration_id=f"{CUSTOM_MEMORY_KV_PROVIDER_ID}:key_value_cache",
        external_operation_id="kv.get",
    )


def _kv_observation(
    *,
    tenant_id: str = "tenant-a",
    observed_at: datetime = _TS,
) -> CustomMemoryKvContractObservation:
    return CustomMemoryKvContractObservation(
        subject=_kv_subject(tenant_id=tenant_id),
        observed_at=observed_at,
        observed_contract=ExternalContractPin(_KV_CONTRACT_REF, _KV_CONTRACT_VERSION),
        schema_ref="schema/custom_memory_kv/get",
        schema_fingerprint="fp-observed",
        validation_status=SchemaValidationStatus.PASS,
        evidence_refs=("probe/custom_memory_kv/schema/1",),
    )


def test_cert_20_provider_observation_tenant_rebinding_attack() -> None:
    observation = _kv_observation(tenant_id="tenant-a")
    provider = CustomMemoryKvContractEvidenceProvider(observation)
    collection_request = ExternalContractEvidenceCollectionRequest(
        expectation=ExternalContractCompatibilityExpectation(
            expectation_id="exp-tenant-b",
            subject=_kv_subject(tenant_id="tenant-b"),
            expected_contract=ExternalContractPin(
                _KV_CONTRACT_REF, _KV_CONTRACT_VERSION
            ),
            required_dimensions=frozenset(
                {ExternalContractCompatibilityDimension.SCHEMA}
            ),
            schema_expectation_ref="schema/custom_memory_kv/get",
        ),
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    assert provider.collect(collection_request) == ()
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider,),
        evaluators=(CustomMemoryKvSchemaCompatibilityEvaluator(),),
    )
    assessment_expectation = collection_request.expectation
    result = assess_external_contract_compatibility(
        extensions,
        evidence_policy=_AcceptAllPolicy(),
        collection_request=collection_request,
        assessment_request=ExternalContractCompatibilityAssessmentRequest(
            assessment_id="cert-20",
            expectation=assessment_expectation,
            evidence=(),
            assessed_at=_TS,
            assessment_window=_WINDOW,
        ),
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE


def test_cert_21_stale_provider_observation_remains_stale() -> None:
    observation = _kv_observation(
        tenant_id="tenant-a", observed_at=_TS_OBSERVED_EARLIER
    )
    extensions = custom_memory_kv_compatibility_extensions(observation)
    subject = _kv_subject(tenant_id="tenant-a")
    expectation = ExternalContractCompatibilityExpectation(
        expectation_id="exp-stale",
        subject=subject,
        expected_contract=ExternalContractPin(_KV_CONTRACT_REF, _KV_CONTRACT_VERSION),
        required_dimensions=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        schema_expectation_ref="schema/custom_memory_kv/get",
    )
    result = assess_external_contract_compatibility(
        extensions,
        evidence_policy=_MaxAgeEvidencePolicy(),
        collection_request=ExternalContractEvidenceCollectionRequest(
            expectation=expectation,
            assessed_at=_TS_ASSESSED_LATER,
            assessment_window=_STALE_WINDOW,
        ),
        assessment_request=ExternalContractCompatibilityAssessmentRequest(
            assessment_id="cert-21",
            expectation=expectation,
            evidence=(),
            assessed_at=_TS_ASSESSED_LATER,
            assessment_window=_STALE_WINDOW,
        ),
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.STALE_EVIDENCE
    provider = CustomMemoryKvContractEvidenceProvider(observation)
    collected = provider.collect(
        ExternalContractEvidenceCollectionRequest(
            expectation=expectation,
            assessed_at=_TS_ASSESSED_LATER,
            assessment_window=_STALE_WINDOW,
        )
    )
    assert collected[0].observed_at == _TS_OBSERVED_EARLIER


def test_cert_22_resolver_tenant_isolation() -> None:
    subject_a = _subject(tenant_id="tenant-a")
    subject_b = _subject(tenant_id="tenant-b")
    key_a = ExternalContractExpectationKey(
        subject=subject_a,
        expected_contract=ExternalContractPin("contract/ref", "v1"),
    )
    key_b = ExternalContractExpectationKey(
        subject=subject_b,
        expected_contract=ExternalContractPin("contract/ref", "v1"),
    )
    assert key_a != key_b
    assert key_a.subject.tenant_id == "tenant-a"
    assert key_b.subject.tenant_id == "tenant-b"


def test_cert_meta_five_outcome_model_unchanged() -> None:
    assert frozenset(ExternalContractCompatibilityOutcome) == frozenset(
        {
            ExternalContractCompatibilityOutcome.COMPATIBLE,
            ExternalContractCompatibilityOutcome.SCHEMA_INCOMPATIBLE,
            ExternalContractCompatibilityOutcome.PROTOCOL_INCOMPATIBLE,
            ExternalContractCompatibilityOutcome.SEMANTIC_INCOMPATIBLE,
            ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE,
        }
    )


def test_cert_meta_authority_boundary_no_widening() -> None:
    paths = (
        _REPO_ROOT
        / "intergrax/integrations/external_contract_compatibility_service.py",
        _REPO_ROOT
        / "intergrax/integrations/external_contract_compatibility_extensions.py",
        _REPO_ROOT
        / "intergrax/integrations/contracts/external_contract_compatibility.py",
    )
    forbidden_fragments = (
        "CapabilityGap",
        "ToolRuntime",
        "grant_permission",
        "IntegrationCatalog",
    )
    forbidden_tokens = ("ACQUIRE", "CONFIGURE_EXISTING", "ADAPT")
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source
        for token in forbidden_tokens:
            assert re.search(rf"\b{token}\b", source) is None
        assert "runtime.execution" not in source
        assert "runtime.tool_runtime" not in source


def test_cert_meta_generic_core_vendor_purity() -> None:
    for rel in (
        "intergrax/integrations/external_contract_compatibility_service.py",
        "intergrax/integrations/external_contract_compatibility_extensions.py",
    ):
        source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        for token in (
            "custom_memory_kv",
            "jira",
            "stripe",
            "salesforce",
            "provider_id ==",
        ):
            assert token not in source
        tree = ast.parse(source)
        import_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any("examples.custom_memory_kv" in mod for mod in import_modules)


def test_cert_meta_strong_typing_surfaces() -> None:
    contracts_path = (
        _REPO_ROOT
        / "intergrax/integrations/contracts/external_contract_compatibility.py"
    )
    source = contracts_path.read_text(encoding="utf-8")
    for pattern in (
        "dict[str, Any]",
        "Mapping[str, Any]",
        ": Any",
        "getattr(",
        "setattr(",
        "hasattr(",
    ):
        assert pattern not in source


def test_cert_meta_tenant_isolation_local_verdict() -> None:
    """Cross-tenant negative families required by CERT tenant scope."""
    exp_a = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        subject=_subject(tenant_id="tenant-a"),
    )
    wrong_tenant_evidence = _schema_evidence(_subject(tenant_id="tenant-b"))
    result = _service(
        _status_evaluator(
            "schema.v1",
            ExternalContractCompatibilityDimension.SCHEMA,
            DimensionCompatibilityStatus.COMPATIBLE,
        )
    ).assess(_request(exp_a, (wrong_tenant_evidence,)))
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )
    observation = _kv_observation(tenant_id="tenant-a")
    assert (
        CustomMemoryKvContractEvidenceProvider(observation).collect(
            ExternalContractEvidenceCollectionRequest(
                expectation=ExternalContractCompatibilityExpectation(
                    expectation_id="exp-b",
                    subject=_kv_subject(tenant_id="tenant-b"),
                    expected_contract=ExternalContractPin(
                        _KV_CONTRACT_REF, _KV_CONTRACT_VERSION
                    ),
                    required_dimensions=frozenset(
                        {ExternalContractCompatibilityDimension.SCHEMA}
                    ),
                    schema_expectation_ref="schema/custom_memory_kv/get",
                ),
                assessed_at=_TS,
                assessment_window=_WINDOW,
            )
        )
        == ()
    )
