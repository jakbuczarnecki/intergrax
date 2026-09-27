# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import FrozenInstanceError

import pytest

from intergrax.contracts.execution_identity import validate_execution_id, validate_run_id, validate_task_id
from intergrax.integrations.contracts.external_contract_compatibility import (
    DimensionCompatibilityStatus,
    ExternalContractAssessmentWindow,
    ExternalContractCompatibilityAssessmentRequest,
    ExternalContractCompatibilityDimension,
    ExternalContractCompatibilityEvidence,
    ExternalContractCompatibilityExpectation,
    ExternalContractCompatibilityFinding,
    ExternalContractCompatibilityOutcome,
    ExternalContractCompatibilityReasonCode,
    ExternalContractCompatibilitySubject,
    ExternalContractEvidenceAuthority,
    ExternalContractPin,
    ExternalContractProtocolEvidenceFact,
    ExternalContractSchemaEvidenceFact,
    ExternalContractSemanticAssertionResult,
    ExternalContractSemanticEvidenceFact,
    ProtocolValidationStatus,
    SchemaValidationStatus,
    SemanticAssertionStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TS = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _subject(
    *,
    tenant_id: str = "tenant-a",
    provider_id: str = "prov",
    integration_kind: str = "api",
    integration_id: str = "prov:api",
    external_operation_id: str = "op.read",
    host_binding_ref: str | None = None,
    execution_task_id: object = None,
    execution_run_id: object = None,
    execution_id: object = None,
) -> ExternalContractCompatibilitySubject:
    return ExternalContractCompatibilitySubject(
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        integration_id=integration_id,
        external_operation_id=external_operation_id,
        host_binding_ref=host_binding_ref,
        execution_task_id=execution_task_id,  # type: ignore[arg-type]
        execution_run_id=execution_run_id,  # type: ignore[arg-type]
        execution_id=execution_id,  # type: ignore[arg-type]
    )


def _expectation(
    *,
    expected_contract: ExternalContractPin | None = None,
    subject: ExternalContractCompatibilitySubject | None = None,
) -> ExternalContractCompatibilityExpectation:
    return ExternalContractCompatibilityExpectation(
        expectation_id="exp-1",
        subject=subject or _subject(),
        expected_contract=expected_contract or ExternalContractPin("contract/ref", "v1"),
        required_dimensions=frozenset(
            {
                ExternalContractCompatibilityDimension.SCHEMA,
                ExternalContractCompatibilityDimension.PROTOCOL,
                ExternalContractCompatibilityDimension.SEMANTIC,
            }
        ),
        schema_expectation_ref="schema/exp",
        protocol_expectation_ref="protocol/exp",
        semantic_expectation_refs=("semantic/exp",),
    )


def test_enum_values_exact() -> None:
    assert [e.value for e in ExternalContractCompatibilityDimension] == [
        "schema",
        "protocol",
        "semantic",
    ]
    assert [e.value for e in ExternalContractCompatibilityOutcome] == [
        "compatible",
        "schema_incompatible",
        "protocol_incompatible",
        "semantic_incompatible",
        "insufficient_evidence",
    ]
    assert [e.value for e in DimensionCompatibilityStatus] == [
        "compatible",
        "incompatible",
        "insufficient_evidence",
    ]
    assert ExternalContractEvidenceAuthority.PROVIDER_ADAPTER.is_authoritative
    assert not ExternalContractEvidenceAuthority.LLM_ADVISORY.is_authoritative


def test_subject_valid_and_frozen() -> None:
    subject = _subject(
        execution_task_id=validate_task_id("task_" + "a" * 32),
        execution_run_id=validate_run_id("run_" + "b" * 32),
        execution_id=validate_execution_id("exec_" + "c" * 32),
        host_binding_ref="host-1",
    )
    assert subject.integration_id == "prov:api"
    with pytest.raises(FrozenInstanceError):
        subject.tenant_id = "x"  # type: ignore[misc]


def test_subject_rejects_empty_ids() -> None:
    with pytest.raises(ValueError):
        _subject(tenant_id="")


def test_subject_integration_id_mismatch() -> None:
    with pytest.raises(ValueError, match="integration_id"):
        _subject(integration_id="prov:api", provider_id="other", integration_kind="api")


def test_subject_invalid_execution_id() -> None:
    with pytest.raises(ValueError):
        _subject(
            execution_task_id=validate_task_id("task_" + "a" * 32),
            execution_run_id="bad",
        )


def test_contract_pin_validation() -> None:
    ExternalContractPin("ref", "v1")
    with pytest.raises(ValueError):
        ExternalContractPin("", "v1")


def test_expectation_required_dimensions_non_empty() -> None:
    with pytest.raises(ValueError):
        ExternalContractCompatibilityExpectation(
            expectation_id="e",
            subject=_subject(),
            expected_contract=ExternalContractPin("r", "v"),
            required_dimensions=frozenset(),
        )


def test_expectation_required_ref_invariants() -> None:
    with pytest.raises(ValueError, match="schema_expectation_ref"):
        ExternalContractCompatibilityExpectation(
            expectation_id="e",
            subject=_subject(),
            expected_contract=ExternalContractPin("r", "v"),
            required_dimensions=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        )


def test_assessment_window_modes() -> None:
    ExternalContractAssessmentWindow(valid_from=_TS, valid_until=_TS)
    ExternalContractAssessmentWindow(max_age=timedelta(seconds=30))
    ExternalContractAssessmentWindow(evidence_ttl_ref="ttl/policy")


def test_assessment_window_invalid_mixed() -> None:
    with pytest.raises(ValueError):
        ExternalContractAssessmentWindow(valid_from=_TS, max_age=timedelta(seconds=1))


def test_timezone_naive_rejected() -> None:
    with pytest.raises(ValueError):
        ExternalContractCompatibilityAssessmentRequest(
            assessment_id="a1",
            expectation=_expectation(),
            evidence=(),
            assessed_at=datetime(2026, 1, 1),
            assessment_window=ExternalContractAssessmentWindow(max_age=timedelta(seconds=1)),
        )


def test_schema_fact_and_evidence() -> None:
    fact = ExternalContractSchemaEvidenceFact(
        schema_ref="s",
        schema_fingerprint="fp",
        validation_status=SchemaValidationStatus.PASS,
    )
    ExternalContractCompatibilityEvidence(
        evidence_id="ev-1",
        subject=_subject(),
        observed_contract=ExternalContractPin("c", "v2"),
        dimension=ExternalContractCompatibilityDimension.SCHEMA,
        observed_at=_TS,
        authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
        evidence_refs=("ref-1",),
        fact=fact,
    )


def test_evidence_dimension_fact_mismatch() -> None:
    fact = ExternalContractProtocolEvidenceFact(
        protocol_ref=None,
        protocol_version=None,
        method=None,
        content_type=None,
        validation_status=ProtocolValidationStatus.PASS,
    )
    with pytest.raises(TypeError):
        ExternalContractCompatibilityEvidence(
            evidence_id="ev-1",
            subject=_subject(),
            observed_contract=None,
            dimension=ExternalContractCompatibilityDimension.SCHEMA,
            observed_at=_TS,
            authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
            evidence_refs=("ref-1",),
            fact=fact,
        )


def test_observed_contract_none_legal() -> None:
    fact = ExternalContractSchemaEvidenceFact(
        schema_ref=None,
        schema_fingerprint=None,
        validation_status=SchemaValidationStatus.UNKNOWN,
    )
    ExternalContractCompatibilityEvidence(
        evidence_id="ev-2",
        subject=_subject(),
        observed_contract=None,
        dimension=ExternalContractCompatibilityDimension.SCHEMA,
        observed_at=_TS,
        authority=ExternalContractEvidenceAuthority.CONTRACT_SPECIFICATION,
        evidence_refs=("ref-2",),
        fact=fact,
    )


def test_expected_v1_observed_v2_legal() -> None:
    expectation = _expectation(expected_contract=ExternalContractPin("contract/ref", "v1"))
    fact = ExternalContractSchemaEvidenceFact(
        schema_ref="s",
        schema_fingerprint=None,
        validation_status=SchemaValidationStatus.PASS,
    )
    ExternalContractCompatibilityEvidence(
        evidence_id="ev-3",
        subject=expectation.subject,
        observed_contract=ExternalContractPin("contract/ref", "v2"),
        dimension=ExternalContractCompatibilityDimension.SCHEMA,
        observed_at=_TS,
        authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
        evidence_refs=("ref-3",),
        fact=fact,
    )


def test_request_no_evidence_legal() -> None:
    ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-1",
        expectation=_expectation(),
        evidence=(),
        assessed_at=_TS,
        assessment_window=ExternalContractAssessmentWindow(max_age=timedelta(hours=1)),
    )


def test_explicit_evaluator_ids_duplicate_rejected() -> None:
    with pytest.raises(ValueError):
        ExternalContractCompatibilityAssessmentRequest(
            assessment_id="assess-1",
            expectation=_expectation(),
            evidence=(),
            assessed_at=_TS,
            assessment_window=ExternalContractAssessmentWindow(max_age=timedelta(hours=1)),
            explicit_evaluator_ids=("a", "a"),
        )


def test_finding_compatible_requires_none_reason() -> None:
    with pytest.raises(ValueError):
        ExternalContractCompatibilityFinding(
            dimension=ExternalContractCompatibilityDimension.SCHEMA,
            status=DimensionCompatibilityStatus.COMPATIBLE,
            reason_code=ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH,
            evidence_refs=("r",),
            evaluator_id="eval",
            source_authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
        )


def test_semantic_fact_unique_assertion_ids() -> None:
    assertion = ExternalContractSemanticAssertionResult(
        assertion_id="a1",
        status=SemanticAssertionStatus.PASS,
        authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
    )
    with pytest.raises(ValueError):
        ExternalContractSemanticEvidenceFact(assertions=(assertion, assertion))
