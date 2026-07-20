# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import (
    CommercialQuote,
    ExternalContractorIdentity,
    ExternalDeliverableRef,
    ExternalProviderEvidenceKind,
    ExternalProviderEvidenceRef,
    ExternalTaskCorrelation,
    ExternalWorkCapability,
    ExternalWorkCreateRequest,
    ExternalWorkErrorCode,
    ExternalWorkProviderDescriptor,
    ExternalWorkSnapshot,
    ExternalWorkStatus,
    ExternalWorkTimelineEvent,
    QuoteAcceptanceEvidence,
    QuoteLifecycleState,
    is_retryable_external_work_error,
    is_terminal_external_work_status,
    validate_content_digest,
    validate_quote_acceptance_match,
)
from intergrax.contracts.money import MoneyAmount

_CONTRACTS = (
    Path("intergrax/contracts/external_work.py"),
    Path("intergrax/contracts/money.py"),
)

_DIGEST = "sha256:" + ("ab" * 32)
_DIGEST_OTHER = "sha256:" + ("cd" * 32)
_T0 = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)


def _actor() -> ActorIdentity:
    return ActorIdentity(
        kind=ActorKind.USER,
        actor_id="user-1",
        tenant_id="tenant-a",
    )


def _quote(**overrides: object) -> CommercialQuote:
    payload: dict[str, object] = {
        "quote_id": "q-1",
        "task_id": "task-1",
        "run_id": "run-1",
        "external_task_id": "ext-task-1",
        "provider_id": "partner_stub",
        "version": 1,
        "amount": MoneyAmount(amount=Decimal("100.00"), currency="USD"),
        "scope_description": "review PR #12",
        "scope_digest": _DIGEST,
        "created_at": _T0,
        "expires_at": _T0 + timedelta(hours=24),
        "lifecycle_state": QuoteLifecycleState.OFFERED,
    }
    payload.update(overrides)
    return CommercialQuote.model_validate(payload)


def _acceptance(**overrides: object) -> QuoteAcceptanceEvidence:
    payload: dict[str, object] = {
        "acceptance_id": "acc-1",
        "quote_id": "q-1",
        "quote_version": 1,
        "scope_digest": _DIGEST,
        "actor": _actor(),
        "accepted_at": _T0 + timedelta(minutes=5),
        "hitl_decision_id": "hdec_abc",
        "interrupt_id": "int_xyz",
    }
    payload.update(overrides)
    return QuoteAcceptanceEvidence.model_validate(payload)


@pytest.mark.unit
@pytest.mark.gate
def test_external_work_status_values_and_terminal_semantics() -> None:
    expected = {
        "created",
        "initializing",
        "quote_pending",
        "quote_available",
        "waiting_for_acceptance",
        "accepted",
        "executing",
        "waiting_for_human",
        "completed",
        "failed",
        "cancelled",
        "expired",
    }
    assert {s.value for s in ExternalWorkStatus} == expected
    for status in (
        ExternalWorkStatus.COMPLETED,
        ExternalWorkStatus.FAILED,
        ExternalWorkStatus.CANCELLED,
        ExternalWorkStatus.EXPIRED,
    ):
        assert is_terminal_external_work_status(status)
    assert not is_terminal_external_work_status(ExternalWorkStatus.EXECUTING)
    assert not is_terminal_external_work_status(ExternalWorkStatus.WAITING_FOR_ACCEPTANCE)


@pytest.mark.unit
@pytest.mark.gate
def test_contractor_identity_and_correlation_roundtrip() -> None:
    identity = ExternalContractorIdentity(
        provider_id="partner_stub",
        contractor_id="contractor-42",
        external_agent_id="agent-ext-9",
        display_label="Partner Reviewer",
        protocol_id="a2a",
        descriptor_ref="card:partner_stub/contractor-42",
        descriptor_digest=_DIGEST,
    )
    correlation = ExternalTaskCorrelation(
        task_id="task-1",
        run_id="run-1",
        correlation_id="corr-1",
        provider_id=identity.provider_id,
        external_task_id="ext-task-1",
        idempotency_key="task-1:create:v1",
    )
    assert ExternalContractorIdentity.model_validate_json(identity.model_dump_json()) == identity
    assert ExternalTaskCorrelation.model_validate_json(correlation.model_dump_json()) == correlation
    # Intergrax task identity remains distinct field from external foreign key.
    assert correlation.task_id != correlation.external_task_id


@pytest.mark.unit
@pytest.mark.gate
def test_correlation_rejects_empty_ids() -> None:
    with pytest.raises(ValidationError):
        ExternalTaskCorrelation(
            task_id=" ",
            provider_id="p",
            external_task_id="e",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_commercial_quote_roundtrip_and_immutability() -> None:
    quote = _quote()
    restored = CommercialQuote.model_validate_json(quote.model_dump_json())
    assert restored == quote
    with pytest.raises(ValidationError):
        quote.version = 2  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.gate
def test_quote_rejects_invalid_version_digest_expiration() -> None:
    with pytest.raises(ValidationError):
        _quote(version=0)
    with pytest.raises(ValidationError, match="sha256"):
        _quote(scope_digest="sha256:deadbeef")
    with pytest.raises(ValidationError, match="expires_at"):
        _quote(expires_at=_T0)


@pytest.mark.unit
@pytest.mark.gate
def test_quote_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _quote(created_at=datetime(2026, 7, 20, 12, 0, 0))


@pytest.mark.unit
@pytest.mark.gate
def test_quote_acceptance_match_success_and_mismatches() -> None:
    quote = _quote()
    acceptance = _acceptance(hitl_decision_id="hdec_1")
    ok = validate_quote_acceptance_match(
        quote, acceptance, evaluation_time=_T0 + timedelta(hours=1)
    )
    assert ok.valid
    assert ok.errors == []

    id_mismatch = validate_quote_acceptance_match(
        quote,
        _acceptance(quote_id="other"),
        evaluation_time=_T0 + timedelta(hours=1),
    )
    assert not id_mismatch.valid
    assert "quote_id mismatch" in id_mismatch.errors

    version_mismatch = validate_quote_acceptance_match(
        quote,
        _acceptance(quote_version=2),
        evaluation_time=_T0 + timedelta(hours=1),
    )
    assert "quote_version mismatch" in version_mismatch.errors

    scope_mismatch = validate_quote_acceptance_match(
        quote,
        _acceptance(scope_digest=_DIGEST_OTHER),
        evaluation_time=_T0 + timedelta(hours=1),
    )
    assert "scope_digest mismatch" in scope_mismatch.errors

    expired = validate_quote_acceptance_match(
        quote,
        acceptance,
        evaluation_time=_T0 + timedelta(hours=48),
    )
    assert "quote expired" in expired.errors[0]


@pytest.mark.unit
@pytest.mark.gate
def test_acceptance_reuses_actor_identity() -> None:
    acceptance = _acceptance()
    assert isinstance(acceptance.actor, ActorIdentity)
    assert acceptance.actor.kind is ActorKind.USER


@pytest.mark.unit
@pytest.mark.gate
def test_deliverable_ref_validation() -> None:
    deliverable = ExternalDeliverableRef(
        deliverable_id="del-1",
        task_id="task-1",
        external_task_id="ext-task-1",
        kind="report",
        media_type="application/pdf",
        resource_uri="workspace://tenant-a/tasks/task-1/del-1.pdf",
        content_digest=_DIGEST,
        size_bytes=128,
        created_at=_T0,
        metadata={"safe": True},
    )
    restored = ExternalDeliverableRef.model_validate_json(deliverable.model_dump_json())
    assert restored == deliverable
    with pytest.raises(ValidationError):
        ExternalDeliverableRef(
            deliverable_id="del-1",
            task_id="task-1",
            kind="report",
            resource_uri="workspace://x",
            size_bytes=-1,
            created_at=_T0,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_content_digest_helper() -> None:
    assert validate_content_digest(_DIGEST) == _DIGEST
    with pytest.raises(ValueError, match="sha256"):
        validate_content_digest("md5:abc")


@pytest.mark.unit
@pytest.mark.gate
def test_external_work_modules_have_no_applications_or_agents_imports() -> None:
    forbidden = ("applications.", "agents.", "from applications", "from agents")
    for path in _CONTRACTS:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path} contains {token}"


@pytest.mark.unit
@pytest.mark.gate
def test_provider_descriptor_and_snapshot_compose_gec1_types() -> None:
    identity = ExternalContractorIdentity(
        provider_id="partner_stub",
        contractor_id="c-1",
        protocol_id="intergrax.external_work.v1",
    )
    descriptor = ExternalWorkProviderDescriptor(
        identity=identity,
        capabilities=(ExternalWorkCapability.QUOTE_FIRST, ExternalWorkCapability.TIMELINE),
        protocol_id="intergrax.external_work.v1",
    )
    assert descriptor.supports(ExternalWorkCapability.QUOTE_FIRST)
    assert not descriptor.supports(ExternalWorkCapability.CANCELLATION)

    correlation = ExternalTaskCorrelation(
        task_id="task-1",
        provider_id="partner_stub",
        external_task_id="ext-1",
        idempotency_key="idem-1",
    )
    snapshot = ExternalWorkSnapshot(
        correlation=correlation,
        status=ExternalWorkStatus.EXECUTING,
        created_at=_T0,
        updated_at=_T0 + timedelta(minutes=1),
        provider_state_label="running",
    )
    assert snapshot.is_terminal is False
    terminal = snapshot.model_copy(update={"status": ExternalWorkStatus.COMPLETED})
    assert terminal.is_terminal is True


@pytest.mark.unit
@pytest.mark.gate
def test_create_request_timeline_and_provider_evidence_refs() -> None:
    request = ExternalWorkCreateRequest(
        provider_id="partner_stub",
        task_id="task-1",
        requested_capability="external_contractor.adapt",
        scope_description="scope",
        scope_digest=_DIGEST,
        idempotency_key="idem-1",
    )
    restored = ExternalWorkCreateRequest.model_validate(request.model_dump(mode="json"))
    assert restored.idempotency_key == "idem-1"

    event = ExternalWorkTimelineEvent(
        event_id="e-1",
        task_id="task-1",
        external_task_id="ext-1",
        provider_id="partner_stub",
        event_kind="status",
        status=ExternalWorkStatus.EXECUTING,
        provider_timestamp=_T0,
        summary="executing",
        provider_sequence=2,
    )
    assert event.event_kind == "status"

    evidence = ExternalProviderEvidenceRef(
        evidence_id="pev-1",
        task_id="task-1",
        external_task_id="ext-1",
        provider_id="partner_stub",
        kind=ExternalProviderEvidenceKind.TOOL_LOG,
        resource_uri="provider://tools/1",
        created_at=_T0,
    )
    assert evidence.kind is ExternalProviderEvidenceKind.TOOL_LOG
    assert is_retryable_external_work_error(ExternalWorkErrorCode.PROVIDER_UNAVAILABLE)
    assert not is_retryable_external_work_error(ExternalWorkErrorCode.INVALID_REQUEST)
