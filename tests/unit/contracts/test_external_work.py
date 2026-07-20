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
    ExternalTaskCorrelation,
    ExternalWorkStatus,
    QuoteAcceptanceEvidence,
    QuoteLifecycleState,
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
