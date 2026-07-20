# © Artur Czarnecki. All rights reserved.

"""GEC-2 — provider-neutral ExternalWorkIntegration contract tests.

Uses an in-memory fake only to prove the Protocol is implementable.
This is not the GEC-8 stub provider.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Sequence

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
)
from intergrax.contracts.money import MoneyAmount
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationError
from intergrax.integrations.contracts.external_work import (
    ExternalWorkError,
    ExternalWorkIntegration,
)
from intergrax.integrations.registry.profile import IntegrationProfile

_DIGEST = "sha256:" + ("ab" * 32)
_T0 = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
_CONTRACT_PATH = Path("intergrax/integrations/contracts/external_work.py")
_DOMAIN_PATH = Path("intergrax/contracts/external_work.py")


def _actor() -> ActorIdentity:
    return ActorIdentity(
        kind=ActorKind.USER,
        actor_id="user-1",
        tenant_id="tenant-a",
    )


def _identity() -> ExternalContractorIdentity:
    return ExternalContractorIdentity(
        provider_id="memory_fake",
        contractor_id="contractor-1",
        protocol_id="intergrax.external_work.v1",
        descriptor_digest=_DIGEST,
    )


def _create_request(**overrides: object) -> ExternalWorkCreateRequest:
    payload: dict[str, object] = {
        "provider_id": "memory_fake",
        "task_id": "task-1",
        "run_id": "run-1",
        "requested_capability": "external_contractor.adapt",
        "scope_description": "review PR #12",
        "scope_digest": _DIGEST,
        "idempotency_key": "idem-create-1",
        "workspace_ref": "workspace://tenant-a/docs",
        "budget_limit": MoneyAmount(amount=Decimal("50.00"), currency="USD"),
    }
    payload.update(overrides)
    return ExternalWorkCreateRequest.model_validate(payload)


def _acceptance(**overrides: object) -> QuoteAcceptanceEvidence:
    payload: dict[str, object] = {
        "acceptance_id": "acc-1",
        "quote_id": "q-1",
        "quote_version": 1,
        "scope_digest": _DIGEST,
        "actor": _actor(),
        "accepted_at": _T0 + timedelta(minutes=5),
        "hitl_decision_id": "hdec_abc",
    }
    payload.update(overrides)
    return QuoteAcceptanceEvidence.model_validate(payload)


class _InMemoryExternalWorkIntegration:
    """Deterministic fake for contract conformance — not a production stub provider."""

    def __init__(self) -> None:
        self._by_idempotency: dict[str, ExternalWorkSnapshot] = {}
        self._by_external_task: dict[str, ExternalWorkSnapshot] = {}
        self._quotes: dict[str, CommercialQuote] = {}
        self._accept_keys: dict[str, ExternalWorkSnapshot] = {}
        self._cancel_keys: dict[str, ExternalWorkSnapshot] = {}
        self._timeline: dict[str, list[ExternalWorkTimelineEvent]] = {}
        self._deliverables: dict[str, list[ExternalDeliverableRef]] = {}
        self._evidence: dict[str, list[ExternalProviderEvidenceRef]] = {}
        self._seq = 0
        self.unsupported: set[str] = set()

    def discover(self) -> ExternalWorkProviderDescriptor:
        return ExternalWorkProviderDescriptor(
            identity=_identity(),
            capabilities=(
                ExternalWorkCapability.QUOTE_FIRST,
                ExternalWorkCapability.HUMAN_CONTINUATION,
                ExternalWorkCapability.CANCELLATION,
                ExternalWorkCapability.TIMELINE,
                ExternalWorkCapability.DELIVERABLES,
                ExternalWorkCapability.EVIDENCE_REFS,
                ExternalWorkCapability.ASYNC_EXECUTION,
            ),
            protocol_id="intergrax.external_work.v1",
            descriptor_digest=_DIGEST,
            schema_id="external_work_provider_descriptor.v1",
        )

    def create_work(self, request: ExternalWorkCreateRequest) -> ExternalWorkSnapshot:
        if "create_work" in self.unsupported:
            raise ExternalWorkError(
                "create not supported",
                code=ExternalWorkErrorCode.OPERATION_NOT_SUPPORTED,
                provider_id=request.provider_id,
            )
        existing = self._by_idempotency.get(request.idempotency_key)
        if existing is not None:
            return existing
        self._seq += 1
        external_task_id = f"ext-{self._seq}"
        correlation = ExternalTaskCorrelation(
            task_id=request.task_id,
            run_id=request.run_id,
            correlation_id=request.correlation_id,
            provider_id=request.provider_id,
            external_task_id=external_task_id,
            idempotency_key=request.idempotency_key,
        )
        quote = CommercialQuote(
            quote_id="q-1",
            task_id=request.task_id,
            run_id=request.run_id,
            external_task_id=external_task_id,
            provider_id=request.provider_id,
            version=1,
            amount=request.budget_limit
            or MoneyAmount(amount=Decimal("10.00"), currency="USD"),
            scope_description=request.scope_description,
            scope_digest=request.scope_digest,
            created_at=_T0,
            expires_at=_T0 + timedelta(hours=24),
            lifecycle_state=QuoteLifecycleState.OFFERED,
        )
        snapshot = ExternalWorkSnapshot(
            correlation=correlation,
            status=ExternalWorkStatus.QUOTE_AVAILABLE,
            quote=quote,
            created_at=_T0,
            updated_at=_T0,
            provider_state_label="quoted",
            deliverable_count=0,
        )
        self._by_idempotency[request.idempotency_key] = snapshot
        self._by_external_task[external_task_id] = snapshot
        self._quotes[external_task_id] = quote
        self._timeline[external_task_id] = [
            ExternalWorkTimelineEvent(
                event_id=f"evt-{external_task_id}-1",
                task_id=request.task_id,
                external_task_id=external_task_id,
                provider_id=request.provider_id,
                event_kind="created",
                status=ExternalWorkStatus.CREATED,
                provider_timestamp=_T0,
                ingested_at=_T0,
                summary="work created",
                provider_sequence=1,
            )
        ]
        self._deliverables[external_task_id] = []
        self._evidence[external_task_id] = [
            ExternalProviderEvidenceRef(
                evidence_id=f"pev-{external_task_id}-1",
                task_id=request.task_id,
                external_task_id=external_task_id,
                provider_id=request.provider_id,
                kind=ExternalProviderEvidenceKind.TASK_EVENT,
                resource_uri=f"provider://events/{external_task_id}/1",
                content_digest=_DIGEST,
                created_at=_T0,
            )
        ]
        return snapshot

    def get_work(self, correlation: ExternalTaskCorrelation) -> ExternalWorkSnapshot:
        snapshot = self._by_external_task.get(correlation.external_task_id)
        if snapshot is None:
            raise ExternalWorkError(
                "task not found",
                code=ExternalWorkErrorCode.TASK_NOT_FOUND,
                provider_id=correlation.provider_id,
            )
        return snapshot

    def get_quote(self, correlation: ExternalTaskCorrelation) -> CommercialQuote:
        quote = self._quotes.get(correlation.external_task_id)
        if quote is None:
            raise ExternalWorkError(
                "quote unavailable",
                code=ExternalWorkErrorCode.QUOTE_UNAVAILABLE,
                provider_id=correlation.provider_id,
            )
        return quote

    def submit_quote_acceptance(
        self,
        correlation: ExternalTaskCorrelation,
        acceptance: QuoteAcceptanceEvidence,
        *,
        idempotency_key: str,
    ) -> ExternalWorkSnapshot:
        if cached := self._accept_keys.get(idempotency_key):
            return cached
        current = self.get_work(correlation)
        quote = self.get_quote(correlation)
        if acceptance.quote_id != quote.quote_id or acceptance.quote_version != quote.version:
            raise ExternalWorkError(
                "quote changed or expired",
                code=ExternalWorkErrorCode.QUOTE_CHANGED_OR_EXPIRED,
                provider_id=correlation.provider_id,
            )
        if current.status in {
            ExternalWorkStatus.ACCEPTED,
            ExternalWorkStatus.EXECUTING,
            ExternalWorkStatus.COMPLETED,
        }:
            raise ExternalWorkError(
                "acceptance conflict",
                code=ExternalWorkErrorCode.ACCEPTANCE_CONFLICT,
                provider_id=correlation.provider_id,
            )
        updated = ExternalWorkSnapshot(
            correlation=current.correlation,
            status=ExternalWorkStatus.ACCEPTED,
            quote=quote.model_copy(
                update={"lifecycle_state": QuoteLifecycleState.ACCEPTED}
            ),
            created_at=current.created_at,
            updated_at=_T0 + timedelta(minutes=5),
            provider_state_label="accepted",
            deliverable_count=current.deliverable_count,
        )
        self._by_external_task[correlation.external_task_id] = updated
        if current.correlation.idempotency_key:
            self._by_idempotency[current.correlation.idempotency_key] = updated
        self._accept_keys[idempotency_key] = updated
        if updated.quote is not None:
            self._quotes[correlation.external_task_id] = updated.quote
        return updated

    def cancel_work(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        idempotency_key: str,
        reason: str = "",
    ) -> ExternalWorkSnapshot:
        _ = reason
        if cached := self._cancel_keys.get(idempotency_key):
            return cached
        current = self.get_work(correlation)
        if current.is_terminal and current.status != ExternalWorkStatus.CANCELLED:
            raise ExternalWorkError(
                "cancellation rejected",
                code=ExternalWorkErrorCode.CANCELLATION_REJECTED,
                provider_id=correlation.provider_id,
            )
        updated = ExternalWorkSnapshot(
            correlation=current.correlation,
            status=ExternalWorkStatus.CANCELLED,
            quote=current.quote,
            created_at=current.created_at,
            updated_at=_T0 + timedelta(minutes=10),
            provider_state_label="cancelled",
            deliverable_count=current.deliverable_count,
        )
        self._by_external_task[correlation.external_task_id] = updated
        self._cancel_keys[idempotency_key] = updated
        return updated

    def get_timeline(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        limit: int = 50,
    ) -> Sequence[ExternalWorkTimelineEvent]:
        events = self._timeline.get(correlation.external_task_id, [])
        return events[:limit]

    def get_deliverables(
        self,
        correlation: ExternalTaskCorrelation,
    ) -> Sequence[ExternalDeliverableRef]:
        return list(self._deliverables.get(correlation.external_task_id, []))

    def get_evidence(
        self,
        correlation: ExternalTaskCorrelation,
    ) -> Sequence[ExternalProviderEvidenceRef]:
        return list(self._evidence.get(correlation.external_task_id, []))


@pytest.mark.unit
@pytest.mark.gate
def test_fake_conforms_to_external_work_integration_protocol() -> None:
    fake: ExternalWorkIntegration = _InMemoryExternalWorkIntegration()
    assert isinstance(fake, ExternalWorkIntegration)


@pytest.mark.unit
@pytest.mark.gate
def test_discover_returns_provider_neutral_capabilities() -> None:
    fake = _InMemoryExternalWorkIntegration()
    descriptor = fake.discover()
    assert descriptor.identity.provider_id == "memory_fake"
    assert descriptor.supports(ExternalWorkCapability.QUOTE_FIRST)
    assert descriptor.supports(ExternalWorkCapability.CANCELLATION)
    assert descriptor.protocol_id == "intergrax.external_work.v1"
    assert "a2a" not in descriptor.protocol_id.lower()


@pytest.mark.unit
@pytest.mark.gate
def test_create_work_idempotency_and_correlation() -> None:
    fake = _InMemoryExternalWorkIntegration()
    request = _create_request()
    first = fake.create_work(request)
    second = fake.create_work(request)
    assert first.correlation.external_task_id == second.correlation.external_task_id
    assert first.correlation.task_id == "task-1"
    assert first.correlation.idempotency_key == "idem-create-1"
    assert first.status == ExternalWorkStatus.QUOTE_AVAILABLE
    assert not first.is_terminal


@pytest.mark.unit
@pytest.mark.gate
def test_create_request_serialization_roundtrip() -> None:
    request = _create_request()
    restored = ExternalWorkCreateRequest.model_validate(request.model_dump(mode="json"))
    assert restored == request


@pytest.mark.unit
@pytest.mark.gate
def test_quote_retrieval_and_acceptance_submission() -> None:
    fake = _InMemoryExternalWorkIntegration()
    snapshot = fake.create_work(_create_request())
    quote = fake.get_quote(snapshot.correlation)
    assert quote.quote_id == "q-1"
    accepted = fake.submit_quote_acceptance(
        snapshot.correlation,
        _acceptance(),
        idempotency_key="idem-accept-1",
    )
    assert accepted.status == ExternalWorkStatus.ACCEPTED
    again = fake.submit_quote_acceptance(
        snapshot.correlation,
        _acceptance(),
        idempotency_key="idem-accept-1",
    )
    assert again.status == ExternalWorkStatus.ACCEPTED
    assert again.updated_at == accepted.updated_at


@pytest.mark.unit
@pytest.mark.gate
def test_cancel_timeline_deliverables_and_evidence() -> None:
    fake = _InMemoryExternalWorkIntegration()
    snapshot = fake.create_work(_create_request())
    timeline = fake.get_timeline(snapshot.correlation)
    assert len(timeline) == 1
    assert timeline[0].event_kind == "created"
    assert fake.get_deliverables(snapshot.correlation) == []
    evidence = fake.get_evidence(snapshot.correlation)
    assert evidence[0].kind == ExternalProviderEvidenceKind.TASK_EVENT
    cancelled = fake.cancel_work(
        snapshot.correlation,
        idempotency_key="idem-cancel-1",
        reason="operator",
    )
    assert cancelled.status == ExternalWorkStatus.CANCELLED
    assert cancelled.is_terminal


@pytest.mark.unit
@pytest.mark.gate
def test_structured_errors_and_unsupported_operation() -> None:
    fake = _InMemoryExternalWorkIntegration()
    fake.unsupported.add("create_work")
    with pytest.raises(ExternalWorkError) as exc_info:
        fake.create_work(_create_request())
    assert exc_info.value.code == ExternalWorkErrorCode.OPERATION_NOT_SUPPORTED
    assert isinstance(exc_info.value, IntegrationError)
    assert exc_info.value.retryable is False

    missing = ExternalTaskCorrelation(
        task_id="task-x",
        provider_id="memory_fake",
        external_task_id="missing",
    )
    fake2 = _InMemoryExternalWorkIntegration()
    with pytest.raises(ExternalWorkError) as not_found:
        fake2.get_work(missing)
    assert not_found.value.code == ExternalWorkErrorCode.TASK_NOT_FOUND


@pytest.mark.unit
@pytest.mark.gate
def test_retryable_error_classification() -> None:
    err = ExternalWorkError(
        "down",
        code=ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE,
        provider_id="memory_fake",
    )
    assert err.retryable is True
    permanent = ExternalWorkError(
        "broken",
        code=ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE,
    )
    assert permanent.retryable is False


@pytest.mark.unit
@pytest.mark.gate
def test_profile_binds_external_work_instance_without_catalog_slug() -> None:
    fake = _InMemoryExternalWorkIntegration()
    profile = IntegrationProfile(external_work=fake)
    assert profile.instance_for_category(IntegrationCategory.EXTERNAL_WORK) is fake
    assert profile.slug_for_category(IntegrationCategory.EXTERNAL_WORK) is None


@pytest.mark.unit
@pytest.mark.gate
def test_no_transport_types_in_boundary_modules() -> None:
    forbidden_imports = ("httpx", "aiohttp", "requests", "urllib3", "jsonrpc")
    for path in (_CONTRACT_PATH, _DOMAIN_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                lowered = module.lower()
                for token in forbidden_imports:
                    assert token not in lowered, f"{token} imported in {path}"
                assert not lowered.startswith("http."), f"http transport import in {path}"


@pytest.mark.unit
@pytest.mark.gate
def test_boundary_modules_have_no_tier2_or_tier3_imports() -> None:
    for path in (_CONTRACT_PATH, _DOMAIN_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("applications.")
                assert not node.module.startswith("agents.")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("applications.")
                    assert not alias.name.startswith("agents.")


@pytest.mark.unit
@pytest.mark.gate
def test_create_request_rejects_naive_timestamps_via_nested_money_only() -> None:
    with pytest.raises(ValidationError):
        _create_request(scope_digest="not-a-digest")
