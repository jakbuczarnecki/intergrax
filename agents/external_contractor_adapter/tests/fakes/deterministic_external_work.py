# © Artur Czarnecki. All rights reserved.

"""Deterministic in-memory ExternalWorkIntegration for GEC-3 adapter tests.

Not an A2A/REST stub, not a partner simulator, not GEC-8 provider material.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Sequence

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
from intergrax.integrations.contracts.external_work import ExternalWorkError

_DIGEST = "sha256:" + ("cd" * 32)
_T0 = datetime(2026, 7, 20, 14, 0, 0, tzinfo=timezone.utc)


class DeterministicExternalWorkFake:
    """Minimal sync fake proving Protocol sufficiency for Tier-2 mapping."""

    def __init__(
        self,
        *,
        capabilities: tuple[ExternalWorkCapability, ...] | None = None,
        unsupported_ops: frozenset[str] | None = None,
    ) -> None:
        self._capabilities = capabilities or (
            ExternalWorkCapability.QUOTE_FIRST,
            ExternalWorkCapability.HUMAN_CONTINUATION,
            ExternalWorkCapability.CANCELLATION,
            ExternalWorkCapability.TIMELINE,
            ExternalWorkCapability.DELIVERABLES,
            ExternalWorkCapability.EVIDENCE_REFS,
        )
        self._unsupported_ops = unsupported_ops or frozenset()
        self._by_idempotency: dict[str, ExternalWorkSnapshot] = {}
        self._by_external_task: dict[str, ExternalWorkSnapshot] = {}
        self._quotes: dict[str, CommercialQuote] = {}
        self._accept_keys: dict[str, ExternalWorkSnapshot] = {}
        self._timeline: dict[str, list[ExternalWorkTimelineEvent]] = {}
        self._deliverables: dict[str, list[ExternalDeliverableRef]] = {}
        self._evidence: dict[str, list[ExternalProviderEvidenceRef]] = {}
        self._cancel_keys: dict[str, ExternalWorkSnapshot] = {}
        self._seq = 0
        self.create_calls = 0
        self.accept_calls = 0
        self.cancel_calls = 0

    def discover(self) -> ExternalWorkProviderDescriptor:
        return ExternalWorkProviderDescriptor(
            identity=ExternalContractorIdentity(
                provider_id="gec3_deterministic_fake",
                contractor_id="fake-contractor",
                protocol_id="intergrax.external_work.v1",
                descriptor_digest=_DIGEST,
            ),
            capabilities=self._capabilities,
            protocol_id="intergrax.external_work.v1",
            descriptor_digest=_DIGEST,
            schema_id="external_work_provider_descriptor.v1",
        )

    def create_work(self, request: ExternalWorkCreateRequest) -> ExternalWorkSnapshot:
        self.create_calls += 1
        if "create_work" in self._unsupported_ops:
            raise ExternalWorkError(
                "create not supported",
                code=ExternalWorkErrorCode.OPERATION_NOT_SUPPORTED,
                provider_id=request.provider_id,
            )
        existing = self._by_idempotency.get(request.idempotency_key)
        if existing is not None:
            return existing
        self._seq += 1
        external_task_id = f"ext-gec3-{self._seq}"
        correlation = ExternalTaskCorrelation(
            task_id=request.task_id,
            run_id=request.run_id,
            correlation_id=request.correlation_id,
            provider_id=request.provider_id,
            external_task_id=external_task_id,
            idempotency_key=request.idempotency_key,
        )
        quote = CommercialQuote(
            quote_id=f"q-gec3-{self._seq}",
            task_id=request.task_id,
            run_id=request.run_id,
            external_task_id=external_task_id,
            provider_id=request.provider_id,
            version=1,
            amount=request.budget_limit
            or MoneyAmount(amount=Decimal("25.00"), currency="USD"),
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
            deliverable_count=1,
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
            ),
            ExternalWorkTimelineEvent(
                event_id=f"evt-{external_task_id}-2",
                task_id=request.task_id,
                external_task_id=external_task_id,
                provider_id=request.provider_id,
                event_kind="quote_offered",
                status=ExternalWorkStatus.QUOTE_AVAILABLE,
                provider_timestamp=_T0 + timedelta(minutes=1),
                ingested_at=_T0 + timedelta(minutes=1),
                summary="quote available",
                provider_sequence=2,
            ),
        ]
        self._deliverables[external_task_id] = [
            ExternalDeliverableRef(
                deliverable_id=f"del-{external_task_id}-1",
                task_id=request.task_id,
                external_task_id=external_task_id,
                kind="report",
                media_type="text/plain",
                resource_uri=f"workspace://deliverables/{external_task_id}/report.txt",
                content_digest=_DIGEST,
                size_bytes=128,
                created_at=_T0,
            )
        ]
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
        self.accept_calls += 1
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
        updated_quote = quote.model_copy(
            update={"lifecycle_state": QuoteLifecycleState.ACCEPTED}
        )
        updated = ExternalWorkSnapshot(
            correlation=current.correlation,
            status=ExternalWorkStatus.ACCEPTED,
            quote=updated_quote,
            created_at=current.created_at,
            updated_at=_T0 + timedelta(minutes=5),
            provider_state_label="accepted",
            deliverable_count=current.deliverable_count,
        )
        self._by_external_task[correlation.external_task_id] = updated
        if current.correlation.idempotency_key:
            self._by_idempotency[current.correlation.idempotency_key] = updated
        self._accept_keys[idempotency_key] = updated
        self._quotes[correlation.external_task_id] = updated_quote
        return updated

    def cancel_work(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        idempotency_key: str,
        reason: str = "",
    ) -> ExternalWorkSnapshot:
        self.cancel_calls += 1
        _ = reason
        if "cancel_work" in self._unsupported_ops:
            raise ExternalWorkError(
                "cancel not supported",
                code=ExternalWorkErrorCode.OPERATION_NOT_SUPPORTED,
                provider_id=correlation.provider_id,
            )
        if cached := self._cancel_keys.get(idempotency_key):
            return cached
        current = self.get_work(correlation)
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
        if current.correlation.idempotency_key:
            self._by_idempotency[current.correlation.idempotency_key] = updated
        self._cancel_keys[idempotency_key] = updated
        return updated

    def get_timeline(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        limit: int = 50,
    ) -> Sequence[ExternalWorkTimelineEvent]:
        return list(self._timeline.get(correlation.external_task_id, []))[:limit]

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
