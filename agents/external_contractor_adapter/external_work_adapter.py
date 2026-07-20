# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Tier-2 external-work adapter (GEC-3 / GEC-4).

Owns mapping / correlation / normalization and continuation-evidence
forwarding only. Does not own governance, HITL decisions, policy, payments,
receipts, polling, resume engines, or transport.
"""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.contracts.external_work import (
    CommercialQuote,
    ExternalDeliverableRef,
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
)
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.integrations.contracts.external_work import (
    ExternalWorkError,
    ExternalWorkIntegration,
)
from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult

# Statuses that surface a QUOTE continuation blocker (no acceptance evidence yet).
_QUOTE_CONTINUATION_STATUSES: frozenset[ExternalWorkStatus] = frozenset(
    {
        ExternalWorkStatus.QUOTE_AVAILABLE,
        ExternalWorkStatus.WAITING_FOR_ACCEPTANCE,
    }
)

# Metadata keys consumed from AgentRunRequest / AgentStepContext (merged).
META_PROVIDER_ID = "external_work.provider_id"
META_SCOPE_DESCRIPTION = "external_work.scope_description"
META_SCOPE_DIGEST = "external_work.scope_digest"
META_IDEMPOTENCY_KEY = "external_work.idempotency_key"
META_CORRELATION_ID = "external_work.correlation_id"
META_WORKSPACE_REF = "external_work.workspace_ref"
META_BUDGET_LIMIT = "external_work.budget_limit"
META_REQUESTED_CAPABILITY = "external_work.requested_capability"
META_QUOTE_ACCEPTANCE = "external_work.quote_acceptance_evidence"
META_ACCEPTANCE_IDEMPOTENCY_KEY = "external_work.acceptance_idempotency_key"
META_SKIP_ENRICHMENT = "external_work.skip_enrichment"

_DEFAULT_CAPABILITY = "external_contractor.adapt"


class ExternalWorkAdapter:
    """Translator: Intergrax intent → ExternalWorkIntegration → canonical view."""

    def __init__(self, integration: ExternalWorkIntegration) -> None:
        self._integration = integration

    @property
    def integration(self) -> ExternalWorkIntegration:
        return self._integration

    def discover(self) -> ExternalWorkProviderDescriptor:
        return self._integration.discover()

    def build_create_request(
        self,
        *,
        task_id: str,
        run_id: str | None,
        metadata: Mapping[str, Any],
        message: str = "",
    ) -> ExternalWorkCreateRequest:
        """Map Intergrax identity + metadata into a platform create request.

        Forwards existing idempotency material; never invents retry policy or new ids.
        """
        provider_id = _require_meta_str(metadata, META_PROVIDER_ID, field="provider_id")
        scope_description = str(
            metadata.get(META_SCOPE_DESCRIPTION) or message or "external work"
        ).strip()
        scope_digest = _require_meta_str(metadata, META_SCOPE_DIGEST, field="scope_digest")
        idempotency_key = _require_meta_str(
            metadata, META_IDEMPOTENCY_KEY, field="idempotency_key"
        )
        requested = str(
            metadata.get(META_REQUESTED_CAPABILITY) or _DEFAULT_CAPABILITY
        ).strip()
        budget_raw = metadata.get(META_BUDGET_LIMIT)
        budget_limit: MoneyAmount | None = None
        if budget_raw is not None:
            budget_limit = (
                budget_raw
                if isinstance(budget_raw, MoneyAmount)
                else MoneyAmount.model_validate(budget_raw)
            )
        return ExternalWorkCreateRequest(
            provider_id=provider_id,
            task_id=task_id,
            run_id=run_id or None,
            correlation_id=_optional_meta_str(metadata, META_CORRELATION_ID),
            requested_capability=requested,
            scope_description=scope_description,
            scope_digest=scope_digest,
            idempotency_key=idempotency_key,
            workspace_ref=_optional_meta_str(metadata, META_WORKSPACE_REF),
            budget_limit=budget_limit,
            metadata={
                k: v
                for k, v in metadata.items()
                if isinstance(k, str) and k.startswith("external_work.ext.")
            },
        )

    def create_and_map(
        self,
        request: ExternalWorkCreateRequest,
        *,
        acceptance: QuoteAcceptanceEvidence | None = None,
        acceptance_idempotency_key: str | None = None,
        enrich: bool = True,
    ) -> ExternalWorkAdapterResult:
        """Synchronous create/correlate + optional enrich; no poll/retry loops."""
        try:
            provider = self._integration.discover()
            snapshot = self._integration.create_work(request)
            if acceptance is not None:
                key = (acceptance_idempotency_key or "").strip()
                if not key:
                    return ExternalWorkAdapterResult(
                        used=False,
                        reason="acceptance_idempotency_key_required",
                        error_code=ExternalWorkErrorCode.INVALID_REQUEST,
                        error_message=(
                            "quote acceptance forward requires acceptance_idempotency_key"
                        ),
                        error_retryable=False,
                        provider=provider,
                        snapshot=snapshot,
                        status=snapshot.status,
                        quote=snapshot.quote,
                    )
                # Forward already-authorized evidence only — never decide acceptance.
                snapshot = self._integration.submit_quote_acceptance(
                    snapshot.correlation,
                    acceptance,
                    idempotency_key=key,
                )
            if not enrich:
                return ExternalWorkAdapterResult(
                    used=True,
                    reason="mapped",
                    status=snapshot.status,
                    snapshot=snapshot,
                    quote=snapshot.quote,
                    provider=provider,
                )
            return self._enrich(snapshot, provider=provider)
        except ExternalWorkError as exc:
            return _error_result(exc, provider=None)

    def map_existing(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        enrich: bool = True,
    ) -> ExternalWorkAdapterResult:
        """Fetch and normalize an already-correlated external task."""
        try:
            provider = self._integration.discover()
            snapshot = self._integration.get_work(correlation)
            if not enrich:
                return ExternalWorkAdapterResult(
                    used=True,
                    reason="mapped",
                    status=snapshot.status,
                    snapshot=snapshot,
                    quote=snapshot.quote,
                    provider=provider,
                )
            return self._enrich(snapshot, provider=provider)
        except ExternalWorkError as exc:
            return _error_result(exc, provider=None)

    def forward_quote_acceptance(
        self,
        correlation: ExternalTaskCorrelation,
        acceptance: QuoteAcceptanceEvidence,
        *,
        idempotency_key: str,
    ) -> ExternalWorkAdapterResult:
        """Transmit pre-authorized acceptance evidence — does not accept/reject."""
        try:
            provider = self._integration.discover()
            snapshot = self._integration.submit_quote_acceptance(
                correlation,
                acceptance,
                idempotency_key=idempotency_key,
            )
            return self._enrich(snapshot, provider=provider)
        except ExternalWorkError as exc:
            return _error_result(exc, provider=None)

    def forward_continuation_evidence(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        reason: ContinuationReason,
        evidence: QuoteAcceptanceEvidence,
        idempotency_key: str,
    ) -> ExternalWorkAdapterResult:
        """Forward continuation evidence without evaluating governance.

        QUOTE is the first specialization — evidence is ``QuoteAcceptanceEvidence``.
        Other reasons are rejected as unsupported at this adapter (no domain map).
        """
        if reason is not ContinuationReason.QUOTE:
            return ExternalWorkAdapterResult(
                used=False,
                reason="continuation_reason_unsupported",
                error_code=ExternalWorkErrorCode.OPERATION_NOT_SUPPORTED,
                error_message=(
                    f"external work adapter forwards only {ContinuationReason.QUOTE.value} "
                    f"continuation evidence (got {reason.value})"
                ),
                error_retryable=False,
                metadata={"continuation_reason": reason.value},
            )
        # Propagate only — never decide accept/reject or resume Nexus.
        return self.forward_quote_acceptance(
            correlation,
            evidence,
            idempotency_key=idempotency_key,
        )

    def surface_continuation_blocker(
        self,
        result: ExternalWorkAdapterResult,
        *,
        run_id: str,
        source_agent_id: str = "external_contractor_adapter",
        source_step_id: str | None = None,
    ) -> GovernedContinuationRequest | None:
        """Detect a QUOTE continuation blocker from a mapped result.

        Surfaces a ``GovernedContinuationRequest`` for Nexus interrupt composition.
        Requires a real Nexus ``run_id`` — never synthesizes one from ``task_id``.
        Does not create interrupts, call policy, or resume execution.
        """
        if not result.used or result.snapshot is None:
            return None
        if result.status not in _QUOTE_CONTINUATION_STATUSES:
            return None
        if result.quote is None:
            return None
        resolved_run_id = (run_id or "").strip()
        if not resolved_run_id:
            return None
        correlation = result.snapshot.correlation
        return GovernedContinuationRequest(
            reason=ContinuationReason.QUOTE,
            task_id=correlation.task_id,
            run_id=resolved_run_id,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
            prompt="External work quote requires governed continuation before side effects",
            correlation={
                "task_id": correlation.task_id,
                "run_id": correlation.run_id,
                "external_task_id": correlation.external_task_id,
                "provider_id": correlation.provider_id,
                "idempotency_key": correlation.idempotency_key,
                "correlation_id": correlation.correlation_id,
            },
            context={
                "quote_id": result.quote.quote_id,
                "quote_version": result.quote.version,
                "scope_digest": result.quote.scope_digest,
                "external_work_status": (
                    result.status.value if result.status is not None else None
                ),
            },
        )

    def with_continuation_surface(
        self,
        result: ExternalWorkAdapterResult,
        *,
        run_id: str | None,
        source_agent_id: str = "external_contractor_adapter",
        source_step_id: str | None = None,
    ) -> ExternalWorkAdapterResult:
        """Attach continuation blocker when quote awaits governance evidence.

        Fail-closed when a blocker is required but no real Nexus run identity
        is available — never substitutes ``task_id`` for ``run_id``.
        """
        if not _needs_quote_continuation(result):
            return result
        resolved_run_id = (run_id or "").strip()
        if not resolved_run_id:
            return result.model_copy(
                update={
                    "used": False,
                    "reason": "continuation_correlation_failed",
                    "continuation": None,
                    "error_code": ExternalWorkErrorCode.INVALID_REQUEST,
                    "error_message": (
                        "A governed continuation blocker requires a real "
                        "Nexus run identity."
                    ),
                    "error_retryable": False,
                }
            )
        blocker = self.surface_continuation_blocker(
            result,
            run_id=resolved_run_id,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
        )
        if blocker is None:
            return result
        return result.model_copy(
            update={
                "reason": "continuation_blocked",
                "continuation": blocker,
            }
        )

    def _enrich(
        self,
        snapshot: ExternalWorkSnapshot,
        *,
        provider: ExternalWorkProviderDescriptor,
    ) -> ExternalWorkAdapterResult:
        correlation = snapshot.correlation
        unsupported: list[ExternalWorkCapability] = []
        quote = snapshot.quote
        timeline: tuple[ExternalWorkTimelineEvent, ...] = ()
        deliverables: tuple[ExternalDeliverableRef, ...] = ()
        evidence: tuple[ExternalProviderEvidenceRef, ...] = ()

        if provider.supports(ExternalWorkCapability.QUOTE_FIRST):
            if quote is None:
                try:
                    quote = self._integration.get_quote(correlation)
                except ExternalWorkError as exc:
                    if exc.code != ExternalWorkErrorCode.QUOTE_UNAVAILABLE:
                        return _error_result(exc, provider=provider)
        else:
            unsupported.append(ExternalWorkCapability.QUOTE_FIRST)

        if provider.supports(ExternalWorkCapability.TIMELINE):
            timeline = tuple(self._integration.get_timeline(correlation))
        else:
            unsupported.append(ExternalWorkCapability.TIMELINE)

        if provider.supports(ExternalWorkCapability.DELIVERABLES):
            deliverables = tuple(self._integration.get_deliverables(correlation))
        else:
            unsupported.append(ExternalWorkCapability.DELIVERABLES)

        if provider.supports(ExternalWorkCapability.EVIDENCE_REFS):
            evidence = tuple(self._integration.get_evidence(correlation))
        else:
            unsupported.append(ExternalWorkCapability.EVIDENCE_REFS)

        return ExternalWorkAdapterResult(
            used=True,
            reason="mapped",
            status=snapshot.status,
            snapshot=snapshot,
            quote=quote,
            timeline=timeline,
            deliverables=deliverables,
            evidence=evidence,
            provider=provider,
            unsupported_capabilities=tuple(unsupported),
            metadata={
                "external_task_id": correlation.external_task_id,
                "provider_id": correlation.provider_id,
                "idempotency_key": correlation.idempotency_key,
            },
        )


def adapt_from_step_metadata(
    integration: ExternalWorkIntegration | None,
    *,
    task_id: str,
    run_id: str | None,
    message: str,
    metadata: Mapping[str, Any],
) -> ExternalWorkAdapterResult:
    """Entry used by the reflex domain step — injection required, no construction."""
    if integration is None:
        return ExternalWorkAdapterResult(
            used=False,
            reason="external_work_integration_missing",
            metadata={"capability": _DEFAULT_CAPABILITY},
        )

    adapter = ExternalWorkAdapter(integration)
    try:
        request = adapter.build_create_request(
            task_id=task_id,
            run_id=run_id,
            metadata=metadata,
            message=message,
        )
    except ValueError as exc:
        return ExternalWorkAdapterResult(
            used=False,
            reason="invalid_adapt_intent",
            error_code=ExternalWorkErrorCode.INVALID_REQUEST,
            error_message=str(exc),
            error_retryable=False,
        )

    acceptance = _parse_acceptance(metadata.get(META_QUOTE_ACCEPTANCE))
    acceptance_key = _optional_meta_str(metadata, META_ACCEPTANCE_IDEMPOTENCY_KEY)
    enrich = not bool(metadata.get(META_SKIP_ENRICHMENT))
    mapped = adapter.create_and_map(
        request,
        acceptance=acceptance,
        acceptance_idempotency_key=acceptance_key,
        enrich=enrich,
    )
    # When no continuation evidence was supplied, surface the blocker for Nexus.
    # Forward the real Nexus run_id from execution context — never invent one.
    if acceptance is None and mapped.used:
        return adapter.with_continuation_surface(mapped, run_id=run_id)
    return mapped


def _needs_quote_continuation(result: ExternalWorkAdapterResult) -> bool:
    """True when mapped state requires a QUOTE governed-continuation blocker."""
    return (
        result.used
        and result.snapshot is not None
        and result.quote is not None
        and result.status in _QUOTE_CONTINUATION_STATUSES
    )


def _error_result(
    exc: ExternalWorkError,
    *,
    provider: ExternalWorkProviderDescriptor | None,
) -> ExternalWorkAdapterResult:
    return ExternalWorkAdapterResult(
        used=False,
        reason="external_work_error",
        error_code=exc.code,
        error_message=str(exc),
        error_retryable=exc.retryable,
        provider=provider,
    )


def _require_meta_str(metadata: Mapping[str, Any], key: str, *, field: str) -> str:
    raw = metadata.get(key)
    if raw is None or not str(raw).strip():
        raise ValueError(f"missing required metadata {key!r} ({field})")
    return str(raw).strip()


def _optional_meta_str(metadata: Mapping[str, Any], key: str) -> str | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    normalized = str(raw).strip()
    return normalized or None


def _parse_acceptance(raw: object) -> QuoteAcceptanceEvidence | None:
    if raw is None:
        return None
    if isinstance(raw, QuoteAcceptanceEvidence):
        return raw
    return QuoteAcceptanceEvidence.model_validate(raw)
