# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral external work integration boundary (GEC-2).

Transport-agnostic Protocol for agent-to-agent, HTTP APIs, and future providers.
Domain types live in ``intergrax.contracts.external_work``. This module owns the
integration surface and structured boundary errors only — not partner SDKs, HITL,
policy, or receipts.

Idempotency rules (contract):

- ``create_work`` — mutating; must honour ``request.idempotency_key``
- ``submit_quote_acceptance`` — mutating; must honour ``idempotency_key``
- ``cancel_work`` — mutating; should be idempotent when the provider allows
- reads (``discover``, ``get_*``) — safe to retry
- do not blindly retry mutating calls; use ``ExternalWorkError.retryable`` /
  ``is_retryable_external_work_error``

Governance decisions (accept quote, pay, publish) are **not** created here — only
transmitted when already authorized as platform evidence.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from intergrax.contracts.external_work import (
    CommercialQuote,
    ExternalDeliverableRef,
    ExternalProviderEvidenceRef,
    ExternalTaskCorrelation,
    ExternalWorkCreateRequest,
    ExternalWorkErrorCode,
    ExternalWorkProviderDescriptor,
    ExternalWorkSnapshot,
    ExternalWorkTimelineEvent,
    QuoteAcceptanceEvidence,
    is_retryable_external_work_error,
)
from intergrax.integrations.contracts.base import IntegrationError


class ExternalWorkError(IntegrationError):
    """Structured boundary error — no transport-specific exception leakage."""

    def __init__(
        self,
        message: str,
        *,
        code: ExternalWorkErrorCode,
        retryable: bool | None = None,
        provider_id: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = (
            is_retryable_external_work_error(code) if retryable is None else retryable
        )
        self.provider_id = provider_id


@runtime_checkable
class ExternalWorkIntegration(Protocol):
    """
    Canonical provider-neutral external-work facade.

    Implementations: future protocol / HTTP / partner mappers. GEC-2 ships the
    Protocol and contract tests only — no live transport.
    """

    def discover(self) -> ExternalWorkProviderDescriptor:
        """Return provider identity and supported external-work capabilities."""
        ...

    def create_work(self, request: ExternalWorkCreateRequest) -> ExternalWorkSnapshot:
        """Create or correlate external work (idempotent on ``request.idempotency_key``)."""
        ...

    def get_work(self, correlation: ExternalTaskCorrelation) -> ExternalWorkSnapshot:
        """Fetch the canonical current-state snapshot for correlated work."""
        ...

    def get_quote(self, correlation: ExternalTaskCorrelation) -> CommercialQuote:
        """Retrieve the current commercial quote for correlated work."""
        ...

    def submit_quote_acceptance(
        self,
        correlation: ExternalTaskCorrelation,
        acceptance: QuoteAcceptanceEvidence,
        *,
        idempotency_key: str,
    ) -> ExternalWorkSnapshot:
        """Transmit already-authorized acceptance evidence (does not decide acceptance)."""
        ...

    def cancel_work(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        idempotency_key: str,
        reason: str = "",
    ) -> ExternalWorkSnapshot:
        """Request cancellation when the provider supports it (idempotent where allowed)."""
        ...

    def get_timeline(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        limit: int = 50,
    ) -> Sequence[ExternalWorkTimelineEvent]:
        """Return provider-observed timeline facts (not Intergrax runtime traces)."""
        ...

    def get_deliverables(
        self,
        correlation: ExternalTaskCorrelation,
    ) -> Sequence[ExternalDeliverableRef]:
        """Return workspace-safe deliverable references."""
        ...

    def get_evidence(
        self,
        correlation: ExternalTaskCorrelation,
    ) -> Sequence[ExternalProviderEvidenceRef]:
        """Return provider-supplied evidence references (not Intergrax proof)."""
        ...