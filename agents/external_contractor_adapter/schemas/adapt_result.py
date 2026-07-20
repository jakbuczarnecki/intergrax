# © Artur Czarnecki. All rights reserved.

"""Canonical Tier-2 adapter result — composition of GEC-1/GEC-2 platform contracts."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.external_work import (
    CommercialQuote,
    ExternalDeliverableRef,
    ExternalProviderEvidenceRef,
    ExternalWorkCapability,
    ExternalWorkErrorCode,
    ExternalWorkProviderDescriptor,
    ExternalWorkSnapshot,
    ExternalWorkStatus,
    ExternalWorkTimelineEvent,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.runtime_policy import PolicyDecision


class ExternalWorkAdapterResult(BaseModel):
    """Normalized adapter view for Nexus / callers — not a governance decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_work_adapter_result.v1"] = (
        "external_work_adapter_result.v1"
    )
    used: bool
    reason: str = Field(min_length=1)
    capability: str = "external_contractor.adapt"
    status: ExternalWorkStatus | None = None
    snapshot: ExternalWorkSnapshot | None = None
    quote: CommercialQuote | None = None
    timeline: tuple[ExternalWorkTimelineEvent, ...] = ()
    deliverables: tuple[ExternalDeliverableRef, ...] = ()
    evidence: tuple[ExternalProviderEvidenceRef, ...] = ()
    provider: ExternalWorkProviderDescriptor | None = None
    unsupported_capabilities: tuple[ExternalWorkCapability, ...] = ()
    continuation: GovernedContinuationRequest | None = None
    policy_decision: PolicyDecision | None = None
    error_code: ExternalWorkErrorCode | None = None
    error_message: str | None = None
    error_retryable: bool | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    def to_domain_summary(self) -> dict[str, object]:
        """Compact step output fragment (JSON-safe)."""
        summary: dict[str, object] = {
            "used": self.used,
            "reason": self.reason,
            "capability": self.capability,
            "status": self.status.value if self.status is not None else None,
            "unsupported_capabilities": [c.value for c in self.unsupported_capabilities],
            "error_code": self.error_code.value if self.error_code is not None else None,
            "error_message": self.error_message,
            "error_retryable": self.error_retryable,
        }
        if self.snapshot is not None:
            summary["correlation"] = self.snapshot.correlation.model_dump(mode="json")
            summary["snapshot"] = self.snapshot.model_dump(mode="json")
        if self.quote is not None:
            summary["quote"] = self.quote.model_dump(mode="json")
        if self.timeline:
            summary["timeline"] = [e.model_dump(mode="json") for e in self.timeline]
        if self.deliverables:
            summary["deliverables"] = [d.model_dump(mode="json") for d in self.deliverables]
        if self.evidence:
            summary["evidence"] = [e.model_dump(mode="json") for e in self.evidence]
        if self.provider is not None:
            summary["provider"] = self.provider.model_dump(mode="json")
        if self.continuation is not None:
            summary["continuation"] = self.continuation.model_dump(mode="json")
        if self.policy_decision is not None:
            summary["policy_decision"] = self.policy_decision.model_dump(mode="json")
        if self.metadata:
            summary["metadata"] = dict(self.metadata)
        return summary
