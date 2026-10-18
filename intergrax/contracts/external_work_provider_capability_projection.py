# © Artur Czarnecki. All rights reserved.

"""Descriptor-mapped vs GEC-2 integration invariant capability projection (R3-F02)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.external_work import (
    ExternalWorkCapability,
    ExternalWorkProviderDescriptor,
)
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)


@dataclass(frozen=True, slots=True)
class ExternalWorkIntegrationOrchestratorInvariants:
    """Mandatory GEC-2 ``ExternalWorkIntegration`` obligations (not descriptor tokens)."""

    supports_create: bool
    supports_idempotency: bool


def external_work_integration_orchestrator_invariants() -> (
    ExternalWorkIntegrationOrchestratorInvariants
):
    """Contractual obligations from ``intergrax.integrations.contracts.external_work``."""
    return ExternalWorkIntegrationOrchestratorInvariants(
        supports_create=True,
        supports_idempotency=True,
    )


def project_descriptor_mapped_provider_capabilities(
    descriptor: ExternalWorkProviderDescriptor,
) -> ExternalWorkProviderCapabilities:
    """Map only ``ExternalWorkCapability`` descriptor tokens to orchestrator flags."""
    caps = frozenset(descriptor.capabilities)
    provider_id = descriptor.identity.provider_id.strip()
    if not provider_id:
        raise ValueError("discovered provider identity lacks provider_id")
    return ExternalWorkProviderCapabilities(
        provider_id=provider_id,
        supports_create=False,
        supports_quote=ExternalWorkCapability.QUOTE_FIRST in caps,
        supports_accept=ExternalWorkCapability.QUOTE_FIRST in caps,
        supports_cancel=ExternalWorkCapability.CANCELLATION in caps,
        supports_status_polling=ExternalWorkCapability.ASYNC_EXECUTION in caps
        or ExternalWorkCapability.TIMELINE in caps,
        supports_idempotency=False,
        supports_native_provider_request_id=ExternalWorkCapability.ASYNC_EXECUTION in caps,
        supports_tool_logs=ExternalWorkCapability.EVIDENCE_REFS in caps,
        supports_receipt_timeline=ExternalWorkCapability.TIMELINE in caps
        or ExternalWorkCapability.EVIDENCE_REFS in caps,
        supports_payment_state=ExternalWorkCapability.QUOTE_FIRST in caps,
        supports_human_wait_state=ExternalWorkCapability.HUMAN_CONTINUATION in caps,
    )


def merge_descriptor_and_integration_invariants(
    descriptor: ExternalWorkProviderDescriptor,
    *,
    integration_admitted: bool,
) -> ExternalWorkProviderCapabilities:
    """Combine descriptor projection with integration invariant flags when admitted."""
    mapped = project_descriptor_mapped_provider_capabilities(descriptor)
    if not integration_admitted:
        return mapped
    invariants = external_work_integration_orchestrator_invariants()
    return mapped.model_copy(
        update={
            "supports_create": invariants.supports_create,
            "supports_idempotency": invariants.supports_idempotency,
        },
    )


__all__ = [
    "ExternalWorkIntegrationOrchestratorInvariants",
    "external_work_integration_orchestrator_invariants",
    "merge_descriptor_and_integration_invariants",
    "project_descriptor_mapped_provider_capabilities",
]
