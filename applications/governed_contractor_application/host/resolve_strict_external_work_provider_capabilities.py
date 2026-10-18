# © Artur Czarnecki. All rights reserved.

"""Strict production provider identity + capability projection (R2-F03)."""

from __future__ import annotations

import json

from intergrax.contracts.external_work import (
    ExternalWorkCapability,
    ExternalWorkProviderDescriptor,
)
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.validation import compute_sha256_content_digest
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from governed_contractor_application.host.settings import GovernedContractorBackendSettings


class StrictExternalWorkProviderResolutionError(ValueError):
    """Strict external-work provider pin or capability validation failed."""


def project_external_work_provider_capabilities(
    descriptor: ExternalWorkProviderDescriptor,
) -> ExternalWorkProviderCapabilities:
    """Map discovered descriptor capabilities to orchestrator feature flags."""
    caps = frozenset(descriptor.capabilities)
    provider_id = descriptor.identity.provider_id.strip()
    if not provider_id:
        raise StrictExternalWorkProviderResolutionError(
            "discovered provider identity lacks provider_id",
        )
    return ExternalWorkProviderCapabilities(
        provider_id=provider_id,
        supports_create=True,
        supports_quote=ExternalWorkCapability.QUOTE_FIRST in caps,
        supports_accept=ExternalWorkCapability.QUOTE_FIRST in caps,
        supports_cancel=ExternalWorkCapability.CANCELLATION in caps,
        supports_status_polling=ExternalWorkCapability.ASYNC_EXECUTION in caps
        or ExternalWorkCapability.TIMELINE in caps,
        supports_idempotency=True,
        supports_native_provider_request_id=ExternalWorkCapability.ASYNC_EXECUTION in caps,
        supports_tool_logs=ExternalWorkCapability.EVIDENCE_REFS in caps,
        supports_receipt_timeline=ExternalWorkCapability.TIMELINE in caps
        or ExternalWorkCapability.EVIDENCE_REFS in caps,
        supports_payment_state=ExternalWorkCapability.QUOTE_FIRST in caps,
        supports_human_wait_state=ExternalWorkCapability.HUMAN_CONTINUATION in caps,
    )


def _descriptor_digest(descriptor: ExternalWorkProviderDescriptor) -> str:
    if descriptor.descriptor_digest is not None:
        return descriptor.descriptor_digest
    canonical = json.dumps(
        descriptor.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return compute_sha256_content_digest(canonical)


def resolve_strict_external_work_provider_capabilities(
    integration: ExternalWorkIntegration,
    *,
    settings: GovernedContractorBackendSettings,
) -> ExternalWorkProviderCapabilities:
    """Discover, pin-validate, and project provider capabilities for strict composition."""
    pin_id = settings.strict_external_work_provider_id
    pin_digest = settings.strict_external_work_provider_descriptor_digest
    if pin_id is None or not pin_id.strip():
        raise StrictExternalWorkProviderResolutionError(
            "strict external work requires GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_ID",
        )
    if pin_digest is None or not pin_digest.strip():
        raise StrictExternalWorkProviderResolutionError(
            "strict external work requires "
            "GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST",
        )
    try:
        descriptor = integration.discover()
    except Exception as exc:
        raise StrictExternalWorkProviderResolutionError(
            f"external work integration discover failed: {exc}",
        ) from exc
    discovered_id = descriptor.identity.provider_id.strip()
    if discovered_id != pin_id.strip():
        raise StrictExternalWorkProviderResolutionError(
            "strict external work provider_id pin mismatch with discover() identity",
        )
    discovered_digest = _descriptor_digest(descriptor)
    if discovered_digest != pin_digest.strip():
        raise StrictExternalWorkProviderResolutionError(
            "strict external work provider descriptor digest pin mismatch",
        )
    projected = project_external_work_provider_capabilities(descriptor)
    if not projected.supports_create:
        raise StrictExternalWorkProviderResolutionError(
            "strict external work requires provider create capability",
        )
    if ExternalWorkCapability.QUOTE_FIRST in descriptor.capabilities:
        if not (
            projected.supports_quote
            and projected.supports_accept
            and projected.supports_idempotency
        ):
            raise StrictExternalWorkProviderResolutionError(
                "quote-first provider lacks required orchestrator capability projection",
            )
    return projected


__all__ = [
    "StrictExternalWorkProviderResolutionError",
    "project_external_work_provider_capabilities",
    "resolve_strict_external_work_provider_capabilities",
]
