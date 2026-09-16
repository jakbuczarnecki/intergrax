# © Artur Czarnecki. All rights reserved.

"""Reference Tool domain lifecycle authority for ME-14 (host-profile scoped activation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from intergrax.contracts.capability_catalog import CapabilityReleaseIdentity
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.tools.marketplace_lifecycle_handoff import (
    ToolLifecycleHandoffPayload,
)
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_DIGEST_V2,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_PACKAGE_REFERENCE_V2,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    ME14_VERSION_V2,
    register_me14_echo_for_release,
)

ME14_HOST_PROFILE_ID: Final = "host-profile-me14"


@dataclass(frozen=True, slots=True)
class ToolActivationRequest:
    """Caller-built activation slice from marketplace envelope (exact release preserved)."""

    payload: ToolLifecycleHandoffPayload
    selected_release: CapabilityReleaseIdentity


@dataclass
class ReferenceToolHostLifecycleService:
    """
    Tool-owned lifecycle: acquire/activate into a host-scoped ToolRegistry.

    Marketplace never mutates this registry; only ``activate`` after handoff.
    """

    host_profile_id: str
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    _activated_releases: dict[str, CapabilityReleaseIdentity] = field(default_factory=dict)
    _handoff_operation_ids: set[str] = field(default_factory=set)
    trust_allowed: bool = True

    def registry_read(self) -> ToolRegistry:
        return self.registry

    def is_active(self, logical_tool_id: str) -> bool:
        return self.registry.has(logical_tool_id)

    def activate(self, request: ToolActivationRequest) -> DomainLifecycleHandoffAck:
        payload = request.payload
        release = request.selected_release
        if payload.host_profile_id != self.host_profile_id:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="host_profile_id mismatch",
            )
        identity_key = CapabilityIdentityKey.from_discovery_identity(release.discovery)
        if payload.capability_identity_key != identity_key:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="capability_identity_key mismatch",
            )
        if not self.trust_allowed:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="tool trust policy denied activation",
            )
        if release.discovery.logical.logical_id != ME14_TOOL_LOGICAL_ID:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="unsupported tool logical_id for reference lifecycle",
            )
        validation_error = _validate_exact_release(release)
        if validation_error is not None:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail=validation_error,
            )
        if payload.operation_id in self._handoff_operation_ids:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
                domain_reference=_domain_reference(release),
                reason_detail="idempotent handoff replay",
            )
        if self.registry.has(ME14_TOOL_LOGICAL_ID):
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="tool already active for host profile",
            )
        register_me14_echo_for_release(self.registry, release)
        self._handoff_operation_ids.add(payload.operation_id)
        self._activated_releases[ME14_TOOL_LOGICAL_ID] = release
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference=_domain_reference(release),
            reason_detail="tool activated for host profile",
        )

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: ToolLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
        selected_release: CapabilityReleaseIdentity,
    ) -> DomainLifecycleHandoffAck:
        del request_id, correlation_id
        return self.activate(
            ToolActivationRequest(payload=payload, selected_release=selected_release),
        )


def _domain_reference(release: CapabilityReleaseIdentity) -> str:
    version = release.version_label or "unknown"
    digest = release.content_digest or "unknown"
    return f"tool:{ME14_TOOL_LOGICAL_ID}@{version}:{digest}"


def _validate_exact_release(release: CapabilityReleaseIdentity) -> str | None:
    package_reference = release.package_reference
    version_label = release.version_label
    digest = release.content_digest
    if package_reference not in (ME14_PACKAGE_REFERENCE_V1, ME14_PACKAGE_REFERENCE_V2):
        return "package_reference mismatch"
    if version_label not in (ME14_VERSION_V1, ME14_VERSION_V2):
        return "version_label mismatch"
    if digest not in (ME14_DIGEST_V1, ME14_DIGEST_V2):
        return "content_digest mismatch"
    if version_label == ME14_VERSION_V1 and digest != ME14_DIGEST_V1:
        return "version/digest binding mismatch for v1"
    if version_label == ME14_VERSION_V2 and digest != ME14_DIGEST_V2:
        return "version/digest binding mismatch for v2"
    return None


__all__ = [
    "ME14_HOST_PROFILE_ID",
    "ReferenceToolHostLifecycleService",
    "ToolActivationRequest",
]
