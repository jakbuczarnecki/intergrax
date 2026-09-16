# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production Tool host lifecycle activation authority."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.tools.catalog import ToolPackageResolution
from intergrax.tools.dynamic_acquisition import ToolHostActivationMaterializer, ToolHostActivationPort
from intergrax.tools.identity import ToolPackageIdentity
from intergrax.tools.registry.provenance import ToolRuntimeActivationMetadata
from intergrax.tools.registry.runtime import ToolRegistry


@dataclass
class ToolHostLifecycleService(ToolHostActivationPort):
    """Host-profile scoped Tool activation into a runtime registry projection."""

    host_profile_id: str
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    _handoff_operation_ids: set[str] = field(default_factory=set)

    def registry_read(self) -> ToolRegistry:
        return self.registry

    def is_active(self, logical_tool_id: str) -> bool:
        return self.registry.has(logical_tool_id)

    def activation_metadata(self, logical_tool_id: str) -> ToolRuntimeActivationMetadata | None:
        return self.registry.activation_metadata(logical_tool_id)

    def activate(
        self,
        *,
        operation_id: str,
        host_profile_id: str,
        resolved: ToolPackageResolution,
        materializer: ToolHostActivationMaterializer,
    ) -> DomainLifecycleHandoffAck:
        if host_profile_id != self.host_profile_id:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="host_profile_id mismatch",
            )
        candidate = resolved.package_candidate
        if candidate.package_digest is None:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="resolved package lacks digest",
            )
        package_identity = ToolPackageIdentity.from_candidate(candidate)

        if operation_id in self._handoff_operation_ids:
            existing = self.registry.activation_metadata(package_identity.logical_tool_id)
            if existing is None:
                return DomainLifecycleHandoffAck(
                    disposition=DomainLifecycleHandoffDisposition.REJECTED,
                    reason_detail="idempotent handoff without active tool",
                )
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
                domain_reference=_domain_reference(package_identity),
                reason_detail="idempotent handoff replay",
            )

        if self.registry.has(package_identity.logical_tool_id):
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="tool already active for host profile",
            )

        tool_id, activation = materializer.materialize(resolved)
        if tool_id != package_identity.logical_tool_id:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="materialized tool id mismatch",
            )
        self._handoff_operation_ids.add(operation_id)
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference=_domain_reference(package_identity),
            reason_detail="tool activated for host profile",
        )


def _domain_reference(identity: ToolPackageIdentity) -> str:
    return (
        f"tool:{identity.logical_tool_id}@"
        f"{identity.package_version}:{identity.package_digest}"
    )


__all__ = ["ToolHostLifecycleService"]
