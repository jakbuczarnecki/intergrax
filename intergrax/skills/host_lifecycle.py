# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production Skill host lifecycle binding authority."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.skills.catalog import SkillPackageResolution
from intergrax.skills.dynamic_acquisition import SkillHostBindingMaterializer, SkillHostBindingPort
from intergrax.skills.identity import SkillPackageIdentity
from intergrax.skills.registry.profile import SkillProfile, is_skill_enabled
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.read import SkillRegistryRead
from intergrax.skills.registry.runtime import SkillRegistry


@dataclass
class SkillHostLifecycleService(SkillHostBindingPort):
    """Host-profile scoped Skill binding into registry projection and profile enablement."""

    host_profile_id: str
    registry: SkillRegistry = field(default_factory=SkillRegistry)
    skill_profile: SkillProfile = field(default_factory=SkillProfile)
    _handoff_operation_ids: set[str] = field(default_factory=set)

    def registry_read(self) -> SkillRegistryRead:
        return self.registry

    def is_bound(self, logical_skill_id: str) -> bool:
        if not self.registry.has(logical_skill_id):
            return False
        return is_skill_enabled(self.skill_profile, logical_skill_id)

    def binding_metadata(self, logical_skill_id: str) -> SkillRuntimeBindingMetadata | None:
        return self.registry.binding_metadata(logical_skill_id)

    def bind(
        self,
        *,
        operation_id: str,
        host_profile_id: str,
        resolved: SkillPackageResolution,
        materializer: SkillHostBindingMaterializer,
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
        package_identity = SkillPackageIdentity.from_candidate(candidate)

        if operation_id in self._handoff_operation_ids:
            existing = self.registry.binding_metadata(package_identity.logical_skill_id)
            if existing is None or not self.is_bound(package_identity.logical_skill_id):
                return DomainLifecycleHandoffAck(
                    disposition=DomainLifecycleHandoffDisposition.REJECTED,
                    reason_detail="idempotent handoff without bound skill",
                )
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
                domain_reference=_domain_reference(package_identity),
                reason_detail="idempotent handoff replay",
            )

        if self.is_bound(package_identity.logical_skill_id):
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="skill already bound for host profile",
            )

        skill_id, binding = materializer.materialize(resolved)
        if skill_id != package_identity.logical_skill_id:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="materialized skill id mismatch",
            )
        enabled = list(self.skill_profile.enabled)
        if skill_id not in enabled:
            enabled.append(skill_id)
            self.skill_profile = self.skill_profile.model_copy(update={"enabled": enabled})
        self._handoff_operation_ids.add(operation_id)
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference=_domain_reference(package_identity),
            reason_detail="skill bound for host profile",
        )


def _domain_reference(identity: SkillPackageIdentity) -> str:
    return (
        f"skill:{identity.logical_skill_id}@"
        f"{identity.package_version}:{identity.package_digest}"
    )


__all__ = ["SkillHostLifecycleService"]
