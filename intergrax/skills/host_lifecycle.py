# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production Skill host lifecycle binding authority."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.skills.catalog import SkillPackageResolution
from intergrax.skills.dynamic_acquisition import SkillHostBindingMaterializer, SkillHostBindingPort
from intergrax.skills.identity import SkillPackageIdentity
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.read import SkillRegistryRead
from intergrax.skills.registry.runtime import SkillRegistry


def _binding_matches_identity(
    binding: SkillRuntimeBindingMetadata,
    identity: SkillPackageIdentity,
) -> bool:
    return (
        binding.logical_skill_id == identity.logical_skill_id
        and binding.package_reference == identity.package_reference
        and binding.version_label == identity.package_version
        and binding.content_digest == identity.package_digest
    )


@dataclass
class SkillHostLifecycleService(SkillHostBindingPort):
    """Host-profile scoped Skill installation into registry projection (not profile enablement).

    Registry mutations during ``bind`` are serialized per service instance (single-writer).
    """

    host_profile_id: str
    registry: SkillRegistry = field(default_factory=SkillRegistry)
    skill_profile: SkillProfile = field(default_factory=SkillProfile)
    _handoff_operation_ids: set[str] = field(default_factory=set)
    _bind_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _revert_registry_mutations_since(self, prior_skill_ids: frozenset[str]) -> None:
        for skill_id in set(self.registry.skill_ids()) - prior_skill_ids:
            self.registry.unregister(skill_id)

    def registry_read(self) -> SkillRegistryRead:
        return self.registry

    def is_bound(self, logical_skill_id: str) -> bool:
        """True when the skill is installed on this host profile registry projection."""
        return self.registry.has(logical_skill_id)

    def binding_metadata(self, logical_skill_id: str) -> SkillRuntimeBindingMetadata | None:
        return self.registry.binding_metadata(logical_skill_id)

    def enable_installed_skill(self, logical_skill_id: str) -> None:
        """Explicit profile enablement for an installed skill (sanctioned configuration)."""
        if not self.registry.has(logical_skill_id):
            raise ValueError(f"skill not installed: {logical_skill_id}")
        enabled = list(self.skill_profile.enabled)
        if logical_skill_id not in enabled:
            enabled.append(logical_skill_id)
            self.skill_profile = self.skill_profile.model_copy(update={"enabled": enabled})

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
        skill_id = package_identity.logical_skill_id

        with self._bind_lock:
            if operation_id in self._handoff_operation_ids:
                existing = self.registry.binding_metadata(skill_id)
                if existing is None or not self.registry.has(skill_id):
                    return DomainLifecycleHandoffAck(
                        disposition=DomainLifecycleHandoffDisposition.REJECTED,
                        reason_detail="idempotent handoff without bound skill",
                    )
                if not _binding_matches_identity(existing, package_identity):
                    return DomainLifecycleHandoffAck(
                        disposition=DomainLifecycleHandoffDisposition.REJECTED,
                        reason_detail="idempotent handoff package identity mismatch",
                    )
                return DomainLifecycleHandoffAck(
                    disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
                    domain_reference=_domain_reference(package_identity),
                    reason_detail="idempotent handoff replay",
                )

            if self.registry.has(skill_id):
                existing = self.registry.binding_metadata(skill_id)
                if existing is not None and _binding_matches_identity(existing, package_identity):
                    return DomainLifecycleHandoffAck(
                        disposition=DomainLifecycleHandoffDisposition.REJECTED,
                        reason_detail="skill already bound for host profile",
                    )
                return DomainLifecycleHandoffAck(
                    disposition=DomainLifecycleHandoffDisposition.REJECTED,
                    reason_detail="incompatible package identity for installed skill",
                )

            prior_skill_ids = frozenset(self.registry.skill_ids())
            try:
                materialized_id, binding = materializer.materialize(resolved)
            except Exception:
                self._revert_registry_mutations_since(prior_skill_ids)
                return DomainLifecycleHandoffAck(
                    disposition=DomainLifecycleHandoffDisposition.REJECTED,
                    reason_detail="skill materialization failed",
                )

            if materialized_id != skill_id:
                self._revert_registry_mutations_since(prior_skill_ids)
                return DomainLifecycleHandoffAck(
                    disposition=DomainLifecycleHandoffDisposition.REJECTED,
                    reason_detail="materialized skill id mismatch",
                )
            if not _binding_matches_identity(binding, package_identity):
                self._revert_registry_mutations_since(prior_skill_ids)
                return DomainLifecycleHandoffAck(
                    disposition=DomainLifecycleHandoffDisposition.REJECTED,
                    reason_detail="materialized binding identity mismatch",
                )

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
