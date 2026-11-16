# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production Skill host lifecycle binding authority."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.semver import SemVer
from intergrax.skills.catalog import SkillPackageResolution
from intergrax.skills.contracts.registered_skill import RegisteredSkill
from intergrax.skills.dynamic_acquisition import SkillHostBindingMaterializer, SkillHostBindingPort
from intergrax.skills.identity import SkillPackageIdentity
from intergrax.skills.registry.profile import SkillProfile, is_skill_enabled
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.read import SkillRegistryRead
from intergrax.skills.registry.runtime import SkillRegistry

# Canonical skill lifecycle evidence (PDC-D1-R017/R018): exactly one event per applied transition.
SKILL_EVENT_DISABLED: Final = "skill.disabled"
SKILL_EVENT_REMOVED: Final = "skill.removed"
SKILL_EVENT_UPGRADED: Final = "skill.upgraded"


class SkillLifecycleOutcome(StrEnum):
    APPLIED = "applied"
    NOOP = "noop"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SkillLifecycleTransition:
    """Typed outcome of a skill disable/remove/upgrade; ``event_type`` is set when APPLIED."""

    outcome: SkillLifecycleOutcome
    logical_skill_id: str
    event_type: str | None = None
    reason_detail: str = ""
    domain_reference: str | None = None


def _upgrade_denial(prior_version: str, candidate_version: str) -> str | None:
    """Skill upgrade version policy: strictly newer, same major (SemVer)."""
    try:
        prior = SemVer.parse(prior_version)
        candidate = SemVer.parse(candidate_version)
    except ValueError:
        return "skill versions must be SemVer to evaluate an upgrade"
    if candidate.major != prior.major:
        return "major version change is not an in-place skill upgrade"
    if (candidate.minor, candidate.patch) <= (prior.minor, prior.patch):
        return "candidate skill version is not newer than the installed version"
    return None


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

    def disable_skill(self, logical_skill_id: str) -> SkillLifecycleTransition:
        """Withdraw explicit profile enablement; the skill stays installed (PDC-D1-R018)."""
        with self._bind_lock:
            if not self.registry.has(logical_skill_id):
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    logical_skill_id,
                    reason_detail="skill not installed",
                )
            if logical_skill_id not in self.skill_profile.enabled:
                if is_skill_enabled(self.skill_profile, logical_skill_id):
                    return SkillLifecycleTransition(
                        SkillLifecycleOutcome.REJECTED,
                        logical_skill_id,
                        reason_detail="skill is enabled through a bundle; disable the bundle",
                    )
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.NOOP,
                    logical_skill_id,
                    reason_detail="skill already disabled",
                )
            remaining = [item for item in self.skill_profile.enabled if item != logical_skill_id]
            self.skill_profile = self.skill_profile.model_copy(update={"enabled": remaining})
            if is_skill_enabled(self.skill_profile, logical_skill_id):
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.APPLIED,
                    logical_skill_id,
                    event_type=SKILL_EVENT_DISABLED,
                    reason_detail="explicit enablement withdrawn; still enabled through a bundle",
                )
            return SkillLifecycleTransition(
                SkillLifecycleOutcome.APPLIED,
                logical_skill_id,
                event_type=SKILL_EVENT_DISABLED,
                reason_detail="skill disabled",
            )

    def remove_skill(self, logical_skill_id: str) -> SkillLifecycleTransition:
        """Uninstall a skill from the host registry projection (PDC-D1-R018).

        Refused while another installed skill still requires it. Replaying an old
        install handoff after removal fails closed (``idempotent handoff without bound skill``).
        """
        with self._bind_lock:
            if not self.registry.has(logical_skill_id):
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.NOOP,
                    logical_skill_id,
                    reason_detail="skill not installed",
                )
            dependents = sorted(
                registered.manifest.skill_id
                for registered in self.registry.list()
                if logical_skill_id in registered.manifest.requires_skills
                and registered.manifest.skill_id != logical_skill_id
            )
            if dependents:
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    logical_skill_id,
                    reason_detail="skill is required by installed skills: " + ", ".join(dependents),
                )
            self.registry.unregister(logical_skill_id)
            remaining = [item for item in self.skill_profile.enabled if item != logical_skill_id]
            self.skill_profile = self.skill_profile.model_copy(update={"enabled": remaining})
            return SkillLifecycleTransition(
                SkillLifecycleOutcome.APPLIED,
                logical_skill_id,
                event_type=SKILL_EVENT_REMOVED,
                reason_detail="skill removed",
            )

    def upgrade_skill(
        self,
        *,
        operation_id: str,
        host_profile_id: str,
        resolved: SkillPackageResolution,
        materializer: SkillHostBindingMaterializer,
    ) -> SkillLifecycleTransition:
        """Replace the installed release of a skill with a newer compatible one (PDC-D1-R018).

        The previous registry entry is restored when materialization fails, so a rejected
        upgrade never leaves the host without the skill. Profile enablement is preserved.
        """
        candidate = resolved.package_candidate
        skill_id = candidate.logical_skill_id
        if host_profile_id != self.host_profile_id:
            return SkillLifecycleTransition(
                SkillLifecycleOutcome.REJECTED,
                skill_id,
                reason_detail="host_profile_id mismatch",
            )
        if candidate.package_digest is None:
            return SkillLifecycleTransition(
                SkillLifecycleOutcome.REJECTED,
                skill_id,
                reason_detail="resolved package lacks digest",
            )
        package_identity = SkillPackageIdentity.from_candidate(candidate)

        with self._bind_lock:
            if operation_id in self._handoff_operation_ids:
                current = self.registry.binding_metadata(skill_id)
                if current is not None and _binding_matches_identity(current, package_identity):
                    return SkillLifecycleTransition(
                        SkillLifecycleOutcome.NOOP,
                        skill_id,
                        reason_detail="idempotent upgrade replay",
                        domain_reference=_domain_reference(package_identity),
                    )
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail="operation id already used for a different handoff",
                )
            if not self.registry.has(skill_id):
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail="skill not installed",
                )
            previous = self.registry.get(skill_id)
            prior_binding = previous.binding
            if prior_binding is None:
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail="installed skill has no release binding",
                )
            if prior_binding.package_reference != package_identity.package_reference:
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail="candidate belongs to a different package reference",
                )
            if prior_binding.content_digest == package_identity.package_digest:
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail="candidate release is already installed",
                )
            denial = _upgrade_denial(prior_binding.version_label, package_identity.package_version)
            if denial is not None:
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail=denial,
                )

            self.registry.unregister(skill_id)
            try:
                materialized_id, binding = materializer.materialize(resolved)
            except Exception:
                self._restore_registered_skill(skill_id, previous)
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail="skill materialization failed; previous release restored",
                )
            if materialized_id != skill_id or not _binding_matches_identity(binding, package_identity):
                self._restore_registered_skill(skill_id, previous)
                return SkillLifecycleTransition(
                    SkillLifecycleOutcome.REJECTED,
                    skill_id,
                    reason_detail="materialized binding identity mismatch; previous release restored",
                )
            self._handoff_operation_ids.add(operation_id)
            return SkillLifecycleTransition(
                SkillLifecycleOutcome.APPLIED,
                skill_id,
                event_type=SKILL_EVENT_UPGRADED,
                reason_detail="skill upgraded",
                domain_reference=_domain_reference(package_identity),
            )

    def _restore_registered_skill(self, skill_id: str, previous: RegisteredSkill) -> None:
        self.registry.unregister(skill_id)
        self.registry.register_or_replace(previous.manifest, binding=previous.binding)

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
    return f"skill:{identity.logical_skill_id}@{identity.package_version}:{identity.package_digest}"


__all__ = [
    "SKILL_EVENT_DISABLED",
    "SKILL_EVENT_REMOVED",
    "SKILL_EVENT_UPGRADED",
    "SkillHostLifecycleService",
    "SkillLifecycleOutcome",
    "SkillLifecycleTransition",
]
