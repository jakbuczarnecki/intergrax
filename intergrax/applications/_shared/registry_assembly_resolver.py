# © Artur Czarnecki. All rights reserved.

"""Registry assembly validation for Tier-3 environment wiring (Phase REG-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.artifact_lifecycle_state import is_resolution_allowed
from intergrax.runtime.registry.semver_compat import is_compatible_runtime
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile


@dataclass(frozen=True, slots=True)
class RegistryAssemblyValidationResult:
    """Outcome of harness registry assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class RegistryAssemblyError(ValueError):
    """Raised when a harness registry snapshot fails assembly validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def _profile_requires_tool_registry(profile: ToolProfile) -> bool:
    return bool(
        profile.enabled
        or profile.enabled_bundles
        or profile.register_all_catalog_bundles
    )


def _profile_requires_skill_registry(profile: SkillProfile) -> bool:
    return bool(
        profile.enabled
        or profile.enabled_bundles
        or profile.register_all_catalog_bundles
    )


def validate_registry_snapshot(
    snapshot: HarnessRegistrySnapshot,
    env: ApplicationEnvironmentProfile,
) -> RegistryAssemblyValidationResult:
    """Validate registry handles required by the environment profile."""
    errors: list[str] = []

    if snapshot.policy_bundle is None:
        errors.append("policy_bundle must be wired for harness hosts")

    if _profile_requires_tool_registry(env.tool_profile):
        if snapshot.tool_registry is None:
            errors.append("tool_registry required when tool_profile enables tools or bundles")
        elif not snapshot.tool_ids():
            errors.append("tool_registry is empty but tool_profile requests catalog tools")

    if _profile_requires_skill_registry(env.skill_profile):
        if snapshot.skill_registry is None:
            errors.append("skill_registry required when skill_profile enables skills or bundles")
        elif not snapshot.skill_ids():
            errors.append("skill_registry is empty but skill_profile requests catalog skills")

    if env.prompt_profile.load_on_startup and snapshot.prompt_registry is None:
        errors.append("prompt_registry required when prompt_profile.load_on_startup is True")

    if snapshot.integration_profile is None:
        errors.append("integration_profile must be present on harness registry snapshot")

    if snapshot.agent_registry is not None:
        for contract in snapshot.agent_registry.list_contracts():
            if not is_resolution_allowed(contract.lifecycle_state):
                errors.append(
                    f"agent {contract.id} lifecycle {contract.lifecycle_state.value} blocks resolution"
                )
            compat = is_compatible_runtime("1.0.0", contract.version)
            if not compat.compatible and contract.lifecycle_state is AgentLifecycleState.PRODUCTION:
                errors.append(
                    f"agent {contract.id} version {contract.version} incompatible with runtime baseline"
                )

    return RegistryAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_registry_assembly_valid(
    snapshot: HarnessRegistrySnapshot,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Raise :class:`RegistryAssemblyError` when registry assembly validation fails."""
    result = validate_registry_snapshot(snapshot, env)
    if not result.valid:
        raise RegistryAssemblyError(result.errors)
