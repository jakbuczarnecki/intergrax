# © Artur Czarnecki. All rights reserved.

"""Harness host registry authority (AGENT-CONSOLIDATION-3)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionEvidence,
)
from intergrax.applications._shared.wiring import (
    BuilderMap,
    build_manifest_development_registry,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.agent_registry import AgentRegistry

if TYPE_CHECKING:
    pass


class RegistryAssemblyMode(str, Enum):
    """Explicit registry authority mode for harness host assembly."""

    REVISION_BOUND = "revision_bound"
    MANIFEST_DEVELOPMENT = "manifest_development"


class HarnessHostRegistryAuthorityError(ValueError):
    """Revision-bound production registry authority violation."""


def resolve_registry_assembly_mode(
    environment: ApplicationEnvironmentProfile,
    *,
    explicit: RegistryAssemblyMode | None = None,
) -> RegistryAssemblyMode:
    """Infer assembly mode from explicit override or environment execution posture."""
    if environment.execution_mode == ExecutionMode.STRICT:
        if explicit is RegistryAssemblyMode.MANIFEST_DEVELOPMENT:
            raise HarnessHostRegistryAuthorityError(
                "STRICT execution mode requires revision-bound registry authority"
            )
        return RegistryAssemblyMode.REVISION_BOUND
    if explicit is not None:
        return explicit
    return RegistryAssemblyMode.MANIFEST_DEVELOPMENT


def validate_registry_projection_scope(
    projection: MaterializedRegistryProjection,
    *,
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
) -> None:
    """Fail closed when projection evidence does not match host identity."""
    evidence = projection.evidence
    if evidence.application_id != manifest.app_id:
        raise HarnessHostRegistryAuthorityError(
            "registry projection application_id "
            f"{evidence.application_id!r} does not match manifest {manifest.app_id!r}"
        )
    if evidence.application_environment_id != environment.profile_id:
        raise HarnessHostRegistryAuthorityError(
            "registry projection application_environment_id "
            f"{evidence.application_environment_id!r} does not match host "
            f"environment {environment.profile_id!r}"
        )


def resolve_harness_host_registry(
    *,
    manifest: ApplicationManifest,
    build_context: ApplicationBuildContext,
    environment: ApplicationEnvironmentProfile,
    assembly_mode: RegistryAssemblyMode,
    registry_projection: MaterializedRegistryProjection | None = None,
    registry: AgentRegistry | None = None,
    builders: BuilderMap | None = None,
) -> tuple[AgentRegistry, RegistryProjectionEvidence | None]:
    """Resolve harness host registry authority without silent manifest fallback."""
    if assembly_mode is RegistryAssemblyMode.REVISION_BOUND:
        if registry_projection is None:
            raise HarnessHostRegistryAuthorityError(
                "revision-bound host assembly requires MaterializedRegistryProjection; "
                "manifest-only registry fallback is forbidden"
            )
        if registry is not None:
            raise HarnessHostRegistryAuthorityError(
                "anonymous AgentRegistry override is forbidden for revision-bound assembly"
            )
        if builders is not None:
            raise HarnessHostRegistryAuthorityError(
                "host builders fallback is forbidden for revision-bound assembly"
            )
        validate_registry_projection_scope(
            registry_projection,
            manifest=manifest,
            environment=environment,
        )
        return registry_projection.agent_registry, registry_projection.evidence

    if registry_projection is not None:
        raise HarnessHostRegistryAuthorityError(
            "registry_projection is forbidden for manifest development assembly; "
            "use RegistryAssemblyMode.REVISION_BOUND for production authority"
        )
    if registry is not None:
        return registry, None
    return (
        build_manifest_development_registry(
            manifest,
            build_context,
            builders=builders,
        ),
        None,
    )


__all__ = [
    "HarnessHostRegistryAuthorityError",
    "RegistryAssemblyMode",
    "resolve_harness_host_registry",
    "resolve_registry_assembly_mode",
    "validate_registry_projection_scope",
]
