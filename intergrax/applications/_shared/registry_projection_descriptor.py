# © Artur Czarnecki. All rights reserved.

"""Durable revision-bound registry projection descriptor authority (AP-10 rehydration)."""

from __future__ import annotations

import threading
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile

_NON_EMPTY = Field(min_length=1)
SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1: Final = (
    "runtime_registry_projection_descriptor.v1"
)
PROJECTION_DESCRIPTOR_CONTRACT_VERSION: Final = "projection_descriptor_contract.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class RegistryProjectionDescriptorError(ValueError):
    """Durable projection descriptor authority failed."""


class EnvironmentIdentitySnapshot(BaseModel):
    """Revision-bound environment identity for projection rebuild.

    Registry projection validates only ``profile_id`` against
    ``runtime_revision.application_environment_id``; richer environment
    configuration does not participate in projection fingerprinting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = _NON_EMPTY

    @field_validator("profile_id")
    @classmethod
    def _strip_profile_id(cls, value: str) -> str:
        return _strip_required(value)


class BuildContextDescriptorSnapshot(BaseModel):
    """Pinned build-context identity required for deterministic projection rebuild."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strict_harness: bool = False
    skill_profile: SkillProfile | None = None
    tool_profile: ToolProfile | None = None
    environment_identity: EnvironmentIdentitySnapshot | None = None

    @classmethod
    def from_build_context(cls, build_context: ApplicationBuildContext) -> BuildContextDescriptorSnapshot:
        environment_identity = None
        environment = build_context.environment
        if environment is not None:
            environment_identity = EnvironmentIdentitySnapshot(profile_id=environment.profile_id)
        return cls(
            strict_harness=build_context.strict_harness,
            skill_profile=build_context.skill_profile,
            tool_profile=build_context.tool_profile,
            environment_identity=environment_identity,
        )

    def to_build_context(self, manifest: ApplicationManifest) -> ApplicationBuildContext:
        from intergrax.applications.contracts.environment_profile import (
            ApplicationEnvironmentProfile,
        )
        from intergrax.applications.contracts.environment_profile.bundles import HostMeta

        environment = None
        if self.environment_identity is not None:
            profile_id = self.environment_identity.profile_id
            environment = ApplicationEnvironmentProfile(
                meta=HostMeta(profile_id=profile_id),
            )
        return ApplicationBuildContext(
            manifest=manifest,
            strict_harness=self.strict_harness,
            skill_profile=self.skill_profile,
            tool_profile=self.tool_profile,
            environment=environment,
        )


class RuntimeRegistryProjectionDescriptor(BaseModel):
    """Immutable durable authority for process-local projection rehydration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1
    descriptor_version: str = PROJECTION_DESCRIPTOR_CONTRACT_VERSION
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    application_release_id: str = _NON_EMPTY
    effective_roster_revision_id: str = _NON_EMPTY
    materialized_runtime_lock_id: str = _NON_EMPTY
    materialized_runtime_lock_digest: str = _NON_EMPTY
    materialization_artifact_locator: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    materialization_topology: MaterializationTopology
    manifest: ApplicationManifest
    build_context_snapshot: BuildContextDescriptorSnapshot

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        normalized = _strip_required(value)
        if normalized != SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1:
            raise ValueError(
                f"unsupported schema version {normalized!r}; "
                f"expected {SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1!r}"
            )
        return normalized

    @field_validator("descriptor_version")
    @classmethod
    def _validate_descriptor_version(cls, value: str) -> str:
        normalized = _strip_required(value)
        if normalized != PROJECTION_DESCRIPTOR_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported descriptor contract {normalized!r}; "
                f"expected {PROJECTION_DESCRIPTOR_CONTRACT_VERSION!r}"
            )
        return normalized

    @field_validator(
        "application_id",
        "application_environment_id",
        "runtime_revision_id",
        "application_release_id",
        "effective_roster_revision_id",
        "materialized_runtime_lock_id",
        "materialized_runtime_lock_digest",
        "materialization_artifact_locator",
        "materialization_artifact_digest",
    )
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)


class RuntimeRegistryProjectionDescriptorStore(Protocol):
    """Revision-keyed durable projection descriptor persistence."""

    def put(self, descriptor: RuntimeRegistryProjectionDescriptor) -> None:
        """Persist one immutable descriptor keyed by runtime_revision_id."""

    def get_for_revision(
        self,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> RuntimeRegistryProjectionDescriptor | None:
        """Load descriptor for one revision within one application environment."""


def build_runtime_registry_projection_descriptor(
    bundle: RegistryProjectionInputBundle,
    *,
    artifact_locator: str,
    materialization_topology: MaterializationTopology,
) -> RuntimeRegistryProjectionDescriptor:
    """Build descriptor from canonical projection input bundle at activation boundary."""
    revision = bundle.runtime_revision
    roster = bundle.effective_roster
    artifact_digest = bundle.materialization_artifact_digest
    lock_id = revision.materialized_runtime_lock_id
    lock_digest = revision.materialized_runtime_lock_digest
    roster_revision_id = roster.effective_roster_revision_id
    if artifact_digest is None:
        raise RegistryProjectionDescriptorError(
            "projection input requires materialization_artifact_digest"
        )
    if lock_id is None or lock_digest is None:
        raise RegistryProjectionDescriptorError(
            "runtime revision requires materialized runtime lock identity"
        )
    if roster_revision_id is None:
        raise RegistryProjectionDescriptorError(
            "effective roster requires effective_roster_revision_id"
        )
    manifest = bundle.manifest
    if not isinstance(manifest, ApplicationManifest):
        raise RegistryProjectionDescriptorError(
            "projection descriptor requires ApplicationManifest authority"
        )
    return RuntimeRegistryProjectionDescriptor(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        application_release_id=revision.application_release_id,
        effective_roster_revision_id=roster_revision_id,
        materialized_runtime_lock_id=lock_id,
        materialized_runtime_lock_digest=lock_digest,
        materialization_artifact_locator=artifact_locator,
        materialization_artifact_digest=artifact_digest,
        materialization_topology=materialization_topology,
        manifest=manifest,
        build_context_snapshot=BuildContextDescriptorSnapshot.from_build_context(
            bundle.build_context,
        ),
    )


class InMemoryRuntimeRegistryProjectionDescriptorStore:
    """Process-local descriptor store for tests and reference composition."""

    def __init__(self) -> None:
        self._descriptors: dict[str, RuntimeRegistryProjectionDescriptor] = {}
        self._lock = threading.Lock()

    def put(self, descriptor: RuntimeRegistryProjectionDescriptor) -> None:
        revision_id = descriptor.runtime_revision_id
        with self._lock:
            existing = self._descriptors.get(revision_id)
            if existing is not None and existing != descriptor:
                raise RegistryProjectionDescriptorError(
                    f"conflicting projection descriptor for {revision_id!r}"
                )
            self._descriptors[revision_id] = descriptor

    def get_for_revision(
        self,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> RuntimeRegistryProjectionDescriptor | None:
        with self._lock:
            descriptor = self._descriptors.get(runtime_revision_id)
        if descriptor is None:
            return None
        if descriptor.application_id != application_id:
            raise RegistryProjectionDescriptorError(
                "projection descriptor application_id mismatch with lookup scope"
            )
        if descriptor.application_environment_id != application_environment_id:
            raise RegistryProjectionDescriptorError(
                "projection descriptor application_environment_id mismatch with lookup scope"
            )
        return descriptor


__all__ = [
    "BuildContextDescriptorSnapshot",
    "EnvironmentIdentitySnapshot",
    "InMemoryRuntimeRegistryProjectionDescriptorStore",
    "PROJECTION_DESCRIPTOR_CONTRACT_VERSION",
    "RegistryProjectionDescriptorError",
    "RuntimeRegistryProjectionDescriptor",
    "RuntimeRegistryProjectionDescriptorStore",
    "SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1",
    "build_runtime_registry_projection_descriptor",
]
