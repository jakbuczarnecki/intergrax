# © Artur Czarnecki. All rights reserved.

"""Durable revision-bound registry projection descriptor authority (AP-10 rehydration)."""

from __future__ import annotations

import threading
from typing import Any, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest

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


class BuildContextDescriptorSnapshot(BaseModel):
    """Pinned build-context identity required for deterministic projection rebuild."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strict_harness: bool = False
    skill_profile_json: dict[str, Any] | None = None
    tool_profile_json: dict[str, Any] | None = None
    environment_profile_id: str | None = None

    @classmethod
    def from_build_context(cls, build_context: ApplicationBuildContext) -> BuildContextDescriptorSnapshot:
        environment_profile_id = None
        environment = build_context.environment
        if environment is not None:
            environment_profile_id = environment.profile_id
        return cls(
            strict_harness=build_context.strict_harness,
            skill_profile_json=(
                build_context.skill_profile.model_dump(mode="json")
                if build_context.skill_profile is not None
                else None
            ),
            tool_profile_json=(
                build_context.tool_profile.model_dump(mode="json")
                if build_context.tool_profile is not None
                else None
            ),
            environment_profile_id=environment_profile_id,
        )

    def to_build_context(self, manifest: ApplicationManifest) -> ApplicationBuildContext:
        from intergrax.applications.contracts.environment_profile import (
            ApplicationEnvironmentProfile,
        )
        from intergrax.skills.registry.profile import SkillProfile
        from intergrax.tools.registry.profile import ToolProfile

        skill_profile = (
            SkillProfile.model_validate(self.skill_profile_json)
            if self.skill_profile_json is not None
            else None
        )
        tool_profile = (
            ToolProfile.model_validate(self.tool_profile_json)
            if self.tool_profile_json is not None
            else None
        )
        environment = None
        if self.environment_profile_id is not None:
            environment = ApplicationEnvironmentProfile(
                profile_id=self.environment_profile_id,
                display_name=self.environment_profile_id,
            )
        return ApplicationBuildContext(
            manifest=manifest,
            strict_harness=self.strict_harness,
            skill_profile=skill_profile,
            tool_profile=tool_profile,
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
    manifest_json: dict[str, Any]
    build_context_snapshot: BuildContextDescriptorSnapshot

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
        manifest_json=manifest.model_dump(mode="json"),
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
    "InMemoryRuntimeRegistryProjectionDescriptorStore",
    "PROJECTION_DESCRIPTOR_CONTRACT_VERSION",
    "RegistryProjectionDescriptorError",
    "RuntimeRegistryProjectionDescriptor",
    "RuntimeRegistryProjectionDescriptorStore",
    "SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1",
    "build_runtime_registry_projection_descriptor",
]
