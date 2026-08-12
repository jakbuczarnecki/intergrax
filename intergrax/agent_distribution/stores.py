# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable store ports for Agent Distribution (AGENT_DISTRIBUTION §23)."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.installation import AgentInstallationRecord
from intergrax.agent_distribution.runtime_revision import RuntimeRevision


class AgentArtifactMetadata(BaseModel):
    """Artifact metadata owned by the distribution plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_digest: str = Field(min_length=1)
    artifact_store_ref: str = Field(min_length=1)
    distribution_package_id: str = Field(min_length=1)
    tombstoned: bool = False


class AgentInstallationStore(Protocol):
    """Installation record persistence — domain-shaped access only."""

    def get_installation(self, installation_id: str) -> AgentInstallationRecord | None:
        """Load one immutable installation revision by id."""

    def get_active_installation_for_slot(
        self,
        installation_slot_id: str,
    ) -> AgentInstallationRecord | None:
        """Resolve the active digest-pinned installation for a slot."""

    def list_installations_for_slot(
        self,
        installation_slot_id: str,
    ) -> list[AgentInstallationRecord]:
        """List installation revisions for a slot (audit / rollback)."""

    def persist_installation(
        self,
        record: AgentInstallationRecord,
        *,
        expected_active_installation_id: str | None = None,
    ) -> AgentInstallationRecord:
        """Persist installation with serialized slot updates."""


class ApplicationAgentBindingStore(Protocol):
    """Durable application agent binding persistence."""

    def get_binding(self, application_binding_id: str) -> ApplicationAgentBinding | None:
        """Load one binding by stable id."""

    def list_bindings_for_environment(
        self,
        application_environment_id: str,
    ) -> list[ApplicationAgentBinding]:
        """List bindings scoped to an application environment."""

    def persist_binding(
        self,
        binding: ApplicationAgentBinding,
        *,
        expected_revision: int | None = None,
    ) -> ApplicationAgentBinding:
        """Persist binding with optimistic revision concurrency."""


class MaterializedRuntimeLockStore(Protocol):
    """Immutable lock artifact persistence."""

    def get_lock(self, lock_id: str) -> MaterializedRuntimeLock | None:
        """Load lock by content id."""

    def get_lock_by_digest(self, lock_digest: str) -> MaterializedRuntimeLock | None:
        """Load lock by digest."""

    def persist_lock(self, lock: MaterializedRuntimeLock) -> MaterializedRuntimeLock:
        """Persist immutable lock artifact (reject digest mismatch)."""


class RuntimeRevisionStore(Protocol):
    """Runtime revision and activation pointer persistence."""

    def get_revision(self, runtime_revision_id: str) -> RuntimeRevision | None:
        """Load one runtime revision by immutable id."""

    def get_active_revision(
        self,
        application_environment_id: str,
    ) -> RuntimeRevision | None:
        """Resolve the active runtime revision for an environment."""

    def persist_candidate_revision(
        self,
        revision: RuntimeRevision,
        *,
        expected_revision_state: str | None = None,
    ) -> RuntimeRevision:
        """Persist candidate/validated revision with optimistic state guard."""

    def swap_active_revision(
        self,
        *,
        application_environment_id: str,
        new_active_revision_id: str,
        prior_active_revision_id: str | None = None,
    ) -> RuntimeRevision:
        """Atomically promote validated revision to active."""


class AgentArtifactMetadataStore(Protocol):
    """Digest-pinned artifact metadata persistence."""

    def get_by_digest(self, package_digest: str) -> AgentArtifactMetadata | None:
        """Resolve artifact metadata by immutable digest."""

    def persist_metadata(self, metadata: AgentArtifactMetadata) -> AgentArtifactMetadata:
        """Persist artifact metadata record."""
