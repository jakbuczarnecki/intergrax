# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable store ports for Agent Distribution (AGENT_DISTRIBUTION §23)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
)
from intergrax.agent_distribution.installation import AgentInstallationRecord
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
)
from intergrax.agent_distribution.runtime_revision import (
    RuntimeRevision,
    RuntimeRevisionState,
)

_NON_EMPTY = Field(min_length=1)


class AgentArtifactMetadata(BaseModel):
    """Artifact metadata owned by the distribution plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_digest: str = _NON_EMPTY
    artifact_store_ref: str = Field(min_length=1)
    distribution_package_id: str = Field(min_length=1)
    agent_project_metadata_ref: str = _NON_EMPTY
    tombstoned: bool = False

    @field_validator("agent_project_metadata_ref")
    @classmethod
    def _strip_metadata_ref(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("package_digest")
    @classmethod
    def _validate_package_digest(cls, value: str) -> str:
        return normalize_package_digest(value)


class AgentInstallationStore(Protocol):
    """Installation record persistence — domain-shaped access only."""

    def get_installation(self, installation_id: str) -> AgentInstallationRecord | None:
        """Load one immutable installation revision by id."""

    def get_active_installation_for_slot(
        self,
        environment_id: str,
        installation_slot_id: str,
    ) -> AgentInstallationRecord | None:
        """Resolve the active digest-pinned installation for one environment slot."""

    def list_installations_for_slot(
        self,
        environment_id: str,
        installation_slot_id: str,
    ) -> list[AgentInstallationRecord]:
        """List installation revisions for one environment slot (audit / rollback)."""

    def list_installations_for_environment(
        self,
        environment_id: str,
    ) -> list[AgentInstallationRecord]:
        """List installation records scoped to one environment id."""

    def persist_installation(
        self,
        record: AgentInstallationRecord,
        *,
        expected_active_installation_id: str | None = None,
    ) -> AgentInstallationRecord:
        """Persist installation with serialized slot updates."""

    def atomic_promote_active_installation(
        self,
        *,
        demoted_prior: AgentInstallationRecord | None,
        promoted: AgentInstallationRecord,
        expected_active_installation_id: str | None,
    ) -> tuple[AgentInstallationRecord, AgentInstallationRecord | None]:
        """Atomically demote prior active and promote verified installation."""


class ApplicationAgentBindingStore(Protocol):
    """Durable application agent binding persistence."""

    def get_binding(
        self, application_binding_id: str
    ) -> ApplicationAgentBinding | None:
        """Load one binding by stable id."""

    def list_bindings_for_environment(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> list[ApplicationAgentBinding]:
        """List bindings scoped to one application environment."""

    def list_bindings_for_slot(
        self,
        installation_slot_id: str,
    ) -> list[ApplicationAgentBinding]:
        """List bindings anchored to an installation slot."""

    def persist_binding(
        self,
        binding: ApplicationAgentBinding,
        *,
        expected_revision: int | None = None,
    ) -> ApplicationAgentBinding:
        """Persist binding with optimistic revision concurrency."""


class RuntimeMaterializationStore(Protocol):
    """Immutable runtime materialization authority persistence."""

    def get_by_revision(
        self, runtime_revision_id: str
    ) -> RuntimeMaterializationRecord | None:
        """Load canonical materialization record for one runtime revision."""

    def persist(
        self, record: RuntimeMaterializationRecord
    ) -> RuntimeMaterializationRecord:
        """Persist immutable materialization authority (reject authority mismatch)."""


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
        application_id: str,
        application_environment_id: str,
    ) -> RuntimeRevision | None:
        """Resolve the active runtime revision for an application environment."""

    def list_revisions_for_environment(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> list[RuntimeRevision]:
        """List persisted runtime revisions for one application environment."""

    def persist_candidate_revision(
        self,
        revision: RuntimeRevision,
        *,
        expected_revision_state: RuntimeRevisionState | None = None,
    ) -> RuntimeRevision:
        """Persist candidate/validated revision with optimistic state guard."""

    def swap_active_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        new_active_revision_id: str,
        prior_active_revision_id: str | None = None,
    ) -> RuntimeRevision:
        """Atomically promote validated revision to active."""

    def atomic_activate_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        promoted: RuntimeRevision,
        demoted_prior: RuntimeRevision | None,
        expected_prior_active_revision_id: str | None,
    ) -> tuple[RuntimeRevision, RuntimeRevision | None]:
        """Atomically supersede prior active revision and activate validated revision."""


class AgentArtifactMetadataStore(Protocol):
    """Digest-pinned artifact metadata persistence."""

    def get_by_digest(self, package_digest: str) -> AgentArtifactMetadata | None:
        """Resolve artifact metadata by immutable digest."""

    def persist_metadata(
        self, metadata: AgentArtifactMetadata
    ) -> AgentArtifactMetadata:
        """Persist artifact metadata record."""


class ApplicationEnvironmentServingRecord(BaseModel):
    """Authoritative traffic serving pointer for one environment (§20.5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    traffic_serving_revision_id: str | None = None
    serving_pointer_revision: int = Field(default=0, ge=0)
    prior_traffic_revision_id: str | None = None
    committed_at: datetime | None = None

    @field_validator(
        "application_id",
        "application_environment_id",
        "traffic_serving_revision_id",
        "prior_traffic_revision_id",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class DeploymentInstanceStore(Protocol):
    """Durable deployment instance persistence."""

    def get_instance(
        self,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> DeploymentInstanceRecord | None:
        """Load deployment instance for one revision in an application environment."""

    def persist_instance(
        self, instance: DeploymentInstanceRecord
    ) -> DeploymentInstanceRecord:
        """Create or replace deployment instance record."""

    def update_instance(
        self,
        instance: DeploymentInstanceRecord,
        *,
        expected_state: DeploymentInstanceState | None = None,
        expected_record_revision: int | None = None,
    ) -> DeploymentInstanceRecord:
        """Update deployment instance with optimistic concurrency."""


class ApplicationEnvironmentServingStore(Protocol):
    """Durable traffic serving pointer persistence."""

    def get_serving_record(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> ApplicationEnvironmentServingRecord | None:
        """Load authoritative serving record for an application environment."""

    def atomic_swap_serving_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str | None,
        expected_pointer_revision: int,
        new_revision_id: str,
        prior_revision_id: str | None,
        committed_at: datetime,
    ) -> ApplicationEnvironmentServingRecord:
        """CAS-protected traffic pointer swap (§24.5)."""


class ActivationAtomicCommitResult(BaseModel):
    """Outcome of one durable activation commit boundary (§20.5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    serving_record: ApplicationEnvironmentServingRecord
    activated_revision: RuntimeRevision
    candidate_instance: DeploymentInstanceRecord
    prior_instance: DeploymentInstanceRecord | None = None
    demoted_prior_revision: RuntimeRevision | None = None


class RollbackAtomicCommitResult(BaseModel):
    """Outcome of one durable rollback commit boundary (§20.7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    serving_record: ApplicationEnvironmentServingRecord
    restored_revision: RuntimeRevision
    restored_instance: DeploymentInstanceRecord
    demoted_current_revision: RuntimeRevision
    superseded_instance: DeploymentInstanceRecord | None = None


class ApplicationEnvironmentActivationStore(Protocol):
    """Atomic activation / rollback commit boundary across serving, revision, deployment."""

    def atomic_commit_activation(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str | None,
        expected_pointer_revision: int,
        candidate_revision_id: str,
        expected_artifact_digest: str,
        committed_at: datetime,
    ) -> ActivationAtomicCommitResult:
        """Atomically commit traffic pointer, revision, and deployment activation states."""

    def atomic_commit_rollback(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str,
        expected_pointer_revision: int,
        target_revision_id: str,
        committed_at: datetime,
    ) -> RollbackAtomicCommitResult:
        """Atomically commit traffic pointer rollback and restored serving states."""
