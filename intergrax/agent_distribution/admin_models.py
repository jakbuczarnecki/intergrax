# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed Agent Platform admin command/response models (AP-11)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._config_validation import validate_non_secret_distribution_config
from intergrax.agent_distribution._immutable_json import DistributionJsonValue
from intergrax.agent_distribution.binding import (
    AgentBindingFactoryReference,
    AgentBindingPolicyOverrides,
)
from intergrax.agent_distribution.catalog import AgentCatalogEntry, CatalogEntryFilters
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.trust import AgentInstallationTrustRecord

_NON_EMPTY = Field(min_length=1)


class AgentPlatformAdminBlockedError(Exception):
    """Admin operation cannot proceed because a required host port is missing."""

    def __init__(self, blocker_code: str, message: str) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code


class InstallAgentRequest(BaseModel):
    """Install a digest-pinned package into one environment slot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    installation_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY
    package_identity: AgentPackageIdentity
    artifact_store_ref: str = _NON_EMPTY
    trust_record: AgentInstallationTrustRecord
    agent_project_metadata_ref: str = _NON_EMPTY
    catalog_entry_id: str | None = None
    version_selector: str | None = None

    @field_validator(
        "installation_id",
        "installation_slot_id",
        "artifact_store_ref",
        "agent_project_metadata_ref",
        "catalog_entry_id",
        "version_selector",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class BindAgentRequest(BaseModel):
    """Bind an installed slot to one application environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_binding_id: str = _NON_EMPTY
    logical_agent_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY
    config: Mapping[str, DistributionJsonValue] = Field(default_factory=dict)
    secret_refs: tuple[str, ...] = ()
    policy_overrides: AgentBindingPolicyOverrides | None = None
    factory_reference: AgentBindingFactoryReference | None = None
    enablement: bool = False
    builtin_package_ref: str | None = None

    @field_validator("config", mode="before")
    @classmethod
    def _reject_secret_config(cls, value: object) -> Mapping[str, DistributionJsonValue]:
        if not isinstance(value, Mapping):
            raise ValueError("config must be a mapping")
        return validate_non_secret_distribution_config(
            value,
            context_label="binding config",
        )

    @field_validator(
        "application_binding_id",
        "logical_agent_id",
        "installation_slot_id",
        "builtin_package_ref",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class UpdateAgentBindingRequest(BaseModel):
    """Replace non-secret binding config with CAS revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_revision: int = Field(ge=0)
    config: Mapping[str, DistributionJsonValue]

    @field_validator("config", mode="before")
    @classmethod
    def _reject_secret_config(cls, value: object) -> Mapping[str, DistributionJsonValue]:
        if not isinstance(value, Mapping):
            raise ValueError("config must be a mapping")
        return validate_non_secret_distribution_config(
            value,
            context_label="binding config",
        )


class SetAgentEnablementRequest(BaseModel):
    """Enable or disable desired-state binding without activating traffic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_revision: int = Field(ge=0)


class BuildApplicationRevisionRequest(BaseModel):
    """Freeze desired state into a candidate RuntimeRevision (no activation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_revision_id: str = _NON_EMPTY
    application_release_id: str = _NON_EMPTY
    platform_version: str = _NON_EMPTY
    python_version: str = _NON_EMPTY
    source_context_root: str = _NON_EMPTY
    output_root: str = _NON_EMPTY
    application_source_root: str = _NON_EMPTY
    materialization_topology: MaterializationTopology
    repository_declaration: RepositoryDependencyDeclaration
    resolver_algorithm_id: str = _NON_EMPTY
    resolver_algorithm_version: str = _NON_EMPTY
    agent_source_roots: tuple[tuple[str, str], ...] = ()

    @field_validator(
        "runtime_revision_id",
        "application_release_id",
        "platform_version",
        "python_version",
        "source_context_root",
        "output_root",
        "application_source_root",
        "resolver_algorithm_id",
        "resolver_algorithm_version",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ActivateRuntimeRevisionRequest(BaseModel):
    """Commit an explicit validated RuntimeRevision to serving traffic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_revision_id: str = _NON_EMPTY
    artifact_locator: str = _NON_EMPTY
    expected_artifact_digest: str = _NON_EMPTY
    expected_serving_pointer_revision: int = Field(ge=0)
    expected_prior_traffic_revision_id: str | None = None

    @field_validator(
        "runtime_revision_id",
        "artifact_locator",
        "expected_artifact_digest",
        "expected_prior_traffic_revision_id",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class RollbackRuntimeRevisionRequest(BaseModel):
    """Restore the immutable prior RuntimeRevision (no desired-state rebuild)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_current_traffic_revision_id: str = _NON_EMPTY
    expected_serving_pointer_revision: int = Field(ge=0)
    target_runtime_revision_id: str | None = None

    @field_validator("expected_current_traffic_revision_id", "target_runtime_revision_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class CatalogListRequest(BaseModel):
    """Optional catalog listing filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filters: CatalogEntryFilters | None = None


class InstallationView(BaseModel):
    """Stable installation identities for admin clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    installation_id: str
    installation_slot_id: str
    environment_id: str
    distribution_package_id: str
    package_version: str
    package_digest: str
    installation_state: InstallationState
    active_for_slot: bool
    installed: bool


class BindingView(BaseModel):
    """Stable binding identities for admin clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_binding_id: str
    application_id: str
    application_environment_id: str
    logical_agent_id: str
    installation_slot_id: str
    active_installation_id: str | None = None
    enablement: bool
    binding_revision: int
    tombstone: bool


class RosterEntryView(BaseModel):
    """Derived effective-roster row — not a stored authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_agent_id: str
    installation_slot_id: str
    active_installation_id: str | None = None
    distribution_package_id: str
    package_digest: str
    effective_enablement: bool
    application_binding_id: str | None = None


class EffectiveRosterView(BaseModel):
    """Desired-state roster projection for one application environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    application_environment_id: str
    manifest_release_id: str
    effective_roster_revision_id: str | None = None
    entries: tuple[RosterEntryView, ...] = ()


class RuntimeRevisionView(BaseModel):
    """Stable runtime revision identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_revision_id: str
    application_environment_id: str
    application_release_id: str
    revision_state: RuntimeRevisionState
    effective_roster_revision_id: str
    materialized_runtime_lock_id: str | None = None
    materialized_runtime_lock_digest: str | None = None
    runtime_graph_digest: str | None = None
    materialization_artifact_digest: str | None = None
    materialization_topology: MaterializationTopology | None = None
    installed_agent_package_digests: tuple[str, ...] = ()
    supersedes_revision_id: str | None = None
    rollback_target_revision_id: str | None = None


class ServingStateView(BaseModel):
    """Authoritative serving pointer — distinct from desired enablement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    application_environment_id: str
    traffic_serving_revision_id: str | None = None
    prior_traffic_revision_id: str | None = None
    serving_pointer_revision: int = 0
    active_revision: RuntimeRevisionView | None = None


class ActivationStatusView(BaseModel):
    """Activation/rollback evidence from serving + deployment instance stores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    serving: ServingStateView
    candidate_revision: RuntimeRevisionView | None = None
    serving_instance_state: str | None = None
    serving_readiness_evidence_ref: str | None = None
    projection_readiness_token: str | None = None


class BuildRevisionResult(BaseModel):
    """Candidate build outcome — never changes serving traffic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_revision_id: str
    revision_state: RuntimeRevisionState
    effective_roster_revision_id: str
    materialized_runtime_lock_id: str | None = None
    materialized_runtime_lock_digest: str | None = None
    runtime_graph_digest: str | None = None
    materialization_artifact_digest: str | None = None
    artifact_locator: str | None = None
    materialization_topology: MaterializationTopology | None = None
    audit_event_types: tuple[str, ...] = ()


class ActivationResultView(BaseModel):
    """Traffic commit evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    traffic_serving_revision_id: str | None = None
    serving_pointer_revision: int
    activated_revision_id: str
    revision_state: RuntimeRevisionState
    prior_traffic_revision_id: str | None = None
    audit_event_types: tuple[str, ...] = ()


class RollbackResultView(BaseModel):
    """Immutable rollback evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    traffic_serving_revision_id: str | None = None
    serving_pointer_revision: int
    restored_revision_id: str
    revision_state: RuntimeRevisionState
    superseded_revision_id: str | None = None
    audit_event_types: tuple[str, ...] = ()


class MutationResultView(BaseModel):
    """Generic mutation envelope with audit event types from domain services."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_event_types: tuple[str, ...] = ()


class InstallationMutationResult(MutationResultView):
    installation: InstallationView


class BindingMutationResult(MutationResultView):
    binding: BindingView


class AgentStatusView(BaseModel):
    """Derived agent status read model — not a stored source of truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_agent_id: str
    available: bool | None = None
    installed: bool
    bound: bool
    enabled_in_desired_state: bool
    included_in_active_revision: bool
    traffic_serving_revision_id: str | None = None
    pending_candidate_revision_id: str | None = None


class CatalogListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[AgentCatalogEntry, ...] = ()


class InstallationListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    installations: tuple[InstallationView, ...] = ()


class BindingListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bindings: tuple[BindingView, ...] = ()


class RevisionHistoryView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    traffic_serving_revision_id: str | None = None
    prior_traffic_revision_id: str | None = None
    revisions: tuple[RuntimeRevisionView, ...] = ()
