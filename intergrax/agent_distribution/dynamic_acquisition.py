# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dynamic agent acquisition orchestration bridge into AC-3 lifecycle (AC-4 Phase 6)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    ActivationResultView,
    BindAgentRequest,
    BindingMutationResult,
    BuildApplicationRevisionRequest,
    BuildRevisionResult,
    InstallAgentRequest,
    InstallationMutationResult,
    RuntimeRevisionView,
    ServingStateView,
)
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    AgentDiscoveryCandidateIdentity,
    CatalogPackageResolution,
    CatalogSourceProvider,
)
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.trust import AgentInstallationTrustRecord
from intergrax.contracts.agent_run import RequestIdentity

_NON_EMPTY = Field(min_length=1)

SCHEMA_DYNAMIC_AGENT_ACQUISITION_REQUEST_V1: Final = (
    "dynamic_agent_acquisition_request.v1"
)
SCHEMA_DYNAMIC_AGENT_ACQUISITION_RESULT_V1: Final = (
    "dynamic_agent_acquisition_result.v1"
)
SCHEMA_DYNAMIC_AGENT_ACQUISITION_INSTALL_INTENT_V1: Final = (
    "dynamic_agent_acquisition_install_intent.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class DynamicAgentAcquisitionError(AgentDistributionError):
    """Base error for dynamic agent acquisition orchestration."""


class DynamicAgentAcquisitionContractError(DynamicAgentAcquisitionError):
    """Malformed acquisition request or unsupported mode."""


class DynamicAgentAcquisitionResolutionError(DynamicAgentAcquisitionError):
    """Exact package resolution failed or stale discovery candidate mismatch."""


class DynamicAgentAcquisitionActivationError(DynamicAgentAcquisitionError):
    """Install/bind succeeded but runtime build or activation failed."""


class AcquisitionMode(StrEnum):
    """Acquisition lifecycle mode — Phase 6 supports persistent application scope only."""

    PERSISTENT = "persistent"


class DynamicAgentAcquisitionOutcome(StrEnum):
    """Explicit acquisition terminal semantics."""

    ACQUIRED_ACTIVE = "acquired_active"
    ALREADY_ACTIVE = "already_active"
    DESIRED_STATE_UPDATED_ACTIVATION_FAILED = "desired_state_updated_activation_failed"


class DynamicAgentAcquisitionInstallIntent(BaseModel):
    """Canonical installation authority fields — package identity comes from resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DYNAMIC_AGENT_ACQUISITION_INSTALL_INTENT_V1
    mutation_id: str = _NON_EMPTY
    installation_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY
    artifact_store_ref: str = _NON_EMPTY
    trust_record: AgentInstallationTrustRecord
    agent_project_metadata_ref: str = _NON_EMPTY

    @field_validator(
        "mutation_id",
        "installation_id",
        "installation_slot_id",
        "artifact_store_ref",
        "agent_project_metadata_ref",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class DynamicAgentAcquisitionRequest(BaseModel):
    """Typed immutable acquisition request for one selected discovery candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DYNAMIC_AGENT_ACQUISITION_REQUEST_V1
    mode: AcquisitionMode = AcquisitionMode.PERSISTENT
    selected_identity: AgentDiscoveryCandidateIdentity
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    catalog_entry_id: str | None = None
    install: DynamicAgentAcquisitionInstallIntent
    bind: BindAgentRequest
    build: BuildApplicationRevisionRequest
    activate: ActivateRuntimeRevisionRequest

    @field_validator("application_id", "application_environment_id", "catalog_entry_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: AcquisitionMode) -> AcquisitionMode:
        if value is not AcquisitionMode.PERSISTENT:
            raise ValueError(
                "only persistent application-scoped acquisition is supported"
            )
        return value


class DynamicAgentAcquisitionResult(BaseModel):
    """Audit-friendly acquisition outcome using AC-3 authority identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DYNAMIC_AGENT_ACQUISITION_RESULT_V1
    outcome: DynamicAgentAcquisitionOutcome
    selected_identity: AgentDiscoveryCandidateIdentity
    resolved_package_identity: AgentPackageIdentity
    catalog_entry_id: str = _NON_EMPTY
    artifact_locator: str = _NON_EMPTY
    installation_id: str = _NON_EMPTY
    application_binding_id: str = _NON_EMPTY
    runtime_revision_id: str = _NON_EMPTY
    traffic_serving_revision_id: str | None = None
    installation_reused: bool = False
    binding_reused: bool = False
    activation_view: ActivationResultView | None = None

    @field_validator(
        "catalog_entry_id",
        "artifact_locator",
        "installation_id",
        "application_binding_id",
        "runtime_revision_id",
        "traffic_serving_revision_id",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class CatalogSourceProviderRegistry:
    """Plugin registry mapping catalog source ids to catalog providers."""

    def __init__(
        self,
        providers: Mapping[str, CatalogSourceProvider],
    ) -> None:
        self._providers = dict(providers)

    def require(self, catalog_source_id: str) -> CatalogSourceProvider:
        provider = self._providers.get(catalog_source_id)
        if provider is None:
            raise DynamicAgentAcquisitionResolutionError(
                f"catalog source {catalog_source_id} has no registered provider",
            )
        if provider.catalog_source_id != catalog_source_id:
            raise DynamicAgentAcquisitionResolutionError(
                "catalog provider instance id does not match registry key",
            )
        return provider


def assert_exact_discovery_candidate_match(
    *,
    identity: AgentDiscoveryCandidateIdentity,
    resolution: CatalogPackageResolution,
) -> None:
    """Fail closed when catalog resolution no longer matches selected identity."""
    entry_source = resolution.entry.catalog_source
    selected_source = identity.source
    if entry_source.catalog_source_id != selected_source.catalog_source_id:
        raise DynamicAgentAcquisitionResolutionError(
            "resolved catalog source id does not match selected identity",
        )
    if entry_source.provider_kind != selected_source.provider_kind:
        raise DynamicAgentAcquisitionResolutionError(
            "resolved catalog provider kind does not match selected identity",
        )

    resolved = resolution.package_candidate
    expected = identity.package
    if resolved.distribution_package_id != expected.distribution_package_id:
        raise DynamicAgentAcquisitionResolutionError(
            "resolved distribution package id does not match selected candidate",
        )
    if resolved.package_version != expected.package_version:
        raise DynamicAgentAcquisitionResolutionError(
            "resolved package version does not match selected candidate",
        )
    if (
        expected.package_digest is not None
        and resolved.package_digest != expected.package_digest
    ):
        raise DynamicAgentAcquisitionResolutionError(
            "resolved package digest does not match selected candidate",
        )


def resolve_discovery_candidate_exact(
    *,
    identity: AgentDiscoveryCandidateIdentity,
    catalog_entry_id: str | None,
    registry: CatalogSourceProviderRegistry,
) -> CatalogPackageResolution:
    """Resolve selected source-qualified identity through canonical catalog authority."""
    provider = registry.require(identity.source.catalog_source_id)
    matched_entry: AgentCatalogEntry | None = None
    for entry in provider.list_entries():
        if catalog_entry_id is not None and entry.catalog_entry_id != catalog_entry_id:
            continue
        if entry.package_id_line != identity.package.distribution_package_id:
            continue
        if entry.catalog_source != identity.source:
            continue
        matched_entry = entry
        break
    if matched_entry is None:
        raise DynamicAgentAcquisitionResolutionError(
            "no catalog entry matches selected discovery candidate",
        )

    resolution = provider.resolve_package(
        matched_entry,
        version_selector=identity.package.package_version,
    )
    assert_exact_discovery_candidate_match(identity=identity, resolution=resolution)
    return resolution


class AgentPlatformLifecyclePort(Protocol):
    """AC-3 public lifecycle facade used by dynamic acquisition — no service locator."""

    def install_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: InstallAgentRequest,
        principal: RequestIdentity,
    ) -> InstallationMutationResult: ...

    def bind_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: BindAgentRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult: ...

    def enable_binding(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        application_binding_id: str,
        request: object,
        principal: RequestIdentity,
    ) -> BindingMutationResult: ...

    def build_application_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: BuildApplicationRevisionRequest,
        principal: RequestIdentity,
    ) -> BuildRevisionResult: ...

    def activate_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: ActivateRuntimeRevisionRequest,
        principal: RequestIdentity,
    ) -> ActivationResultView: ...

    def inspect_serving(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> ServingStateView: ...

    def inspect_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> RuntimeRevisionView: ...


class DynamicAgentAcquisitionService:
    """Orchestrates exact discovery candidate into canonical AC-3 lifecycle."""

    def __init__(
        self,
        *,
        catalog_registry: CatalogSourceProviderRegistry,
        lifecycle: AgentPlatformLifecyclePort,
    ) -> None:
        self._catalog_registry = catalog_registry
        self._lifecycle = lifecycle

    def acquire(
        self,
        request: DynamicAgentAcquisitionRequest,
        *,
        principal: RequestIdentity,
    ) -> DynamicAgentAcquisitionResult:
        resolution = resolve_discovery_candidate_exact(
            identity=request.selected_identity,
            catalog_entry_id=request.catalog_entry_id,
            registry=self._catalog_registry,
        )
        if resolution.package_candidate.package_digest is None:
            raise DynamicAgentAcquisitionResolutionError(
                "resolved package candidate lacks digest required for installation",
            )
        package_identity = resolution.package_candidate.to_digest_pinned()

        install_request = InstallAgentRequest(
            mutation_id=request.install.mutation_id,
            installation_id=request.install.installation_id,
            installation_slot_id=request.install.installation_slot_id,
            package_identity=package_identity,
            artifact_store_ref=request.install.artifact_store_ref,
            trust_record=request.install.trust_record,
            agent_project_metadata_ref=request.install.agent_project_metadata_ref,
        )
        install_result = self._lifecycle.install_agent(
            application_id=request.application_id,
            application_environment_id=request.application_environment_id,
            request=install_request,
            principal=principal,
        )
        installation_reused = not install_result.audit_event_types

        bind_result = self._lifecycle.bind_agent(
            application_id=request.application_id,
            application_environment_id=request.application_environment_id,
            request=request.bind,
            principal=principal,
        )
        binding_reused = not bind_result.audit_event_types

        if request.bind.enablement and not bind_result.binding.enablement:
            from intergrax.agent_distribution.admin_models import (
                SetAgentEnablementRequest,
            )

            self._lifecycle.enable_binding(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                application_binding_id=request.bind.application_binding_id,
                request=SetAgentEnablementRequest(
                    mutation_id=f"{request.bind.mutation_id}-enable",
                    expected_revision=bind_result.binding.binding_revision,
                ),
                principal=principal,
            )

        base_result = DynamicAgentAcquisitionResult(
            outcome=DynamicAgentAcquisitionOutcome.DESIRED_STATE_UPDATED_ACTIVATION_FAILED,
            selected_identity=request.selected_identity,
            resolved_package_identity=package_identity,
            catalog_entry_id=resolution.entry.catalog_entry_id,
            artifact_locator=resolution.artifact_locator,
            installation_id=request.install.installation_id,
            application_binding_id=request.bind.application_binding_id,
            runtime_revision_id=request.build.runtime_revision_id,
            installation_reused=installation_reused,
            binding_reused=binding_reused,
        )

        if installation_reused and binding_reused:
            serving = self._lifecycle.inspect_serving(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
            )
            if serving.traffic_serving_revision_id == request.build.runtime_revision_id:
                revision = self._lifecycle.inspect_revision(
                    application_id=request.application_id,
                    application_environment_id=request.application_environment_id,
                    runtime_revision_id=request.build.runtime_revision_id,
                )
                if revision.revision_state is RuntimeRevisionState.ACTIVE:
                    return base_result.model_copy(
                        update={
                            "outcome": DynamicAgentAcquisitionOutcome.ALREADY_ACTIVE,
                            "traffic_serving_revision_id": serving.traffic_serving_revision_id,
                        },
                    )

        try:
            build_result = self._lifecycle.build_application_revision(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                request=request.build,
                principal=principal,
            )
            activate_view = self._lifecycle.activate_revision(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                request=request.activate.model_copy(
                    update={
                        "runtime_revision_id": request.build.runtime_revision_id,
                        "artifact_locator": build_result.artifact_locator
                        or request.activate.artifact_locator,
                        "expected_artifact_digest": (
                            build_result.materialization_artifact_digest
                            or request.activate.expected_artifact_digest
                        ),
                    },
                ),
                principal=principal,
            )
        except AgentDistributionError as exc:
            raise DynamicAgentAcquisitionActivationError(
                "desired state updated but runtime activation failed",
            ) from exc

        return base_result.model_copy(
            update={
                "outcome": DynamicAgentAcquisitionOutcome.ACQUIRED_ACTIVE,
                "traffic_serving_revision_id": activate_view.traffic_serving_revision_id,
                "activation_view": activate_view,
            },
        )
