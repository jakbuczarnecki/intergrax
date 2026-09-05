# © Artur Czarnecki. All rights reserved.

"""Production delegated subtask lifecycle plan factories (AC-4 Phase 9)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Protocol
from uuid import uuid4

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.catalog import AgentDiscoveryCandidateIdentity
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskLifecyclePlan,
    DelegatedSubtaskReleaseContext,
    DelegationId,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.dynamic_acquisition import (
    DynamicAgentAcquisitionInstallIntent,
    DynamicAgentAcquisitionRequest,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentAcquisitionRequest,
    TaskScopedAgentLeaseId,
    TaskScopedAgentReleaseRequest,
    TaskScopeId,
)
from intergrax.agent_distribution.trust import AgentInstallationTrustRecord

_ARTIFACT_DIGEST: Final = "sha256:" + ("d" * 64)


class DelegatedSubtaskTrustRecordFactory(Protocol):
    """Application-scoped trust authority for delegated specialist install."""

    def build_trust_record(
        self,
        *,
        package_digest: str,
        package_id: str,
    ) -> AgentInstallationTrustRecord: ...


@dataclass(frozen=True, slots=True)
class ProductionDelegatedSubtaskPlanConfig:
    """Immutable delegated lifecycle identity and build inputs."""

    application_id: str
    application_environment_id: str
    application_release_id: str
    package_metadata_refs: dict[str, str]
    package_logical_agents: dict[str, str]
    package_binding_ids: dict[str, str] | None = None
    installation_slot_prefix: str = "slot-delegated"
    binding_id_prefix: str = "bind-delegated"
    materialization_topology: MaterializationTopology = (
        MaterializationTopology.OCI_IMAGE
    )
    platform_version: str = "0.1.0"
    python_version: str = "3.12"
    source_context_root: str = "/tmp/src"
    output_root: str = "/tmp/out"
    application_source_root: str = "applications/app-a"
    resolver_algorithm_id: str = "intergrax.production-resolver"
    resolver_algorithm_version: str = "1.0.0"


def _fresh_mutation_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _binding_id_for_identity(
    *,
    config: ProductionDelegatedSubtaskPlanConfig,
    selected_identity: AgentDiscoveryCandidateIdentity,
    logical_agent_id: str,
) -> str:
    digest_input = "|".join(
        (
            config.application_id,
            config.application_environment_id,
            logical_agent_id,
            *selected_identity.sort_key,
        ),
    )
    suffix = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    return f"{config.binding_id_prefix}-{suffix}"


def _installation_slot_for_package(
    *, config: ProductionDelegatedSubtaskPlanConfig, package_id: str
) -> str:
    suffix = hashlib.sha256(package_id.encode("utf-8")).hexdigest()[:12]
    return f"{config.installation_slot_prefix}-{suffix}"


class ProductionDelegatedSubtaskAcquisitionPlanFactory:
    """Build canonical task-scoped acquisition requests from selected identity."""

    def __init__(
        self,
        *,
        admin_service: AgentPlatformAdminService,
        config: ProductionDelegatedSubtaskPlanConfig,
        trust_record_factory: DelegatedSubtaskTrustRecordFactory,
    ) -> None:
        self._admin_service = admin_service
        self._config = config
        self._trust_record_factory = trust_record_factory

    def build_acquisition_plan(
        self,
        *,
        delegation_id: DelegationId,
        task_scope_id: TaskScopeId,
        application_id: str,
        application_environment_id: str,
        lease_id: TaskScopedAgentLeaseId,
        selected_identity: AgentDiscoveryCandidateIdentity,
    ) -> DelegatedSubtaskLifecyclePlan:
        del delegation_id
        package_id = selected_identity.package.distribution_package_id
        metadata_ref = self._config.package_metadata_refs.get(package_id)
        if metadata_ref is None:
            raise ValueError(
                f"no metadata ref configured for delegated package {package_id!r}",
            )
        logical_agent_id = self._config.package_logical_agents.get(package_id)
        if logical_agent_id is None:
            raise ValueError(
                f"no logical agent configured for delegated package {package_id!r}",
            )
        serving = self._admin_service.inspect_serving(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        install_mutation_id = _fresh_mutation_id("mut-delegated-install")
        bind_mutation_id = _fresh_mutation_id("mut-delegated-bind")
        revision_id = f"rev-acquire-{lease_id}"
        installation_id = f"inst-{lease_id}"
        slot_id = _installation_slot_for_package(
            config=self._config, package_id=package_id
        )
        binding_id = _binding_id_for_identity(
            config=self._config,
            selected_identity=selected_identity,
            logical_agent_id=logical_agent_id,
        )
        configured_binding = (self._config.package_binding_ids or {}).get(package_id)
        if configured_binding is not None:
            binding_id = configured_binding
        package_digest = selected_identity.package.package_digest or _ARTIFACT_DIGEST
        return DelegatedSubtaskLifecyclePlan(
            acquisition_request=TaskScopedAgentAcquisitionRequest(
                lease_id=lease_id,
                task_scope_id=task_scope_id,
                acquisition_request=DynamicAgentAcquisitionRequest(
                    selected_identity=selected_identity,
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                    catalog_entry_id=None,
                    install=DynamicAgentAcquisitionInstallIntent(
                        mutation_id=install_mutation_id,
                        installation_id=installation_id,
                        installation_slot_id=slot_id,
                        artifact_store_ref=f"store://artifacts/{installation_id}",
                        trust_record=self._trust_record_factory.build_trust_record(
                            package_digest=package_digest,
                            package_id=package_id,
                        ),
                        agent_project_metadata_ref=metadata_ref,
                    ),
                    bind=BindAgentRequest(
                        mutation_id=bind_mutation_id,
                        application_binding_id=binding_id,
                        logical_agent_id=logical_agent_id,
                        installation_slot_id=slot_id,
                        enablement=True,
                    ),
                    build=BuildApplicationRevisionRequest(
                        mutation_id=f"{install_mutation_id}-build",
                        runtime_revision_id=revision_id,
                        application_release_id=self._config.application_release_id,
                        platform_version=self._config.platform_version,
                        python_version=self._config.python_version,
                        source_context_root=self._config.source_context_root,
                        output_root=self._config.output_root,
                        application_source_root=self._config.application_source_root,
                        materialization_topology=self._config.materialization_topology,
                        repository_declaration=RepositoryDependencyDeclaration(
                            application_release_id=self._config.application_release_id,
                            direct_dependencies=(),
                        ),
                        resolver_algorithm_id=self._config.resolver_algorithm_id,
                        resolver_algorithm_version=self._config.resolver_algorithm_version,
                    ),
                    activate=ActivateRuntimeRevisionRequest(
                        mutation_id=f"{install_mutation_id}-activate",
                        runtime_revision_id=revision_id,
                        artifact_locator="production://artifact",
                        expected_artifact_digest=_ARTIFACT_DIGEST,
                        expected_serving_pointer_revision=serving.serving_pointer_revision,
                        expected_prior_traffic_revision_id=serving.traffic_serving_revision_id,
                    ),
                ),
            ),
        )


class ProductionDelegatedSubtaskReleasePlanFactory:
    """Build release requests from current lease state immediately before release."""

    def __init__(
        self,
        *,
        admin_service: AgentPlatformAdminService,
        config: ProductionDelegatedSubtaskPlanConfig,
    ) -> None:
        self._admin_service = admin_service
        self._config = config

    def build_release_request(
        self,
        *,
        context: DelegatedSubtaskReleaseContext,
    ) -> TaskScopedAgentReleaseRequest:
        serving = self._admin_service.inspect_serving(
            application_id=context.application_id,
            application_environment_id=context.application_environment_id,
        )
        binding = _require_binding_view(
            admin_service=self._admin_service,
            application_id=context.application_id,
            application_environment_id=context.application_environment_id,
            application_binding_id=context.lease.application_binding_id,
        )
        disable_mutation_id = _fresh_mutation_id("mut-delegated-disable")
        revision_id = f"rev-release-{context.lease.lease_id}"
        return TaskScopedAgentReleaseRequest(
            lease_id=context.lease.lease_id,
            task_scope_id=context.task_scope_id,
            application_id=context.application_id,
            application_environment_id=context.application_environment_id,
            disable=SetAgentEnablementRequest(
                mutation_id=disable_mutation_id,
                expected_revision=binding.binding_revision,
            ),
            build=BuildApplicationRevisionRequest(
                mutation_id=f"{disable_mutation_id}-build",
                runtime_revision_id=revision_id,
                application_release_id=self._config.application_release_id,
                platform_version=self._config.platform_version,
                python_version=self._config.python_version,
                source_context_root=self._config.source_context_root,
                output_root=self._config.output_root,
                application_source_root=self._config.application_source_root,
                materialization_topology=self._config.materialization_topology,
                repository_declaration=RepositoryDependencyDeclaration(
                    application_release_id=self._config.application_release_id,
                    direct_dependencies=(),
                ),
                resolver_algorithm_id=self._config.resolver_algorithm_id,
                resolver_algorithm_version=self._config.resolver_algorithm_version,
            ),
            activate=ActivateRuntimeRevisionRequest(
                mutation_id=f"{disable_mutation_id}-activate",
                runtime_revision_id=revision_id,
                artifact_locator="production://artifact",
                expected_artifact_digest=_ARTIFACT_DIGEST,
                expected_serving_pointer_revision=serving.serving_pointer_revision,
                expected_prior_traffic_revision_id=serving.traffic_serving_revision_id,
            ),
        )


def _require_binding_view(
    *,
    admin_service: AgentPlatformAdminService,
    application_id: str,
    application_environment_id: str,
    application_binding_id: str,
):
    bindings = admin_service.list_bindings(
        application_id=application_id,
        application_environment_id=application_environment_id,
    ).bindings
    for binding in bindings:
        if binding.application_binding_id == application_binding_id:
            return binding
    raise ValueError(
        f"binding {application_binding_id!r} not found for release plan",
    )


def derive_production_delegated_installation_slot(
    *,
    config: ProductionDelegatedSubtaskPlanConfig,
    package_id: str,
) -> str:
    return _installation_slot_for_package(config=config, package_id=package_id)


def derive_production_delegated_binding_id(
    *,
    config: ProductionDelegatedSubtaskPlanConfig,
    selected_identity: AgentDiscoveryCandidateIdentity,
    package_id: str,
) -> str:
    logical_agent_id = config.package_logical_agents[package_id]
    configured = (config.package_binding_ids or {}).get(package_id)
    if configured is not None:
        return configured
    return _binding_id_for_identity(
        config=config,
        selected_identity=selected_identity,
        logical_agent_id=logical_agent_id,
    )


__all__ = [
    "DelegatedSubtaskTrustRecordFactory",
    "ProductionDelegatedSubtaskAcquisitionPlanFactory",
    "ProductionDelegatedSubtaskPlanConfig",
    "ProductionDelegatedSubtaskReleasePlanFactory",
    "derive_production_delegated_binding_id",
    "derive_production_delegated_installation_slot",
]
