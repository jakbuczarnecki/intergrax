# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed Agent Platform admin orchestration facade (AP-11)."""

from __future__ import annotations

from intergrax.agent_distribution.activation import ActivationService
from intergrax.agent_distribution.admin_models import (
    ActivationResultView,
    ActivationStatusView,
    ActivateRuntimeRevisionRequest,
    AgentPlatformAdminBlockedError,
    AgentPlatformAdminGovernanceBlockedError,
    AgentStatusView,
    BindAgentRequest,
    BindingListResult,
    BindingMutationResult,
    BindingView,
    BuildApplicationRevisionRequest,
    BuildRevisionResult,
    CatalogListResult,
    EffectiveRosterView,
    InstallAgentRequest,
    InstallationListResult,
    InstallationMutationResult,
    InstallationView,
    RevisionHistoryView,
    RollbackResultView,
    RollbackRuntimeRevisionRequest,
    RosterEntryView,
    RuntimeRevisionView,
    ServingStateView,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadataProvider
from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
    authorize_scoped_control_plane_mutation,
    binding_absent_token,
    binding_config_digest,
    build_activation_mutation_request,
    build_bind_agent_mutation_request,
    build_disable_binding_mutation_request,
    build_enable_binding_mutation_request,
    build_input_digest,
    build_install_agent_mutation_request,
    build_runtime_revision_identity_digest,
    build_runtime_revision_mutation_request,
    build_rollback_mutation_request,
    build_update_binding_config_mutation_request,
    installation_absent_token,
    installation_state_token,
)
from intergrax.agent_distribution.catalog import CatalogEntryFilters, CatalogSourceProvider
from intergrax.agent_distribution.dependency import DependencyResolverInput
from intergrax.agent_distribution.dependency_specification import (
    build_candidate_dependency_specification,
)
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    BindingRevisionConflict,
    InstallationSlotConflict,
    RuntimeActivationConflict,
    RuntimeRevisionConflict,
    RuntimeRollbackError,
)
from intergrax.agent_distribution.events import TransitionResult
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    installation_state_is_installed,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.materialization import (
    ApplicationBuildContext,
    MaterializationInput,
)
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.resolver import DependencyResolver
from intergrax.agent_distribution.roster import EffectiveRoster, ManifestDefaultAgentDeclaration
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
    CandidateRuntimeGraphValidator,
)
from intergrax.agent_distribution.runtime_lock import MaterializedRuntimeLockService
from intergrax.agent_distribution.runtime_revision import RuntimeRevision, RuntimeRevisionState
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.stores import (
    AgentArtifactMetadata,
    AgentArtifactMetadataStore,
    AgentInstallationStore,
    ApplicationAgentBindingStore,
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    MaterializedRuntimeLockStore,
    RuntimeRevisionStore,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


def _event_types(*results: TransitionResult[object]) -> tuple[str, ...]:
    events: list[str] = []
    for result in results:
        for event in result.events:
            events.append(event.event_type)
    return tuple(events)


def _installation_view(record: AgentInstallationRecord) -> InstallationView:
    identity = record.package_identity
    return InstallationView(
        installation_id=record.installation_id,
        installation_slot_id=record.installation_slot_id,
        environment_id=record.environment_id,
        distribution_package_id=identity.distribution_package_id,
        package_version=identity.package_version,
        package_digest=identity.package_digest,
        installation_state=record.installation_state,
        active_for_slot=record.active_for_slot,
        installed=installation_state_is_installed(record.installation_state),
    )


def _binding_view(binding: ApplicationAgentBinding) -> BindingView:
    return BindingView(
        application_binding_id=binding.application_binding_id,
        application_id=binding.application_id,
        application_environment_id=binding.application_environment_id,
        logical_agent_id=binding.logical_agent_id,
        installation_slot_id=binding.installation_slot_id,
        active_installation_id=binding.active_installation_id,
        enablement=binding.enablement,
        binding_revision=binding.binding_revision,
        tombstone=binding.tombstone,
    )


def _revision_view(revision: RuntimeRevision) -> RuntimeRevisionView:
    return RuntimeRevisionView(
        runtime_revision_id=revision.runtime_revision_id,
        application_environment_id=revision.application_environment_id,
        application_release_id=revision.application_release_id,
        revision_state=revision.revision_state,
        effective_roster_revision_id=revision.effective_roster_revision_id,
        materialized_runtime_lock_id=revision.materialized_runtime_lock_id,
        materialized_runtime_lock_digest=revision.materialized_runtime_lock_digest,
        runtime_graph_digest=revision.runtime_graph_digest,
        materialization_artifact_digest=revision.materialization_artifact_digest,
        materialization_topology=revision.materialization_topology,
        installed_agent_package_digests=revision.installed_agent_package_digests,
        supersedes_revision_id=revision.supersedes_revision_id,
        rollback_target_revision_id=revision.rollback_target_revision_id,
    )


class AgentPlatformAdminService:
    """One control-plane facade over AP-3..AP-10 services — no HTTP concerns."""

    def __init__(
        self,
        *,
        installation_store: AgentInstallationStore,
        binding_store: ApplicationAgentBindingStore,
        revision_store: RuntimeRevisionStore,
        serving_store: ApplicationEnvironmentServingStore,
        deployment_instance_store: DeploymentInstanceStore,
        lock_store: MaterializedRuntimeLockStore,
        artifact_metadata_store: AgentArtifactMetadataStore,
        installation_service: InstallationService,
        binding_service: BindingService,
        revision_service: RuntimeRevisionService,
        roster_builder: EffectiveRosterBuilder,
        requirement_set_builder: InstalledAgentRequirementSetBuilder,
        activation_service: ActivationService,
        lock_service: MaterializedRuntimeLockService | None = None,
        graph_builder: CandidateRuntimeGraphBuilder | None = None,
        graph_validator: CandidateRuntimeGraphValidator | None = None,
        materialization_service: RuntimeMaterializationService | None = None,
        metadata_provider: AgentProjectMetadataProvider | None = None,
        catalog_provider: CatalogSourceProvider | None = None,
        dependency_resolver: DependencyResolver | None = None,
        manifest_defaults: tuple[ManifestDefaultAgentDeclaration, ...] = (),
        mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
        environment_tenant_resolver: ApplicationEnvironmentTenantResolver | None = None,
    ) -> None:
        self._installation_store = installation_store
        self._binding_store = binding_store
        self._revision_store = revision_store
        self._serving_store = serving_store
        self._deployment_instance_store = deployment_instance_store
        self._lock_store = lock_store
        self._artifact_metadata_store = artifact_metadata_store
        self._installation_service = installation_service
        self._binding_service = binding_service
        self._revision_service = revision_service
        self._roster_builder = roster_builder
        self._requirement_set_builder = requirement_set_builder
        self._activation_service = activation_service
        if lock_service is not None:
            self._lock_service = lock_service
        elif dependency_resolver is not None:
            self._lock_service = MaterializedRuntimeLockService(dependency_resolver)
        else:
            self._lock_service = None
        self._graph_builder = graph_builder
        self._graph_validator = graph_validator or CandidateRuntimeGraphValidator()
        self._materialization_service = materialization_service
        self._metadata_provider = metadata_provider
        self._catalog_provider = catalog_provider
        self._manifest_defaults = manifest_defaults
        self._mutation_authorization_boundary = mutation_authorization_boundary
        self._environment_tenant_resolver = environment_tenant_resolver

    def list_catalog(
        self,
        filters: CatalogEntryFilters | None = None,
    ) -> CatalogListResult:
        provider = self._catalog_provider
        if provider is None:
            raise AgentPlatformAdminBlockedError(
                "AP-11_BLOCKED_BY_MISSING_CATALOG_PROVIDER",
                "catalog list requires an injected CatalogSourceProvider",
            )
        return CatalogListResult(entries=tuple(provider.list_entries(filters)))

    def list_installed(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> InstallationListResult:
        del application_id
        records = self._installation_store.list_installations_for_environment(
            application_environment_id
        )
        return InstallationListResult(
            installations=tuple(_installation_view(record) for record in records)
        )

    def inspect_installation(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        installation_id: str,
    ) -> InstallationView:
        record = self._installation_store.get_installation(installation_id)
        if record is None or record.environment_id != application_environment_id:
            raise AgentDistributionNotFoundError(
                f"installation {installation_id} was not found"
            )
        self._assert_installation_application_scope(
            application_id=application_id,
            application_environment_id=application_environment_id,
            record=record,
        )
        return _installation_view(record)

    def list_bindings(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> BindingListResult:
        bindings = self._binding_service.list_bindings_for_environment(
            application_id,
            application_environment_id,
        )
        return BindingListResult(bindings=tuple(_binding_view(item) for item in bindings))

    def inspect_effective_roster(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        manifest_release_id: str = "unspecified",
    ) -> EffectiveRosterView:
        roster = self._build_roster(
            application_id=application_id,
            application_environment_id=application_environment_id,
            manifest_release_id=manifest_release_id,
        )
        return EffectiveRosterView(
            application_id=roster.application_id,
            application_environment_id=roster.application_environment_id,
            manifest_release_id=roster.manifest_release_id,
            effective_roster_revision_id=roster.effective_roster_revision_id,
            entries=tuple(
                RosterEntryView(
                    logical_agent_id=entry.logical_agent_id,
                    installation_slot_id=entry.installation_slot_id,
                    active_installation_id=entry.active_installation_id,
                    distribution_package_id=entry.distribution_package_id,
                    package_digest=entry.package_digest,
                    effective_enablement=entry.effective_enablement,
                    application_binding_id=entry.application_binding_id,
                )
                for entry in roster.entries
            ),
        )

    def inspect_serving(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> ServingStateView:
        serving = self._serving_store.get_serving_record(
            application_id,
            application_environment_id,
        )
        active = self._revision_service.get_active_revision(
            application_id,
            application_environment_id,
        )
        return ServingStateView(
            application_id=application_id,
            application_environment_id=application_environment_id,
            traffic_serving_revision_id=(
                serving.traffic_serving_revision_id if serving is not None else None
            ),
            prior_traffic_revision_id=(
                serving.prior_traffic_revision_id if serving is not None else None
            ),
            serving_pointer_revision=(
                serving.serving_pointer_revision if serving is not None else 0
            ),
            active_revision=_revision_view(active) if active is not None else None,
        )

    def inspect_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> RuntimeRevisionView:
        revision = self._revision_store.get_revision(runtime_revision_id)
        if (
            revision is None
            or revision.application_id != application_id
            or revision.application_environment_id != application_environment_id
        ):
            raise AgentDistributionNotFoundError(
                f"runtime revision {runtime_revision_id} was not found"
            )
        return _revision_view(revision)

    def inspect_revision_history(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> RevisionHistoryView:
        serving = self.inspect_serving(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        revisions = self._revision_store.list_revisions_for_environment(
            application_id,
            application_environment_id,
        )
        return RevisionHistoryView(
            traffic_serving_revision_id=serving.traffic_serving_revision_id,
            prior_traffic_revision_id=serving.prior_traffic_revision_id,
            revisions=tuple(_revision_view(item) for item in revisions),
        )

    def inspect_activation(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> ActivationStatusView:
        serving = self.inspect_serving(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        pending = self._pending_candidate(
            application_id=application_id,
            application_environment_id=application_environment_id,
            traffic_serving_revision_id=serving.traffic_serving_revision_id,
        )
        instance_state: str | None = None
        readiness: str | None = None
        if serving.traffic_serving_revision_id is not None:
            instance = self._deployment_instance_store.get_instance(
                application_id,
                application_environment_id,
                serving.traffic_serving_revision_id,
            )
            if instance is not None:
                instance_state = instance.instance_state.value
                readiness = instance.readiness_evidence_ref
        return ActivationStatusView(
            serving=serving,
            candidate_revision=_revision_view(pending) if pending is not None else None,
            serving_instance_state=instance_state,
            serving_readiness_evidence_ref=readiness,
        )

    def inspect_agent_status(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        logical_agent_id: str,
    ) -> AgentStatusView:
        bindings = [
            binding
            for binding in self._binding_service.list_bindings_for_environment(
                application_id,
                application_environment_id,
            )
            if binding.logical_agent_id == logical_agent_id
            and not binding.tombstone
        ]
        binding = bindings[0] if bindings else None
        installed = False
        package_digest: str | None = None
        distribution_package_id: str | None = None
        if binding is not None:
            active = self._installation_service.resolve_active_for_slot(
                application_environment_id,
                binding.installation_slot_id,
            )
            installed = active is not None and installation_state_is_installed(
                active.installation_state
            )
            if active is not None:
                package_digest = active.package_identity.package_digest
                distribution_package_id = active.package_identity.distribution_package_id
        serving = self.inspect_serving(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        included = False
        active_revision = serving.active_revision
        if (
            active_revision is not None
            and package_digest is not None
            and package_digest in active_revision.installed_agent_package_digests
        ):
            included = True
        pending = self._pending_candidate(
            application_id=application_id,
            application_environment_id=application_environment_id,
            traffic_serving_revision_id=serving.traffic_serving_revision_id,
        )
        available: bool | None = None
        if self._catalog_provider is not None and distribution_package_id is not None:
            available = any(
                entry.package_id_line == distribution_package_id
                for entry in self._catalog_provider.list_entries()
            )
        roster = self._build_roster(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        enabled = any(
            entry.logical_agent_id == logical_agent_id and entry.effective_enablement
            for entry in roster.entries
        )
        return AgentStatusView(
            logical_agent_id=logical_agent_id,
            available=available,
            installed=installed,
            bound=binding is not None,
            enabled_in_desired_state=enabled,
            included_in_active_revision=included,
            traffic_serving_revision_id=serving.traffic_serving_revision_id,
            pending_candidate_revision_id=(
                pending.runtime_revision_id if pending is not None else None
            ),
        )

    def install_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: InstallAgentRequest,
        principal: RequestIdentity,
    ) -> InstallationMutationResult:
        self._require_environment_tenant_scope(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            operation="install_agent",
        )
        identity = self._resolve_install_identity(request)
        existing = self._installation_store.get_installation(request.installation_id)
        if existing is not None:
            if (
                existing.installation_slot_id == request.installation_slot_id
                and existing.environment_id == application_environment_id
                and existing.package_identity == identity
            ):
                return InstallationMutationResult(
                    installation=_installation_view(existing),
                    audit_event_types=(),
                )
            raise InstallationSlotConflict(
                "installation id already used with different identity"
            )

        active_for_slot = self._installation_service.resolve_active_for_slot(
            application_environment_id,
            request.installation_slot_id,
        )
        if active_for_slot is None:
            current_revision = installation_absent_token(
                installation_slot_id=request.installation_slot_id,
            )
        else:
            current_revision = installation_state_token(
                installation_slot_id=active_for_slot.installation_slot_id,
                installation_id=active_for_slot.installation_id,
                installation_state=active_for_slot.installation_state.value,
                package_digest=active_for_slot.package_identity.package_digest,
            )
        self._authorize_desired_state_mutation(
            build_install_agent_mutation_request(
                principal=principal,
                application_id=application_id,
                application_environment_id=application_environment_id,
                mutation_id=request.mutation_id,
                installation_slot_id=request.installation_slot_id,
                installation_id=request.installation_id,
                package_digest=identity.package_digest,
                current_revision=current_revision,
            ),
            operation="install_agent",
        )

        created = self._installation_service.create_candidate_installation(
            installation_id=request.installation_id,
            installation_slot_id=request.installation_slot_id,
            environment_id=application_environment_id,
            package_identity=identity,
        )
        verified = self._installation_service.mark_verified(
            request.installation_id,
            artifact_store_ref=request.artifact_store_ref,
            trust_record=request.trust_record,
        )
        promoted = self._installation_service.promote_verified_to_active(
            request.installation_id
        )
        self._artifact_metadata_store.persist_metadata(
            AgentArtifactMetadata(
                package_digest=identity.package_digest,
                artifact_store_ref=request.artifact_store_ref,
                distribution_package_id=identity.distribution_package_id,
                agent_project_metadata_ref=request.agent_project_metadata_ref,
            )
        )
        return InstallationMutationResult(
            installation=_installation_view(promoted.value),
            audit_event_types=_event_types(created, verified, promoted),
        )

    def bind_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: BindAgentRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult:
        self._require_environment_tenant_scope(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            operation="bind_agent",
        )
        existing = self._binding_store.get_binding(request.application_binding_id)
        if existing is not None:
            if (
                existing.application_id == application_id
                and existing.application_environment_id == application_environment_id
                and existing.logical_agent_id == request.logical_agent_id
                and existing.installation_slot_id == request.installation_slot_id
            ):
                return BindingMutationResult(
                    binding=_binding_view(existing),
                    audit_event_types=(),
                )
            raise BindingRevisionConflict("application_binding_id already used")
        self._authorize_desired_state_mutation(
            build_bind_agent_mutation_request(
                principal=principal,
                application_id=application_id,
                application_environment_id=application_environment_id,
                mutation_id=request.mutation_id,
                application_binding_id=request.application_binding_id,
                logical_agent_id=request.logical_agent_id,
                installation_slot_id=request.installation_slot_id,
                enablement=request.enablement,
                current_revision=binding_absent_token(),
            ),
            operation="bind_agent",
        )
        result = self._binding_service.create_binding(
            application_binding_id=request.application_binding_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
            logical_agent_id=request.logical_agent_id,
            installation_slot_id=request.installation_slot_id,
            config=request.config,
            secret_refs=request.secret_refs,
            policy_overrides=request.policy_overrides,
            factory_reference=request.factory_reference,
            builtin_package_ref=request.builtin_package_ref,
            enablement=request.enablement,
        )
        return BindingMutationResult(
            binding=_binding_view(result.value),
            audit_event_types=_event_types(result),
        )

    def update_binding_config(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        application_binding_id: str,
        request: UpdateAgentBindingRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult:
        self._require_environment_tenant_scope(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            operation="update_binding_config",
        )
        binding = self._require_binding_scope(
            application_binding_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        config_digest_value = binding_config_digest(request.config)
        self._authorize_desired_state_mutation(
            build_update_binding_config_mutation_request(
                principal=principal,
                application_id=application_id,
                application_environment_id=application_environment_id,
                mutation_id=request.mutation_id,
                application_binding_id=application_binding_id,
                expected_revision=request.expected_revision,
                config_digest_value=config_digest_value,
            ),
            operation="update_binding_config",
        )
        result = self._binding_service.update_config(
            application_binding_id,
            request.config,
            expected_revision=request.expected_revision,
        )
        return BindingMutationResult(
            binding=_binding_view(result.value),
            audit_event_types=_event_types(result),
        )

    def enable_binding(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        application_binding_id: str,
        request: SetAgentEnablementRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult:
        self._require_environment_tenant_scope(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            operation="enable_binding",
        )
        binding = self._require_binding_scope(
            application_binding_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        self._authorize_desired_state_mutation(
            build_enable_binding_mutation_request(
                principal=principal,
                application_id=application_id,
                application_environment_id=application_environment_id,
                mutation_id=request.mutation_id,
                application_binding_id=application_binding_id,
                expected_revision=request.expected_revision,
                current_enablement=binding.enablement,
            ),
            operation="enable_binding",
        )
        result = self._binding_service.enable(
            application_binding_id,
            expected_revision=request.expected_revision,
        )
        return BindingMutationResult(
            binding=_binding_view(result.value),
            audit_event_types=_event_types(result),
        )

    def disable_binding(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        application_binding_id: str,
        request: SetAgentEnablementRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult:
        self._require_environment_tenant_scope(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            operation="disable_binding",
        )
        binding = self._require_binding_scope(
            application_binding_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        self._authorize_desired_state_mutation(
            build_disable_binding_mutation_request(
                principal=principal,
                application_id=application_id,
                application_environment_id=application_environment_id,
                mutation_id=request.mutation_id,
                application_binding_id=application_binding_id,
                expected_revision=request.expected_revision,
                current_enablement=binding.enablement,
            ),
            operation="disable_binding",
        )
        result = self._binding_service.disable(
            application_binding_id,
            expected_revision=request.expected_revision,
        )
        return BindingMutationResult(
            binding=_binding_view(result.value),
            audit_event_types=_event_types(result),
        )

    def build_application_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: BuildApplicationRevisionRequest,
        principal: RequestIdentity,
    ) -> BuildRevisionResult:
        if self._materialization_service is None or self._graph_builder is None:
            raise AgentPlatformAdminBlockedError(
                "AP-11_BLOCKED_BY_MISSING_BUILD_APPLY_SERVICE",
                "build/apply requires graph builder and materialization service",
            )
        if self._metadata_provider is None:
            raise AgentPlatformAdminBlockedError(
                "AP-11_BLOCKED_BY_MISSING_BUILD_APPLY_SERVICE",
                "build/apply requires AgentProjectMetadataProvider",
            )
        if self._lock_service is None:
            raise AgentPlatformAdminBlockedError(
                "AP-11_BLOCKED_BY_MISSING_DEPENDENCY_RESOLVER",
                "build/apply requires an injected DependencyResolver",
            )
        self._require_environment_tenant_scope(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            operation="build_application_revision",
        )

        existing = self._revision_store.get_revision(request.runtime_revision_id)
        roster = self._build_roster(
            application_id=application_id,
            application_environment_id=application_environment_id,
            manifest_release_id=request.application_release_id,
        )
        requirement_set = self._requirement_set_builder.build(roster)
        specification = build_candidate_dependency_specification(
            repository_declaration=request.repository_declaration,
            installed_agent_requirement_set=requirement_set,
            platform_version=request.platform_version,
        )
        resolver_input = DependencyResolverInput(
            specification=specification,
            resolver_algorithm_id=request.resolver_algorithm_id,
            resolver_algorithm_version=request.resolver_algorithm_version,
        )
        lock = self._lock_service.produce_lock(resolver_input)
        metadata_refs = {
            package.distribution_package_id: package.agent_project_metadata_ref
            for package in requirement_set.agent_packages
        }
        graph = self._graph_builder.build(
            lock=lock,
            effective_roster=roster,
            repository_declaration=request.repository_declaration,
            agent_metadata_refs=metadata_refs,
        )
        graph = self._graph_validator.validate(
            lock=lock,
            effective_roster=roster,
            graph=graph,
        )
        if roster.effective_roster_revision_id is None:
            raise AgentDistributionNotFoundError("effective roster lacks revision identity")
        build_input_digest_value = build_input_digest(
            application_release_id=request.application_release_id,
            platform_version=request.platform_version,
            python_version=request.python_version,
            source_context_root=request.source_context_root,
            application_source_root=request.application_source_root,
            agent_source_roots=request.agent_source_roots,
            materialization_topology=request.materialization_topology.value,
            repository_declaration=request.repository_declaration,
            resolver_algorithm_id=request.resolver_algorithm_id,
            resolver_algorithm_version=request.resolver_algorithm_version,
        )
        identity_digest = build_runtime_revision_identity_digest(
            runtime_revision_id=request.runtime_revision_id,
            application_release_id=request.application_release_id,
            platform_version=request.platform_version,
            effective_roster_revision_id=roster.effective_roster_revision_id,
            lock_digest=lock.lock_digest,
            graph_digest=graph.runtime_graph_digest,
            materialization_topology=request.materialization_topology.value,
            build_input_digest=build_input_digest_value,
        )

        if existing is not None:
            if (
                existing.application_id != application_id
                or existing.application_environment_id != application_environment_id
            ):
                raise RuntimeRevisionConflict("runtime_revision_id already used")
            if self._existing_revision_matches_proposed_build(
                existing,
                request=request,
                effective_roster_revision_id=roster.effective_roster_revision_id,
                lock_digest=lock.lock_digest,
                graph_digest=graph.runtime_graph_digest,
            ):
                if existing.revision_state in {
                    RuntimeRevisionState.VALIDATED,
                    RuntimeRevisionState.CANDIDATE,
                }:
                    return self._build_revision_result_from_existing(existing)
            raise RuntimeRevisionConflict(
                "runtime_revision_id conflicts with requested build identity"
            )

        authorization = self._authorize_build(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=request.mutation_id,
            runtime_revision_id=request.runtime_revision_id,
            identity_digest=identity_digest,
        )

        persisted_lock = self._lock_store.persist_lock(lock)
        enabled_digests = tuple(
            sorted(
                entry.package_digest
                for entry in roster.entries
                if entry.effective_enablement
            )
        )
        candidate = RuntimeRevision(
            runtime_revision_id=request.runtime_revision_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
            application_release_id=request.application_release_id,
            platform_version=request.platform_version,
            effective_roster_revision_id=roster.effective_roster_revision_id,
            installed_agent_package_digests=enabled_digests,
            materialized_runtime_lock_id=persisted_lock.lock_id,
            materialized_runtime_lock_digest=persisted_lock.lock_digest,
            runtime_graph_digest=graph.runtime_graph_digest,
            materialization_topology=request.materialization_topology,
            revision_state=RuntimeRevisionState.CANDIDATE,
        )
        persisted = self._revision_service.persist_candidate_revision(candidate)
        build_context = ApplicationBuildContext(
            application_id=application_id,
            application_release_id=request.application_release_id,
            application_environment_id=application_environment_id,
            source_context_root=request.source_context_root,
            platform_version=request.platform_version,
            python_version=request.python_version,
            output_root=request.output_root,
            application_source_root=request.application_source_root,
            agent_source_roots=request.agent_source_roots,
        )
        output = self._materialization_service.materialize(
            MaterializationInput(
                runtime_revision=persisted.value,
                materialized_runtime_lock=persisted_lock,
                candidate_runtime_graph=graph,
                effective_roster=roster,
                application_build_context=build_context,
            )
        )
        validated = persisted.value.model_copy(
            update={
                "revision_state": RuntimeRevisionState.VALIDATED,
                "materialization_artifact_digest": output.materialization_artifact_digest,
                "materialization_topology": output.topology,
            }
        )
        marked = self._revision_service.mark_validated(
            request.runtime_revision_id,
            validated_revision=validated,
        )
        serving = self._serving_store.get_serving_record(
            application_id,
            application_environment_id,
        )
        if (
            serving is not None
            and serving.traffic_serving_revision_id == marked.value.runtime_revision_id
        ):
            raise RuntimeActivationConflict("build must not activate serving traffic")
        return BuildRevisionResult(
            runtime_revision_id=marked.value.runtime_revision_id,
            revision_state=marked.value.revision_state,
            effective_roster_revision_id=marked.value.effective_roster_revision_id,
            materialized_runtime_lock_id=marked.value.materialized_runtime_lock_id,
            materialized_runtime_lock_digest=marked.value.materialized_runtime_lock_digest,
            runtime_graph_digest=marked.value.runtime_graph_digest,
            materialization_artifact_digest=marked.value.materialization_artifact_digest,
            artifact_locator=output.artifact_locator,
            materialization_topology=marked.value.materialization_topology,
            authorization_evidence=authorization.evidence,
            audit_event_types=_event_types(persisted, marked),
        )

    def activate_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: ActivateRuntimeRevisionRequest,
        principal: RequestIdentity,
    ) -> ActivationResultView:
        self._require_revision_scope(
            runtime_revision_id=request.runtime_revision_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        prepared = self._activation_service.prepare_candidate(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=request.runtime_revision_id,
            artifact_locator=request.artifact_locator,
        )
        serving = self._serving_store.get_serving_record(
            application_id,
            application_environment_id,
        )
        current_traffic_revision_id = (
            serving.traffic_serving_revision_id if serving is not None else None
        )
        current_pointer_revision = serving.serving_pointer_revision if serving is not None else 0
        authorization = self._authorize_activation(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=request.mutation_id,
            current_traffic_revision_id=current_traffic_revision_id,
            current_serving_pointer_revision=current_pointer_revision,
            target_runtime_revision_id=request.runtime_revision_id,
        )
        committed = self._activation_service.commit_activation(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=request.runtime_revision_id,
            expected_prior_traffic_revision_id=request.expected_prior_traffic_revision_id,
            expected_serving_pointer_revision=request.expected_serving_pointer_revision,
            expected_artifact_digest=request.expected_artifact_digest,
        )
        serving_after = committed.value.serving_record
        activated = committed.value.activated_revision
        return ActivationResultView(
            traffic_serving_revision_id=serving_after.traffic_serving_revision_id,
            serving_pointer_revision=serving_after.serving_pointer_revision,
            activated_revision_id=activated.runtime_revision_id,
            revision_state=activated.revision_state,
            prior_traffic_revision_id=serving_after.prior_traffic_revision_id,
            authorization_evidence=authorization.evidence,
            audit_event_types=_event_types(prepared, committed),
        )

    def rollback_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: RollbackRuntimeRevisionRequest,
        principal: RequestIdentity,
    ) -> RollbackResultView:
        serving = self._serving_store.get_serving_record(
            application_id,
            application_environment_id,
        )
        if serving is None or serving.prior_traffic_revision_id is None:
            raise RuntimeRollbackError("no prior traffic revision available for rollback")
        if (
            request.target_runtime_revision_id is not None
            and request.target_runtime_revision_id != serving.prior_traffic_revision_id
        ):
            raise RuntimeActivationConflict(
                "rollback target does not match immutable prior traffic revision"
            )
        if serving.traffic_serving_revision_id is None:
            raise RuntimeRollbackError("no current serving revision to rollback from")
        target_revision_id = serving.prior_traffic_revision_id
        authorization = self._authorize_rollback(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=request.mutation_id,
            current_traffic_revision_id=serving.traffic_serving_revision_id,
            current_serving_pointer_revision=serving.serving_pointer_revision,
            target_runtime_revision_id=target_revision_id,
        )
        rolled = self._activation_service.rollback(
            application_id=application_id,
            application_environment_id=application_environment_id,
            expected_current_traffic_revision_id=request.expected_current_traffic_revision_id,
            expected_serving_pointer_revision=request.expected_serving_pointer_revision,
        )
        serving_after = rolled.value.serving_record
        restored = rolled.value.restored_revision
        superseded_id: str | None = None
        if rolled.value.superseded_instance is not None:
            superseded_id = rolled.value.superseded_instance.runtime_revision_id
        return RollbackResultView(
            traffic_serving_revision_id=serving_after.traffic_serving_revision_id,
            serving_pointer_revision=serving_after.serving_pointer_revision,
            restored_revision_id=restored.runtime_revision_id,
            revision_state=restored.revision_state,
            superseded_revision_id=superseded_id,
            authorization_evidence=authorization.evidence,
            audit_event_types=_event_types(rolled),
        )

    def _resolve_install_identity(self, request: InstallAgentRequest) -> AgentPackageIdentity:
        if request.catalog_entry_id is None:
            return request.package_identity
        provider = self._catalog_provider
        if provider is None:
            raise AgentPlatformAdminBlockedError(
                "AP-11_BLOCKED_BY_MISSING_CATALOG_PROVIDER",
                "catalog-backed install requires an injected CatalogSourceProvider",
            )
        selector = request.version_selector or request.package_identity.package_version
        for entry in provider.list_entries():
            if entry.catalog_entry_id != request.catalog_entry_id:
                continue
            resolution = provider.resolve_package(entry, version_selector=selector)
            return resolution.package_candidate.to_digest_pinned()
        raise AgentDistributionNotFoundError(
            f"catalog entry {request.catalog_entry_id} was not found"
        )

    def _build_roster(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        manifest_release_id: str = "unspecified",
    ) -> EffectiveRoster:
        bindings = self._binding_service.list_bindings_for_environment(
            application_id,
            application_environment_id,
        )
        return self._roster_builder.build(
            application_id=application_id,
            application_environment_id=application_environment_id,
            manifest_release_id=manifest_release_id,
            manifest_defaults=self._manifest_defaults,
            durable_bindings=bindings,
        )

    def _assert_installation_application_scope(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        record: AgentInstallationRecord,
    ) -> None:
        bindings = self._binding_service.list_bindings_for_environment(
            application_id,
            application_environment_id,
        )
        if not any(
            binding.installation_slot_id == record.installation_slot_id
            for binding in bindings
        ):
            raise AgentDistributionNotFoundError(
                f"installation {record.installation_id} was not found"
            )

    def _require_revision_scope(
        self,
        *,
        runtime_revision_id: str,
        application_id: str,
        application_environment_id: str,
    ) -> RuntimeRevision:
        revision = self._revision_store.get_revision(runtime_revision_id)
        if (
            revision is None
            or revision.application_id != application_id
            or revision.application_environment_id != application_environment_id
        ):
            raise AgentDistributionNotFoundError(
                f"runtime revision {runtime_revision_id} was not found"
            )
        return revision

    def _require_binding_scope(
        self,
        application_binding_id: str,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> ApplicationAgentBinding:
        binding = self._binding_store.get_binding(application_binding_id)
        if (
            binding is None
            or binding.application_id != application_id
            or binding.application_environment_id != application_environment_id
        ):
            raise AgentDistributionNotFoundError(
                f"binding {application_binding_id} was not found"
            )
        return binding

    def _pending_candidate(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        traffic_serving_revision_id: str | None,
    ) -> RuntimeRevision | None:
        pending: list[RuntimeRevision] = []
        for revision in self._revision_store.list_revisions_for_environment(
            application_id,
            application_environment_id,
        ):
            if revision.runtime_revision_id == traffic_serving_revision_id:
                continue
            if revision.revision_state in {
                RuntimeRevisionState.CANDIDATE,
                RuntimeRevisionState.VALIDATED,
            }:
                pending.append(revision)
        if not pending:
            return None
        pending.sort(key=lambda item: item.runtime_revision_id)
        return pending[-1]

    def _require_mutation_authorization_boundary(
        self,
    ) -> ControlPlaneMutationAuthorizationBoundary:
        boundary = self._mutation_authorization_boundary
        if boundary is None:
            raise AgentPlatformAdminBlockedError(
                "AP-11_BLOCKED_BY_MISSING_MUTATION_AUTHORIZATION_BOUNDARY",
                "control-plane mutations require ControlPlaneMutationAuthorizationBoundary",
            )
        return boundary

    def _require_environment_tenant_resolver(
        self,
    ) -> ApplicationEnvironmentTenantResolver:
        resolver = self._environment_tenant_resolver
        if resolver is None:
            raise AgentPlatformAdminBlockedError(
                "AP-11_BLOCKED_BY_MISSING_TENANT_AUTHORITY",
                "control-plane mutations require ApplicationEnvironmentTenantResolver",
            )
        return resolver

    def _require_environment_tenant_scope(
        self,
        *,
        principal: RequestIdentity,
        application_id: str,
        application_environment_id: str,
        operation: str,
    ) -> None:
        resolver = self._require_environment_tenant_resolver()
        environment_tenant = resolver.resolve_tenant_id(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        if environment_tenant != principal.tenant_id:
            raise AgentPlatformAdminGovernanceBlockedError(
                "AP-11_BLOCKED_BY_TENANT_AUTHORITY",
                f"{operation} denied by tenant authority scope",
                policy_action=PolicyAction.DENY.value,
                authorization_evidence=self._synthetic_tenant_deny_evidence(
                    principal=principal,
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                ),
            )

    def _authorize_desired_state_mutation(
        self,
        request: ControlPlaneMutationRequest,
        *,
        operation: str,
    ) -> ControlPlaneMutationAuthorizationResult:
        resolver = self._require_environment_tenant_resolver()
        boundary = self._require_mutation_authorization_boundary()
        return self._enforce_authorization_result(
            authorize_scoped_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=resolver,
                request=request,
            ),
            operation=operation,
        )

    def _authorize_activation(
        self,
        *,
        principal: RequestIdentity,
        application_id: str,
        application_environment_id: str,
        mutation_id: str,
        current_traffic_revision_id: str | None,
        current_serving_pointer_revision: int,
        target_runtime_revision_id: str,
    ) -> ControlPlaneMutationAuthorizationResult:
        resolver = self._require_environment_tenant_resolver()
        boundary = self._require_mutation_authorization_boundary()
        request = build_activation_mutation_request(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=mutation_id,
            current_traffic_revision_id=current_traffic_revision_id,
            current_serving_pointer_revision=current_serving_pointer_revision,
            target_runtime_revision_id=target_runtime_revision_id,
        )
        return self._enforce_authorization_result(
            authorize_scoped_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=resolver,
                request=request,
            ),
            operation="activation",
        )

    def _authorize_build(
        self,
        *,
        principal: RequestIdentity,
        application_id: str,
        application_environment_id: str,
        mutation_id: str,
        runtime_revision_id: str,
        identity_digest: str,
    ) -> ControlPlaneMutationAuthorizationResult:
        resolver = self._require_environment_tenant_resolver()
        boundary = self._require_mutation_authorization_boundary()
        request = build_runtime_revision_mutation_request(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=mutation_id,
            runtime_revision_id=runtime_revision_id,
            identity_digest=identity_digest,
        )
        return self._enforce_authorization_result(
            authorize_scoped_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=resolver,
                request=request,
            ),
            operation="build_application_revision",
        )

    @staticmethod
    def _existing_revision_matches_proposed_build(
        existing: RuntimeRevision,
        *,
        request: BuildApplicationRevisionRequest,
        effective_roster_revision_id: str,
        lock_digest: str,
        graph_digest: str,
    ) -> bool:
        return (
            existing.application_release_id == request.application_release_id
            and existing.platform_version == request.platform_version
            and existing.effective_roster_revision_id == effective_roster_revision_id
            and existing.materialized_runtime_lock_digest == lock_digest
            and existing.runtime_graph_digest == graph_digest
            and existing.materialization_topology == request.materialization_topology
        )

    @staticmethod
    def _build_revision_result_from_existing(existing: RuntimeRevision) -> BuildRevisionResult:
        return BuildRevisionResult(
            runtime_revision_id=existing.runtime_revision_id,
            revision_state=existing.revision_state,
            effective_roster_revision_id=existing.effective_roster_revision_id,
            materialized_runtime_lock_id=existing.materialized_runtime_lock_id,
            materialized_runtime_lock_digest=existing.materialized_runtime_lock_digest,
            runtime_graph_digest=existing.runtime_graph_digest,
            materialization_artifact_digest=existing.materialization_artifact_digest,
            materialization_topology=existing.materialization_topology,
        )

    def _authorize_rollback(
        self,
        *,
        principal: RequestIdentity,
        application_id: str,
        application_environment_id: str,
        mutation_id: str,
        current_traffic_revision_id: str,
        current_serving_pointer_revision: int,
        target_runtime_revision_id: str,
    ) -> ControlPlaneMutationAuthorizationResult:
        resolver = self._require_environment_tenant_resolver()
        boundary = self._require_mutation_authorization_boundary()
        request = build_rollback_mutation_request(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=mutation_id,
            current_traffic_revision_id=current_traffic_revision_id,
            current_serving_pointer_revision=current_serving_pointer_revision,
            target_runtime_revision_id=target_runtime_revision_id,
        )
        return self._enforce_authorization_result(
            authorize_scoped_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=resolver,
                request=request,
            ),
            operation="rollback",
        )

    @staticmethod
    def _enforce_authorization_result(
        result: ControlPlaneMutationAuthorizationResult,
        *,
        operation: str,
    ) -> ControlPlaneMutationAuthorizationResult:
        if result.permitted:
            return result
        action = result.decision.action
        if result.decision.reason == "tenant_authority_mismatch":
            raise AgentPlatformAdminGovernanceBlockedError(
                "AP-11_BLOCKED_BY_TENANT_AUTHORITY",
                f"{operation} denied by tenant authority scope",
                policy_action=action.value,
                authorization_evidence=result.evidence,
                authorization_scope=result.authorization_scope,
            )
        if action is PolicyAction.REQUIRE_HUMAN:
            raise AgentPlatformAdminGovernanceBlockedError(
                "AP-11_BLOCKED_BY_REQUIRE_HUMAN",
                f"{operation} requires governed human approval",
                policy_action=action.value,
                authorization_evidence=result.evidence,
                authorization_scope=result.authorization_scope,
            )
        if action is PolicyAction.ESCALATE:
            raise AgentPlatformAdminGovernanceBlockedError(
                "AP-11_BLOCKED_BY_ESCALATE",
                f"{operation} requires escalation",
                policy_action=action.value,
                authorization_evidence=result.evidence,
                authorization_scope=result.authorization_scope,
            )
        raise AgentPlatformAdminGovernanceBlockedError(
            "AP-11_BLOCKED_BY_GOVERNANCE_DENY",
            f"{operation} denied by control-plane governance",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )

    @staticmethod
    def _synthetic_tenant_deny_evidence(
        *,
        principal: RequestIdentity,
        application_id: str,
        application_environment_id: str,
    ):
        from intergrax.agent_distribution.control_plane_governance import (
            AGENT_DISTRIBUTION_RESOURCE_TYPE,
            application_environment_resource_id,
            application_environment_resource_scope,
            serving_revision_token,
        )
        from intergrax.contracts.control_plane_mutation import (
            ControlPlaneMutationAuthorizationEvidence,
            ControlPlaneMutationRisk,
        )

        return ControlPlaneMutationAuthorizationEvidence(
            request_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            mutation_id="tenant-authority-deny",
            mutation_type="agent_distribution.tenant_authority",
            tenant_id=principal.tenant_id,
            resource_scope=application_environment_resource_scope(
                application_id=application_id,
                application_environment_id=application_environment_id,
            ),
            resource_type=AGENT_DISTRIBUTION_RESOURCE_TYPE,
            resource_id=application_environment_resource_id(
                application_id=application_id,
                application_environment_id=application_environment_id,
            ),
            current_revision=serving_revision_token(
                traffic_revision_id=None,
                serving_pointer_revision=0,
            ),
            target_revision=serving_revision_token(
                traffic_revision_id=None,
                serving_pointer_revision=0,
            ),
            risk_classification=ControlPlaneMutationRisk.HIGH,
            principal_type=principal.principal_type,
            principal_user_id=principal.user_id,
            principal_auth_subject=principal.auth_subject,
            policy_action=PolicyAction.DENY,
            policy_rule_id="agent_distribution.tenant_scope",
            policy_decision_id="tenant-authority-deny",
        )
