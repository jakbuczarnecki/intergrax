# © Artur Czarnecki. All rights reserved.

"""AC-4 dynamic acquisition lifecycle port with reference production registry projection."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    ActivationResultView,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    BindingMutationResult,
    BuildRevisionResult,
    InstallAgentRequest,
    InstallationMutationResult,
    RuntimeRevisionView,
    ServingStateView,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.applications._shared.registry_projection_authority_resolver import (
    RegistryProjectionAuthorityResolver,
)


@dataclass(frozen=True, slots=True)
class ReferenceProductionAcquisitionLifecyclePort:
    """Delegates AC-3 mutations to admin; activation commits via AP-10 projection launcher."""

    admin: AgentPlatformAdminService
    launcher: ReferenceProductionLifecycleLauncher
    manifest: ApplicationManifest
    registry_projection_authority: RegistryProjectionAuthorityResolver
    principal: RequestIdentity

    def install_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: InstallAgentRequest,
        principal: RequestIdentity,
    ) -> InstallationMutationResult:
        return self.admin.install_agent(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=request,
            principal=principal,
        )

    def bind_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: BindAgentRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult:
        return self.admin.bind_agent(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=request,
            principal=principal,
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
        return self.admin.enable_binding(
            application_id=application_id,
            application_environment_id=application_environment_id,
            application_binding_id=application_binding_id,
            request=request,
            principal=principal,
        )

    def build_application_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: BuildApplicationRevisionRequest,
        principal: RequestIdentity,
    ) -> BuildRevisionResult:
        return self.admin.build_application_revision(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=request,
            principal=principal,
        )

    def activate_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: ActivateRuntimeRevisionRequest,
        principal: RequestIdentity,
    ) -> ActivationResultView:
        del principal
        bundle = build_production_registry_projection_input_bundle_for_revision(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=request.runtime_revision_id,
            manifest=self.manifest,
            build_context=ApplicationBuildContext.for_manifest(self.manifest),
            authority=self.registry_projection_authority,
        )
        serving_before = self.admin.inspect_serving(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        activation_request = request.model_copy(
            update={
                "expected_serving_pointer_revision": serving_before.serving_pointer_revision,
                "expected_prior_traffic_revision_id": serving_before.traffic_serving_revision_id,
            },
        )
        result = self.launcher.deploy_and_activate(
            projection_input=bundle,
            activation_request=activation_request,
            principal=self.principal,
            admission_mutation_id=reference_admission_mutation_id(
                request.runtime_revision_id,
            ),
        )
        return ActivationResultView(
            traffic_serving_revision_id=result.runtime_revision_id,
            serving_pointer_revision=result.serving_pointer_revision,
            activated_revision_id=result.runtime_revision_id,
            revision_state=RuntimeRevisionState.ACTIVE,
            prior_traffic_revision_id=serving_before.traffic_serving_revision_id,
        )

    def inspect_serving(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> ServingStateView:
        return self.admin.inspect_serving(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )

    def inspect_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> RuntimeRevisionView:
        return self.admin.inspect_revision(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=runtime_revision_id,
        )


__all__ = ["ReferenceProductionAcquisitionLifecyclePort"]
