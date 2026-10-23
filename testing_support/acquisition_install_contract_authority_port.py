# © Artur Czarnecki. All rights reserved.

"""Harness-only lifecycle port: supplies install contract authority for proof agents."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.dynamic_acquisition import AgentPlatformLifecyclePort
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
from intergrax.contracts.agent_run import RequestIdentity
from testing_support.canonical_agent_lifecycle_composition import (
    CanonicalLifecycleProofConfig,
    lifecycle_install_package_contract_authority_records,
)


@dataclass(frozen=True, slots=True)
class ProofInstallContractAuthorityLifecyclePort:
    """Delegates AC-3 mutations; enriches install with digest-bound proof authority only."""

    inner: AgentPlatformLifecyclePort
    proof_config: CanonicalLifecycleProofConfig

    def install_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: InstallAgentRequest,
        principal: RequestIdentity,
    ) -> InstallationMutationResult:
        enriched = request
        if not request.package_contract_authority:
            identity = request.package_identity
            enriched = request.model_copy(
                update={
                    "package_contract_authority": lifecycle_install_package_contract_authority_records(
                        self.proof_config,
                        package_digest=identity.package_digest,
                        distribution_package_id=identity.distribution_package_id,
                        artifact_store_ref=request.artifact_store_ref,
                        agent_project_metadata_ref=request.agent_project_metadata_ref,
                    ),
                },
            )
        return self.inner.install_agent(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=enriched,
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
        return self.inner.bind_agent(
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
        return self.inner.enable_binding(
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
        return self.inner.build_application_revision(
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
        return self.inner.activate_revision(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=request,
            principal=principal,
        )

    def inspect_serving(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> ServingStateView:
        return self.inner.inspect_serving(
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
        return self.inner.inspect_revision(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=runtime_revision_id,
        )


__all__ = ["ProofInstallContractAuthorityLifecyclePort"]
