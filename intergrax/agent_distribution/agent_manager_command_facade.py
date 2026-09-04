# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent Manager mutation facade — strict 1:1 delegation to AgentPlatformAdminService."""

from __future__ import annotations

from intergrax.agent_distribution.admin_models import (
    ActivationResultView,
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BindingMutationResult,
    BuildApplicationRevisionRequest,
    BuildRevisionResult,
    InstallAgentRequest,
    InstallationMutationResult,
    RollbackResultView,
    RollbackRuntimeRevisionRequest,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.contracts.agent_run import RequestIdentity


class AgentManagerCommandFacade:
    """Typed control-plane facade — no direct lifecycle store writes."""

    def __init__(self, admin_service: AgentPlatformAdminService) -> None:
        self._admin = admin_service

    def install_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: InstallAgentRequest,
        principal: RequestIdentity,
    ) -> InstallationMutationResult:
        return self._admin.install_agent(
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
        return self._admin.bind_agent(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=request,
            principal=principal,
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
        return self._admin.update_binding_config(
            application_id=application_id,
            application_environment_id=application_environment_id,
            application_binding_id=application_binding_id,
            request=request,
            principal=principal,
        )

    def enable_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        application_binding_id: str,
        request: SetAgentEnablementRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult:
        return self._admin.enable_binding(
            application_id=application_id,
            application_environment_id=application_environment_id,
            application_binding_id=application_binding_id,
            request=request,
            principal=principal,
        )

    def disable_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        application_binding_id: str,
        request: SetAgentEnablementRequest,
        principal: RequestIdentity,
    ) -> BindingMutationResult:
        return self._admin.disable_binding(
            application_id=application_id,
            application_environment_id=application_environment_id,
            application_binding_id=application_binding_id,
            request=request,
            principal=principal,
        )

    def build_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: BuildApplicationRevisionRequest,
        principal: RequestIdentity,
    ) -> BuildRevisionResult:
        return self._admin.build_application_revision(
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
        return self._admin.activate_revision(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=request,
            principal=principal,
        )

    def rollback_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        request: RollbackRuntimeRevisionRequest,
        principal: RequestIdentity,
    ) -> RollbackResultView:
        return self._admin.rollback_revision(
            application_id=application_id,
            application_environment_id=application_environment_id,
            request=request,
            principal=principal,
        )
