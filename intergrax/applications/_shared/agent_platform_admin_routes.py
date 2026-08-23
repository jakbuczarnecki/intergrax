# © Artur Czarnecki. All rights reserved.

"""Shared V1 Agent Platform administration HTTP routes (AP-11)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import ValidationError

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
    RuntimeRevisionView,
    ServingStateView,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.catalog import CatalogEntryFilters
from intergrax.agent_distribution.errors import (
    AgentDistributionError,
    AgentDistributionNotFoundError,
    BindingLifecycleError,
    BindingRevisionConflict,
    EffectiveRosterConflict,
    InstallationLifecycleError,
    InstallationSlotConflict,
    MaterializationError,
    MaterializedRuntimeLockConflict,
    RuntimeActivationConflict,
    RuntimeActivationError,
    RuntimeReadinessError,
    RuntimeRevisionConflict,
    RuntimeRevisionLifecycleError,
    RuntimeRollbackError,
)
from intergrax.applications._shared.harness_auth import (
    _expected_api_key_for_request,
    _harness_auth_state_from_request,
    _identity_provider_from_request,
    _local_dev_auth_bypass_allowed,
    is_harness_api_key_valid,
    resolve_harness_authenticated_principal,
    verify_harness_bearer_identity,
)
from intergrax.applications._shared.harness_principal import harness_principal_to_request_identity
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.integrations.contracts.identity_provider import (
    identity_user_has_agent_platform_admin_authority,
)

_CONFLICT_ERRORS = (
    BindingRevisionConflict,
    BindingLifecycleError,
    InstallationSlotConflict,
    InstallationLifecycleError,
    RuntimeActivationConflict,
    RuntimeActivationError,
    RuntimeReadinessError,
    RuntimeRevisionConflict,
    RuntimeRevisionLifecycleError,
    RuntimeRollbackError,
    EffectiveRosterConflict,
    MaterializedRuntimeLockConflict,
    MaterializationError,
)


def _raise_admin_http(exc: Exception) -> None:
    if isinstance(exc, AgentPlatformAdminGovernanceBlockedError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.governance_http_detail(),
        ) from exc
    if isinstance(exc, AgentPlatformAdminBlockedError):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=exc.blocker_code,
        ) from exc
    if isinstance(exc, AgentDistributionNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, _CONFLICT_ERRORS):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if isinstance(exc, AgentDistributionError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    raise exc


def _require_agent_platform_admin_auth(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    resolve_agent_platform_admin_request_identity(
        request,
        x_api_key=x_api_key,
        authorization=authorization,
    )


def resolve_agent_platform_admin_request_identity(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    authorization: str | None = Header(default=None),
) -> RequestIdentity:
    """Authenticate admin routes and return canonical RequestIdentity (AD17)."""
    expected_api_key = _expected_api_key_for_request(request)
    if expected_api_key is not None and is_harness_api_key_valid(
        x_api_key=x_api_key,
        authorization=authorization,
        expected_api_key=expected_api_key,
    ):
        principal = resolve_harness_authenticated_principal(
            request,
            x_api_key=x_api_key,
            authorization=authorization,
        )
        if principal is not None:
            return harness_principal_to_request_identity(principal)
    identity_provider = _identity_provider_from_request(request)
    if identity_provider is not None:
        user = verify_harness_bearer_identity(
            authorization=authorization,
            identity_provider=identity_provider,
        )
        if user is not None:
            if not identity_user_has_agent_platform_admin_authority(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Agent platform admin authorization required",
                )
            principal = resolve_harness_authenticated_principal(
                request,
                x_api_key=x_api_key,
                authorization=authorization,
            )
            if principal is not None:
                return harness_principal_to_request_identity(principal)
    state = _harness_auth_state_from_request(request)
    if _local_dev_auth_bypass_allowed(state):
        return RequestIdentity(
            tenant_id="default",
            user_id="local-dev-admin",
            principal_type=PrincipalType.USER,
            auth_subject="local-dev-admin",
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing agent platform admin credentials",
    )


def mount_agent_platform_admin_routes(
    app: FastAPI,
    *,
    admin_service: AgentPlatformAdminService,
    prefix: str = "/v1/agent-platform",
) -> APIRouter:
    router = APIRouter(
        prefix=prefix,
        tags=["agent-platform-admin"],
        dependencies=[Depends(_require_agent_platform_admin_auth)],
    )
    env_prefix = "/applications/{application_id}/environments/{environment_id}"

    @router.get("/catalog/agents", response_model=CatalogListResult)
    def list_catalog() -> CatalogListResult:
        try:
            return admin_service.list_catalog(CatalogEntryFilters())
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(f"{env_prefix}/installations", response_model=InstallationListResult)
    def list_installations(
        application_id: str,
        environment_id: str,
    ) -> InstallationListResult:
        try:
            return admin_service.list_installed(
                application_id=application_id,
                application_environment_id=environment_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(
        f"{env_prefix}/installations/{{installation_id}}",
        response_model=InstallationView,
    )
    def inspect_installation(
        application_id: str,
        environment_id: str,
        installation_id: str,
    ) -> InstallationView:
        try:
            return admin_service.inspect_installation(
                application_id=application_id,
                application_environment_id=environment_id,
                installation_id=installation_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.post(f"{env_prefix}/installations", response_model=InstallationMutationResult)
    def install_agent(
        application_id: str,
        environment_id: str,
        body: InstallAgentRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> InstallationMutationResult:
        try:
            return admin_service.install_agent(
                application_id=application_id,
                application_environment_id=environment_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(f"{env_prefix}/bindings", response_model=BindingListResult)
    def list_bindings(application_id: str, environment_id: str) -> BindingListResult:
        try:
            return admin_service.list_bindings(
                application_id=application_id,
                application_environment_id=environment_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.post(f"{env_prefix}/bindings", response_model=BindingMutationResult)
    def bind_agent(
        application_id: str,
        environment_id: str,
        body: BindAgentRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> BindingMutationResult:
        try:
            return admin_service.bind_agent(
                application_id=application_id,
                application_environment_id=environment_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.patch(
        f"{env_prefix}/bindings/{{application_binding_id}}/config",
        response_model=BindingMutationResult,
    )
    def update_binding_config(
        application_id: str,
        environment_id: str,
        application_binding_id: str,
        body: UpdateAgentBindingRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> BindingMutationResult:
        try:
            return admin_service.update_binding_config(
                application_id=application_id,
                application_environment_id=environment_id,
                application_binding_id=application_binding_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.post(
        f"{env_prefix}/bindings/{{application_binding_id}}/enable",
        response_model=BindingMutationResult,
    )
    def enable_binding(
        application_id: str,
        environment_id: str,
        application_binding_id: str,
        body: SetAgentEnablementRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> BindingMutationResult:
        try:
            return admin_service.enable_binding(
                application_id=application_id,
                application_environment_id=environment_id,
                application_binding_id=application_binding_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.post(
        f"{env_prefix}/bindings/{{application_binding_id}}/disable",
        response_model=BindingMutationResult,
    )
    def disable_binding(
        application_id: str,
        environment_id: str,
        application_binding_id: str,
        body: SetAgentEnablementRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> BindingMutationResult:
        try:
            return admin_service.disable_binding(
                application_id=application_id,
                application_environment_id=environment_id,
                application_binding_id=application_binding_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(f"{env_prefix}/roster", response_model=EffectiveRosterView)
    def inspect_roster(application_id: str, environment_id: str) -> EffectiveRosterView:
        try:
            return admin_service.inspect_effective_roster(
                application_id=application_id,
                application_environment_id=environment_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(f"{env_prefix}/serving", response_model=ServingStateView)
    def inspect_serving(application_id: str, environment_id: str) -> ServingStateView:
        try:
            return admin_service.inspect_serving(
                application_id=application_id,
                application_environment_id=environment_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(f"{env_prefix}/revisions", response_model=RevisionHistoryView)
    def inspect_revision_history(
        application_id: str,
        environment_id: str,
    ) -> RevisionHistoryView:
        try:
            return admin_service.inspect_revision_history(
                application_id=application_id,
                application_environment_id=environment_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(
        f"{env_prefix}/revisions/{{runtime_revision_id}}",
        response_model=RuntimeRevisionView,
    )
    def inspect_revision(
        application_id: str,
        environment_id: str,
        runtime_revision_id: str,
    ) -> RuntimeRevisionView:
        try:
            return admin_service.inspect_revision(
                application_id=application_id,
                application_environment_id=environment_id,
                runtime_revision_id=runtime_revision_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(f"{env_prefix}/activation", response_model=ActivationStatusView)
    def inspect_activation(
        application_id: str,
        environment_id: str,
    ) -> ActivationStatusView:
        try:
            return admin_service.inspect_activation(
                application_id=application_id,
                application_environment_id=environment_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(
        f"{env_prefix}/agents/{{logical_agent_id}}/status",
        response_model=AgentStatusView,
    )
    def inspect_agent_status(
        application_id: str,
        environment_id: str,
        logical_agent_id: str,
    ) -> AgentStatusView:
        try:
            return admin_service.inspect_agent_status(
                application_id=application_id,
                application_environment_id=environment_id,
                logical_agent_id=logical_agent_id,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.post(f"{env_prefix}/revisions/build", response_model=BuildRevisionResult)
    def build_revision(
        application_id: str,
        environment_id: str,
        body: BuildApplicationRevisionRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> BuildRevisionResult:
        try:
            return admin_service.build_application_revision(
                application_id=application_id,
                application_environment_id=environment_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.post(f"{env_prefix}/revisions/activate", response_model=ActivationResultView)
    def activate_revision(
        application_id: str,
        environment_id: str,
        body: ActivateRuntimeRevisionRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> ActivationResultView:
        try:
            return admin_service.activate_revision(
                application_id=application_id,
                application_environment_id=environment_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.post(f"{env_prefix}/revisions/rollback", response_model=RollbackResultView)
    def rollback_revision(
        application_id: str,
        environment_id: str,
        body: RollbackRuntimeRevisionRequest,
        principal: RequestIdentity = Depends(resolve_agent_platform_admin_request_identity),
    ) -> RollbackResultView:
        try:
            return admin_service.rollback_revision(
                application_id=application_id,
                application_environment_id=environment_id,
                request=body,
                principal=principal,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    app.include_router(router)
    return router
