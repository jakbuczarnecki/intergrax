# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R015 (xd_06): tenant data and credentials do not cross to an unapproved provider.

Scope note: this module proves the controls that exist on the security plane (tool allow-list,
credential egress scope, tenant scope). The provider-egress *sensitivity* policy of the context
plane belongs to batch B-D2 (``intergrax/context/policy/egress.py``) and is reconciled at
integration (PDC-P6); it is deliberately not duplicated or imported here.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest
from intergrax.applications._shared.application_security_wiring import (
    TenantSecurityMiddleware,
    ToolInjectionDefenseMiddleware,
    default_tool_invocation_policy,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.middleware_hook_semantics import ToolCallHookPayload
from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialScopeAdmissionDecision,
    CredentialScopeMismatchError,
    CredentialUseGrant,
    CredentialUseScope,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from intergrax.utils.time_provider import TimeProvider
from tests.support.middleware_hook_test_context import tenant_intake_hook_context_for_test

from ._support import CountingSecretsStore, generated_secret

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET_PATH = "secrets/tenant-a/approved-provider"
_APPROVED_HOST = "https://api.approved-provider.example:443"
_UNAPPROVED_HOST = "https://collector.unapproved-provider.example:443"


class _Clock(TimeProvider):
    now: ClassVar[datetime] = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)

    @classmethod
    def utc_now(cls) -> datetime:
        return cls.now


class _AllowAll:
    def admit(self, grant: CredentialUseGrant, scope: CredentialUseScope):
        return CredentialScopeAdmissionDecision.ALLOW


def _grant(tenant: str = "tenant-a") -> CredentialUseGrant:
    return CredentialUseGrant(
        grant_id="grant-1",
        credential_ref=CredentialRef.from_secret_path(
            provider_id="approved_provider", secret_path=_SECRET_PATH, tenant_id=tenant
        ),
        tenant_id=tenant,
        provider_id="approved_provider",
        integration_id="approved_provider:chat",
        operation="provider.chat.complete",
        execution_id="exec-a",
        target_scope=canonicalize_network_egress_allowlist((_APPROVED_HOST,)),
        expires_at=_Clock.now + timedelta(minutes=5),
    )


def _scope(host: str, tenant: str = "tenant-a") -> CredentialUseScope:
    return CredentialUseScope(
        tenant_id=tenant,
        provider_id="approved_provider",
        integration_id="approved_provider:chat",
        operation="provider.chat.complete",
        execution_id="exec-a",
        target_scope=canonicalize_network_egress_allowlist((host,)),
    )


def _broker(store: CountingSecretsStore) -> ScopedCredentialBroker:
    return ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(store), admission=_AllowAll(), time_provider=_Clock
    )


def test_xd06_credential_is_not_released_for_an_unapproved_provider_host() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    with pytest.raises(CredentialScopeMismatchError):
        _broker(store).resolve_scoped(_grant(), _scope(_UNAPPROVED_HOST))
    assert store.reads == []


def test_xd06_credential_is_released_for_the_approved_provider_host_only() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    result = _broker(store).resolve_scoped(_grant(), _scope(_APPROVED_HOST))
    assert result.use_evidence.provider_id == "approved_provider"
    assert len(store.reads) == 1


def test_xd06_tenant_credential_is_not_released_for_another_tenant_scope() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    with pytest.raises(CredentialScopeMismatchError):
        _broker(store).resolve_scoped(_grant("tenant-a"), _scope(_APPROVED_HOST, tenant="tenant-b"))
    assert store.reads == []


def _tool_call(tool_id: str, allowed: list[str]) -> HookContext:
    return HookContext(
        task_id="run-1",
        run_id="run-1",
        phase=ExecutionPhase.STEP_EXECUTION,
        payload=ToolCallHookPayload(
            tool_id=tool_id,
            arguments={"tenant_records": "customer list"},
            allowed_tool_ids=allowed,
        ),
        runtime_state={},
    )


def test_xd06_tool_call_to_an_unapproved_provider_is_blocked() -> None:
    middleware = ToolInjectionDefenseMiddleware(default_tool_invocation_policy())
    approved = ["approved_provider.chat"]
    blocked = asyncio.run(
        middleware.before(HookPoint.BEFORE_TOOL_CALL, _tool_call("unapproved_provider.upload", approved))
    )
    assert blocked.action is HookAction.BLOCK
    allowed = asyncio.run(middleware.before(HookPoint.BEFORE_TOOL_CALL, _tool_call("approved_provider.chat", approved)))
    assert allowed.action is not HookAction.BLOCK


def test_xd06_cross_tenant_task_intake_is_blocked_before_any_provider_call() -> None:
    middleware = TenantSecurityMiddleware()
    ctx = tenant_intake_hook_context_for_test(
        tenant_id="tenant-a",
        resource_tenant_id="tenant-b",
        user_id="user-1",
        phase=ExecutionPhase.INTAKE,
    )
    result = asyncio.run(middleware.before(HookPoint.BEFORE_TASK_INTAKE, ctx))
    assert result.action is HookAction.BLOCK
    same_tenant = tenant_intake_hook_context_for_test(
        tenant_id="tenant-a",
        resource_tenant_id="tenant-a",
        user_id="user-1",
        phase=ExecutionPhase.INTAKE,
    )
    assert asyncio.run(middleware.before(HookPoint.BEFORE_TASK_INTAKE, same_tenant)).action is not HookAction.BLOCK
