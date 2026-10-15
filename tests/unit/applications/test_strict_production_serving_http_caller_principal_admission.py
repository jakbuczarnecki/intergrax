# © Artur Czarnecki. All rights reserved.

"""P3-A2-R2 — admitted root identity from HTTP caller principal (AUTH-06 seam)."""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from intergrax.applications._shared.strict_production_host_task_execution_wiring import (
    build_strict_production_serving_host_task_execution_governance,
)
from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    StrictProductionServingExecutionTenantBinding,
)
from intergrax.applications._shared.strict_production_serving_http_execution_principal import (
    bind_strict_serving_http_caller_principal,
    get_strict_serving_http_caller_principal,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BINDING = StrictProductionServingExecutionTenantBinding(
    authoritative_execution_tenant_id="env-prod",
    application_id="app_a",
    application_environment_id="env-prod",
    control_plane_environment_tenant_id="tenant-b",
)

_HTTP_CALLER = RequestIdentity(
    tenant_id="env-prod",
    user_id="api-key-operator",
    principal_type=PrincipalType.USER,
    auth_subject="api-key-operator",
)

_HOST_SERVICE_PRINCIPAL = RequestIdentity(
    tenant_id="tenant-b",
    user_id="p3a-production-launcher",
    principal_type=PrincipalType.SERVICE,
    auth_subject="p3a-production-launcher",
)


def _task() -> Task:
    return Task(
        tenant_id=_BINDING.authoritative_execution_tenant_id,
        user_id="body-user-not-admitted-principal",
        session_id="s",
        agent_id="a",
        message="m",
    )


def test_strict_governance_admits_bound_http_caller_not_task_body_or_host_principal() -> None:
    governance = build_strict_production_serving_host_task_execution_governance(
        binding=_BINDING,
    )
    with bind_strict_serving_http_caller_principal(_HTTP_CALLER):
        admitted = governance.admit_root_governance_identity(_task())
    assert admitted.principal_id == _HTTP_CALLER.user_id
    assert admitted.principal_id != _task().user_id
    assert admitted.principal_id != _HOST_SERVICE_PRINCIPAL.user_id
    assert admitted.tenant_id == _BINDING.authoritative_execution_tenant_id


def test_strict_governance_missing_caller_principal_fails_closed() -> None:
    governance = build_strict_production_serving_host_task_execution_governance(
        binding=_BINDING,
    )
    with pytest.raises(ValueError, match="authenticated caller principal"):
        governance.admit_root_governance_identity(_task())


def test_bind_strict_serving_http_caller_principal_resets_after_success() -> None:
    with bind_strict_serving_http_caller_principal(_HTTP_CALLER):
        assert get_strict_serving_http_caller_principal().user_id == _HTTP_CALLER.user_id
    with pytest.raises(ValueError, match="authenticated caller principal"):
        get_strict_serving_http_caller_principal()


def test_bind_strict_serving_http_caller_principal_resets_after_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with bind_strict_serving_http_caller_principal(_HTTP_CALLER):
            raise RuntimeError("boom")
    with pytest.raises(ValueError, match="authenticated caller principal"):
        get_strict_serving_http_caller_principal()


def test_caller_principal_contextvar_isolated_across_concurrent_requests() -> None:
    def _read_principal_after_delay(user_id: str) -> str:
        principal = RequestIdentity(
            tenant_id="env-prod",
            user_id=user_id,
            principal_type=PrincipalType.USER,
            auth_subject=user_id,
        )
        with bind_strict_serving_http_caller_principal(principal):
            time.sleep(0.02)
            return get_strict_serving_http_caller_principal().user_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(_read_principal_after_delay, "caller-left")
        right = pool.submit(_read_principal_after_delay, "caller-right")
        assert left.result() == "caller-left"
        assert right.result() == "caller-right"
