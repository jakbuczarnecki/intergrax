# © Artur Czarnecki. All rights reserved.

"""G-09 gate fixtures — tenant-scoped online evaluation registry handles."""

from __future__ import annotations

from contextvars import ContextVar

import pytest
from intergrax.runtime.architecture.online_evaluation_registry import (
    InMemoryOnlineEvaluationRegistry,
    TenantScopedOnlineEvaluationRegistry,
)

__all__ = [
    "CI_SMOKE_PRODUCT_TENANT",
    "df5_production_memory_tenant_binding",
    "tenant_scoped_online_evaluation_registry",
]

_SCENARIO_MEMORY_TENANT: ContextVar[str | None] = ContextVar(
    "g09_scenario_memory_tenant",
    default=None,
)

# Shared with DF5 destructive proof and harness architecture gates.
CI_SMOKE_PRODUCT_TENANT = "df5-destructive"


def tenant_scoped_online_evaluation_registry(
    tenant_id: str = CI_SMOKE_PRODUCT_TENANT,
) -> TenantScopedOnlineEvaluationRegistry:
    """In-memory registry view bound to one tenant (G-09 fail-closed contract)."""
    return TenantScopedOnlineEvaluationRegistry(
        InMemoryOnlineEvaluationRegistry(),
        tenant_id,
    )


@pytest.fixture
def df5_production_memory_tenant_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate scenario tenant_id into memory admission during STRICT composition."""
    from intergrax.applications._shared import environment_wiring as environment_wiring_module
    from intergrax.applications._shared import scenario_runtime_baseline as scenario_runtime_baseline_module

    original_wire = scenario_runtime_baseline_module.wire_application_environment
    original_resolve = environment_wiring_module.resolve_memory_platform_wiring

    def wire_application_environment(manifest, env, **kwargs):
        token = _SCENARIO_MEMORY_TENANT.set(kwargs.get("tenant_id"))
        try:
            return original_wire(manifest, env, **kwargs)
        finally:
            _SCENARIO_MEMORY_TENANT.reset(token)

    def resolve_memory_platform_wiring(
        env,
        *,
        integration_profile=None,
        tenant_id=None,
        **kwargs,
    ):
        bound_tenant = _SCENARIO_MEMORY_TENANT.get()
        effective_tenant = tenant_id if tenant_id is not None and str(tenant_id).strip() else bound_tenant
        return original_resolve(
            env,
            integration_profile=integration_profile,
            tenant_id=effective_tenant,
            **kwargs,
        )

    monkeypatch.setattr(
        scenario_runtime_baseline_module,
        "wire_application_environment",
        wire_application_environment,
    )
    monkeypatch.setattr(
        environment_wiring_module,
        "resolve_memory_platform_wiring",
        resolve_memory_platform_wiring,
    )
