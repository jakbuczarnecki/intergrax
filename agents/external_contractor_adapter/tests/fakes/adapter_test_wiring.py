# © Artur Czarnecki. All rights reserved.

"""Shared External Work adapter wiring for unit tests (PG-FIX-A)."""

from __future__ import annotations

from external_contractor_adapter.external_work_adapter import (
    META_WORKSPACE_REF,
    ExternalWorkAdapter,
)
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from external_contractor_adapter.tests.fakes.external_work_authorization_boundary import (
    allow_external_work_boundary,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)

_DEFAULT_TENANT = "tenant-a"
_DEFAULT_WORKSPACE = "workspace-a"
_DEFAULT_PRINCIPAL = "u1"


def default_workspace_meta() -> dict[str, str]:
    return {META_WORKSPACE_REF: _DEFAULT_WORKSPACE}


def allow_adapter(
    integration: ExternalWorkIntegration,
    *,
    policy: DeterministicMeaningfulSideEffectPolicy | None = None,
    principal_id: str = _DEFAULT_PRINCIPAL,
    tenant_id: str = _DEFAULT_TENANT,
    workspace_id: str = _DEFAULT_WORKSPACE,
    authorization_boundary: MeaningfulSideEffectAuthorizationBoundary | None = None,
) -> tuple[ExternalWorkAdapter, DeterministicMeaningfulSideEffectPolicy | None]:
    """Return adapter + optional runtime policy fake wired through canonical boundary."""
    runtime = policy
    if authorization_boundary is None:
        runtime = runtime or DeterministicMeaningfulSideEffectPolicy(
            default=PolicyAction.ALLOW
        )
        authorization_boundary = allow_external_work_boundary(
            runtime_policy_evaluator=runtime,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
        )
    return ExternalWorkAdapter(
        integration,
        authorization_boundary=authorization_boundary,
    ), runtime
