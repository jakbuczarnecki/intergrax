# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict production serving HTTP execution tenant authority (PROD-Q-R1-R7-Q-P3-A1).

Semantic ownership: ``assert_task_tenant_matches_strict_production_serving_authority`` is the
single tenant-authorization decision for HTTP/task execution. It is enforced at
``StrictProductionServingTenantAuthorityTaskExecutionPort`` before colocated physical admission.
``admit_strict_production_serving_root_governance_identity`` maps an already-authorized binding
and trusted per-request HTTP ``RequestIdentity`` (P3-A2) to admitted root identity only.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.serving_runtime_handoff import ServingRuntimeHandoff
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class StrictProductionServingExecutionTenantBinding:
    """Trusted execution-tenant binding for one opened strict serving session."""

    authoritative_execution_tenant_id: str
    application_id: str
    application_environment_id: str
    control_plane_environment_tenant_id: str


def resolve_strict_production_serving_execution_tenant_binding(
    *,
    handoff: ServingRuntimeHandoff,
    governance: ProductionControlPlaneGovernance,
) -> StrictProductionServingExecutionTenantBinding:
    """Derive HTTP execution tenant authority from handoff identity and control-plane governance."""
    identity = handoff.identity
    resolver = governance.environment_tenant_resolver
    if resolver is None:
        raise ValueError(
            "strict production serving requires ProductionControlPlaneGovernance."
            "environment_tenant_resolver",
        )
    environment_tenant = resolver.resolve_tenant_id(
        application_id=identity.application_id,
        application_environment_id=identity.application_environment_id,
    ).strip()
    if not environment_tenant:
        raise ValueError("strict production serving environment tenant authority missing")
    execution_tenant = identity.application_environment_id.strip()
    if not execution_tenant:
        raise ValueError("strict production serving handoff lacks application_environment_id")
    return StrictProductionServingExecutionTenantBinding(
        authoritative_execution_tenant_id=execution_tenant,
        application_id=identity.application_id,
        application_environment_id=identity.application_environment_id,
        control_plane_environment_tenant_id=environment_tenant,
    )


def assert_task_tenant_matches_strict_production_serving_authority(
    task: Task,
    *,
    binding: StrictProductionServingExecutionTenantBinding,
) -> None:
    """Fail closed when HTTP/task tenant does not match authoritative serving binding."""
    request_tenant = task.tenant_id.strip()
    if not request_tenant:
        raise ValueError("strict production serving requires task.tenant_id")
    if request_tenant != binding.authoritative_execution_tenant_id:
        raise ValueError("strict production serving tenant authority mismatch")


def admit_strict_production_serving_root_governance_identity(
    task: Task,
    *,
    binding: StrictProductionServingExecutionTenantBinding,
    principal: RequestIdentity,
) -> AdmittedRootGovernanceIdentity:
    """Map trusted serving authority to admitted root governance identity (not harness/lab).

    Caller must have enforced task tenant via ``StrictProductionServingTenantAuthorityTaskExecutionPort``
    (or an equivalent call to ``assert_task_tenant_matches_strict_production_serving_authority``).
    """
    trusted_principal = (principal.user_id or principal.auth_subject or "").strip()
    if not trusted_principal:
        raise ValueError("strict production serving requires governance principal identity")
    workspace = (
        f"strict-serving:{binding.application_id}:{binding.application_environment_id}"
    )
    return AdmittedRootGovernanceIdentity(
        tenant_id=binding.authoritative_execution_tenant_id,
        workspace_id=workspace,
        principal_id=trusted_principal,
    )


__all__ = [
    "StrictProductionServingExecutionTenantBinding",
    "admit_strict_production_serving_root_governance_identity",
    "assert_task_tenant_matches_strict_production_serving_authority",
    "resolve_strict_production_serving_execution_tenant_binding",
]
