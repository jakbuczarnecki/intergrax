# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Control-plane mutation helpers for Agent Distribution activation/rollback (CLA-CPM)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

AGENT_DISTRIBUTION_RESOURCE_TYPE = "agent_distribution.application_environment"
MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION = "agent_distribution.activate_runtime_revision"
MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION = "agent_distribution.rollback_runtime_revision"


class ApplicationEnvironmentTenantResolver(Protocol):
    """Resolve tenant authority scope for one application environment."""

    def resolve_tenant_id(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> str:
        """Return canonical tenant id owning ``application_id`` / ``application_environment_id``."""


def application_environment_resource_id(
    *,
    application_id: str,
    application_environment_id: str,
) -> str:
    return f"{application_id}:{application_environment_id}"


def application_environment_resource_scope(
    *,
    application_id: str,
    application_environment_id: str,
) -> str:
    return (
        f"agent_distribution.application:{application_id}"
        f".environment:{application_environment_id}"
    )


def serving_revision_token(
    *,
    traffic_revision_id: str | None,
    serving_pointer_revision: int,
) -> str:
    revision_token = traffic_revision_id or "__none__"
    return f"rev:{revision_token}|ptr:{serving_pointer_revision}"


def build_activation_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    current_traffic_revision_id: str | None,
    current_serving_pointer_revision: int,
    target_runtime_revision_id: str,
) -> ControlPlaneMutationRequest:
    expected_pointer = current_serving_pointer_revision + 1
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
        principal=principal,
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
            traffic_revision_id=current_traffic_revision_id,
            serving_pointer_revision=current_serving_pointer_revision,
        ),
        target_revision=serving_revision_token(
            traffic_revision_id=target_runtime_revision_id,
            serving_pointer_revision=expected_pointer,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def build_rollback_mutation_request(
    *,
    principal: RequestIdentity,
    application_id: str,
    application_environment_id: str,
    mutation_id: str,
    current_traffic_revision_id: str,
    current_serving_pointer_revision: int,
    target_runtime_revision_id: str,
) -> ControlPlaneMutationRequest:
    expected_pointer = current_serving_pointer_revision + 1
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION,
        principal=principal,
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
            traffic_revision_id=current_traffic_revision_id,
            serving_pointer_revision=current_serving_pointer_revision,
        ),
        target_revision=serving_revision_token(
            traffic_revision_id=target_runtime_revision_id,
            serving_pointer_revision=expected_pointer,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


class TenantScopedControlPlaneMutationEvaluator:
    """Fail closed when principal tenant does not own the environment scope."""

    def __init__(
        self,
        *,
        tenant_resolver: ApplicationEnvironmentTenantResolver | None = None,
        inner: object | None = None,
    ) -> None:
        self._tenant_resolver = tenant_resolver
        self._inner = inner

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        if self._tenant_resolver is not None:
            application_id, application_environment_id = _parse_resource_id(request.resource_id)
            environment_tenant = self._tenant_resolver.resolve_tenant_id(
                application_id=application_id,
                application_environment_id=application_environment_id,
            )
            if environment_tenant != request.principal.tenant_id:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="tenant_authority_mismatch",
                    policy_rule_id="agent_distribution.tenant_scope",
                )
        if self._inner is not None and hasattr(self._inner, "evaluate"):
            return self._inner.evaluate(request)
        return PolicyDecision(action=PolicyAction.ALLOW, reason="tenant_scope_ok")


def _parse_resource_id(resource_id: str) -> tuple[str, str]:
    application_id, separator, application_environment_id = resource_id.partition(":")
    if not separator:
        raise ValueError("invalid agent distribution resource_id")
    return application_id, application_environment_id
