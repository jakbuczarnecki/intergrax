# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Single validation boundary for admitted scoped adaptive integration effect requests."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationEffectRequest,
)


def _network_subset(
    admitted: NetworkEgressAllowlist,
    candidate: NetworkEgressAllowlist,
) -> bool:
    admitted_hosts = {host.canonical_form() for host in admitted.hosts}
    return all(host.canonical_form() in admitted_hosts for host in candidate.hosts)


def validate_admitted_scoped_adaptive_integration_effect_request(
    *,
    effect_request: ScopedAdaptedIntegrationEffectRequest,
    handoff: ScopedAdaptiveIntegrationExecutionHandoff,
    execution_id: ExecutionId,
    tenant_id: str,
    admitted_network_allowlist: NetworkEgressAllowlist,
) -> str | None:
    """Return error detail when invalid; None when fully admitted (no downstream widening)."""
    if effect_request.tenant_id != tenant_id:
        return "effect request tenant mismatch"
    if effect_request.tenant_id != handoff.tenant_id:
        return "effect request handoff tenant mismatch"
    if effect_request.artifact_id != handoff.artifact.artifact_id:
        return "effect request artifact_id mismatch"
    if effect_request.artifact_fingerprint != handoff.artifact.artifact_fingerprint:
        return "effect request artifact_fingerprint mismatch"
    if effect_request.provider_id != handoff.provider_id:
        return "effect request provider_id mismatch"
    if effect_request.resource_scope != handoff.resource_scope:
        return "effect request resource_scope mismatch"
    if effect_request.integration_category != handoff.integration_category:
        return "effect request integration_category mismatch"
    if effect_request.requested_operation != handoff.requested_operation:
        return "effect request operation mismatch"
    if effect_request.requested_operation not in handoff.permitted_operations:
        return "effect request operation not permitted"
    if effect_request.execution_id != str(execution_id):
        return "effect request execution_id mismatch"
    if not _network_subset(admitted_network_allowlist, effect_request.network_allowlist):
        return "effect request network_allowlist widening"
    if not _network_subset(admitted_network_allowlist, effect_request.target_scope):
        return "effect request target_scope widening"
    art_scope = handoff.artifact.scope
    if effect_request.requested_operation not in art_scope.permitted_operations:
        return "effect request operation outside artifact scope"
    if not _network_subset(art_scope.network_allowlist, effect_request.network_allowlist):
        return "effect request network outside artifact scope"
    return None


__all__ = ["validate_admitted_scoped_adaptive_integration_effect_request"]
