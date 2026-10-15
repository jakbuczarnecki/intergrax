# © Artur Czarnecki. All rights reserved.

"""Strict production HTTP ingress composition guards (P3-A2)."""

from __future__ import annotations

from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
    HarnessHostOrchestrationTopologyReliabilityCompositionError,
)
from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    StrictProductionServingExecutionTenantBinding,
)
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity


def validate_strict_production_serving_http_ingress_composition(
    *,
    binding: StrictProductionServingExecutionTenantBinding,
    api_keys_map: dict[str, ApiKeyIdentity] | None,
    include_task_control: bool,
    include_interaction_routes: bool,
    include_mcp: bool,
    identity_source: str,
) -> None:
    """Fail closed when strict serving session cannot enforce authenticated HTTP ingress."""
    if not api_keys_map:
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "strict governed contractor production serving requires API keys for HTTP ingress; "
            "configure GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_API_KEY or "
            "GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON",
        )
    if identity_source != "context_only":
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "strict governed contractor production serving requires identity_source=context_only",
        )
    if include_task_control or include_interaction_routes or include_mcp:
        raise HarnessHostOrchestrationTopologyReliabilityCompositionError(
            "strict governed contractor production serving rejects alternate execution-bearing "
            "HTTP surfaces (task control, interaction intake, MCP); disable them for this profile",
        )


__all__ = ["validate_strict_production_serving_http_ingress_composition"]
