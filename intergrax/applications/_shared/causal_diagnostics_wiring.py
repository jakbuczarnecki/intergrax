# © Artur Czarnecki. All rights reserved.

"""Causal diagnostics wiring for product ops surfaces (AUDIT-IDEAL-21.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.observability.causal_diagnostics import (
    CausalDiagnosticChain,
    CausalDiagnosticLink,
    build_causal_diagnostic_chain,
)
from intergrax.runtime.observability.trace_scope import TraceScopeState


@dataclass(frozen=True, slots=True)
class CausalDiagnosticsWiring:
    enabled: bool
    chain: CausalDiagnosticChain | None


def resolve_causal_diagnostics_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    scope: TraceScopeState | None = None,
) -> CausalDiagnosticsWiring:
    """Expose causal diagnostic chains when observability profile enables ops tooling."""
    obs = env.observability_profile
    if not obs.causal_diagnostics_enabled:
        return CausalDiagnosticsWiring(enabled=False, chain=None)
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return CausalDiagnosticsWiring(enabled=False, chain=None)

    active_scope = scope or TraceScopeState(
        run_id="bootstrap",
        task_id="bootstrap",
        tenant_id=env.profile_id,
        correlation_id=f"{env.profile_id}:causal",
        parent_event_id="bootstrap.parent",
    )
    chain = build_causal_diagnostic_chain(
        active_scope,
        links=[
            CausalDiagnosticLink(
                event_id="bootstrap.parent",
                parent_event_id=None,
                component="hos",
                summary="product host causal diagnostics ready",
            )
        ],
    )
    return CausalDiagnosticsWiring(enabled=True, chain=chain)
