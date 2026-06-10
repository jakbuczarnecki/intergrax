# © Artur Czarnecki. All rights reserved.

"""Causal diagnostic chains beyond trace bridge (AUDIT-IDEAL-21.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.observability.trace_scope import TraceScopeState


class CausalDiagnosticLink(BaseModel):
    """Single hop in a causal diagnostic chain."""

    event_id: str
    parent_event_id: str | None = None
    component: str
    summary: str


class CausalDiagnosticChain(BaseModel):
    """Ops-facing causal chain correlated to an active trace scope."""

    schema_version: str = "1.0.0"
    run_id: str
    task_id: str
    tenant_id: str
    correlation_id: str
    links: list[CausalDiagnosticLink] = Field(default_factory=list)


def build_causal_diagnostic_chain(
    scope: TraceScopeState,
    *,
    links: list[CausalDiagnosticLink] | None = None,
) -> CausalDiagnosticChain:
    """Materialize a causal chain from the active trace scope and optional link events."""
    chain_links = list(links or [])
    if scope.parent_event_id is not None and not any(
        link.event_id == scope.parent_event_id for link in chain_links
    ):
        chain_links.insert(
            0,
            CausalDiagnosticLink(
                event_id=scope.parent_event_id,
                parent_event_id=None,
                component="trace_scope",
                summary="active parent event anchor",
            ),
        )
    return CausalDiagnosticChain(
        run_id=scope.run_id,
        task_id=scope.task_id,
        tenant_id=scope.tenant_id,
        correlation_id=scope.correlation_id,
        links=chain_links,
    )
