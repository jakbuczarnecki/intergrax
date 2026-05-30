# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 tool catalog wiring helpers (Phase O.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry import ToolProfile, ToolRegistry, ToolWiringContext, build_registry_from_profile
from intergrax.tools.registry.bootstrap import register_default_tools


@dataclass(frozen=True)
class ApplicationToolWiring:
    """Resolved tool profile + wiring context + materialized registry."""

    profile: ToolProfile
    wiring_context: ToolWiringContext
    registry: ToolRegistry


def build_application_tool_wiring(
    profile: ToolProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
    wiring_context: ToolWiringContext | None = None,
    vectorstore_manager: Any | None = None,
    embedding_manager: Any | None = None,
    websearch_executor: Any | None = None,
    rag_manager: Any | None = None,
    sandbox_session: Any | None = None,
    extras: dict[str, Any] | None = None,
) -> ApplicationToolWiring:
    """
    Build a catalog registry for Tier-3 hosts (lab, product, MCP export).

    Call ``register_default_tools()`` once, compose ``ToolWiringContext`` from
    integrations + runtime managers, then enable tools via ``ToolProfile``.
    """
    register_default_tools()
    ctx = wiring_context
    if ctx is None and integration_profile is not None:
        ctx = ToolWiringContext.from_integration_profile(
            integration_profile,
            rag_manager=rag_manager,
            vectorstore_manager=vectorstore_manager,
            embedding_manager=embedding_manager,
            websearch_executor=websearch_executor,
            extras=extras,
        )
    if ctx is None:
        ctx = ToolWiringContext(extras=dict(extras or {}))

    ctx = ToolWiringContext(
        issue_tracker=ctx.issue_tracker,
        search_provider=ctx.search_provider,
        wiki_knowledge=ctx.wiki_knowledge,
        notification_channel=ctx.notification_channel,
        observability_backend=ctx.observability_backend,
        rag_manager=ctx.rag_manager or rag_manager,
        vectorstore_manager=ctx.vectorstore_manager or vectorstore_manager,
        embedding_manager=ctx.embedding_manager or embedding_manager,
        websearch_executor=ctx.websearch_executor or websearch_executor,
        sandbox_session=ctx.sandbox_session or sandbox_session,
        extras=dict(ctx.extras),
    )
    registry = build_registry_from_profile(profile, ctx=ctx)
    return ApplicationToolWiring(profile=profile, wiring_context=ctx, registry=registry)
