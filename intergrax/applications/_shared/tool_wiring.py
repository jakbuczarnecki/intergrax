# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 tool catalog wiring helpers (Phase O.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry import ToolProfile, ToolRegistry, ToolWiringContext, build_registry_from_profile


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
    retriever_manager: Any | None = None,
    reranker_manager: Any | None = None,
    rag_profile: Any | None = None,
    retrieval_service: Any | None = None,
    toc_vectorstore_manager: Any | None = None,
    sandbox_session: Any | None = None,
    security_profile: Any | None = None,
    extras: dict[str, Any] | None = None,
) -> ApplicationToolWiring:
    """
    Build a catalog registry for Tier-3 hosts (lab, product, MCP export).

    Call ``bootstrap_catalogs()`` once (Tier-0 integrations, tools, skills), compose
    ``ToolWiringContext`` from integrations + runtime managers, then enable tools via
    ``ToolProfile``.
    """
    tool_bundle_ids = tuple(profile.enabled_bundles) if profile.enabled_bundles else None
    bootstrap_catalogs(
        register_shipped=True,
        tool_bundle_ids=tool_bundle_ids,
        discover_entry_points=discover_plugins_enabled(),
    )
    ctx = wiring_context
    if ctx is None and integration_profile is not None:
        ctx = ToolWiringContext.from_integration_profile(
            integration_profile,
            rag_manager=rag_manager,
            vectorstore_manager=vectorstore_manager,
            embedding_manager=embedding_manager,
            retriever_manager=retriever_manager,
            reranker_manager=reranker_manager,
            rag_profile=rag_profile,
            retrieval_service=retrieval_service,
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
        observability_backends=dict(ctx.observability_backends),
        object_storage=ctx.object_storage,
        relational_store=ctx.relational_store,
        document_store=ctx.document_store,
        browser_automation=ctx.browser_automation,
        document_parser=ctx.document_parser,
        secrets_store=ctx.secrets_store,
        feature_flag_backend=ctx.feature_flag_backend,
        ci_cd_backend=ctx.ci_cd_backend,
        message_bus=ctx.message_bus,
        graph_store=ctx.graph_store,
        collaboration_suite=ctx.collaboration_suite,
        key_value_cache=ctx.key_value_cache,
        shadow_workspace=ctx.shadow_workspace,
        human_decision_store=ctx.human_decision_store,
        session_storage=ctx.session_storage,
        scheduled_notification_store=ctx.scheduled_notification_store,
        memory_view=ctx.memory_view,
        trace_reader=ctx.trace_reader,
        evaluation_registry=ctx.evaluation_registry,
        integration_profile=ctx.integration_profile or integration_profile,
        rag_manager=ctx.rag_manager or rag_manager,
        vectorstore_manager=ctx.vectorstore_manager or vectorstore_manager,
        embedding_manager=ctx.embedding_manager or embedding_manager,
        retriever_manager=ctx.retriever_manager or retriever_manager,
        reranker_manager=ctx.reranker_manager or reranker_manager,
        rag_profile=ctx.rag_profile or rag_profile,
        retrieval_service=ctx.retrieval_service or retrieval_service,
        toc_vectorstore_manager=toc_vectorstore_manager or ctx.toc_vectorstore_manager,
        security_profile=security_profile or ctx.security_profile,
        websearch_executor=ctx.websearch_executor or websearch_executor,
        sandbox_session=ctx.sandbox_session or sandbox_session,
        security_scanner=ctx.security_scanner,
        sandbox_host=ctx.sandbox_host,
        identity_provider=ctx.identity_provider,
        speech_provider=ctx.speech_provider,
        workflow_orchestrator=ctx.workflow_orchestrator,
        billing_meter=ctx.billing_meter,
        crm_backend=ctx.crm_backend,
        read_allowlist_roots=ctx.read_allowlist_roots,
        run_budget=ctx.run_budget,
        cost_envelopes=ctx.cost_envelopes,
        cost_quotas=ctx.cost_quotas,
        extras=dict(ctx.extras),
    )
    registry = build_registry_from_profile(profile, ctx=ctx)
    return ApplicationToolWiring(profile=profile, wiring_context=ctx, registry=registry)
