# © Artur Czarnecki. All rights reserved.

"""Wire scheduled notification catalog tools when a notification channel is configured."""

from __future__ import annotations

from intergrax.tools.registry.scheduled_notification_binding import (
    InMemoryScheduledNotificationStore,
    scheduled_notification_binding,
)
from intergrax.tools.registry.wiring import ToolWiringContext


def wire_scheduled_notification_tool_binding(ctx: ToolWiringContext) -> ToolWiringContext:
    """Attach in-memory schedule store when notify tools are usable."""
    if ctx.notification_channel is None:
        return ctx
    binding = scheduled_notification_binding(ctx.scheduled_notification_store)
    if binding is None:
        binding = InMemoryScheduledNotificationStore()
    if ctx.scheduled_notification_store is binding:
        return ctx
    return ToolWiringContext(
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
        scheduled_notification_store=binding,
        memory_view=ctx.memory_view,
        trace_reader=ctx.trace_reader,
        evaluation_registry=ctx.evaluation_registry,
        integration_profile=ctx.integration_profile,
        rag_manager=ctx.rag_manager,
        vectorstore_manager=ctx.vectorstore_manager,
        embedding_manager=ctx.embedding_manager,
        retriever_manager=ctx.retriever_manager,
        reranker_manager=ctx.reranker_manager,
        rag_profile=ctx.rag_profile,
        retrieval_service=ctx.retrieval_service,
        websearch_executor=ctx.websearch_executor,
        sandbox_session=ctx.sandbox_session,
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
