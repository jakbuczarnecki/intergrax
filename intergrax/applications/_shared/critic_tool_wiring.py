# © Artur Czarnecki. All rights reserved.

"""Build L1 critic tool client from Tier-3 tool wiring (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

from typing import Any

from intergrax.applications._shared.critic_llm_resolver import resolve_critic_llm_adapter
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.critic.tool_registry_client import ToolRegistryCriticEvalClient
from intergrax.tools.registry.wiring import ToolWiringContext


def _critic_layers_enabled(env: ApplicationEnvironmentProfile) -> bool:
    profile = env.critic_profile
    return profile.semantic_judge_enabled or profile.trajectory_eval_enabled


def build_critic_tool_wiring_context(
    env: ApplicationEnvironmentProfile,
    base_ctx: ToolWiringContext,
    *,
    evaluation_registry: Any | None = None,
    trace_reader: Any | None = None,
) -> ToolWiringContext:
    """Clone tool wiring context with critic-specific LLM and eval bindings."""
    producer = resolve_llm_adapter(env)
    critic_llm = resolve_critic_llm_adapter(env, producer_adapter=producer)
    extras = dict(base_ctx.extras)
    extras["llm_adapter"] = critic_llm
    return ToolWiringContext(
        issue_tracker=base_ctx.issue_tracker,
        search_provider=base_ctx.search_provider,
        wiki_knowledge=base_ctx.wiki_knowledge,
        notification_channel=base_ctx.notification_channel,
        observability_backend=base_ctx.observability_backend,
        observability_backends=dict(base_ctx.observability_backends),
        object_storage=base_ctx.object_storage,
        relational_store=base_ctx.relational_store,
        document_store=base_ctx.document_store,
        browser_automation=base_ctx.browser_automation,
        document_parser=base_ctx.document_parser,
        secrets_store=base_ctx.secrets_store,
        feature_flag_backend=base_ctx.feature_flag_backend,
        ci_cd_backend=base_ctx.ci_cd_backend,
        message_bus=base_ctx.message_bus,
        graph_store=base_ctx.graph_store,
        collaboration_suite=base_ctx.collaboration_suite,
        key_value_cache=base_ctx.key_value_cache,
        shadow_workspace=base_ctx.shadow_workspace,
        human_decision_store=base_ctx.human_decision_store,
        session_storage=base_ctx.session_storage,
        scheduled_notification_store=base_ctx.scheduled_notification_store,
        memory_view=base_ctx.memory_view,
        trace_reader=trace_reader or base_ctx.trace_reader,
        evaluation_registry=evaluation_registry or base_ctx.evaluation_registry,
        integration_profile=base_ctx.integration_profile,
        rag_manager=base_ctx.rag_manager,
        vectorstore_manager=base_ctx.vectorstore_manager,
        embedding_manager=base_ctx.embedding_manager,
        retriever_manager=base_ctx.retriever_manager,
        reranker_manager=base_ctx.reranker_manager,
        rag_profile=base_ctx.rag_profile,
        retrieval_service=base_ctx.retrieval_service,
        websearch_executor=base_ctx.websearch_executor,
        sandbox_session=base_ctx.sandbox_session,
        security_scanner=base_ctx.security_scanner,
        sandbox_host=base_ctx.sandbox_host,
        identity_provider=base_ctx.identity_provider,
        speech_provider=base_ctx.speech_provider,
        workflow_orchestrator=base_ctx.workflow_orchestrator,
        billing_meter=base_ctx.billing_meter,
        crm_backend=base_ctx.crm_backend,
        cloud_platform=base_ctx.cloud_platform,
        read_allowlist_roots=base_ctx.read_allowlist_roots,
        run_budget=base_ctx.run_budget,
        cost_envelopes=base_ctx.cost_envelopes,
        cost_quotas=base_ctx.cost_quotas,
        extras=extras,
    )


def build_critic_eval_tool_client(
    env: ApplicationEnvironmentProfile,
    tool_wiring: ApplicationToolWiring,
    *,
    evaluation_registry: Any | None = None,
    trace_reader: Any | None = None,
) -> CriticEvalToolClient | None:
    """Materialize L1 client when semantic or trajectory critics are enabled."""
    if not _critic_layers_enabled(env):
        return None
    ctx = build_critic_tool_wiring_context(
        env,
        tool_wiring.wiring_context,
        evaluation_registry=evaluation_registry,
        trace_reader=trace_reader,
    )
    return ToolRegistryCriticEvalClient(ctx)
