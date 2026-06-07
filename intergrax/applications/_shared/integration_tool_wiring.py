# © Artur Czarnecki. All rights reserved.

"""Wire Phase M.6 P6 integration categories into ``ToolWiringContext``."""

from __future__ import annotations

from intergrax.integrations._shared.speech_integration_bridge import (
    IntegrationSpeechAdapter,
    speech_provider_for_slug,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.speech_adapters.registry.profile import SPEECH_PROFILE_EXTRA_KEY
from intergrax.tools.providers.speech.backends import SPEECH_BACKEND_EXTRA_KEY
from intergrax.tools.registry.wiring import ToolWiringContext


def wire_integration_tool_context(
    ctx: ToolWiringContext,
    integration_profile: IntegrationProfile,
) -> ToolWiringContext:
    """
    Resolve P6 integration categories and populate typed tool wiring slots.

    Reuses ``ToolWiringContext.from_integration_profile`` outputs when present.
    """
    updated = ToolWiringContext(
        issue_tracker=ctx.issue_tracker,
        search_provider=ctx.search_provider,
        wiki_knowledge=ctx.wiki_knowledge,
        notification_channel=ctx.notification_channel,
        observability_backend=ctx.observability_backend,
        observability_backends=dict(ctx.observability_backends),
        object_storage=ctx.object_storage or _resolve_optional(integration_profile, IntegrationCategory.OBJECT_STORAGE),
        relational_store=ctx.relational_store
        or _resolve_optional(integration_profile, IntegrationCategory.RELATIONAL_STORE),
        document_store=ctx.document_store
        or _resolve_optional(integration_profile, IntegrationCategory.DOCUMENT_STORE),
        browser_automation=ctx.browser_automation
        or _resolve_optional(integration_profile, IntegrationCategory.BROWSER_AUTOMATION),
        document_parser=ctx.document_parser or _resolve_optional(integration_profile, IntegrationCategory.DOCUMENT_PARSER),
        secrets_store=ctx.secrets_store or _resolve_optional(integration_profile, IntegrationCategory.SECRETS_STORE),
        feature_flag_backend=ctx.feature_flag_backend
        or _resolve_optional(integration_profile, IntegrationCategory.FEATURE_FLAG),
        ci_cd_backend=ctx.ci_cd_backend or _resolve_optional(integration_profile, IntegrationCategory.CI_CD),
        message_bus=ctx.message_bus or _resolve_optional(integration_profile, IntegrationCategory.MESSAGE_BUS),
        graph_store=ctx.graph_store or _resolve_optional(integration_profile, IntegrationCategory.GRAPH_STORE),
        collaboration_suite=ctx.collaboration_suite
        or _resolve_optional(integration_profile, IntegrationCategory.COLLABORATION_SUITE),
        key_value_cache=ctx.key_value_cache or _resolve_optional(integration_profile, IntegrationCategory.KEY_VALUE_CACHE),
        shadow_workspace=ctx.shadow_workspace,
        memory_view=ctx.memory_view,
        trace_reader=ctx.trace_reader,
        evaluation_registry=ctx.evaluation_registry,
        integration_profile=integration_profile,
        rag_manager=ctx.rag_manager,
        vectorstore_manager=ctx.vectorstore_manager,
        embedding_manager=ctx.embedding_manager,
        retriever_manager=ctx.retriever_manager,
        reranker_manager=ctx.reranker_manager,
        rag_profile=ctx.rag_profile,
        retrieval_service=ctx.retrieval_service,
        websearch_executor=ctx.websearch_executor,
        sandbox_session=ctx.sandbox_session,
        security_scanner=ctx.security_scanner or _resolve_optional(integration_profile, IntegrationCategory.SECURITY_SCANNER),
        sandbox_host=ctx.sandbox_host or _resolve_optional(integration_profile, IntegrationCategory.SANDBOX_HOST),
        identity_provider=ctx.identity_provider or _resolve_optional(integration_profile, IntegrationCategory.IDENTITY_PROVIDER),
        speech_provider=ctx.speech_provider or _resolve_optional(integration_profile, IntegrationCategory.SPEECH_PROVIDER),
        workflow_orchestrator=ctx.workflow_orchestrator
        or _resolve_optional(integration_profile, IntegrationCategory.WORKFLOW_ORCHESTRATOR),
        billing_meter=ctx.billing_meter or _resolve_optional(integration_profile, IntegrationCategory.BILLING_METER),
        crm_backend=ctx.crm_backend or _resolve_optional(integration_profile, IntegrationCategory.CRM),
        read_allowlist_roots=ctx.read_allowlist_roots,
        run_budget=ctx.run_budget,
        cost_envelopes=ctx.cost_envelopes,
        cost_quotas=ctx.cost_quotas,
        extras=dict(ctx.extras),
    )

    speech_slug = integration_profile.slug_for_category(IntegrationCategory.SPEECH_PROVIDER)
    if updated.speech_provider is not None:
        provider_slug = speech_slug or "stub"
        updated.extras[SPEECH_BACKEND_EXTRA_KEY] = IntegrationSpeechAdapter(
            updated.speech_provider,
            provider=speech_provider_for_slug(provider_slug),
        )
        updated.extras.pop(SPEECH_PROFILE_EXTRA_KEY, None)

    return updated


def _resolve_optional(profile: IntegrationProfile, category: IntegrationCategory) -> object | None:
    instance = profile.instance_for_category(category)
    if instance is not None:
        return instance
    slug = profile.slug_for_category(category)
    if slug is None:
        return None
    try:
        return profile.resolve(category)
    except Exception:
        return None
