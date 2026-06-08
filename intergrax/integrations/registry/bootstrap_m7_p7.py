# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Phase M.7 P7 integration slugs (18 agent-developer providers)."""

from __future__ import annotations


def register_m7_p7_integrations(*, override: bool = False) -> None:
    from intergrax.integrations.providers.search_provider.perplexity.register import register_perplexity_integration
    from intergrax.integrations.providers.search_provider.arxiv.register import register_arxiv_integration
    from intergrax.integrations.providers.search_provider.semantic_scholar.register import register_semantic_scholar_integration
    from intergrax.integrations.providers.document_parser.llamaparse.register import register_llamaparse_integration
    from intergrax.integrations.providers.vector_store.lancedb.register import register_lancedb_integration
    from intergrax.integrations.providers.notification_channel.telegram.register import register_telegram_integration
    from intergrax.integrations.providers.browser_automation.browserbase.register import register_browserbase_integration
    from intergrax.integrations.providers.object_storage.google_drive.register import register_google_drive_integration
    from intergrax.integrations.providers.workflow_orchestrator.n8n.register import register_n8n_integration
    from intergrax.integrations.providers.wiki_knowledge.wikipedia.register import register_wikipedia_integration
    from intergrax.integrations.providers.identity_provider.clerk.register import register_clerk_integration
    from intergrax.integrations.providers.key_value_cache.upstash_redis.register import register_upstash_redis_integration
    from intergrax.integrations.providers.message_bus.upstash_qstash.register import register_upstash_qstash_integration
    from intergrax.integrations.providers.identity_provider.okta.register import register_okta_integration
    from intergrax.integrations.providers.relational_store.bigquery.register import register_bigquery_integration
    from intergrax.integrations.providers.relational_store.motherduck.register import register_motherduck_integration
    from intergrax.integrations.providers.workflow_orchestrator.airbyte.register import register_airbyte_integration
    from intergrax.integrations.providers.browser_automation.apify.register import register_apify_integration

    register_perplexity_integration(override=override)
    register_arxiv_integration(override=override)
    register_semantic_scholar_integration(override=override)
    register_llamaparse_integration(override=override)
    register_lancedb_integration(override=override)
    register_telegram_integration(override=override)
    register_browserbase_integration(override=override)
    register_google_drive_integration(override=override)
    register_n8n_integration(override=override)
    register_wikipedia_integration(override=override)
    register_clerk_integration(override=override)
    register_upstash_redis_integration(override=override)
    register_upstash_qstash_integration(override=override)
    register_okta_integration(override=override)
    register_bigquery_integration(override=override)
    register_motherduck_integration(override=override)
    register_airbyte_integration(override=override)
    register_apify_integration(override=override)
