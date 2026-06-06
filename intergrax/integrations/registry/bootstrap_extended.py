# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Extended integration catalog beyond lab core (Phase P-Ext.4)."""

from __future__ import annotations


def register_extended_integrations(*, override: bool = False) -> None:
    """Register all shipped providers not included in ``bootstrap_core``."""
    from intergrax.integrations.providers.key_value_cache.redis.register import register_redis_integration
    from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
    from intergrax.integrations.providers.message_bus.kafka.register import register_kafka_integration
    from intergrax.integrations.providers.message_bus.celery.register import register_celery_integration
    from intergrax.integrations.providers.message_bus.rabbitmq.register import register_rabbitmq_integration
    from intergrax.integrations.providers.search_provider.google_cse.register import register_google_cse_integration
    from intergrax.integrations.providers.search_provider.bing.register import register_bing_integration
    from intergrax.integrations.providers.notification_channel.slack.register import register_slack_integration
    from intergrax.integrations.providers.notification_channel.teams.register import register_teams_integration
    from intergrax.integrations.providers.notification_channel.webhook.register import register_webhook_integration
    from intergrax.integrations.providers.interaction_surface.lab_json.register import register_lab_json_integration
    from intergrax.integrations.providers.interaction_surface.slash_command.register import register_slash_command_integration
    from intergrax.integrations.providers.notification_channel.log.register import register_log_integration
    from intergrax.integrations.providers.relational_store.postgresql.register import register_postgresql_integration
    from intergrax.integrations.providers.relational_store.mysql.register import register_mysql_integration
    from intergrax.integrations.providers.issue_tracker.jira.register import register_jira_integration
    from intergrax.integrations.providers.wiki_knowledge.confluence.register import register_confluence_integration
    from intergrax.integrations.providers.observability_backend.prometheus.register import register_prometheus_integration
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.register import register_ms365_graph_integration
    from intergrax.integrations.providers.document_store.cassandra.register import register_cassandra_integration
    from intergrax.integrations.providers.cloud_platform.aws.register import register_aws_integration
    from intergrax.integrations.providers.cloud_platform.azure.register import register_azure_integration
    from intergrax.integrations.providers.cloud_platform.gcp.register import register_gcp_integration
    from intergrax.integrations.providers.observability_backend.elasticsearch.register import register_elasticsearch_integration
    from intergrax.integrations.providers.relational_store.databricks.register import register_databricks_integration
    from intergrax.integrations.providers.document_store.mongodb.register import register_mongodb_integration
    from intergrax.integrations.providers.vector_store.pinecone.register import register_pinecone_integration
    from intergrax.integrations.providers.vector_store.qdrant.register import register_qdrant_integration
    from intergrax.integrations.providers.vector_store.chroma.register import register_chroma_integration
    from intergrax.integrations.providers.object_storage.s3.register import register_s3_integration
    from intergrax.integrations.providers.object_storage.azure_blob.register import register_azure_blob_integration
    from intergrax.integrations.providers.object_storage.gcs.register import register_gcs_integration
    from intergrax.integrations.providers.document_store.dynamodb.register import register_dynamodb_integration
    from intergrax.integrations.providers.message_bus.sqs.register import register_sqs_integration
    from intergrax.integrations.providers.message_bus.service_bus.register import register_service_bus_integration
    from intergrax.integrations.providers.message_bus.pubsub.register import register_pubsub_integration
    from intergrax.integrations.providers.key_value_cache.memcached.register import register_memcached_integration
    from intergrax.integrations.providers.key_value_cache.elasticache.register import register_elasticache_integration
    from intergrax.integrations.providers.relational_store.oracle.register import register_oracle_integration
    from intergrax.integrations.providers.relational_store.mssql.register import register_mssql_integration
    from intergrax.integrations.providers.relational_store.azure_sql.register import register_azure_sql_integration
    from intergrax.integrations.providers.relational_store.cloud_sql.register import register_cloud_sql_integration
    from intergrax.integrations.providers.notification_channel.email_smtp.register import register_email_smtp_integration
    from intergrax.integrations.providers.observability_backend.otel.register import register_otel_integration
    from intergrax.integrations.providers.issue_tracker.github.register import register_github_integration
    from intergrax.integrations.providers.issue_tracker.linear.register import register_linear_integration
    from intergrax.integrations.providers.issue_tracker.azure_devops.register import register_azure_devops_integration
    from intergrax.integrations.providers.wiki_knowledge.notion.register import register_notion_integration
    from intergrax.integrations.providers.wiki_knowledge.sharepoint.register import register_sharepoint_integration
    from intergrax.integrations.providers.collaboration_suite.google_workspace.register import register_google_workspace_integration
    from intergrax.integrations.providers.search_provider.brave.register import register_brave_integration
    from intergrax.integrations.providers.search_provider.serpapi.register import register_serpapi_integration
    from intergrax.integrations.providers.browser_automation.playwright.register import register_playwright_integration
    from intergrax.integrations.providers.search_provider.tavily.register import register_tavily_integration
    from intergrax.integrations.providers.search_provider.exa.register import register_exa_integration
    from intergrax.integrations.providers.vector_store.weaviate.register import register_weaviate_integration
    from intergrax.integrations.providers.vector_store.milvus.register import register_milvus_integration
    from intergrax.integrations.providers.vector_store.inmemory.register import register_inmemory_integration
    from intergrax.integrations.providers.secrets_store.vault.register import register_vault_integration
    from intergrax.integrations.providers.observability_backend.langfuse.register import register_langfuse_integration
    from intergrax.integrations.providers.observability_backend.datadog.register import register_datadog_integration
    from intergrax.integrations.providers.observability_backend.clickhouse.register import register_clickhouse_integration
    from intergrax.integrations.providers.observability_backend.sentry.register import register_sentry_integration
    from intergrax.integrations.providers.message_bus.temporal.register import register_temporal_integration
    from intergrax.integrations.providers.message_bus.nats.register import register_nats_integration
    from intergrax.integrations.providers.graph_store.neo4j.register import register_neo4j_integration
    from intergrax.integrations.providers.relational_store.snowflake.register import register_snowflake_integration
    from intergrax.integrations.providers.relational_store.supabase.register import register_supabase_integration
    from intergrax.integrations.providers.object_storage.minio.register import register_minio_integration
    from intergrax.integrations.providers.object_storage.filesystem.register import register_filesystem_integration
    from intergrax.integrations.providers.notification_channel.discord.register import register_discord_integration
    from intergrax.integrations.providers.notification_channel.twilio.register import register_twilio_integration
    from intergrax.integrations.providers.browser_automation.firecrawl.register import register_firecrawl_integration
    from intergrax.integrations.providers.browser_automation.selenium.register import register_selenium_integration
    from intergrax.integrations.providers.document_parser.docling.register import register_docling_integration
    from intergrax.integrations.providers.document_parser.pymupdf.register import register_pymupdf_integration
    from intergrax.integrations.providers.document_parser.unstructured.register import register_unstructured_integration
    from intergrax.integrations.providers.document_parser.python_docx.register import register_python_docx_integration
    from intergrax.integrations.providers.document_parser.openpyxl.register import register_openpyxl_integration
    from intergrax.integrations.providers.document_parser.whisper.register import register_whisper_integration
    from intergrax.integrations.providers.document_parser.yt_dlp.register import register_yt_dlp_integration
    from intergrax.integrations.providers.rerank_provider.cohere_rerank.register import register_cohere_rerank_integration
    from intergrax.integrations.providers.rerank_provider.jina_rerank.register import register_jina_rerank_integration
    from intergrax.integrations.providers.search_provider.reddit.register import register_reddit_integration
    from intergrax.integrations.providers.search_provider.google_places.register import register_google_places_integration
    from intergrax.integrations.providers.observability_backend.langsmith.register import register_langsmith_integration
    from intergrax.integrations.providers.observability_backend.helicone.register import register_helicone_integration
    from intergrax.integrations.providers.observability_backend.posthog.register import register_posthog_integration
    from intergrax.integrations.providers.observability_backend.braintrust.register import register_braintrust_integration
    from intergrax.integrations.providers.observability_backend.signoz.register import register_signoz_integration
    from intergrax.integrations.providers.observability_backend.honeycomb.register import register_honeycomb_integration
    from intergrax.integrations.providers.observability_backend.arize.register import register_arize_integration
    from intergrax.integrations.providers.observability_backend.phoenix.register import register_phoenix_integration
    from intergrax.integrations.providers.observability_backend.wandb.register import register_wandb_integration
    from intergrax.integrations.providers.observability_backend.opensearch.register import register_opensearch_integration
    from intergrax.integrations.providers.notification_channel.pagerduty.register import register_pagerduty_integration
    from intergrax.integrations.providers.notification_channel.opsgenie.register import register_opsgenie_integration
    from intergrax.integrations.providers.issue_tracker.gitlab.register import register_gitlab_integration
    from intergrax.integrations.providers.vector_store.vespa.register import register_vespa_integration

    register_kafka_integration(override=override)
    register_celery_integration(override=override)
    register_rabbitmq_integration(override=override)
    register_teams_integration(override=override)
    register_slash_command_integration(override=override)
    register_postgresql_integration(override=override)
    register_mysql_integration(override=override)
    register_jira_integration(override=override)
    register_confluence_integration(override=override)
    register_ms365_graph_integration(override=override)
    register_cassandra_integration(override=override)
    register_aws_integration(override=override)
    register_azure_integration(override=override)
    register_gcp_integration(override=override)
    register_elasticsearch_integration(override=override)
    register_databricks_integration(override=override)
    register_mongodb_integration(override=override)
    register_pinecone_integration(override=override)
    register_chroma_integration(override=override)
    register_s3_integration(override=override)
    register_azure_blob_integration(override=override)
    register_gcs_integration(override=override)
    register_dynamodb_integration(override=override)
    register_sqs_integration(override=override)
    register_service_bus_integration(override=override)
    register_pubsub_integration(override=override)
    register_memcached_integration(override=override)
    register_elasticache_integration(override=override)
    register_oracle_integration(override=override)
    register_mssql_integration(override=override)
    register_azure_sql_integration(override=override)
    register_cloud_sql_integration(override=override)
    register_email_smtp_integration(override=override)
    register_github_integration(override=override)
    register_linear_integration(override=override)
    register_azure_devops_integration(override=override)
    register_notion_integration(override=override)
    register_sharepoint_integration(override=override)
    register_google_workspace_integration(override=override)
    register_brave_integration(override=override)
    register_serpapi_integration(override=override)
    register_playwright_integration(override=override)
    register_tavily_integration(override=override)
    register_exa_integration(override=override)
    register_weaviate_integration(override=override)
    register_milvus_integration(override=override)
    register_vault_integration(override=override)
    register_langfuse_integration(override=override)
    register_datadog_integration(override=override)
    register_clickhouse_integration(override=override)
    register_sentry_integration(override=override)
    register_temporal_integration(override=override)
    register_nats_integration(override=override)
    register_neo4j_integration(override=override)
    register_snowflake_integration(override=override)
    register_supabase_integration(override=override)
    register_minio_integration(override=override)
    register_filesystem_integration(override=override)
    register_discord_integration(override=override)
    register_twilio_integration(override=override)
    register_firecrawl_integration(override=override)
    register_selenium_integration(override=override)
    register_docling_integration(override=override)
    register_pymupdf_integration(override=override)
    register_unstructured_integration(override=override)
    register_python_docx_integration(override=override)
    register_openpyxl_integration(override=override)
    register_whisper_integration(override=override)
    register_yt_dlp_integration(override=override)
    register_cohere_rerank_integration(override=override)
    register_jina_rerank_integration(override=override)
    register_reddit_integration(override=override)
    register_google_places_integration(override=override)
    register_langsmith_integration(override=override)
    register_helicone_integration(override=override)
    register_posthog_integration(override=override)
    register_braintrust_integration(override=override)
    register_signoz_integration(override=override)
    register_honeycomb_integration(override=override)
    register_arize_integration(override=override)
    register_phoenix_integration(override=override)
    register_wandb_integration(override=override)
    register_opensearch_integration(override=override)
    register_pagerduty_integration(override=override)
    register_opsgenie_integration(override=override)
    register_gitlab_integration(override=override)
    register_vespa_integration(override=override)
    from intergrax.integrations.registry.bootstrap_m6_p4 import register_m6_p4_integrations

    register_m6_p4_integrations(override=override)
    from intergrax.integrations.registry.bootstrap_m6_p5 import register_m6_p5_integrations

    register_m6_p5_integrations(override=override)
