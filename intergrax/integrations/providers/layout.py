# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider filesystem layout — slug → primary category folder (Phase M.6).

Primary lookup remains ``SLUG_CATEGORY`` (one primary folder per slug).
Secondary category packages use ``SECONDARY_PROVIDER_CATEGORIES`` so the same
``provider_id`` can register under multiple categories without overwriting the
primary mapping (e.g. slack notification_channel + conversation_channel).
"""

from __future__ import annotations

# Primary IntegrationCategory folder for each catalog slug.
SLUG_CATEGORY: dict[str, str] = {
    "sqlite": "relational_store",
    "postgresql": "relational_store",
    "mysql": "relational_store",
    "databricks": "relational_store",
    "oracle": "relational_store",
    "mssql": "relational_store",
    "azure_sql": "relational_store",
    "cloud_sql": "relational_store",
    "cassandra": "document_store",
    "mongodb": "document_store",
    "dynamodb": "document_store",
    "redis": "key_value_cache",
    "memcached": "key_value_cache",
    "elasticache": "key_value_cache",
    "kafka": "message_bus",
    "celery": "message_bus",
    "rabbitmq": "message_bus",
    "sqs": "message_bus",
    "service_bus": "message_bus",
    "pubsub": "message_bus",
    "s3": "object_storage",
    "azure_blob": "object_storage",
    "gcs": "object_storage",
    "pinecone": "vector_store",
    "qdrant": "vector_store",
    "chroma": "vector_store",
    "google_cse": "search_provider",
    "bing": "search_provider",
    "brave": "search_provider",
    "serpapi": "search_provider",
    "slack": "notification_channel",
    "teams": "notification_channel",
    "webhook": "notification_channel",
    "log": "notification_channel",
    "email_smtp": "notification_channel",
    "ms365_graph": "collaboration_suite",
    "google_workspace": "collaboration_suite",
    "jira": "issue_tracker",
    "github": "issue_tracker",
    "linear": "issue_tracker",
    "azure_devops": "issue_tracker",
    "confluence": "wiki_knowledge",
    "notion": "wiki_knowledge",
    "sharepoint": "wiki_knowledge",
    "prometheus": "observability_backend",
    "elasticsearch": "observability_backend",
    "otel": "observability_backend",
    "playwright": "browser_automation",
    "aws": "cloud_platform",
    "azure": "cloud_platform",
    "gcp": "cloud_platform",
    "tavily": "search_provider",
    "exa": "search_provider",
    "weaviate": "vector_store",
    "milvus": "vector_store",
    "inmemory": "vector_store",
    "vault": "secrets_store",
    "langfuse": "observability_backend",
    "datadog": "observability_backend",
    "clickhouse": "observability_backend",
    "sentry": "observability_backend",
    "temporal": "message_bus",
    "nats": "message_bus",
    "neo4j": "graph_store",
    "snowflake": "relational_store",
    "supabase": "relational_store",
    "minio": "object_storage",
    "filesystem": "object_storage",
    "discord": "notification_channel",
    "twilio": "notification_channel",
    "firecrawl": "browser_automation",
    "selenium": "browser_automation",
    "docling": "document_parser",
    "pymupdf": "document_parser",
    "unstructured": "document_parser",
    "python_docx": "document_parser",
    "openpyxl": "document_parser",
    "whisper": "document_parser",
    "yt_dlp": "document_parser",
    "cohere_rerank": "rerank_provider",
    "jina_rerank": "rerank_provider",
    "langsmith": "observability_backend",
    "helicone": "observability_backend",
    "posthog": "observability_backend",
    "braintrust": "observability_backend",
    "signoz": "observability_backend",
    "honeycomb": "observability_backend",
    "arize": "observability_backend",
    "phoenix": "observability_backend",
    "wandb": "observability_backend",
    "opensearch": "observability_backend",
    "pagerduty": "notification_channel",
    "opsgenie": "notification_channel",
    "gitlab": "issue_tracker",
    "vespa": "vector_store",
    "reddit": "search_provider",
    "google_places": "search_provider",
    "pgvector": "vector_store",
    "duckdb": "relational_store",
    "influxdb": "observability_backend",
    "timescaledb": "relational_store",
    "grafana": "observability_backend",
    "loki": "observability_backend",
    "tempo": "observability_backend",
    "aws_secrets_manager": "secrets_store",
    "azure_key_vault": "secrets_store",
    "gcp_secret_manager": "secrets_store",
    "doppler": "secrets_store",
    "unleash": "feature_flag",
    "launchdarkly": "feature_flag",
    "github_actions": "ci_cd",
    "redpanda": "message_bus",
    "cloudflare_r2": "object_storage",
    "memgraph": "graph_store",
    "falkordb": "graph_store",
    "incident_io": "notification_channel",
    "kubernetes": "cloud_platform",
    "servicenow": "issue_tracker",
    "bitbucket": "issue_tracker",
    "asana": "issue_tracker",
    "sendgrid": "notification_channel",
    "mailgun": "notification_channel",
    "mlflow": "observability_backend",
    "huggingface_hub": "object_storage",
    "ollama": "model_serving_runtime",
    "gitlab_ci": "ci_cd",
    "circleci": "ci_cd",
    "azure_pipelines": "ci_cd",
    "mailpit": "notification_channel",
    "localstack": "cloud_platform",
    "codecov": "ci_cd",
    "grafana_oncall": "notification_channel",
    "opentelemetry_collector": "observability_backend",
    "trivy": "security_scanner",
    "snyk": "security_scanner",
    "semgrep": "security_scanner",
    "infisical": "secrets_store",
    "e2b": "sandbox_host",
    "modal": "sandbox_host",
    "daytona": "sandbox_host",
    "auth0": "identity_provider",
    "keycloak": "identity_provider",
    "workos": "identity_provider",
    "argocd": "ci_cd",
    "buildkite": "ci_cd",
    "jenkins": "ci_cd",
    "elevenlabs": "speech_provider",
    "deepgram": "speech_provider",
    "newrelic": "observability_backend",
    "splunk": "observability_backend",
    "zendesk": "issue_tracker",
    "statsig": "feature_flag",
    "prefect": "workflow_orchestrator",
    "airflow": "workflow_orchestrator",
    "typesense": "vector_store",
    "neon": "relational_store",
    "pulsar": "message_bus",
    "algolia": "search_provider",
    "confluent": "message_bus",
    "backblaze_b2": "object_storage",
    "triton": "vision_serving",
    "replicate": "ml_inference_host",
    "stripe": "billing_meter",
    "salesforce": "crm",
    "hubspot": "crm",
    "perplexity": "search_provider",
    "arxiv": "search_provider",
    "semantic_scholar": "search_provider",
    "llamaparse": "document_parser",
    "lancedb": "vector_store",
    "telegram": "notification_channel",
    "mattermost": "conversation_channel",
    "rocket_chat": "conversation_channel",
    "google_chat": "conversation_channel",
    "browserbase": "browser_automation",
    "google_drive": "object_storage",
    "n8n": "workflow_orchestrator",
    "wikipedia": "wiki_knowledge",
    "clerk": "identity_provider",
    "upstash_redis": "key_value_cache",
    "upstash_qstash": "message_bus",
    "okta": "identity_provider",
    "bigquery": "relational_store",
    "motherduck": "relational_store",
    "airbyte": "workflow_orchestrator",
    "apify": "browser_automation",
    "llm_guard": "llm_guardrail",
    "guardrails_ai": "llm_guardrail",
    "nemo_guardrails": "llm_guardrail",
    "openguardrails": "llm_guardrail",
    "presidio": "llm_guardrail",
    "llama_guard": "llm_guardrail",
    "lakera": "llm_guardrail",
    "azure_content_safety": "llm_guardrail",
    "bedrock_guardrails": "llm_guardrail",
    "openai": "managed_retrieval",
    "hf": "embedding_provider",
    "vllm": "embedding_provider",
    "llama_cpp": "embedding_provider",
}

# Extra (provider_id, category) memberships beyond the primary SLUG_CATEGORY entry.
# Do not store lists in SLUG_CATEGORY — many consumers expect a single string.
SECONDARY_PROVIDER_CATEGORIES: dict[str, tuple[str, ...]] = {
    "slack": ("conversation_channel",),
    "teams": ("conversation_channel",),
    "discord": ("conversation_channel",),
    "telegram": ("conversation_channel",),
    "openai": ("embedding_provider",),
    "ollama": ("embedding_provider",),
}


def categories_for_provider(slug: str) -> tuple[str, ...]:
    """Return primary category first, then secondary memberships."""
    primary = SLUG_CATEGORY[slug]
    secondary = SECONDARY_PROVIDER_CATEGORIES.get(slug, ())
    return (primary, *secondary)


def provider_category_keys() -> tuple[tuple[str, str], ...]:
    """All valid ``(provider_id, category)`` identities from taxonomy."""
    keys: list[tuple[str, str]] = []
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            keys.append((slug, category))
    return tuple(keys)


def provider_import_path(slug: str, category: str | None = None) -> str:
    """Dotted import path for a provider package, e.g. ``...providers.object_storage.s3``."""
    resolved = category or SLUG_CATEGORY[slug]
    return f"intergrax.integrations.providers.{resolved}.{slug}"


def provider_package_path(slug: str, category: str | None = None) -> str:
    """Repo-relative path to provider directory."""
    resolved = category or SLUG_CATEGORY[slug]
    return f"intergrax/integrations/providers/{resolved}/{slug}"


__all__ = [
    "SECONDARY_PROVIDER_CATEGORIES",
    "SLUG_CATEGORY",
    "categories_for_provider",
    "provider_category_keys",
    "provider_import_path",
    "provider_package_path",
]
