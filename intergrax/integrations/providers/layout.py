# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider filesystem layout — slug → primary category folder (Phase M.6)."""

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
    "lab_json": "interaction_surface",
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
}


def provider_import_path(slug: str) -> str:
    """Dotted import path for a provider package, e.g. ``...providers.object_storage.s3``."""
    category = SLUG_CATEGORY[slug]
    return f"intergrax.integrations.providers.{category}.{slug}"


def provider_package_path(slug: str) -> str:
    """Repo-relative path to provider directory."""
    category = SLUG_CATEGORY[slug]
    return f"intergrax/integrations/providers/{category}/{slug}"


__all__ = ["SLUG_CATEGORY", "provider_import_path", "provider_package_path"]
