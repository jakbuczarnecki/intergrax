# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed integration slugs — no magic strings in Tier-3 code (§7.1.4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Union

from intergrax.integrations.contracts.base import IntegrationCategory


class IntegrationSlug(StrEnum):
    """
    Canonical catalog slug identifiers.

    Extend this enum when registering a new provider in ``registry.catalog``.
    Env/YAML may still pass strings — they are coerced and validated here.
    """

    # relational_store
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    ORACLE = "oracle"
    MSSQL = "mssql"
    DATABRICKS = "databricks"
    AZURE_SQL = "azure_sql"
    CLOUD_SQL = "cloud_sql"
    SNOWFLAKE = "snowflake"
    SUPABASE = "supabase"

    # document_store
    MONGODB = "mongodb"
    CASSANDRA = "cassandra"
    DYNAMODB = "dynamodb"

    # key_value_cache
    REDIS = "redis"
    MEMCACHED = "memcached"
    ELASTICACHE = "elasticache"

    # message_bus
    KAFKA = "kafka"
    CELERY = "celery"
    RABBITMQ = "rabbitmq"
    SQS = "sqs"
    SERVICE_BUS = "service_bus"
    PUBSUB = "pubsub"
    TEMPORAL = "temporal"
    NATS = "nats"

    # object_storage
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    FILESYSTEM = "filesystem"
    MINIO = "minio"

    # vector_store (registry pointer → rag/)
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    INMEMORY = "inmemory"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"

    # search_provider
    GOOGLE_CSE = "google_cse"
    BING = "bing"
    REDDIT = "reddit"
    GOOGLE_PLACES = "google_places"
    BRAVE = "brave"
    SERPAPI = "serpapi"
    TAVILY = "tavily"
    EXA = "exa"

    # notification_channel
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL_SMTP = "email_smtp"
    WEBHOOK = "webhook"
    LOG = "log"
    DISCORD = "discord"
    TWILIO = "twilio"

    # secrets_store
    VAULT = "vault"

    # graph_store
    NEO4J = "neo4j"

    # interaction_surface
    LAB_JSON = "lab_json"
    SLASH_COMMAND = "slash_command"

    # collaboration_suite
    MS365_GRAPH = "ms365_graph"
    GOOGLE_WORKSPACE = "google_workspace"

    # issue_tracker
    JIRA = "jira"
    AZURE_DEVOPS = "azure_devops"
    GITHUB = "github"
    LINEAR = "linear"

    # wiki_knowledge
    CONFLUENCE = "confluence"
    NOTION = "notion"
    SHAREPOINT = "sharepoint"

    # observability_backend
    PROMETHEUS = "prometheus"
    ELASTICSEARCH = "elasticsearch"
    OTEL = "otel"
    LANGFUSE = "langfuse"
    DATADOG = "datadog"
    CLICKHOUSE = "clickhouse"
    SENTRY = "sentry"

    # browser_automation
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    FIRECRAWL = "firecrawl"

    # document_parser
    DOCLING = "docling"
    PYMUPDF = "pymupdf"
    UNSTRUCTURED = "unstructured"
    PYTHON_DOCX = "python_docx"
    OPENPYXL = "openpyxl"
    WHISPER = "whisper"
    YT_DLP = "yt_dlp"

    # rerank_provider
    COHERE_RERANK = "cohere_rerank"
    JINA_RERANK = "jina_rerank"

    # cloud_platform
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


SlugInput = Union[IntegrationSlug, str]


# Profile field → slugs allowed for that category (compile-time catalog).
FIELD_SLUGS: dict[str, frozenset[IntegrationSlug]] = {
    "relational_store": frozenset(
        {
            IntegrationSlug.SQLITE,
            IntegrationSlug.POSTGRESQL,
            IntegrationSlug.MYSQL,
            IntegrationSlug.ORACLE,
            IntegrationSlug.MSSQL,
            IntegrationSlug.DATABRICKS,
            IntegrationSlug.AZURE_SQL,
            IntegrationSlug.CLOUD_SQL,
            IntegrationSlug.SNOWFLAKE,
            IntegrationSlug.SUPABASE,
        }
    ),
    "document_store": frozenset(
        {
            IntegrationSlug.MONGODB,
            IntegrationSlug.CASSANDRA,
            IntegrationSlug.DYNAMODB,
        }
    ),
    "key_value_cache": frozenset(
        {
            IntegrationSlug.REDIS,
            IntegrationSlug.MEMCACHED,
            IntegrationSlug.ELASTICACHE,
        }
    ),
    "message_bus": frozenset(
        {
            IntegrationSlug.KAFKA,
            IntegrationSlug.CELERY,
            IntegrationSlug.RABBITMQ,
            IntegrationSlug.SQS,
            IntegrationSlug.SERVICE_BUS,
            IntegrationSlug.PUBSUB,
            IntegrationSlug.TEMPORAL,
            IntegrationSlug.NATS,
        }
    ),
    "object_storage": frozenset(
        {
            IntegrationSlug.S3,
            IntegrationSlug.AZURE_BLOB,
            IntegrationSlug.GCS,
            IntegrationSlug.FILESYSTEM,
            IntegrationSlug.MINIO,
        }
    ),
    "vector_store": frozenset(
        {
            IntegrationSlug.QDRANT,
            IntegrationSlug.PINECONE,
            IntegrationSlug.CHROMA,
            IntegrationSlug.INMEMORY,
            IntegrationSlug.WEAVIATE,
            IntegrationSlug.MILVUS,
        }
    ),
    "search_provider": frozenset(
        {
            IntegrationSlug.GOOGLE_CSE,
            IntegrationSlug.BING,
            IntegrationSlug.REDDIT,
            IntegrationSlug.GOOGLE_PLACES,
            IntegrationSlug.BRAVE,
            IntegrationSlug.SERPAPI,
            IntegrationSlug.TAVILY,
            IntegrationSlug.EXA,
        }
    ),
    "notification_channel": frozenset(
        {
            IntegrationSlug.SLACK,
            IntegrationSlug.TEAMS,
            IntegrationSlug.EMAIL_SMTP,
            IntegrationSlug.WEBHOOK,
            IntegrationSlug.LOG,
            IntegrationSlug.DISCORD,
            IntegrationSlug.TWILIO,
        }
    ),
    "secrets_store": frozenset({IntegrationSlug.VAULT}),
    "graph_store": frozenset({IntegrationSlug.NEO4J}),
    "interaction_surface": frozenset(
        {
            IntegrationSlug.SLACK,
            IntegrationSlug.TEAMS,
            IntegrationSlug.LAB_JSON,
            IntegrationSlug.SLASH_COMMAND,
        }
    ),
    "collaboration_suite": frozenset(
        {
            IntegrationSlug.MS365_GRAPH,
            IntegrationSlug.GOOGLE_WORKSPACE,
        }
    ),
    "issue_tracker": frozenset(
        {
            IntegrationSlug.JIRA,
            IntegrationSlug.AZURE_DEVOPS,
            IntegrationSlug.GITHUB,
            IntegrationSlug.LINEAR,
        }
    ),
    "wiki_knowledge": frozenset(
        {
            IntegrationSlug.CONFLUENCE,
            IntegrationSlug.NOTION,
            IntegrationSlug.SHAREPOINT,
        }
    ),
    "observability_backend": frozenset(
        {
            IntegrationSlug.PROMETHEUS,
            IntegrationSlug.ELASTICSEARCH,
            IntegrationSlug.OTEL,
            IntegrationSlug.LANGFUSE,
            IntegrationSlug.DATADOG,
            IntegrationSlug.CLICKHOUSE,
            IntegrationSlug.SENTRY,
        }
    ),
    "browser_automation": frozenset(
        {
            IntegrationSlug.PLAYWRIGHT,
            IntegrationSlug.SELENIUM,
            IntegrationSlug.FIRECRAWL,
        }
    ),
    "document_parser": frozenset(
        {
            IntegrationSlug.DOCLING,
            IntegrationSlug.PYMUPDF,
            IntegrationSlug.UNSTRUCTURED,
            IntegrationSlug.PYTHON_DOCX,
            IntegrationSlug.OPENPYXL,
            IntegrationSlug.WHISPER,
            IntegrationSlug.YT_DLP,
        }
    ),
    "rerank_provider": frozenset(
        {
            IntegrationSlug.COHERE_RERANK,
            IntegrationSlug.JINA_RERANK,
        }
    ),
    "cloud_platform": frozenset(
        {
            IntegrationSlug.AWS,
            IntegrationSlug.AZURE,
            IntegrationSlug.GCP,
        }
    ),
}


CLOUD_PLATFORM_DEFAULTS: dict[IntegrationSlug, dict[IntegrationCategory, IntegrationSlug]] = {
    IntegrationSlug.AWS: {
        IntegrationCategory.OBJECT_STORAGE: IntegrationSlug.S3,
        IntegrationCategory.MESSAGE_BUS: IntegrationSlug.SQS,
        IntegrationCategory.DOCUMENT_STORE: IntegrationSlug.DYNAMODB,
        IntegrationCategory.KEY_VALUE_CACHE: IntegrationSlug.ELASTICACHE,
    },
    IntegrationSlug.AZURE: {
        IntegrationCategory.OBJECT_STORAGE: IntegrationSlug.AZURE_BLOB,
        IntegrationCategory.MESSAGE_BUS: IntegrationSlug.SERVICE_BUS,
        IntegrationCategory.RELATIONAL_STORE: IntegrationSlug.AZURE_SQL,
    },
    IntegrationSlug.GCP: {
        IntegrationCategory.OBJECT_STORAGE: IntegrationSlug.GCS,
        IntegrationCategory.MESSAGE_BUS: IntegrationSlug.PUBSUB,
        IntegrationCategory.RELATIONAL_STORE: IntegrationSlug.CLOUD_SQL,
    },
}


def coerce_slug(value: SlugInput) -> IntegrationSlug:
    """Parse env/YAML/code input into a validated ``IntegrationSlug``."""
    if isinstance(value, IntegrationSlug):
        return value
    normalized = str(value).strip().lower()
    try:
        return IntegrationSlug(normalized)
    except ValueError as exc:
        known = ", ".join(sorted(member.value for member in IntegrationSlug))
        raise ValueError(
            f"Unknown integration slug {value!r}. Known slugs: {known}"
        ) from exc


def slug_value(value: SlugInput | None) -> str | None:
    if value is None:
        return None
    return coerce_slug(value).value


def validate_field_slug(field_name: str, value: SlugInput | None) -> IntegrationSlug | None:
    if value is None:
        return None
    slug = coerce_slug(value)
    allowed = FIELD_SLUGS.get(field_name)
    if allowed is not None and slug not in allowed:
        allowed_values = ", ".join(sorted(member.value for member in allowed))
        raise ValueError(
            f"Slug {slug.value!r} is not valid for profile field '{field_name}'. "
            f"Allowed: {allowed_values}"
        )
    return slug
