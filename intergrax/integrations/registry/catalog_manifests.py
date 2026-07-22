# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Tier-3 preset manifests (lightweight — no provider package imports).

Provider packages own the canonical ``manifest.MANIFEST`` used at registration; these
copies are preset shortcuts for ``IntegrationProfile`` only and avoid import cycles.
"""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

SQLITE = IntegrationManifest(
    slug="sqlite",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_SQLITE",
    description="SQLite relational facade for lab and product defaults.",
)

POSTGRESQL = IntegrationManifest(
    slug="postgresql",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_POSTGRESQL",
)

LOG = IntegrationManifest(
    slug="log",
    categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
    status=IntegrationStatus.STABLE,
)

DOCLING = IntegrationManifest(
    slug="docling",
    categories=(IntegrationCategory.DOCUMENT_PARSER,),
    status=IntegrationStatus.STABLE,
)

OTEL = IntegrationManifest(
    slug="otel",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
)

REDIS = IntegrationManifest(
    slug="redis",
    categories=(IntegrationCategory.KEY_VALUE_CACHE,),
    status=IntegrationStatus.STABLE,
)

QDRANT = IntegrationManifest(
    slug="qdrant",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.STABLE,
)

INMEMORY = IntegrationManifest(
    slug="inmemory",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.STABLE,
)

GOOGLE_CSE = IntegrationManifest(
    slug="google_cse",
    categories=(IntegrationCategory.SEARCH_PROVIDER,),
    status=IntegrationStatus.STABLE,
)

COHERE_RERANK = IntegrationManifest(
    slug="cohere_rerank",
    categories=(IntegrationCategory.RERANK_PROVIDER,),
    status=IntegrationStatus.STABLE,
)

JINA_RERANK = IntegrationManifest(
    slug="jina_rerank",
    categories=(IntegrationCategory.RERANK_PROVIDER,),
    status=IntegrationStatus.STABLE,
)

PAGERDUTY = IntegrationManifest(
    slug="pagerduty",
    categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
    status=IntegrationStatus.STABLE,
)

SENTRY = IntegrationManifest(
    slug="sentry",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.BETA,
)

LANGSMITH = IntegrationManifest(
    slug="langsmith",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.BETA,
)

AWS = IntegrationManifest(
    slug="aws",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.STABLE,
)

AZURE = IntegrationManifest(
    slug="azure",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.STABLE,
)

GCP = IntegrationManifest(
    slug="gcp",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.STABLE,
)

GRAFANA = IntegrationManifest(
    slug="grafana",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_GRAFANA",
)

LOKI = IntegrationManifest(
    slug="loki",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_LOKI",
)

TEMPO = IntegrationManifest(
    slug="tempo",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_TEMPO",
)

PGVECTOR = IntegrationManifest(
    slug="pgvector",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_PGVECTOR",
)

DUCKDB = IntegrationManifest(
    slug="duckdb",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_DUCKDB",
)

DOPPLER = IntegrationManifest(
    slug="doppler",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_DOPPLER",
)

UNLEASH = IntegrationManifest(
    slug="unleash",
    categories=(IntegrationCategory.FEATURE_FLAG,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_UNLEASH",
)

GITHUB_ACTIONS = IntegrationManifest(
    slug="github_actions",
    categories=(IntegrationCategory.CI_CD,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_GITHUB_ACTIONS",
)

PROMETHEUS = IntegrationManifest(
    slug="prometheus",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_PROMETHEUS",
)

CLICKHOUSE = IntegrationManifest(
    slug="clickhouse",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_CLICKHOUSE",
)

VAULT = IntegrationManifest(
    slug="vault",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_VAULT",
)

GITHUB = IntegrationManifest(
    slug="github",
    categories=(IntegrationCategory.ISSUE_TRACKER,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_GITHUB",
)

LANGFUSE = IntegrationManifest(
    slug="langfuse",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_LANGFUSE",
)

MINIO = IntegrationManifest(
    slug="minio",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_MINIO",
)

REDPANDA = IntegrationManifest(
    slug="redpanda",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_REDPANDA",
)

KAFKA = IntegrationManifest(
    slug="kafka",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.STABLE,
)

GITLAB_CI = IntegrationManifest(
    slug="gitlab_ci",
    categories=(IntegrationCategory.CI_CD,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_GITLAB_CI",
)

__all__ = [
    "AWS",
    "AZURE",
    "CLICKHOUSE",
    "COHERE_RERANK",
    "DOCLING",
    "DOPPLER",
    "DUCKDB",
    "GCP",
    "GITHUB",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "GOOGLE_CSE",
    "GRAFANA",
    "INMEMORY",
    "JINA_RERANK",
    "KAFKA",
    "LANGFUSE",
    "LANGSMITH",
    "LOG",
    "LOKI",
    "MINIO",
    "OTEL",
    "PAGERDUTY",
    "PGVECTOR",
    "POSTGRESQL",
    "PROMETHEUS",
    "QDRANT",
    "REDIS",
    "REDPANDA",
    "SENTRY",
    "SQLITE",
    "TEMPO",
    "UNLEASH",
    "VAULT",
]
