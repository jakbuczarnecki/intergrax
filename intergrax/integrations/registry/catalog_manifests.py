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

LAB_JSON = IntegrationManifest(
    slug="lab_json",
    categories=(IntegrationCategory.INTERACTION_SURFACE,),
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
    status=IntegrationStatus.BETA,
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

__all__ = [
    "AWS",
    "AZURE",
    "COHERE_RERANK",
    "DOCLING",
    "GCP",
    "GOOGLE_CSE",
    "INMEMORY",
    "JINA_RERANK",
    "LAB_JSON",
    "LANGSMITH",
    "LOG",
    "OTEL",
    "PAGERDUTY",
    "POSTGRESQL",
    "QDRANT",
    "REDIS",
    "SENTRY",
    "SQLITE",
]
