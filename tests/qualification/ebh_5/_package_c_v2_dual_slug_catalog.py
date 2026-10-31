# © Artur Czarnecki. All rights reserved.

"""RPL-B1–B3 integration dual-slug catalog — sanctioned catalog providers only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from intergrax.integrations.contracts.base import IntegrationCategory

RPL_BATCH_B1: Final[str] = "RPL-B1"
RPL_BATCH_B2: Final[str] = "RPL-B2"
RPL_BATCH_B3: Final[str] = "RPL-B3"

_LEGACY_FACTORY_BLOCKER: Final[str] = (
    "Catalog factory for one or both slugs does not return PlatformIntegrationContract @ resolve_from_profile"
)


@dataclass(frozen=True, slots=True)
class IntegrationDualSlugCatalogEntry:
    concern_id: str
    batch: str
    category: IntegrationCategory
    slug_a: str
    slug_b: str
    operational_lifecycle: bool
    """True when connect/close (or equivalent) is exercised on both backends."""


def _entry(
    suffix: str,
    batch: str,
    slug_a: str,
    slug_b: str,
    *,
    operational_lifecycle: bool = False,
) -> IntegrationDualSlugCatalogEntry:
    return IntegrationDualSlugCatalogEntry(
        concern_id=f"integration.{suffix}",
        batch=batch,
        category=IntegrationCategory(suffix),
        slug_a=slug_a,
        slug_b=slug_b,
        operational_lifecycle=operational_lifecycle,
    )


INTEGRATION_DUAL_SLUG_CATALOG: Final[tuple[IntegrationDualSlugCatalogEntry, ...]] = (
    _entry("relational_store", RPL_BATCH_B1, "sqlite", "duckdb", operational_lifecycle=True),
    _entry("document_store", RPL_BATCH_B1, "dynamodb", "cassandra"),
    _entry("message_bus", RPL_BATCH_B1, "pulsar", "sqs"),
    _entry("object_storage", RPL_BATCH_B1, "filesystem", "cloudflare_r2"),
    _entry("graph_store", RPL_BATCH_B1, "neo4j", "memgraph"),
    _entry("notification_channel", RPL_BATCH_B2, "log", "pagerduty"),
    _entry("collaboration_suite", RPL_BATCH_B2, "ms365_graph", "google_workspace"),
    _entry("issue_tracker", RPL_BATCH_B2, "jira", "gitlab"),
    _entry("wiki_knowledge", RPL_BATCH_B2, "confluence", "notion"),
    _entry("browser_automation", RPL_BATCH_B2, "apify", "firecrawl"),
    _entry("cloud_platform", RPL_BATCH_B2, "aws", "gcp"),
    _entry("secrets_store", RPL_BATCH_B2, "doppler", "infisical"),
    _entry("sandbox_host", RPL_BATCH_B2, "modal", "daytona"),
    _entry("speech_provider", RPL_BATCH_B2, "deepgram", "elevenlabs"),
    _entry("embedding_provider", RPL_BATCH_B2, "ollama", "hf"),
    _entry("vector_store", RPL_BATCH_B1, "inmemory", "lancedb"),
    _entry("conversation_channel", RPL_BATCH_B2, "slack", "discord"),
    _entry("workflow_orchestrator", RPL_BATCH_B2, "n8n", "prefect"),
    _entry("ci_cd", RPL_BATCH_B2, "github_actions", "gitlab_ci"),
    _entry("feature_flag", RPL_BATCH_B2, "unleash", "launchdarkly"),
    _entry("crm", RPL_BATCH_B2, "hubspot", "salesforce"),
    _entry("observability_backend", RPL_BATCH_B3, "otel", "sentry"),
    _entry("security_scanner", RPL_BATCH_B3, "trivy", "semgrep"),
    _entry("identity_provider", RPL_BATCH_B3, "auth0", "okta"),
    _entry("document_parser", RPL_BATCH_B3, "pymupdf", "python_docx"),
    _entry("rerank_provider", RPL_BATCH_B3, "cohere_rerank", "jina_rerank"),
    _entry("search_provider", RPL_BATCH_B3, "tavily", "brave"),
    _entry("llm_guardrail", RPL_BATCH_B3, "presidio", "lakera"),
)


def catalog_by_concern_id() -> dict[str, IntegrationDualSlugCatalogEntry]:
    return {row.concern_id: row for row in INTEGRATION_DUAL_SLUG_CATALOG}
