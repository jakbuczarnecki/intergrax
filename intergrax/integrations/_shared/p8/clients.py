# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.7 P7 catalog adapters for agent-developer integrations."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.websearch.schemas.search_hit import SearchHit


def hits_from_generic_results(
    query: str,
    payload: Mapping[str, Any],
    *,
    provider: str,
    limit: int,
    results_key: str = "results",
    title_key: str = "title",
    url_key: str = "url",
    snippet_key: str = "snippet",
) -> Sequence[SearchHit]:
    rows = payload.get(results_key) or payload.get("hits") or payload.get("data") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    hits: list[SearchHit] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        hits.append(
            SearchHit(
                provider=provider,
                query_issued=query,
                rank=idx + 1,
                title=str(row.get(title_key) or row.get("name") or ""),
                url=str(row.get(url_key) or row.get("link") or row.get("id") or ""),
                snippet=str(row.get(snippet_key) or row.get("content") or row.get("abstract") or ""),
            )
        )
        if len(hits) >= limit:
            break
    return hits


class HttpDocumentParser:
    """REST document parser facade (LlamaParse and similar SaaS parsers)."""

    def __init__(self, client: Any, *, parser_id: str) -> None:
        self._client = client
        self._parser_id = parser_id

    def parser_id(self) -> str:
        return self._parser_id

    def is_available(self) -> bool:
        if hasattr(self._client, "health"):
            try:
                return bool(self._client.health())
            except Exception:  # noqa: BLE001 — availability probe
                return False
        return True

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        payload = self._client.parse_file(source)
        if isinstance(payload, list):
            return [
                ParsedDocumentFragment(
                    text=str(row.get("text") or row.get("content") or ""),
                    metadata=dict(row.get("metadata") or {}),
                )
                for row in payload
                if isinstance(row, dict)
            ]
        if isinstance(payload, dict):
            text = str(payload.get("text") or payload.get("content") or payload.get("markdown") or "")
            return [ParsedDocumentFragment(text=text, metadata=dict(payload.get("metadata") or {}))]
        return [ParsedDocumentFragment(text=str(payload or ""), metadata={"source": source})]

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._parser_id)


class HttpBrowserAutomation:
    """Managed browser session facade (Browserbase, Apify actor runs)."""

    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider
        self._closed = False

    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        self._require_open()
        payload = self._client.fetch_page(url, wait_until=wait_until)
        if isinstance(payload, PageContent):
            return payload
        if isinstance(payload, dict):
            return PageContent(
                url=str(payload.get("url") or url),
                title=str(payload.get("title") or ""),
                text=str(payload.get("text") or payload.get("content") or ""),
                html=str(payload.get("html") or ""),
                status_code=int(payload.get("status_code") or 200),
                metadata={k: str(v) for k, v in dict(payload.get("metadata") or {}).items()},
            )
        return PageContent(url=url, text=str(payload or ""))

    def close(self) -> None:
        self._closed = True
        if hasattr(self._client, "close"):
            self._client.close()

    def health(self) -> HealthStatus:
        if self._closed:
            return HealthStatus(slug=self._provider, healthy=False, detail="browser closed")
        return probe_client_health(self._client, slug=self._provider)

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError(f"{self._provider} browser automation is closed")


class UpstashKeyValueCache:
    """Upstash Redis REST API facade."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._closed = False

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        self._require_open()
        value = self._client.get(f"{tenant_id}:{key}")
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._require_open()
        self._client.set(f"{tenant_id}:{key}", value, ttl_seconds=ttl_seconds)

    def delete(self, tenant_id: str, key: str) -> None:
        self._require_open()
        self._client.delete(f"{tenant_id}:{key}")

    def set_if_absent(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        self._require_open()
        return bool(self._client.setnx(f"{tenant_id}:{key}", value, ttl_seconds=ttl_seconds))

    def close(self) -> None:
        self._closed = True

    def health(self) -> HealthStatus:
        if self._closed:
            return HealthStatus(slug="upstash_redis", healthy=False, detail="cache closed")
        return probe_client_health(self._client, slug="upstash_redis")

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Upstash Redis cache is closed")


class HttpBigQueryRelationalStore:
    """BigQuery jobs API facade implementing RelationalStore read paths."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._closed = False

    def connect(self) -> None:
        return None

    def execute(self, statement: str, parameters: Optional[Mapping[str, Any]] = None) -> None:
        self._require_open()
        self._client.execute(statement, dict(parameters or {}))

    def fetch_all(self, statement: str, parameters: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
        self._require_open()
        rows = self._client.fetch_all(statement, dict(parameters or {}))
        return [dict(row) for row in rows if isinstance(row, dict)]

    def close(self) -> None:
        self._closed = True

    def health(self) -> HealthStatus:
        if self._closed:
            return HealthStatus(slug="bigquery", healthy=False, detail="store closed")
        return probe_client_health(self._client, slug="bigquery")

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("BigQuery relational store is closed")


__all__ = [
    "HttpBigQueryRelationalStore",
    "HttpBrowserAutomation",
    "HttpDocumentParser",
    "UpstashKeyValueCache",
    "hits_from_generic_results",
]
