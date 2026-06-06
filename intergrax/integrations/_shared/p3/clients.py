# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.7 catalog adapters over duck-typed backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from intergrax.integrations._shared.rest_search import hits_from_exa_payload, hits_from_tavily_payload
from intergrax.integrations._shared.vector_store_bridge import VectorStoreBridge
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
from intergrax.integrations.contracts.graph_store import GraphNodeRecord, GraphQueryResult, GraphStore
from intergrax.integrations.contracts.observability_backend import MetricPoint, MetricQueryResult, MetricSeries, TraceQueryResult, TraceRecord
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations._shared.p2.clients import RestSearchProvider
from intergrax.websearch.schemas.search_hit import SearchHit


class TraceQueryClient(Protocol):
    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult: ...


class VaultSecretsStore:
    def __init__(self, client: Any, *, mount: str) -> None:
        self._client = client
        self._mount = mount
        self._closed = False

    def get_secret(self, path: str, *, version: Optional[str] = None) -> str:
        self._require_open()
        return str(self._client.read_secret(self._mount, path, version=version))

    def put_secret(self, path: str, value: str) -> None:
        self._require_open()
        self._client.write_secret(self._mount, path, value)

    def delete_secret(self, path: str) -> None:
        self._require_open()
        self._client.delete_secret(self._mount, path)

    def close(self) -> None:
        self._closed = True

    def health(self) -> HealthStatus | bool:
        from intergrax.integrations._shared.health import probe_client_health

        if self._closed:
            return False
        return probe_client_health(self._client, slug="vault")

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Vault secrets store is closed")


class Neo4jGraphStore:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._closed = False

    def run_query(
        self,
        statement: str,
        *,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> GraphQueryResult:
        self._require_open()
        records = self._client.run(statement, dict(parameters or {}))
        return GraphQueryResult(records=[dict(r) for r in records])

    def get_node(self, node_id: str) -> Optional[GraphNodeRecord]:
        self._require_open()
        payload = self._client.get_node(node_id)
        if payload is None:
            return None
        return GraphNodeRecord(
            id=str(payload.get("id") or node_id),
            labels=list(payload.get("labels") or []),
            properties=dict(payload.get("properties") or payload),
        )

    def close(self) -> None:
        self._closed = True

    def health(self) -> HealthStatus:
        from intergrax.integrations._shared.health import probe_client_health

        if self._closed:
            return HealthStatus(slug="neo4j", healthy=False, detail="graph store closed")
        return probe_client_health(self._client, slug="neo4j")

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Neo4j graph store is closed")


class HttpObservabilityBackend:
    """Metrics/traces facade for Langfuse, Datadog, ClickHouse HTTP APIs."""

    def __init__(
        self,
        client: Any,
        *,
        provider: str,
        instant_fn: Callable[[str, Optional[float]], float],
        range_fn: Callable[[str, float, float, str], list[dict[str, float]]],
    ) -> None:
        self._client = client
        self._provider = provider
        self._instant_fn = instant_fn
        self._range_fn = range_fn

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        value = float(self._instant_fn(promql, eval_time))
        ts = float(eval_time or 0.0)
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={"provider": self._provider}, points=[MetricPoint(timestamp=ts, value=value)])],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        rows = self._range_fn(promql, start, end, step)
        points = [MetricPoint(timestamp=float(r["timestamp"]), value=float(r["value"])) for r in rows]
        return MetricQueryResult(
            result_type="matrix",
            series=[MetricSeries(metric={"provider": self._provider}, points=points)],
        )

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        if isinstance(self._client, TraceQueryClient):
            return self._client.query_traces(limit=limit, name=name)
        return TraceQueryResult()

    def health(self) -> HealthStatus:
        from intergrax.integrations._shared.health import probe_client_health

        return probe_client_health(self._client, slug=self._provider)


class SentryObservabilityBackend:
    """
    Error tracking + issue stats facade for Sentry.

    Implements ``ObservabilityBackend`` (issue counts via REST) and exposes
    ``capture_exception`` / ``capture_message`` for runtime error reporting.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        value = float(self._client.query_instant(promql, eval_time=eval_time))
        ts = float(eval_time or 0.0)
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={"provider": "sentry"}, points=[MetricPoint(timestamp=ts, value=value)])],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        rows = self._client.query_range(promql, start=start, end=end, step=step)
        points = [MetricPoint(timestamp=float(r["timestamp"]), value=float(r["value"])) for r in rows]
        return MetricQueryResult(
            result_type="matrix",
            series=[MetricSeries(metric={"provider": "sentry"}, points=points)],
        )

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        _ = limit, name
        return TraceQueryResult()

    def capture_exception(self, exc: BaseException, *, tags: Optional[dict[str, str]] = None) -> str:
        return str(self._client.capture_exception(exc, tags=tags or {}))

    def capture_message(self, message: str, *, level: str = "info") -> str:
        return str(self._client.capture_message(message, level=level))


class RestVectorStoreIntegration(VectorStoreBridge):
    """Catalog bridge when inner store is already a ``VectorStore`` (Weaviate/Milvus HTTP facades)."""

    def __init__(self, config: Any, inner: VectorStore) -> None:
        super().__init__(config, inner)


class HttpNotificationChannel:
    def __init__(
        self,
        sender: Callable[..., None],
        *,
        provider: str,
        health_client: Optional[Any] = None,
    ) -> None:
        self._sender = sender
        self._provider = provider
        self._health_client = health_client

    async def notify(self, message: Any) -> None:
        from intergrax.runtime.notifications.models import NotificationMessage

        if not isinstance(message, NotificationMessage):
            raise IntegrationConfigurationError(f"{self._provider} expects NotificationMessage")
        self._sender(message=message)

    def health(self) -> HealthStatus:
        from intergrax.integrations._shared.health import probe_client_health

        if self._health_client is not None:
            return probe_client_health(self._health_client, slug=self._provider)
        return HealthStatus(slug=self._provider, healthy=True, detail="no probe")


class FirecrawlBrowserAutomation:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._closed = False

    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        self._require_open()
        payload = self._client.scrape(url)
        return PageContent(
            url=url,
            title=str(payload.get("title") or ""),
            text=str(payload.get("markdown") or payload.get("text") or ""),
            html=str(payload.get("html") or ""),
            status_code=int(payload.get("status_code") or 200),
            metadata={k: str(v) for k, v in dict(payload.get("metadata") or {}).items()},
        )

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Firecrawl browser automation is closed")


class SeleniumBrowserAutomation:
    def __init__(self, driver: Any, *, timeout_ms: int) -> None:
        self._driver = driver
        self._timeout_ms = timeout_ms
        self._closed = False

    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        self._require_open()
        self._driver.get(url)
        title = str(getattr(self._driver, "title", "") or "")
        html = str(getattr(self._driver, "page_source", "") or "")
        text = ""
        try:
            body = self._driver.find_element("tag name", "body")
            text = str(body.text or "")
        except Exception:
            text = ""
        return PageContent(url=url, title=title, text=text, html=html, status_code=200)

    def close(self) -> None:
        if not self._closed:
            if hasattr(self._driver, "quit"):
                self._driver.quit()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Selenium browser is closed")


class FilesystemBlobClient:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def put_object(self, *, Key: str, Body: bytes, ContentType: str, Metadata: Optional[dict[str, str]] = None) -> None:
        path = self._root / Key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Body)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps({"content_type": ContentType, "metadata": Metadata or {}}), encoding="utf-8")

    def get_object(self, *, Key: str) -> Any:
        path = self._root / Key
        if not path.is_file():
            raise FileNotFoundError(Key)

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        content_type = "application/octet-stream"
        metadata: dict[str, str] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            content_type = str(meta.get("content_type") or content_type)
            metadata = {str(k): str(v) for k, v in dict(meta.get("metadata") or {}).items()}

        return {
            "Body": _Body(path.read_bytes()),
            "ContentType": content_type,
            "Metadata": metadata,
        }

    def delete_object(self, *, Key: str) -> None:
        path = self._root / Key
        path.unlink(missing_ok=True)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.unlink(missing_ok=True)


def tavily_hits(query: str, payload: Mapping[str, Any], limit: int) -> Sequence[SearchHit]:
    return hits_from_tavily_payload(query, payload, limit=limit)


def exa_hits(query: str, payload: Mapping[str, Any], limit: int) -> Sequence[SearchHit]:
    return hits_from_exa_payload(query, payload, limit=limit)


def build_rest_search_provider(
    *,
    provider: str,
    search_fn: Callable[[str, int], Mapping[str, Any]],
    hits_fn: Callable[[str, Mapping[str, Any], int], Sequence[SearchHit]],
) -> SearchProvider:
    return RestSearchProvider(provider=provider, search_fn=search_fn, hits_fn=hits_fn)
