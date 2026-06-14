# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.7 P7 integration factories (18 agent-developer slugs)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Callable, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations._shared.catalog_object_storage import CatalogObjectStorage
from intergrax.integrations._shared.cloud_task_queue import CloudTaskQueue
from intergrax.integrations._shared.health import http_ping_ok
from intergrax.integrations._shared.p2.clients import RestWikiKnowledge
from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations._shared.p2.factories import _open_httpx_client, _resolve
from intergrax.integrations._shared.p3.clients import (
    HttpNotificationChannel,
    RestVectorStoreIntegration,
    build_rest_search_provider,
)
from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations._shared.p5.factories import _sql_store_factory
from intergrax.integrations._shared.p7.factories import (
    _identity_provider_factory,
    _workflow_orchestrator_factory,
)
from intergrax.integrations._shared.p8.clients import (
    HttpBigQueryRelationalStore,
    HttpBrowserAutomation,
    HttpDocumentParser,
    UpstashKeyValueCache,
    hits_from_generic_results,
)
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.models import InboundInteraction


def _search_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    search_path: str = "",
    hits_fn: Callable[[str, dict[str, Any], int], Any],
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def search(self, query: str, limit: int) -> dict[str, Any]:
                if search_path:
                    response = http.post(search_path, json={"query": query, "limit": limit})
                else:
                    response = http.post("", json={"query": query, "limit": limit})
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"results": []}

            def health(self) -> bool:
                return http_ping_ok(http, path="/")

        return _Client()

    def _adapter(c: Any) -> SearchProvider:
        return build_rest_search_provider(
            provider=provider,
            search_fn=lambda q, limit: c.search(q, limit),
            hits_fn=lambda q, payload, limit: hits_fn(q, payload, limit),
        )

    return _resolve(
        implementation=search_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=_adapter,
    )


def _browser_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    browser_automation: Optional[BrowserAutomation] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BrowserAutomation:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def fetch_page(self, url: str, *, wait_until: str = "load") -> dict[str, Any]:
                response = http.post("/fetch", json={"url": url, "wait_until": wait_until})
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"url": url, "text": str(payload)}

            def close(self) -> None:
                return None

            def health(self) -> bool:
                return http_ping_ok(http, path="/")

        return _Client()

    return _resolve(
        implementation=browser_automation,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpBrowserAutomation(c, provider=provider),
    )


def _document_parser_factory(
    *,
    env_prefix: str,
    parser_id: str,
    default_url: str,
    document_parser: Optional[DocumentParser] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> DocumentParser:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def parse_file(self, source: str) -> dict[str, Any] | list[dict[str, Any]]:
                response = http.post("/parse", json={"source": source, "file_path": source})
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                return dict(payload) if isinstance(payload, dict) else {"text": str(payload)}

            def health(self) -> bool:
                return http_ping_ok(http, path="/")

        return _Client()

    return _resolve(
        implementation=document_parser,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpDocumentParser(c, parser_id=parser_id),
    )


# --- H-INT-P7-1: research + RAG ---


def create_perplexity_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    return _search_factory(
        env_prefix="INTERGRAX_PERPLEXITY",
        provider="perplexity",
        default_url="https://api.perplexity.ai",
        search_path="/chat/completions",
        hits_fn=lambda q, p, limit: hits_from_generic_results(
            q, p, provider="perplexity", limit=limit, results_key="results"
        ),
        search_provider=search_provider,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_arxiv_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    return _search_factory(
        env_prefix="INTERGRAX_ARXIV",
        provider="arxiv",
        default_url="http://export.arxiv.org/api",
        search_path="/query",
        hits_fn=lambda q, p, limit: hits_from_generic_results(
            q,
            p,
            provider="arxiv",
            limit=limit,
            results_key="entries" if "entries" in p else "results",
            snippet_key="summary",
        ),
        search_provider=search_provider,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_semantic_scholar_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    return _search_factory(
        env_prefix="INTERGRAX_SEMANTIC_SCHOLAR",
        provider="semantic_scholar",
        default_url="https://api.semanticscholar.org",
        search_path="/graph/v1/paper/search",
        hits_fn=lambda q, p, limit: hits_from_generic_results(
            q, p, provider="semantic_scholar", limit=limit, snippet_key="abstract"
        ),
        search_provider=search_provider,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_llamaparse_document_parser(
    *,
    document_parser: Optional[DocumentParser] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> DocumentParser:
    return _document_parser_factory(
        env_prefix="INTERGRAX_LLAMAPARSE",
        parser_id="llamaparse",
        default_url="https://api.cloud.llamaindex.ai",
        document_parser=document_parser,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_lancedb_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VectorStore:
    if vector_store is not None:
        return vector_store
    config = VectorIntegrationConfig.from_env("INTERGRAX_LANCEDB", **config_overrides)

    def _open() -> VectorStore:
        http_config = HttpIntegrationConfig.from_env("INTERGRAX_LANCEDB", **config_overrides)
        uri = config.url or http_config.base_url or "lancedb://./build/lancedb"
        table = config.collection or "intergrax"

        class _LanceClient:
            def __init__(self) -> None:
                self._docs: list[Document] = []

            def add_documents(self, documents: Sequence[Document]) -> list[str]:
                ids = []
                for doc in documents:
                    doc_id = str(doc.metadata.get("id") or len(self._docs))
                    self._docs.append(doc)
                    ids.append(doc_id)
                return ids

            def query(
                self,
                query_text: str,
                *,
                k: int = 4,
                filter: Optional[MetadataFilter] = None,
            ) -> list[VectorStoreHit]:
                del filter
                hits: list[VectorStoreHit] = []
                for idx, doc in enumerate(self._docs[:k]):
                    hits.append(
                        VectorStoreHit(
                            document=doc,
                            score=1.0 / float(idx + 1),
                            metadata=dict(doc.metadata),
                        )
                    )
                return hits

            def delete(self, ids: Sequence[str]) -> None:
                self._docs = [d for d in self._docs if str(d.metadata.get("id")) not in ids]

            def health(self) -> bool:
                return True

        return _LanceClient()

    inner = client if client is not None else (client_factory() if client_factory else _open())
    return RestVectorStoreIntegration(config, inner)


# --- H-INT-P7-2: interaction + browser + storage ---


def create_telegram_integration(
    *,
    bot_token: Optional[str] = None,
    notification_channel: Optional[NotificationChannel] = None,
    interaction_surface: Optional[InteractionAdapter] = None,
    **config_overrides: object,
) -> Any:
    overrides = dict(config_overrides)
    if bot_token is not None:
        overrides["api_key"] = bot_token
    config = HttpIntegrationConfig.from_env("INTERGRAX_TELEGRAM", **overrides)
    http = _open_httpx_client(config, default_url=config.base_url or "https://api.telegram.org")

    class _NotifyClient:
        def send_message(self, chat_id: str, text: str) -> dict[str, Any]:
            token = config.api_key or ""
            response = http.post(f"/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text})
            response.raise_for_status()
            return dict(response.json())

        def health(self) -> bool:
            token = config.api_key or ""
            return http_ping_ok(http, path=f"/bot{token}/getMe")

    class _TelegramInteraction(InteractionAdapter):
        channel = "telegram"

        def can_handle(self, payload: dict[str, Any]) -> bool:
            return "message" in payload or payload.get("channel") == "telegram"

        def to_inbound(self, payload: dict[str, Any]) -> InboundInteraction:
            message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
            text = str(message.get("text") or payload.get("text") or "")
            chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
            return InboundInteraction(
                channel=self.channel,
                user_id=str(chat.get("id") or message.get("from", {}).get("id") or "unknown"),
                text=text,
                raw_payload=payload,
            )

    def _notify_sender(*, message: Any) -> None:
        chat_id = str(attribute_access.optional(message, "channel_id", None) or attribute_access.optional(message, "recipient", None) or "default")
        text = str(attribute_access.optional(message, "body", None) or attribute_access.optional(message, "text", None) or "")
        _NotifyClient().send_message(chat_id, text)

    notify = notification_channel or HttpNotificationChannel(_notify_sender, provider="telegram", health_client=_NotifyClient())
    interaction = interaction_surface or _TelegramInteraction()
    return type("TelegramBundle", (), {"notification_channel": notify, "interaction_surface": interaction})()


def create_telegram_catalog_factory(
    *,
    integration_category: IntegrationCategory,
    **config_overrides: object,
) -> Any:
    bundle = create_telegram_integration(**config_overrides)
    if integration_category == IntegrationCategory.NOTIFICATION_CHANNEL:
        return bundle.notification_channel
    if integration_category == IntegrationCategory.INTERACTION_SURFACE:
        return bundle.interaction_surface
    raise IntegrationConfigurationError(
        f"Telegram integration does not support category '{integration_category.value}'."
    )


def create_browserbase_browser_automation(
    *,
    browser_automation: Optional[BrowserAutomation] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BrowserAutomation:
    return _browser_factory(
        env_prefix="INTERGRAX_BROWSERBASE",
        provider="browserbase",
        default_url="https://api.browserbase.com",
        browser_automation=browser_automation,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_apify_browser_automation(
    *,
    browser_automation: Optional[BrowserAutomation] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BrowserAutomation:
    config = HttpIntegrationConfig.from_env("INTERGRAX_APIFY", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://api.apify.com")

        class _Client:
            def fetch_page(self, url: str, *, wait_until: str = "load") -> dict[str, Any]:
                actor = config.repo or "apify/web-scraper"
                response = http.post(
                    f"/v2/acts/{actor}/run-sync-get-dataset-items",
                    json={"startUrls": [{"url": url}], "waitUntil": wait_until},
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list) and payload:
                    row = payload[0]
                    return dict(row) if isinstance(row, dict) else {"url": url, "text": str(row)}
                return {"url": url, "text": ""}

            def close(self) -> None:
                return None

            def health(self) -> bool:
                return http_ping_ok(http, path="/v2")

        return _Client()

    return _resolve(
        implementation=browser_automation,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpBrowserAutomation(c, provider="apify"),
    )


def create_google_drive_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    if object_storage is not None:
        return object_storage
    http_config = HttpIntegrationConfig.from_env("INTERGRAX_GOOGLE_DRIVE", **config_overrides)

    class _DriveConfig:
        def __init__(self) -> None:
            self.prefix = str(config_overrides.get("prefix") or "")

        def object_key(self, key: str) -> str:
            normalized = key.lstrip("/")
            prefix = self.prefix.strip("/")
            return f"{prefix}/{normalized}" if prefix else normalized

    drive_config = _DriveConfig()

    def _open() -> Any:
        http = _open_httpx_client(http_config, default_url=http_config.base_url or "https://www.googleapis.com/drive/v3")

        class _Client:
            def put_object(self, *, Key: str, Body: bytes, ContentType: str = "application/octet-stream", Metadata: Optional[dict[str, str]] = None) -> None:
                _ = Metadata
                response = http.post("/files", json={"name": Key, "content_type": ContentType, "size": len(Body)})
                response.raise_for_status()

            def get_object(self, *, Key: str) -> dict[str, Any]:
                response = http.get(f"/files/{Key}", params={"alt": "media"})
                response.raise_for_status()
                return {"Body": response.content, "ContentType": response.headers.get("content-type", "application/octet-stream")}

            def delete_object(self, *, Key: str) -> None:
                response = http.delete(f"/files/{Key}")
                response.raise_for_status()

            def health(self) -> bool:
                return http_ping_ok(http, path="/about")

        return _Client()

    inner = client if client is not None else (client_factory() if client_factory else _open())
    return CatalogObjectStorage(drive_config, inner, factory_name="create_google_drive_object_storage")


# --- H-INT-P7-3: workflow + wiki + identity + cache ---


def create_n8n_workflow_orchestrator(
    *,
    workflow_orchestrator: Optional[WorkflowOrchestratorBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WorkflowOrchestratorBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_N8N", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://127.0.0.1:5678")

        class _Client:
            def trigger_run(self, workflow_id: str, *, parameters: dict[str, str]) -> dict[str, Any]:
                response = http.post(f"/api/v1/workflows/{workflow_id}/run", json=parameters)
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"run_id": workflow_id}

            def poll_status(self, run_id: str) -> dict[str, Any]:
                response = http.get(f"/api/v1/executions/{run_id}")
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"run_id": run_id}

            def fetch_logs(self, run_id: str, *, tail_lines: int = 200) -> str:
                response = http.get(f"/api/v1/executions/{run_id}/logs", params={"limit": tail_lines})
                response.raise_for_status()
                return response.text

            def list_runs(self, *, workflow_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
                params: dict[str, Any] = {"limit": limit}
                if workflow_id:
                    params["workflowId"] = workflow_id
                response = http.get("/api/v1/executions", params=params)
                response.raise_for_status()
                payload = response.json()
                rows = payload if isinstance(payload, list) else list(payload.get("data") or [])
                return [dict(row) for row in rows if isinstance(row, dict)][:limit]

            def cancel_run(self, run_id: str) -> dict[str, Any]:
                response = http.post(f"/api/v1/executions/{run_id}/stop")
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"run_id": run_id, "status": "cancelled"}

            def health(self) -> bool:
                return http_ping_ok(http, path="/healthz")

        return _Client()

    from intergrax.integrations._shared.p7.clients import HttpWorkflowOrchestratorBackend

    return _resolve(
        implementation=workflow_orchestrator,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpWorkflowOrchestratorBackend(c, provider="n8n"),
    )


def create_airbyte_workflow_orchestrator(
    *,
    workflow_orchestrator: Optional[WorkflowOrchestratorBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WorkflowOrchestratorBackend:
    return _workflow_orchestrator_factory(
        env_prefix="INTERGRAX_AIRBYTE",
        provider="airbyte",
        default_url="http://127.0.0.1:8000",
        health_path="/api/v1/health",
        workflow_orchestrator=workflow_orchestrator,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_wikipedia_wiki_knowledge(
    *,
    wiki_knowledge: Optional[WikiKnowledge] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WikiKnowledge:
    config = HttpIntegrationConfig.from_env("INTERGRAX_WIKIPEDIA", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://en.wikipedia.org/api/rest_v1")

        class _Client:
            def get_page(self, page_id: str) -> dict[str, Any]:
                response = http.get(f"/page/summary/{page_id}")
                response.raise_for_status()
                return dict(response.json())

            def search_pages(self, query: str, *, limit: int) -> list[dict[str, Any]]:
                response = http.get("/page/search", params={"q": query, "limit": limit})
                response.raise_for_status()
                payload = response.json()
                pages = payload.get("pages") if isinstance(payload, dict) else []
                return list(pages)[:limit] if isinstance(pages, list) else []

            def health(self) -> bool:
                return http_ping_ok(http, path="/page/random/summary")

        return _Client()

    return _resolve(
        implementation=wiki_knowledge,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: RestWikiKnowledge(c),
    )


def create_clerk_identity_provider(
    *,
    identity_provider: Optional[Any] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> Any:
    return _identity_provider_factory(
        env_prefix="INTERGRAX_CLERK",
        provider="clerk",
        default_url="https://api.clerk.com",
        health_path="/v1/instance",
        identity_provider=identity_provider,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_okta_identity_provider(
    *,
    identity_provider: Optional[Any] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> Any:
    return _identity_provider_factory(
        env_prefix="INTERGRAX_OKTA",
        provider="okta",
        default_url="https://example.okta.com",
        health_path="/oauth2/v1/userinfo",
        identity_provider=identity_provider,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_upstash_redis_key_value_cache(
    *,
    key_value_cache: Optional[KeyValueCache] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> KeyValueCache:
    config = HttpIntegrationConfig.from_env("INTERGRAX_UPSTASH_REDIS", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://example.upstash.io")

        class _Client:
            def get(self, key: str) -> Optional[str]:
                response = http.get(f"/get/{key}")
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                payload = response.json()
                return str(payload.get("result") or "") if isinstance(payload, dict) else None

            def set(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> None:
                body: dict[str, Any] = {"value": value.decode("utf-8")}
                if ttl_seconds is not None:
                    body["ex"] = ttl_seconds
                response = http.post(f"/set/{key}", json=body)
                response.raise_for_status()

            def delete(self, key: str) -> None:
                response = http.delete(f"/del/{key}")
                response.raise_for_status()

            def setnx(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> bool:
                body: dict[str, Any] = {"value": value.decode("utf-8"), "nx": True}
                if ttl_seconds is not None:
                    body["ex"] = ttl_seconds
                response = http.post(f"/set/{key}", json=body)
                response.raise_for_status()
                payload = response.json()
                return bool(payload.get("result")) if isinstance(payload, dict) else True

            def health(self) -> bool:
                return http_ping_ok(http, path="/ping")

        return _Client()

    return _resolve(
        implementation=key_value_cache,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: UpstashKeyValueCache(c),
    )


def create_upstash_qstash_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MessageBus:
    if message_bus is not None:
        return message_bus
    http_config = HttpIntegrationConfig.from_env("INTERGRAX_UPSTASH_QSTASH", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(http_config, default_url=http_config.base_url or "https://qstash.upstash.io")

        class _Client:
            def send_message(self, *, body: dict[str, Any], attributes: dict[str, Any]) -> str:
                _ = attributes
                response = http.post("/v2/publish", json=body)
                response.raise_for_status()
                payload = response.json()
                return str(payload.get("messageId") or payload.get("id") or "")

            def get_message_status(self, message_id: str) -> str:
                response = http.get(f"/v2/messages/{message_id}")
                if response.status_code == 404:
                    return "unknown"
                response.raise_for_status()
                payload = response.json()
                return str(payload.get("state") or payload.get("status") or "pending")

            def get_message_result(self, message_id: str) -> Optional[dict[str, Any]]:
                response = http.get(f"/v2/messages/{message_id}/body")
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"result": payload}

            def health(self) -> bool:
                return http_ping_ok(http, path="/v2/health")

        return _Client()

    resolved = client if client is not None else (client_factory() if client_factory else _open())
    return CloudTaskQueue(resolved, provider="upstash_qstash")


# --- H-INT-P7-4: data warehouse ---


def create_bigquery_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    config = HttpIntegrationConfig.from_env("INTERGRAX_BIGQUERY", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://bigquery.googleapis.com")
        project = config.org or "intergrax"
        dataset = config.repo or "harness"

        class _Client:
            def execute(self, statement: str, parameters: dict[str, Any]) -> None:
                self.fetch_all(statement, parameters)

            def fetch_all(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
                response = http.post(
                    f"/bigquery/v2/projects/{project}/queries",
                    json={"query": statement, "parameterMode": "NAMED", "queryParameters": parameters},
                )
                response.raise_for_status()
                payload = response.json()
                rows = []
                for row in list(payload.get("rows") or []):
                    if isinstance(row, dict):
                        rows.append(dict(row.get("f") or row))
                return rows

            def health(self) -> bool:
                return http_ping_ok(http, path=f"/bigquery/v2/projects/{project}/datasets/{dataset}")

        return _Client()

    return _resolve(
        implementation=relational_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpBigQueryRelationalStore(c),
    )


def create_motherduck_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    overrides = dict(config_overrides)
    if "dsn" not in overrides and "connection_string" not in overrides:
        config = HttpIntegrationConfig.from_env("INTERGRAX_MOTHERDUCK", **config_overrides)
        token = config.api_key or ""
        overrides["dsn"] = f"md:intergrax?motherduck_token={token}" if token else "md:intergrax"
    return _sql_store_factory(
        prefix="INTERGRAX_MOTHERDUCK",
        factory_name="create_motherduck_relational_store",
        driver="duckdb",
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        **overrides,
    )


__all__ = [
    "create_airbyte_workflow_orchestrator",
    "create_apify_browser_automation",
    "create_arxiv_search_provider",
    "create_bigquery_relational_store",
    "create_browserbase_browser_automation",
    "create_clerk_identity_provider",
    "create_google_drive_object_storage",
    "create_lancedb_vector_store",
    "create_llamaparse_document_parser",
    "create_motherduck_relational_store",
    "create_n8n_workflow_orchestrator",
    "create_okta_identity_provider",
    "create_perplexity_search_provider",
    "create_semantic_scholar_search_provider",
    "create_telegram_catalog_factory",
    "create_telegram_integration",
    "create_upstash_qstash_message_bus",
    "create_upstash_redis_key_value_cache",
    "create_wikipedia_wiki_knowledge",
]
