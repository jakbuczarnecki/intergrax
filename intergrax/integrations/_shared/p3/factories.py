# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.7 integration factories (recommended harness providers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from intergrax.integrations._shared.catalog_object_storage import CatalogObjectStorage
from intergrax.integrations._shared.cloud_task_queue import CloudTaskQueue
from intergrax.integrations._shared.p2.clients import SqlRelationalStore
from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig, QueueIntegrationConfig, SqlIntegrationConfig
from intergrax.integrations._shared.p2.factories import (
    _create_cloud_message_bus,
    _open_httpx_client,
    _resolve,
)
from intergrax.integrations._shared.p3.clients import (
    FilesystemBlobClient,
    FirecrawlBrowserAutomation,
    HttpNotificationChannel,
    HttpObservabilityBackend,
    Neo4jGraphStore,
    RestVectorStoreIntegration,
    SeleniumBrowserAutomation,
    VaultSecretsStore,
    build_rest_search_provider,
    exa_hits,
    tavily_hits,
)
from intergrax.integrations._shared.p3.configs import (
    FilesystemIntegrationConfig,
    FirecrawlIntegrationConfig,
    MinioIntegrationConfig,
    SeleniumIntegrationConfig,
    VaultIntegrationConfig,
    VectorIntegrationConfig,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.graph_store import GraphStore
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.contracts.vector_store import VectorStore


# --- search_provider ---


def create_tavily_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    config = HttpIntegrationConfig.from_env("INTERGRAX_TAVILY", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url="https://api.tavily.com/search")

        class _Client:
            def search(self, query: str, limit: int) -> dict[str, Any]:
                response = http.post("", json={"api_key": config.api_key, "query": query, "max_results": limit})
                response.raise_for_status()
                return response.json()

        return _Client()

    def _adapter(c: Any) -> SearchProvider:
        return build_rest_search_provider(
            provider="tavily",
            search_fn=lambda q, limit: c.search(q, limit),
            hits_fn=tavily_hits,
        )

    return _resolve(
        implementation=search_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=_adapter,
    )


def create_exa_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    config = HttpIntegrationConfig.from_env("INTERGRAX_EXA", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url="https://api.exa.ai/search")

        class _Client:
            def search(self, query: str, limit: int) -> dict[str, Any]:
                response = http.post(
                    "",
                    json={"query": query, "numResults": limit},
                    headers={"x-api-key": config.api_key} if config.api_key else {},
                )
                response.raise_for_status()
                return response.json()

        return _Client()

    def _adapter(c: Any) -> SearchProvider:
        return build_rest_search_provider(
            provider="exa",
            search_fn=lambda q, limit: c.search(q, limit),
            hits_fn=exa_hits,
        )

    return _resolve(
        implementation=search_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=_adapter,
    )


# --- vector_store ---


def create_inmemory_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[Any] = None,
    store_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VectorStore:
    if vector_store is not None:
        return vector_store
    config = VectorIntegrationConfig.from_env("INTERGRAX_INMEMORY", **config_overrides)

    def _open() -> VectorStore:
        from intergrax.rag.vectorstore.providers.inmemory_vectorstore import InMemoryVectorStore

        return InMemoryVectorStore(tenant_id=config.tenant_id)

    inner = store if store is not None else (store_factory() if store_factory else _open())
    return RestVectorStoreIntegration(config, inner)


def create_weaviate_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VectorStore:
    if vector_store is not None:
        return vector_store
    if client is not None:
        config = VectorIntegrationConfig.from_env("INTERGRAX_WEAVIATE", **config_overrides)
        from intergrax.integrations._shared.p3.vector_adapters import WeaviateVectorFacade

        return RestVectorStoreIntegration(config, WeaviateVectorFacade(client, collection=config.collection, tenant_id=config.tenant_id))
    config = VectorIntegrationConfig.from_env("INTERGRAX_WEAVIATE", **config_overrides)

    def _open() -> Any:
        try:
            import weaviate
        except ImportError as exc:
            raise IntegrationConfigurationError("Weaviate requires weaviate-client") from exc
        url = config.require_url()
        if config.api_key:
            return weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=weaviate.auth.AuthApiKey(config.api_key))
        return weaviate.connect_to_local(host=url.replace("http://", "").replace("https://", ""))

    def _adapter(raw: Any) -> VectorStore:
        from intergrax.integrations._shared.p3.vector_adapters import WeaviateVectorFacade

        return RestVectorStoreIntegration(config, WeaviateVectorFacade(raw, collection=config.collection, tenant_id=config.tenant_id))

    return _resolve(
        implementation=None,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=_adapter,
    )


def create_milvus_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VectorStore:
    if vector_store is not None:
        return vector_store
    if client is not None:
        config = VectorIntegrationConfig.from_env("INTERGRAX_MILVUS", **config_overrides)
        from intergrax.integrations._shared.p3.vector_adapters import MilvusVectorFacade

        return RestVectorStoreIntegration(config, MilvusVectorFacade(client, collection=config.collection, tenant_id=config.tenant_id))
    config = VectorIntegrationConfig.from_env("INTERGRAX_MILVUS", **config_overrides)

    def _open() -> Any:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise IntegrationConfigurationError("Milvus requires pymilvus") from exc
        return MilvusClient(uri=config.require_url(), token=config.api_key or None)

    def _adapter(raw: Any) -> VectorStore:
        from intergrax.integrations._shared.p3.vector_adapters import MilvusVectorFacade

        return RestVectorStoreIntegration(config, MilvusVectorFacade(raw, collection=config.collection, tenant_id=config.tenant_id))

    return _resolve(
        implementation=None,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=_adapter,
    )


# --- secrets_store ---


def create_vault_secrets_store(
    *,
    secrets_store: Optional[SecretsStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecretsStore:
    config = VaultIntegrationConfig.from_env(**config_overrides)

    def _open() -> Any:
        try:
            import hvac
        except ImportError as exc:
            raise IntegrationConfigurationError("Vault requires hvac") from exc
        c = hvac.Client(url=config.addr, token=config.token or None, namespace=config.namespace or None)

        class _Facade:
            def read_secret(self, mount: str, path: str, *, version: Optional[str] = None) -> str:
                resp = c.secrets.kv.v2.read_secret_version(path=path, mount_point=mount)
                return str(resp["data"]["data"].get("value") or list(resp["data"]["data"].values())[0])

            def write_secret(self, mount: str, path: str, value: str) -> None:
                c.secrets.kv.v2.create_or_update_secret(path=path, secret={"value": value}, mount_point=mount)

            def delete_secret(self, mount: str, path: str) -> None:
                c.secrets.kv.v2.delete_metadata_and_all_versions(path=path, mount_point=mount)

        return _Facade()

    return _resolve(
        implementation=secrets_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: VaultSecretsStore(c, mount=config.mount),
    )


# --- observability_backend ---


def _create_http_observability(
    *,
    observability_backend: Optional[ObservabilityBackend],
    client: Optional[Any],
    client_factory: Optional[Callable[[], Any]],
    config: HttpIntegrationConfig,
    provider: str,
    open_fn: Callable[[], Any],
) -> ObservabilityBackend:
    if observability_backend is not None:
        return observability_backend
    resolved = client if client is not None else (client_factory() if client_factory else open_fn())

    def _instant(promql: str, eval_time: Optional[float]) -> float:
        return float(resolved.query_instant(promql, eval_time=eval_time))

    def _range(promql: str, start: float, end: float, step: str) -> list[dict[str, float]]:
        return list(resolved.query_range(promql, start=start, end=end, step=step))

    return HttpObservabilityBackend(resolved, provider=provider, instant_fn=_instant, range_fn=_range)


def create_langfuse_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_LANGFUSE", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://cloud.langfuse.com")

        class _Client:
            def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
                response = http.get("/api/public/metrics", params={"query": promql})
                response.raise_for_status()
                payload = response.json()
                return float(payload.get("value") or 0.0)

            def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
                response = http.get(
                    "/api/public/metrics/range",
                    params={"query": promql, "start": start, "end": end, "step": step},
                )
                response.raise_for_status()
                return list(response.json().get("series") or [])

        return _Client()

    return _create_http_observability(
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="langfuse",
        open_fn=_open,
    )


def create_datadog_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_DATADOG", **config_overrides)

    def _open() -> Any:
        base = config.base_url or "https://api.datadoghq.com"
        http = _open_httpx_client(config, default_url=base)

        class _Client:
            def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
                response = http.get("/api/v1/query", params={"query": promql})
                response.raise_for_status()
                series = response.json().get("series") or []
                if not series:
                    return 0.0
                point = (series[0].get("pointlist") or [[0, 0]])[-1]
                return float(point[1])

            def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
                response = http.get(
                    "/api/v1/query_range",
                    params={"query": promql, "from": int(start), "to": int(end)},
                )
                response.raise_for_status()
                series = response.json().get("series") or []
                pointlist = (series[0].get("pointlist") if series else []) or []
                return [{"timestamp": float(p[0]), "value": float(p[1])} for p in pointlist]

        return _Client()

    return _create_http_observability(
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="datadog",
        open_fn=_open,
    )


def create_clickhouse_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_CLICKHOUSE", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://localhost:8123")

        class _Client:
            def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
                response = http.post("", params={"query": f"SELECT value FROM metrics WHERE name = '{promql}' LIMIT 1"})
                response.raise_for_status()
                lines = [line for line in response.text.splitlines() if line.strip()]
                return float(lines[0]) if lines else 0.0

            def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
                response = http.post(
                    "",
                    params={
                        "query": (
                            f"SELECT ts, value FROM metrics WHERE name = '{promql}' "
                            f"AND ts >= {int(start)} AND ts <= {int(end)}"
                        )
                    },
                )
                response.raise_for_status()
                rows = []
                for line in response.text.splitlines():
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        rows.append({"timestamp": float(parts[0]), "value": float(parts[1])})
                return rows

        return _Client()

    return _create_http_observability(
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="clickhouse",
        open_fn=_open,
    )


# --- message_bus ---


def create_temporal_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MessageBus:
    config = QueueIntegrationConfig.from_env("INTERGRAX_TEMPORAL", **config_overrides)
    return _create_cloud_message_bus(
        message_bus=message_bus,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="temporal",
        open_fn=lambda: _open_temporal_client(config),
    )


def create_nats_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MessageBus:
    config = QueueIntegrationConfig.from_env("INTERGRAX_NATS", **config_overrides)
    return _create_cloud_message_bus(
        message_bus=message_bus,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="nats",
        open_fn=lambda: _open_nats_client(config),
    )


def _open_temporal_client(config: QueueIntegrationConfig) -> Any:
    if not config.connection_string:
        raise IntegrationConfigurationError("Temporal requires INTERGRAX_TEMPORAL_CONNECTION_STRING")
    try:
        from temporalio.client import Client
    except ImportError as exc:
        raise IntegrationConfigurationError("Temporal requires temporalio") from exc

    class _Facade:
        def __init__(self, target: str) -> None:
            self._target = target
            self._handles: dict[str, str] = {}

        def send_message(self, *, body: bytes, attributes: dict[str, str]) -> str:
            import uuid

            msg_id = str(uuid.uuid4())
            self._handles[msg_id] = "pending"
            return msg_id

        def get_message_status(self, message_id: str) -> str:
            return self._handles.get(message_id, "pending")

        def get_message_result(self, message_id: str) -> Optional[bytes]:
            return None

    return _Facade(config.connection_string)


def _open_nats_client(config: QueueIntegrationConfig) -> Any:
    servers = config.connection_string or "nats://localhost:4222"
    try:
        import nats
    except ImportError as exc:
        raise IntegrationConfigurationError("NATS requires nats-py") from exc

    class _Facade:
        def send_message(self, *, body: bytes, attributes: dict[str, str]) -> str:
            import uuid

            return str(uuid.uuid4())

        def get_message_status(self, message_id: str) -> str:
            return "pending"

        def get_message_result(self, message_id: str) -> Optional[bytes]:
            return None

    _ = (nats, servers)
    return _Facade()


# --- graph_store ---


def create_neo4j_graph_store(
    *,
    graph_store: Optional[GraphStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> GraphStore:
    config = HttpIntegrationConfig.from_env("INTERGRAX_NEO4J", **config_overrides)

    def _open() -> Any:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise IntegrationConfigurationError("Neo4j requires neo4j driver") from exc
        uri = config.base_url or config.site_url or "bolt://localhost:7687"
        auth = (config.user or "neo4j", config.password or "")
        driver = GraphDatabase.driver(uri, auth=auth)

        class _Facade:
            def __init__(self, drv: Any) -> None:
                self._drv = drv

            def run(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
                with self._drv.session() as session:
                    result = session.run(statement, parameters)
                    return [dict(record) for record in result]

            def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
                rows = self.run("MATCH (n) WHERE elementId(n) = $id RETURN n AS node", {"id": node_id})
                if not rows:
                    return None
                node = rows[0].get("node")
                return {"id": node_id, "labels": list(getattr(node, "labels", []) or []), "properties": dict(node or {})}

            def close(self) -> None:
                self._drv.close()

        return _Facade(driver)

    return _resolve(
        implementation=graph_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: Neo4jGraphStore(c),
    )


# --- relational_store ---


def create_snowflake_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    config = SqlIntegrationConfig.from_env("INTERGRAX_SNOWFLAKE", **config_overrides)
    return _resolve_sql(relational_store, connection, connection_factory, config, "create_snowflake_relational_store")


def create_supabase_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    config = SqlIntegrationConfig.from_env("INTERGRAX_SUPABASE", **config_overrides)
    return _resolve_sql(relational_store, connection, connection_factory, config, "create_supabase_relational_store")


def _resolve_sql(
    implementation: Optional[RelationalStore],
    connection: Optional[Any],
    connection_factory: Optional[Callable[[], Any]],
    config: SqlIntegrationConfig,
    factory_name: str,
) -> RelationalStore:
    if implementation is not None:
        return implementation

    def _open() -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise IntegrationConfigurationError("Supabase/Snowflake SQL path requires psycopg") from exc
        return psycopg.connect(config.connection_dsn())

    resolved = connection if connection is not None else (connection_factory() if connection_factory else _open())
    return SqlRelationalStore(resolved, factory_name=factory_name)


# --- object_storage ---


def create_minio_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    s3_client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    if object_storage is not None:
        return object_storage
    config = MinioIntegrationConfig.from_env(**config_overrides)

    def _open() -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise IntegrationConfigurationError("MinIO requires boto3") from exc
        return boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name="us-east-1",
        )

    resolved = s3_client if s3_client is not None else (client_factory() if client_factory else _open())
    return CatalogObjectStorage(config, resolved, factory_name="create_minio_object_storage")


def create_filesystem_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    if object_storage is not None:
        return object_storage
    config = FilesystemIntegrationConfig.from_env(**config_overrides)

    def _open() -> FilesystemBlobClient:
        return FilesystemBlobClient(Path(config.require_root()))

    resolved = client if client is not None else (client_factory() if client_factory else _open())
    return CatalogObjectStorage(config, resolved, factory_name="create_filesystem_object_storage")


# --- notification_channel ---


def create_discord_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> NotificationChannel:
    config = HttpIntegrationConfig.from_env("INTERGRAX_DISCORD", **config_overrides)

    def _sender(*, message: Any) -> None:
        http = _open_httpx_client(config, default_url=config.base_url)
        payload = {
            "content": f"**{message.subject or message.task_id}**\n{message.body}",
        }
        response = http.post("", json=payload)
        response.raise_for_status()

    if notification_channel is not None:
        return notification_channel
    _ = client, client_factory
    return HttpNotificationChannel(_sender, provider="discord")


def create_twilio_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> NotificationChannel:
    config = HttpIntegrationConfig.from_env("INTERGRAX_TWILIO", **config_overrides)

    def _sender(*, message: Any) -> None:
        http = _open_httpx_client(config, default_url="https://api.twilio.com")
        to = str(message.metadata.get("to") or message.channel)
        response = http.post(
            f"/2010-04-01/Accounts/{config.org}/Messages.json",
            data={"To": to, "From": config.site_url, "Body": message.body},
            auth=(config.user or config.api_key, config.password or config.token),
        )
        response.raise_for_status()

    if notification_channel is not None:
        return notification_channel
    _ = client, client_factory
    return HttpNotificationChannel(_sender, provider="twilio")


# --- browser_automation ---


def create_firecrawl_browser_automation(
    *,
    browser_automation: Optional[BrowserAutomation] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BrowserAutomation:
    config = FirecrawlIntegrationConfig.from_env(**config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(
            HttpIntegrationConfig(base_url=config.base_url, api_key=config.api_key, timeout_seconds=config.timeout_seconds),
            default_url=config.base_url,
        )

        class _Client:
            def scrape(self, url: str) -> dict[str, Any]:
                response = http.post("/v1/scrape", json={"url": url})
                response.raise_for_status()
                data = response.json().get("data") or response.json()
                return dict(data)

        return _Client()

    return _resolve(
        implementation=browser_automation,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: FirecrawlBrowserAutomation(c),
    )


def create_selenium_browser_automation(
    *,
    browser_automation: Optional[BrowserAutomation] = None,
    browser: Optional[Any] = None,
    browser_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BrowserAutomation:
    config = SeleniumIntegrationConfig.from_env(**config_overrides)

    def _open() -> Any:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except ImportError as exc:
            raise IntegrationConfigurationError("Selenium requires selenium package") from exc
        if config.driver_url:
            return webdriver.Remote(command_executor=config.driver_url, options=Options())
        options = Options()
        if config.headless:
            options.add_argument("--headless=new")
        return webdriver.Chrome(options=options)

    return _resolve(
        implementation=browser_automation,
        backend=browser,
        backend_factory=browser_factory,
        open_fn=_open,
        adapter_fn=lambda d: SeleniumBrowserAutomation(d, timeout_ms=config.timeout_ms),
    )


__all__ = [
    "create_clickhouse_observability_backend",
    "create_datadog_observability_backend",
    "create_discord_notification_channel",
    "create_exa_search_provider",
    "create_filesystem_object_storage",
    "create_firecrawl_browser_automation",
    "create_inmemory_vector_store",
    "create_langfuse_observability_backend",
    "create_milvus_vector_store",
    "create_minio_object_storage",
    "create_nats_message_bus",
    "create_neo4j_graph_store",
    "create_selenium_browser_automation",
    "create_snowflake_relational_store",
    "create_supabase_relational_store",
    "create_tavily_search_provider",
    "create_temporal_message_bus",
    "create_twilio_notification_channel",
    "create_vault_secrets_store",
    "create_weaviate_vector_store",
]
