# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""P2/P3 integration factory functions (thin provider shells delegate here)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations._shared.catalog_object_storage import CatalogObjectStorage
from intergrax.integrations._shared.cloud_task_queue import CloudTaskQueue
from intergrax.integrations._shared.p2.clients import (
    DynamoDBDocumentStore,
    GoogleWorkspaceCollaborationSuite,
    MemcachedKeyValueCache,
    OtelObservabilityBackend,
    PlaywrightBrowserAutomation,
    RestIssueTracker,
    RestSearchProvider,
    RestWikiKnowledge,
    SmtpNotificationChannel,
    SqlRelationalStore,
    brave_hits,
    serpapi_hits,
)
from intergrax.integrations._shared.p2.configs import (
    DynamoDBIntegrationConfig,
    GcsIntegrationConfig,
    HttpIntegrationConfig,
    MemcachedIntegrationConfig,
    OtelIntegrationConfig,
    PlaywrightIntegrationConfig,
    QueueIntegrationConfig,
    SmtpIntegrationConfig,
    SqlIntegrationConfig,
)
from intergrax.integrations._shared.p2.gcs_blob import build_gcs_object_storage, open_gcs_bucket
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.providers.object_storage.azure_blob.bundle import create_azure_blob_object_storage


def _resolve[T](
    *,
    implementation: Optional[T],
    backend: Optional[Any],
    backend_factory: Optional[Callable[[], Any]],
    open_fn: Callable[[], Any],
    adapter_fn: Callable[[Any], T],
) -> T:
    if implementation is not None:
        return implementation
    resolved = backend if backend is not None else (backend_factory() if backend_factory else open_fn())
    return adapter_fn(resolved)


# --- object_storage ---


def create_gcs_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    gcs_bucket: Optional[Any] = None,
    bucket_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    if object_storage is not None:
        return object_storage
    config = GcsIntegrationConfig.from_env(**config_overrides)
    resolved_bucket = gcs_bucket if gcs_bucket is not None else (bucket_factory() if bucket_factory else open_gcs_bucket(config))
    return build_gcs_object_storage(config, resolved_bucket)


# --- document_store ---


def create_dynamodb_document_store(
    *,
    document_store: Optional[DocumentStore] = None,
    table: Optional[Any] = None,
    table_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> DocumentStore:
    config = DynamoDBIntegrationConfig.from_env(**config_overrides)

    def _open() -> DynamoDBDocumentStore:
        resolved = table if table is not None else (table_factory() if table_factory else _open_dynamodb_table(config))
        return DynamoDBDocumentStore(
            resolved,
            partition_attr=config.partition_attr,
            sort_attr=config.sort_attr,
        )

    return _resolve(
        implementation=document_store,
        backend=table,
        backend_factory=table_factory,
        open_fn=_open,
        adapter_fn=lambda x: x if isinstance(x, DynamoDBDocumentStore) else DynamoDBDocumentStore(
            x, partition_attr=config.partition_attr, sort_attr=config.sort_attr
        ),
    )


def _open_dynamodb_table(config: DynamoDBIntegrationConfig) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "DynamoDB integration requires boto3. Install with: uv pip install boto3"
        ) from exc
    kwargs: dict[str, Any] = {}
    if config.region:
        kwargs["region_name"] = config.region
    resource = boto3.resource("dynamodb", **kwargs)
    return _DynamoDBTableFacade(
        resource.Table(config.table_name),
        partition_attr=config.partition_attr,
        sort_attr=config.sort_attr,
    )


class _DynamoDBTableFacade:
    def __init__(self, table: Any, *, partition_attr: str, sort_attr: str) -> None:
        self._table = table
        self._partition_attr = partition_attr
        self._sort_attr = sort_attr

    def get_item(self, partition_key: str, row_key: str) -> Optional[dict[str, Any]]:
        response = self._table.get_item(
            Key={self._partition_attr: partition_key, self._sort_attr: row_key},
        )
        item = response.get("Item")
        return dict(item) if item else None

    def put_item(self, item: dict[str, Any]) -> None:
        self._table.put_item(Item=item)

    def delete_item(self, partition_key: str, row_key: str) -> None:
        self._table.delete_item(
            Key={self._partition_attr: partition_key, self._sort_attr: row_key},
        )

    def query(
        self,
        partition_key: str,
        *,
        limit: int,
        row_key_prefix: Optional[str],
    ) -> list[dict[str, Any]]:
        from boto3.dynamodb.conditions import Key

        kwargs: dict[str, Any] = {"Limit": limit}
        if row_key_prefix:
            kwargs["KeyConditionExpression"] = Key(self._partition_attr).eq(partition_key) & Key(self._sort_attr).begins_with(
                row_key_prefix
            )
        else:
            kwargs["KeyConditionExpression"] = Key(self._partition_attr).eq(partition_key)
        response = self._table.query(**kwargs)
        return [dict(item) for item in response.get("Items", [])]


# --- message_bus ---


def create_sqs_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MessageBus:
    config = QueueIntegrationConfig.from_env("INTERGRAX_SQS", **config_overrides)
    return _create_cloud_message_bus(
        message_bus=message_bus,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="sqs",
        open_fn=lambda: _open_sqs_client(config),
    )


def create_service_bus_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MessageBus:
    config = QueueIntegrationConfig.from_env("INTERGRAX_SERVICE_BUS", **config_overrides)
    return _create_cloud_message_bus(
        message_bus=message_bus,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="service_bus",
        open_fn=lambda: _open_service_bus_client(config),
    )


def create_pubsub_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MessageBus:
    config = QueueIntegrationConfig.from_env("INTERGRAX_PUBSUB", **config_overrides)
    return _create_cloud_message_bus(
        message_bus=message_bus,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="pubsub",
        open_fn=lambda: _open_pubsub_client(config),
    )


def _create_cloud_message_bus(
    *,
    message_bus: Optional[MessageBus],
    client: Optional[Any],
    client_factory: Optional[Callable[[], Any]],
    config: QueueIntegrationConfig,
    provider: str,
    open_fn: Callable[[], Any],
) -> MessageBus:
    if message_bus is not None:
        return message_bus
    resolved = client if client is not None else (client_factory() if client_factory else open_fn())
    return CloudTaskQueue(resolved, provider=provider)


def _open_sqs_client(config: QueueIntegrationConfig) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise IntegrationConfigurationError("SQS requires boto3") from exc
    kwargs: dict[str, Any] = {}
    if config.region:
        kwargs["region_name"] = config.region
    sqs = boto3.client("sqs", **kwargs)
    queue_url = sqs.get_queue_url(QueueName=config.queue_name)["QueueUrl"]
    return _SqsFacade(sqs, queue_url)


class _SqsFacade:
    def __init__(self, client: Any, queue_url: str) -> None:
        self._client = client
        self._queue_url = queue_url

    def send_message(self, *, body: bytes, attributes: dict[str, str]) -> str:
        response = self._client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=body.decode() if isinstance(body, bytes) else str(body),
            MessageAttributes={
                key: {"DataType": "String", "StringValue": value} for key, value in attributes.items()
            },
        )
        return str(response["MessageId"])

    def get_message_status(self, message_id: str) -> str:
        return "pending"

    def get_message_result(self, message_id: str) -> Optional[bytes]:
        return None


def _open_service_bus_client(config: QueueIntegrationConfig) -> Any:
    if not config.connection_string:
        raise IntegrationConfigurationError("Service Bus requires INTERGRAX_SERVICE_BUS_CONNECTION_STRING")
    try:
        from azure.servicebus import ServiceBusClient, ServiceBusMessage
    except ImportError as exc:
        raise IntegrationConfigurationError("Service Bus requires azure-servicebus") from exc
    client = ServiceBusClient.from_connection_string(config.connection_string)
    return _ServiceBusFacade(client, config.queue_name, ServiceBusMessage)


class _ServiceBusFacade:
    def __init__(self, client: Any, queue_name: str, message_cls: Any) -> None:
        self._client = client
        self._queue_name = queue_name
        self._message_cls = message_cls

    def send_message(self, *, body: bytes, attributes: dict[str, str]) -> str:
        import uuid

        message_id = str(uuid.uuid4())
        with self._client.get_queue_sender(queue_name=self._queue_name) as sender:
            sender.send_messages(self._message_cls(body.decode(), message_id=message_id))
        return message_id

    def get_message_status(self, message_id: str) -> str:
        return "pending"

    def get_message_result(self, message_id: str) -> Optional[bytes]:
        return None


def _open_pubsub_client(config: QueueIntegrationConfig) -> Any:
    try:
        from google.cloud import pubsub_v1
    except ImportError as exc:
        raise IntegrationConfigurationError("Pub/Sub requires google-cloud-pubsub") from exc
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(config.project_id or "", config.topic)
    return _PubSubFacade(publisher, topic_path)


class _PubSubFacade:
    def __init__(self, publisher: Any, topic_path: str) -> None:
        self._publisher = publisher
        self._topic_path = topic_path

    def send_message(self, *, body: bytes, attributes: dict[str, str]) -> str:
        future = self._publisher.publish(self._topic_path, body, **{k: str(v) for k, v in attributes.items()})
        return str(future.result())

    def get_message_status(self, message_id: str) -> str:
        return "pending"

    def get_message_result(self, message_id: str) -> Optional[bytes]:
        return None


# --- key_value_cache ---


def create_memcached_key_value_cache(
    *,
    key_value_cache: Optional[KeyValueCache] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> KeyValueCache:
    config = MemcachedIntegrationConfig.from_env(**config_overrides)
    return _resolve(
        implementation=key_value_cache,
        backend=client,
        backend_factory=client_factory,
        open_fn=lambda: _open_memcached_client(config),
        adapter_fn=lambda c: MemcachedKeyValueCache(c),
    )


def create_elasticache_key_value_cache(
    *,
    key_value_cache: Optional[KeyValueCache] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> KeyValueCache:
    """ElastiCache Redis-compatible endpoint — same adapter as memcached-style duck client."""
    return create_memcached_key_value_cache(
        key_value_cache=key_value_cache,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def _open_memcached_client(config: MemcachedIntegrationConfig) -> Any:
    try:
        from pymemcache.client.base import Client
    except ImportError as exc:
        raise IntegrationConfigurationError("Memcached requires pymemcache") from exc
    raw = Client((config.host, config.port))

    class _Facade:
        def get(self, key: str) -> Optional[bytes]:
            value = raw.get(key)
            return value if isinstance(value, bytes) else None

        def set(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> None:
            raw.set(key, value, expire=ttl_seconds or 0)

        def delete(self, key: str) -> None:
            raw.delete(key)

        def set_if_absent(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> bool:
            return bool(raw.add(key, value, expire=ttl_seconds or 0))

    return _Facade()


# --- relational_store ---


def create_oracle_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    return _create_sql_store(
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        prefix="INTERGRAX_ORACLE",
        factory_name="create_oracle_relational_store",
        driver="oracledb",
        **config_overrides,
    )


def create_mssql_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    return _create_sql_store(
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        prefix="INTERGRAX_MSSQL",
        factory_name="create_mssql_relational_store",
        driver="pyodbc",
        **config_overrides,
    )


def create_azure_sql_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    return _create_sql_store(
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        prefix="INTERGRAX_AZURE_SQL",
        factory_name="create_azure_sql_relational_store",
        driver="pyodbc",
        **config_overrides,
    )


def create_cloud_sql_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    return _create_sql_store(
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        prefix="INTERGRAX_CLOUD_SQL",
        factory_name="create_cloud_sql_relational_store",
        driver="pg8000",
        **config_overrides,
    )


def _create_sql_store(
    *,
    relational_store: Optional[RelationalStore],
    connection: Optional[Any],
    connection_factory: Optional[Callable[[], Any]],
    prefix: str,
    factory_name: str,
    driver: str,
    **config_overrides: object,
) -> RelationalStore:
    config = SqlIntegrationConfig.from_env(prefix, **config_overrides)

    def _open() -> Any:
        dsn = config.connection_dsn()
        if driver == "oracledb":
            import oracledb

            return oracledb.connect(dsn)
        if driver == "pyodbc":
            import pyodbc

            return pyodbc.connect(dsn)
        if driver == "pg8000":
            import pg8000.dbapi

            return pg8000.dbapi.connect(dsn)
        raise IntegrationConfigurationError(f"Unsupported SQL driver: {driver}")

    return _resolve(
        implementation=relational_store,
        backend=connection,
        backend_factory=connection_factory,
        open_fn=_open,
        adapter_fn=lambda c: SqlRelationalStore(c, factory_name=factory_name),
    )


# --- notification_channel ---


def create_email_smtp_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    sender: Optional[Callable[..., None]] = None,
    sender_factory: Optional[Callable[[], Callable[..., None]]] = None,
    **config_overrides: object,
) -> NotificationChannel:
    config = SmtpIntegrationConfig.from_env(**config_overrides)

    def _open_sender() -> Callable[..., None]:
        import smtplib
        from email.message import EmailMessage

        if not config.smtp_host:
            raise IntegrationConfigurationError("SMTP requires INTERGRAX_EMAIL_SMTP_HOST")

        def _send(*, from_address: str, to: str, subject: str, body: str, metadata: dict[str, str]) -> None:
            msg = EmailMessage()
            msg["From"] = from_address or config.from_address
            msg["To"] = to
            msg["Subject"] = subject
            msg.set_content(body)
            with smtplib.SMTP(config.smtp_host, config.smtp_port) as smtp:
                if config.use_tls:
                    smtp.starttls()
                if config.user:
                    smtp.login(config.user, config.password)
                smtp.send_message(msg)

        return _send

    resolved_sender = sender if sender is not None else (sender_factory() if sender_factory else _open_sender())
    if notification_channel is not None:
        return notification_channel
    return SmtpNotificationChannel(resolved_sender, from_address=config.from_address or config.user)


# --- observability_backend ---


def create_otel_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    exporter: Optional[Any] = None,
    exporter_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    config = OtelIntegrationConfig.from_env(**config_overrides)

    def _open() -> Any:
        class _NoopExporter:
            def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
                return 0.0

            def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
                return []

        return _NoopExporter()

    return _resolve(
        implementation=observability_backend,
        backend=exporter,
        backend_factory=exporter_factory,
        open_fn=_open,
        adapter_fn=lambda e: OtelObservabilityBackend(e),
    )


# --- issue_tracker ---


def create_github_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    config = HttpIntegrationConfig.from_env("INTERGRAX_GITHUB", **config_overrides)
    return _create_rest_issue_tracker(
        issue_tracker=issue_tracker,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="github",
        base_path=f"/repos/{config.org}/{config.repo}",
    )


def create_linear_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    config = HttpIntegrationConfig.from_env("INTERGRAX_LINEAR", **config_overrides)
    return _create_rest_issue_tracker(
        issue_tracker=issue_tracker,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="linear",
        base_path="/issues",
    )


def create_azure_devops_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    config = HttpIntegrationConfig.from_env("INTERGRAX_AZURE_DEVOPS", **config_overrides)
    return _create_rest_issue_tracker(
        issue_tracker=issue_tracker,
        client=client,
        client_factory=client_factory,
        config=config,
        provider="azure_devops",
        base_path=f"/{config.org}/{config.repo}/_apis/wit/workitems",
    )


def _create_rest_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker],
    client: Optional[Any],
    client_factory: Optional[Callable[[], Any]],
    config: HttpIntegrationConfig,
    provider: str,
    base_path: str,
) -> IssueTracker:
    return _resolve(
        implementation=issue_tracker,
        backend=client,
        backend_factory=client_factory,
        open_fn=lambda: _open_http_issue_client(config, base_path=base_path),
        adapter_fn=lambda c: RestIssueTracker(c, provider=provider),
    )


def _open_http_issue_client(config: HttpIntegrationConfig, *, base_path: str) -> Any:
    http = _open_httpx_client(config)

    class _Client:
        def get_issue(self, issue_key: str) -> dict[str, Any]:
            response = http.get(f"{base_path}/{issue_key}")
            response.raise_for_status()
            return response.json()

        def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
            response = http.post(f"{base_path}/{issue_key}/comments", json={"body": body})
            response.raise_for_status()
            return response.json()

        def search_issues(self, jql: str, *, limit: int) -> list[dict[str, Any]]:
            response = http.get(f"{base_path}/search", params={"q": jql, "limit": limit})
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            return list(payload.get("issues") or payload.get("items") or [])

        def health(self) -> bool:
            from intergrax.integrations._shared.health import http_ping_ok

            return http_ping_ok(http, path="/zen")

    return _Client()


# --- wiki_knowledge ---


def create_notion_wiki_knowledge(
    *,
    wiki_knowledge: Optional[WikiKnowledge] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WikiKnowledge:
    config = HttpIntegrationConfig.from_env("INTERGRAX_NOTION", **config_overrides)
    return _create_rest_wiki(
        wiki_knowledge=wiki_knowledge,
        client=client,
        client_factory=client_factory,
        config=config,
        base_path="/v1/pages",
    )


def create_sharepoint_wiki_knowledge(
    *,
    wiki_knowledge: Optional[WikiKnowledge] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WikiKnowledge:
    config = HttpIntegrationConfig.from_env("INTERGRAX_SHAREPOINT", **config_overrides)
    return _create_rest_wiki(
        wiki_knowledge=wiki_knowledge,
        client=client,
        client_factory=client_factory,
        config=config,
        base_path="/sites/pages",
    )


def _create_rest_wiki(
    *,
    wiki_knowledge: Optional[WikiKnowledge],
    client: Optional[Any],
    client_factory: Optional[Callable[[], Any]],
    config: HttpIntegrationConfig,
    base_path: str,
) -> WikiKnowledge:
    return _resolve(
        implementation=wiki_knowledge,
        backend=client,
        backend_factory=client_factory,
        open_fn=lambda: _open_http_wiki_client(config, base_path=base_path),
        adapter_fn=lambda c: RestWikiKnowledge(c),
    )


def _open_http_wiki_client(config: HttpIntegrationConfig, *, base_path: str) -> Any:
    http = _open_httpx_client(config)

    class _Client:
        def get_page(self, page_id: str) -> dict[str, Any]:
            response = http.get(f"{base_path}/{page_id}")
            response.raise_for_status()
            return response.json()

        def search_pages(self, query: str, *, limit: int) -> list[dict[str, Any]]:
            response = http.get(f"{base_path}/search", params={"q": query, "limit": limit})
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            return list(payload.get("pages") or payload.get("results") or [])

    return _Client()


# --- collaboration_suite ---


def create_google_workspace_collaboration_suite(
    *,
    collaboration_suite: Optional[CollaborationSuite] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CollaborationSuite:
    config = HttpIntegrationConfig.from_env("INTERGRAX_GOOGLE_WORKSPACE", **config_overrides)
    return _resolve(
        implementation=collaboration_suite,
        backend=client,
        backend_factory=client_factory,
        open_fn=lambda: _open_google_workspace_client(config),
        adapter_fn=lambda c: GoogleWorkspaceCollaborationSuite(c),
    )


def _open_google_workspace_client(config: HttpIntegrationConfig) -> Any:
    http = _open_httpx_client(config)
    base = config.base_url or "https://www.googleapis.com"

    class _Client:
        def get_message(self, user_id: str, message_id: str) -> dict[str, Any]:
            response = http.get(f"{base}/gmail/v1/users/{user_id}/messages/{message_id}")
            response.raise_for_status()
            return response.json()

        def list_messages(self, user_id: str, *, folder: str, limit: int) -> list[dict[str, Any]]:
            response = http.get(
                f"{base}/gmail/v1/users/{user_id}/messages",
                params={"labelIds": folder, "maxResults": limit},
            )
            response.raise_for_status()
            return list(response.json().get("messages") or [])

        def send_mail(self, user_id: str, *, subject: str, body: str, to: list[str]) -> None:
            import base64

            raw = base64.urlsafe_b64encode(f"Subject: {subject}\n\n{body}".encode()).decode()
            response = http.post(
                f"{base}/gmail/v1/users/{user_id}/messages/send",
                json={"raw": raw, "to": to},
            )
            response.raise_for_status()

        def list_calendar_events(
            self, user_id: str, *, start: str, end: str, limit: int
        ) -> list[dict[str, Any]]:
            response = http.get(
                f"{base}/calendar/v3/calendars/{user_id}/events",
                params={"timeMin": start, "timeMax": end, "maxResults": limit},
            )
            response.raise_for_status()
            return list(response.json().get("items") or [])

        def get_user(self, user_id: str) -> dict[str, Any]:
            response = http.get(f"{base}/admin/directory/v1/users/{user_id}")
            response.raise_for_status()
            return response.json()

    return _Client()


# --- search_provider ---


def create_brave_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    config = HttpIntegrationConfig.from_env("INTERGRAX_BRAVE", **config_overrides)

    def _adapter(c: Any) -> RestSearchProvider:
        return RestSearchProvider(
            provider="brave",
            search_fn=lambda q, limit: c.search(q, limit),
            hits_fn=brave_hits,
        )

    return _resolve(
        implementation=search_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=lambda: _open_brave_client(config),
        adapter_fn=_adapter,
    )


def create_serpapi_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    config = HttpIntegrationConfig.from_env("INTERGRAX_SERPAPI", **config_overrides)

    def _adapter(c: Any) -> RestSearchProvider:
        return RestSearchProvider(
            provider="serpapi",
            search_fn=lambda q, limit: c.search(q, limit),
            hits_fn=serpapi_hits,
        )

    return _resolve(
        implementation=search_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=lambda: _open_serpapi_client(config),
        adapter_fn=_adapter,
    )


def _open_brave_client(config: HttpIntegrationConfig) -> Any:
    http = _open_httpx_client(config, default_url="https://api.search.brave.com/res/v1/web/search")

    class _Client:
        def search(self, query: str, limit: int) -> dict[str, Any]:
            response = http.get("", params={"q": query, "count": limit})
            response.raise_for_status()
            return response.json()

    return _Client()


def _open_serpapi_client(config: HttpIntegrationConfig) -> Any:
    http = _open_httpx_client(config, default_url="https://serpapi.com/search.json")

    class _Client:
        def search(self, query: str, limit: int) -> dict[str, Any]:
            response = http.get("", params={"q": query, "num": limit, "api_key": config.api_key})
            response.raise_for_status()
            return response.json()

    return _Client()


# --- browser_automation ---


def create_playwright_browser_automation(
    *,
    browser_automation: Optional[BrowserAutomation] = None,
    browser: Optional[Any] = None,
    browser_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BrowserAutomation:
    config = PlaywrightIntegrationConfig.from_env(**config_overrides)

    def _open() -> Any:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise IntegrationConfigurationError("Playwright requires playwright package") from exc
        pw = sync_playwright().start()
        return pw.chromium.launch(headless=config.headless)

    return _resolve(
        implementation=browser_automation,
        backend=browser,
        backend_factory=browser_factory,
        open_fn=_open,
        adapter_fn=lambda b: PlaywrightBrowserAutomation(b, timeout_ms=config.timeout_ms),
    )


# --- shared HTTP ---


def _open_httpx_client(config: HttpIntegrationConfig, *, default_url: str = "") -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise IntegrationConfigurationError("HTTP integrations require httpx") from exc
    base_url = config.base_url or default_url or config.site_url
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if config.token:
        headers["Authorization"] = f"token {config.token}"
    return httpx.Client(base_url=base_url, headers=headers, timeout=config.timeout_seconds)


__all__ = [
    "create_azure_blob_object_storage",
    "create_azure_devops_issue_tracker",
    "create_azure_sql_relational_store",
    "create_brave_search_provider",
    "create_cloud_sql_relational_store",
    "create_dynamodb_document_store",
    "create_elasticache_key_value_cache",
    "create_email_smtp_notification_channel",
    "create_gcs_object_storage",
    "create_github_issue_tracker",
    "create_google_workspace_collaboration_suite",
    "create_linear_issue_tracker",
    "create_memcached_key_value_cache",
    "create_mssql_relational_store",
    "create_oracle_relational_store",
    "create_otel_observability_backend",
    "create_playwright_browser_automation",
    "create_pubsub_message_bus",
    "create_serpapi_search_provider",
    "create_service_bus_message_bus",
    "create_sharepoint_wiki_knowledge",
    "create_sqs_message_bus",
    "create_notion_wiki_knowledge",
]
