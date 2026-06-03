# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Phase M.6 P2/P3 integration providers."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations._shared.conformance import (
    assert_browser_automation,
    assert_collaboration_suite,
    assert_document_store,
    assert_issue_tracker,
    assert_key_value_cache,
    assert_message_bus,
    assert_notification_channel,
    assert_object_storage,
    assert_observability_backend,
    assert_relational_store,
    assert_search_provider,
    assert_wiki_knowledge,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.browser_automation import PageContent
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.object_storage.azure_blob.bundle import create_azure_blob_object_storage
from intergrax.integrations.providers.search_provider.brave.bundle import create_brave_search_provider
from intergrax.integrations.providers.relational_store.cloud_sql.bundle import create_cloud_sql_relational_store
from intergrax.integrations.providers.document_store.dynamodb.bundle import create_dynamodb_document_store
from intergrax.integrations.providers.key_value_cache.elasticache.bundle import create_elasticache_key_value_cache
from intergrax.integrations.providers.notification_channel.email_smtp.bundle import create_email_smtp_notification_channel
from intergrax.integrations.providers.object_storage.gcs.bundle import create_gcs_object_storage
from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker
from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import create_google_workspace_collaboration_suite
from intergrax.integrations.providers.issue_tracker.linear.bundle import create_linear_issue_tracker
from intergrax.integrations.providers.key_value_cache.memcached.bundle import create_memcached_key_value_cache
from intergrax.integrations.providers.relational_store.mssql.bundle import create_mssql_relational_store
from intergrax.integrations.providers.wiki_knowledge.notion.bundle import create_notion_wiki_knowledge
from intergrax.integrations.providers.relational_store.oracle.bundle import create_oracle_relational_store
from intergrax.integrations.providers.observability_backend.otel.bundle import create_otel_observability_backend
from intergrax.integrations.providers.browser_automation.playwright.bundle import create_playwright_browser_automation
from intergrax.integrations.providers.message_bus.pubsub.bundle import create_pubsub_message_bus
from intergrax.integrations.providers.search_provider.serpapi.bundle import create_serpapi_search_provider
from intergrax.integrations.providers.message_bus.service_bus.bundle import create_service_bus_message_bus
from intergrax.integrations.providers.wiki_knowledge.sharepoint.bundle import create_sharepoint_wiki_knowledge
from intergrax.integrations.providers.message_bus.sqs.bundle import create_sqs_message_bus
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import catalog_snapshot, clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.queueing.contracts.task_queue import TaskRequest
from intergrax.runtime.notifications.models import NotificationMessage

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeBlobContainer:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def upload_blob(self, name: str, data: bytes, **kwargs: Any) -> None:
        self._objects[name] = data

    def download_blob(self, key: str) -> Any:
        class _Blob:
            def __init__(self, data: bytes) -> None:
                self._data = data
                self.properties = type("P", (), {"content_settings": type("C", (), {"content_type": "text/plain"})(), "metadata": {}})()

            def readall(self) -> bytes:
                return self._data

        if key not in self._objects:
            raise FileNotFoundError(key)
        return _Blob(self._objects[key])

    def delete_blob(self, key: str) -> None:
        self._objects.pop(key, None)


class _FakeGcsBucket:
    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}

    def blob(self, key: str) -> Any:
        bucket = self

        class _Blob:
            def upload_from_string(self, body: bytes, *, content_type: str) -> None:
                bucket._objects[key] = {"body": body, "content_type": content_type}

            def exists(self) -> bool:
                return key in bucket._objects

            def download_as_bytes(self) -> bytes:
                return bucket._objects[key]["body"]

            @property
            def content_type(self) -> str:
                return bucket._objects[key]["content_type"]

        return _Blob()


class _FakeDynamoTable:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], dict[str, Any]] = {}

    def get_item(self, partition_key: str, row_key: str) -> Optional[dict[str, Any]]:
        return self._items.get((partition_key, row_key))

    def put_item(self, item: dict[str, Any]) -> None:
        self._items[(item["partition_key"], item["row_key"])] = item

    def delete_item(self, partition_key: str, row_key: str) -> None:
        self._items.pop((partition_key, row_key), None)

    def query(self, partition_key: str, *, limit: int, row_key_prefix: Optional[str]) -> list[dict[str, Any]]:
        rows = [v for (pk, _), v in self._items.items() if pk == partition_key]
        if row_key_prefix:
            rows = [r for r in rows if str(r.get("row_key", "")).startswith(row_key_prefix)]
        return rows[:limit]


class _FakeQueueClient:
    def send_message(self, *, body: bytes, attributes: dict[str, str]) -> str:
        return "msg-1"

    def get_message_status(self, message_id: str) -> str:
        return "pending"

    def get_message_result(self, message_id: str) -> Optional[bytes]:
        return None


class _FakeMemcached:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self._data.get(key)

    def set(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def set_if_absent(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> bool:
        if key in self._data:
            return False
        self._data[key] = value
        return True


class _FakeSqlConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        self.executed.append((sql, params))

        class _Cursor:
            def fetchall(self) -> list[dict[str, Any]]:
                return [{"id": 1}]

        return _Cursor()

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeHttpIssueClient:
    def get_issue(self, issue_key: str) -> dict[str, Any]:
        return {"key": issue_key, "summary": "Test", "status": "open", "url": "https://example/issue/1"}

    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        return {"id": "c1", "author": "bot"}

    def search_issues(self, jql: str, *, limit: int) -> list[dict[str, Any]]:
        return [{"key": "X-1", "summary": "Found", "status": "open"}]


class _FakeWikiClient:
    def get_page(self, page_id: str) -> dict[str, Any]:
        return {"id": page_id, "title": "Doc", "body": "Hello"}

    def search_pages(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        return [{"id": "p1", "title": query, "body": "snippet"}]


class _FakeGoogleClient:
    def get_message(self, user_id: str, message_id: str) -> dict[str, Any]:
        return {"id": message_id, "subject": "Hi", "snippet": "preview"}

    def list_messages(self, user_id: str, *, folder: str, limit: int) -> list[dict[str, Any]]:
        return [{"id": "m1", "subject": "Inbox"}]

    def send_mail(self, user_id: str, *, subject: str, body: str, to: list[str]) -> None:
        return None

    def list_calendar_events(self, user_id: str, *, start: str, end: str, limit: int) -> list[dict[str, Any]]:
        return [{"id": "e1", "summary": "Meet", "start": start, "end": end}]

    def get_user(self, user_id: str) -> dict[str, Any]:
        return {"id": user_id, "name": "User", "email": "u@example.com"}


class _FakeSearchClient:
    def search(self, query: str, limit: int) -> dict[str, Any]:
        return {"web": {"results": [{"title": query, "url": "https://example", "description": "hit"}]}}


class _FakeSerpClient:
    def search(self, query: str, limit: int) -> dict[str, Any]:
        return {"organic_results": [{"title": query, "link": "https://example", "snippet": "hit"}]}


class _FakeBrowser:
    def new_page(self) -> Any:
        browser = self

        class _Page:
            def goto(self, url: str, *, wait_until: str, timeout: int) -> Any:
                return type("R", (), {"status": 200})()

            def title(self) -> str:
                return "Title"

            def content(self) -> str:
                return "<html></html>"

            def inner_text(self, selector: str) -> str:
                return "body text"

            def close(self) -> None:
                return None

        return _Page()

    def close(self) -> None:
        return None


def test_azure_blob_object_storage() -> None:
    store = create_azure_blob_object_storage(container_client=_FakeBlobContainer(), container="docs")
    assert_object_storage(store)
    store.put("a.txt", b"hello", content_type="text/plain")
    obj = store.get("a.txt")
    assert obj is not None and obj.body == b"hello"


def test_gcs_object_storage() -> None:
    store = create_gcs_object_storage(gcs_bucket=_FakeGcsBucket(), bucket="b1")
    assert_object_storage(store)
    store.put("k", b"x")
    assert store.get("k") is not None


def test_dynamodb_document_store() -> None:
    table = _FakeDynamoTable()
    store = create_dynamodb_document_store(table=table)
    assert_document_store(store)
    store.put(DocumentRecord(partition_key="p1", row_key="r1", data={"x": 1}))
    doc = store.get("p1", "r1")
    assert doc is not None and doc.data["x"] == 1


@pytest.mark.parametrize(
    "factory",
    [
        create_sqs_message_bus,
        create_service_bus_message_bus,
        create_pubsub_message_bus,
    ],
)
def test_cloud_message_bus(factory: Any) -> None:
    bus = factory(client=_FakeQueueClient())
    assert_message_bus(bus)
    handle = bus.enqueue(
        TaskRequest(tenant_id="t1", run_id="r1", task_name="x", payload=b"p", idempotency_key=None)
    )
    assert handle.task_id == "msg-1"


def test_memcached_and_elasticache() -> None:
    for factory in (create_memcached_key_value_cache, create_elasticache_key_value_cache):
        cache = factory(client=_FakeMemcached())
        assert_key_value_cache(cache)
        cache.set("t1", "k", b"v")
        assert cache.get("t1", "k") == b"v"


@pytest.mark.parametrize(
    "factory",
    [
        create_oracle_relational_store,
        create_mssql_relational_store,
        create_cloud_sql_relational_store,
    ],
)
def test_sql_relational_store(factory: Any) -> None:
    store = factory(connection=_FakeSqlConnection())
    assert_relational_store(store)
    store.execute("SELECT 1")
    rows = store.fetch_all("SELECT 1")
    assert rows[0]["id"] == 1


@pytest.mark.asyncio
async def test_email_smtp_notification() -> None:
    sent: list[dict[str, Any]] = []

    def _sender(**kwargs: Any) -> None:
        sent.append(kwargs)

    channel = create_email_smtp_notification_channel(
        sender=_sender,
        from_address="noreply@example.com",
    )
    assert_notification_channel(channel)
    await channel.notify(
        NotificationMessage(
            tenant_id="t1",
            channel="#alerts",
            task_id="t1",
            subject="Alert",
            body="Body",
            metadata={"to": "ops@example.com"},
        )
    )
    assert sent[0]["to"] == "ops@example.com"


def test_otel_observability() -> None:
    class _Exporter:
        def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
            return 42.0

        def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
            return [{"timestamp": start, "value": 1.0}]

    backend = create_otel_observability_backend(exporter=_Exporter())
    assert_observability_backend(backend)
    result = backend.query_instant("up")
    assert result.series[0].points[0].value == 42.0


@pytest.mark.parametrize(
    "factory",
    [create_github_issue_tracker, create_linear_issue_tracker],
)
def test_issue_trackers(factory: Any) -> None:
    tracker = factory(client=_FakeHttpIssueClient())
    assert_issue_tracker(tracker)
    issue = tracker.get_issue("X-1")
    assert issue.summary == "Test"


@pytest.mark.parametrize(
    "factory",
    [create_notion_wiki_knowledge, create_sharepoint_wiki_knowledge],
)
def test_wiki_knowledge(factory: Any) -> None:
    wiki = factory(client=_FakeWikiClient())
    assert_wiki_knowledge(wiki)
    page = wiki.get_page("p1")
    assert page.title == "Doc"


def test_google_workspace() -> None:
    suite = create_google_workspace_collaboration_suite(client=_FakeGoogleClient())
    assert_collaboration_suite(suite)
    assert suite.get_user("u1").email == "u@example.com"


def test_search_providers() -> None:
    brave = create_brave_search_provider(client=_FakeSearchClient())
    assert_search_provider(brave)
    hits = brave.search("intergrax", limit=1)
    assert hits[0].url == "https://example"

    serp = create_serpapi_search_provider(client=_FakeSerpClient())
    assert_search_provider(serp)
    assert serp.search("q", limit=1)[0].provider == "serpapi"


def test_playwright_browser_automation() -> None:
    browser = create_playwright_browser_automation(browser=_FakeBrowser())
    assert_browser_automation(browser)
    page = browser.fetch_page("https://example.com")
    assert isinstance(page, PageContent)
    assert page.status_code == 200


def test_register_default_integrations_includes_p2_slugs() -> None:
    register_default_integrations()
    slugs = set(catalog_snapshot().keys())
    for slug in (
        "gcs",
        "dynamodb",
        "brave",
        "playwright",
        "azure_blob",
    ):
        assert slug in slugs


def test_resolve_gcs_via_profile() -> None:
    register_default_integrations()
    profile = IntegrationProfile(object_storage="gcs")
    store = resolve(
        IntegrationCategory.OBJECT_STORAGE,
        profile=profile,
        config={"gcs_bucket": _FakeGcsBucket(), "bucket": "lab"},
    )
    assert_object_storage(store)
