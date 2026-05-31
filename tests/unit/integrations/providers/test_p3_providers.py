# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Phase M.7 harness integration providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from intergrax.integrations._shared.conformance import (
    assert_browser_automation,
    assert_graph_store,
    assert_message_bus,
    assert_notification_channel,
    assert_object_storage,
    assert_observability_backend,
    assert_relational_store,
    assert_search_provider,
    assert_secrets_store,
    assert_vector_store,
)
from intergrax.integrations.contracts.browser_automation import PageContent
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.browser_automation.firecrawl.bundle import create_firecrawl_browser_automation
from intergrax.integrations.providers.browser_automation.selenium.bundle import create_selenium_browser_automation
from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store
from intergrax.integrations.providers.message_bus.nats.bundle import create_nats_message_bus
from intergrax.integrations.providers.message_bus.temporal.bundle import create_temporal_message_bus
from intergrax.integrations.providers.notification_channel.discord.bundle import create_discord_notification_channel
from intergrax.integrations.providers.object_storage.filesystem.bundle import create_filesystem_object_storage
from intergrax.integrations.providers.observability_backend.clickhouse.bundle import create_clickhouse_observability_backend
from intergrax.integrations.providers.observability_backend.datadog.bundle import create_datadog_observability_backend
from intergrax.integrations.providers.observability_backend.langfuse.bundle import create_langfuse_observability_backend
from intergrax.integrations.providers.search_provider.exa.bundle import create_exa_search_provider
from intergrax.integrations.providers.search_provider.tavily.bundle import create_tavily_search_provider
from intergrax.integrations.providers.secrets_store.vault.bundle import create_vault_secrets_store
from intergrax.integrations.providers.vector_store.inmemory.bundle import create_inmemory_vector_store
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import catalog_snapshot, clear_catalog
from intergrax.integrations.registry.slugs import IntegrationSlug
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


class _FakeSearchClient:
    def search(self, query: str, limit: int) -> dict[str, Any]:
        return {"results": [{"title": query, "url": "https://example", "content": "hit"}]}


class _FakeObsClient:
    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
        return 3.0

    def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
        return [{"timestamp": start, "value": 1.0}]


class _FakeQueueClient:
    def send_message(self, *, body: bytes, attributes: dict[str, str]) -> str:
        return "msg-p3"

    def get_message_status(self, message_id: str) -> str:
        return "pending"

    def get_message_result(self, message_id: str) -> Optional[bytes]:
        return None


class _FakeVaultClient:
    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}

    def read_secret(self, mount: str, path: str, *, version: Optional[str] = None) -> str:
        return self._secrets[f"{mount}/{path}"]

    def write_secret(self, mount: str, path: str, value: str) -> None:
        self._secrets[f"{mount}/{path}"] = value

    def delete_secret(self, mount: str, path: str) -> None:
        self._secrets.pop(f"{mount}/{path}", None)


class _FakeNeo4jClient:
    def run(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"statement": statement, "params": parameters}]

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        return {"id": node_id, "labels": ["Agent"], "properties": {"name": "n1"}}


class _FakeFirecrawlClient:
    def scrape(self, url: str) -> dict[str, Any]:
        return {"title": "Page", "markdown": "content", "html": "<p></p>", "status_code": 200}


class _FakeSeleniumDriver:
    def get(self, url: str) -> None:
        self._url = url

    @property
    def title(self) -> str:
        return "Title"

    @property
    def page_source(self) -> str:
        return "<html></html>"

    def find_element(self, by: str, value: str) -> Any:
        return type("B", (), {"text": "body"})()

    def quit(self) -> None:
        return None


def test_tavily_and_exa_search() -> None:
    tavily = create_tavily_search_provider(client=_FakeSearchClient())
    assert_search_provider(tavily)
    assert tavily.search("ai", limit=1)[0].provider == "tavily"

    exa = create_exa_search_provider(client=_FakeSearchClient())
    assert_search_provider(exa)
    assert exa.search("ai", limit=1)[0].provider == "exa"


def test_inmemory_vector_store() -> None:
    store = create_inmemory_vector_store(tenant_id="lab")
    assert_vector_store(store)
    assert store.count() == 0


def test_vault_secrets_store() -> None:
    vault = create_vault_secrets_store(client=_FakeVaultClient(), mount="secret")
    assert_secrets_store(vault)
    vault.put_secret("tenant/api", "key-1")
    assert vault.get_secret("tenant/api") == "key-1"
    vault.delete_secret("tenant/api")


def test_observability_http_backends() -> None:
    for factory in (
        create_langfuse_observability_backend,
        create_datadog_observability_backend,
        create_clickhouse_observability_backend,
    ):
        backend = factory(client=_FakeObsClient())
        assert_observability_backend(backend)
        assert backend.query_instant("up").series[0].points[0].value == 3.0


@pytest.mark.parametrize("factory", [create_temporal_message_bus, create_nats_message_bus])
def test_message_bus_p3(factory: Any) -> None:
    bus = factory(client=_FakeQueueClient())
    assert_message_bus(bus)
    handle = bus.enqueue(
        TaskRequest(tenant_id="t1", run_id="r1", task_name="x", payload=b"p", idempotency_key=None)
    )
    assert handle.task_id == "msg-p3"


def test_neo4j_graph_store() -> None:
    graph = create_neo4j_graph_store(client=_FakeNeo4jClient())
    assert_graph_store(graph)
    result = graph.run_query("MATCH (n) RETURN n LIMIT 1")
    assert result.records
    node = graph.get_node("n1")
    assert node is not None and node.labels == ["Agent"]


def test_filesystem_object_storage(tmp_path: Path) -> None:
    store = create_filesystem_object_storage(root_dir=str(tmp_path))
    assert_object_storage(store)
    store.put("a.txt", b"data")
    obj = store.get("a.txt")
    assert obj is not None and obj.body == b"data"


@pytest.mark.asyncio
async def test_discord_notification() -> None:
    sent: list[Any] = []

    def _sender(*, message: Any) -> None:
        sent.append(message)

    from intergrax.integrations._shared.p3.clients import HttpNotificationChannel

    channel = HttpNotificationChannel(_sender, provider="discord")
    assert_notification_channel(channel)
    await channel.notify(
        NotificationMessage(
            tenant_id="t1",
            channel="#ops",
            task_id="t1",
            subject="Alert",
            body="Body",
            metadata={},
        )
    )
    assert sent[0].body == "Body"


def test_firecrawl_and_selenium() -> None:
    fc = create_firecrawl_browser_automation(client=_FakeFirecrawlClient())
    assert_browser_automation(fc)
    page = fc.fetch_page("https://example.com")
    assert isinstance(page, PageContent)

    sel = create_selenium_browser_automation(browser=_FakeSeleniumDriver())
    assert_browser_automation(sel)
    page2 = sel.fetch_page("https://example.com")
    assert page2.title == "Title"


def test_sentry_observability() -> None:
    class _FakeSentryClient:
        def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
            return 7.0

        def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
            return [{"timestamp": start, "value": 7.0}]

        def capture_exception(self, exc: BaseException, *, tags: dict[str, str]) -> str:
            return "event-1"

        def capture_message(self, message: str, *, level: str) -> str:
            return "msg-1"

    from intergrax.integrations.providers.observability_backend.sentry.bundle import create_sentry_observability_backend

    backend = create_sentry_observability_backend(client=_FakeSentryClient())
    assert_observability_backend(backend)
    assert backend.query_instant("is:unresolved").series[0].points[0].value == 7.0
    assert backend.capture_message("agent failed") == "msg-1"


def test_register_default_integrations_includes_p3_slugs() -> None:
    register_default_integrations()
    slugs = set(catalog_snapshot().keys())
    for slug in (
        IntegrationSlug.TAVILY,
        IntegrationSlug.VAULT,
        IntegrationSlug.NEO4J,
        IntegrationSlug.INMEMORY,
        IntegrationSlug.FIRECRAWL,
        IntegrationSlug.SENTRY,
    ):
        assert slug.value in slugs
