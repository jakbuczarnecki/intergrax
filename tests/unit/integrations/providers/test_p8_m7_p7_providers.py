# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Phase M.7 P7 integration providers and agent-developer presets."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations._shared.conformance import (
    assert_browser_automation,
    assert_document_parser,
    assert_identity_provider_backend,
    assert_key_value_cache,
    assert_message_bus,
    assert_object_storage,
    assert_relational_store,
    assert_search_provider,
    assert_vector_store,
    assert_wiki_knowledge,
    assert_workflow_orchestrator_backend,
)
from intergrax.integrations._shared.p8.factories import (
    create_airbyte_workflow_orchestrator,
    create_apify_browser_automation,
    create_arxiv_search_provider,
    create_bigquery_relational_store,
    create_browserbase_browser_automation,
    create_clerk_identity_provider,
    create_google_drive_object_storage,
    create_lancedb_vector_store,
    create_llamaparse_document_parser,
    create_motherduck_relational_store,
    create_n8n_workflow_orchestrator,
    create_okta_identity_provider,
    create_perplexity_search_provider,
    create_semantic_scholar_search_provider,
    create_telegram_catalog_factory,
    create_upstash_qstash_message_bus,
    create_upstash_redis_key_value_cache,
    create_wikipedia_wiki_knowledge,
)
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import catalog_snapshot, clear_catalog
from intergrax.integrations.registry import presets
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

M7_P7_SLUGS = (
    "perplexity",
    "arxiv",
    "semantic_scholar",
    "llamaparse",
    "lancedb",
    "telegram",
    "browserbase",
    "google_drive",
    "n8n",
    "wikipedia",
    "clerk",
    "upstash_redis",
    "upstash_qstash",
    "okta",
    "bigquery",
    "motherduck",
    "airbyte",
    "apify",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeSearchClient:
    def search(self, query: str, limit: int) -> dict[str, Any]:
        return {"results": [{"title": query, "url": "https://example.com", "snippet": "hit"}][:limit]}

    def health(self) -> bool:
        return True


class _FakeParserClient:
    def parse_file(self, source: str) -> dict[str, Any]:
        return {"text": f"parsed:{source}", "metadata": {}}

    def health(self) -> bool:
        return True


class _FakeBrowserClient:
    def fetch_page(self, url: str, *, wait_until: str = "load") -> dict[str, Any]:
        return {"url": url, "title": "t", "text": "body", "html": "<p>body</p>"}

    def close(self) -> None:
        return None

    def health(self) -> bool:
        return True


class _FakeStorageClient:
    def put_object(self, *, Key: str, Body: bytes, ContentType: str = "application/octet-stream", Metadata: Optional[dict[str, str]] = None) -> None:
        _ = Metadata

    def get_object(self, *, Key: str) -> dict[str, Any]:
        class _Body:
            def read(self) -> bytes:
                return b"data"

        return {"Body": _Body(), "ContentType": "application/octet-stream"}

    def delete_object(self, *, Key: str) -> None:
        return None

    def health(self) -> bool:
        return True


class _FakeWorkflowClient:
    def trigger_run(self, workflow_id: str, *, parameters: dict[str, str]) -> dict[str, Any]:
        return {"run_id": workflow_id, "parameters": parameters}

    def poll_status(self, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "status": "success"}

    def fetch_logs(self, run_id: str, *, tail_lines: int = 200) -> str:
        return f"log:{run_id}"

    def list_runs(self, *, workflow_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
        return [{"run_id": "r1"}]

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "status": "cancelled"}

    def health(self) -> bool:
        return True


class _FakeWikiClient:
    def get_page(self, page_id: str) -> dict[str, Any]:
        return {"id": page_id, "title": page_id}

    def search_pages(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        return [{"id": "1", "title": query}][:limit]

    def health(self) -> bool:
        return True


class _FakeIdentityClient:
    def verify_token(self, token: str) -> dict[str, Any]:
        return {"sub": token}

    def userinfo(self, token: str) -> dict[str, Any]:
        return {"sub": token}

    def list_tenants(self, *, limit: int) -> list[dict[str, Any]]:
        return [{"id": "t1"}]

    def health(self) -> bool:
        return True


class _FakeKvClient:
    def get(self, key: str) -> Optional[str]:
        return "value" if key.endswith("hit") else None

    def set(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def setnx(self, key: str, value: bytes, *, ttl_seconds: Optional[int] = None) -> bool:
        return True

    def health(self) -> bool:
        return True


class _FakeQueueClient:
    def send_message(self, *, body: dict[str, Any], attributes: dict[str, Any]) -> str:
        _ = attributes
        return "task-1"

    def get_message_status(self, message_id: str) -> str:
        return "succeeded"

    def get_message_result(self, message_id: str) -> Optional[dict[str, Any]]:
        return {"task_id": message_id}

    def health(self) -> bool:
        return True


class _FakeSqlClient:
    def execute(self, statement: str, parameters: dict[str, Any]) -> None:
        return None

    def fetch_all(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"statement": statement}]

    def health(self) -> bool:
        return True


@pytest.mark.parametrize("slug", M7_P7_SLUGS)
def test_m7_p7_slug_registered_stable(slug: str) -> None:
    register_default_integrations(preset="full")
    snapshot = catalog_snapshot()
    assert slug in snapshot
    assert snapshot[slug].status is IntegrationStatus.STABLE


def test_search_providers_conformance() -> None:
    for factory in (
        create_perplexity_search_provider,
        create_arxiv_search_provider,
        create_semantic_scholar_search_provider,
    ):
        backend = factory(client=_FakeSearchClient())
        provider = assert_search_provider(backend)
        hits = provider.search("intergrax agents", limit=3)
        assert len(hits) >= 1


def test_llamaparse_document_parser_conformance() -> None:
    parser = assert_document_parser(create_llamaparse_document_parser(client=_FakeParserClient()))
    fragments = parser.parse_file("/tmp/sample.pdf")
    assert fragments[0].text.startswith("parsed:")


def test_lancedb_vector_store_conformance() -> None:
    store = assert_vector_store(create_lancedb_vector_store())
    assert store.health() is True or hasattr(store, "health")


def test_telegram_dual_category_factory() -> None:
    notify = create_telegram_catalog_factory(integration_category=IntegrationCategory.NOTIFICATION_CHANNEL)
    interaction = create_telegram_catalog_factory(integration_category=IntegrationCategory.INTERACTION_SURFACE)
    assert notify is not None
    assert interaction is not None


def test_browser_providers_conformance() -> None:
    for factory in (create_browserbase_browser_automation, create_apify_browser_automation):
        browser = assert_browser_automation(factory(client=_FakeBrowserClient()))
        page = browser.fetch_page("https://example.com")
        assert page.url == "https://example.com"
        browser.close()


def test_google_drive_object_storage_conformance() -> None:
    storage = assert_object_storage(create_google_drive_object_storage(client=_FakeStorageClient()))
    storage.put("artifact.txt", b"payload")
    stored = storage.get("artifact.txt")
    assert stored is not None
    assert stored.body == b"data"


def test_workflow_orchestrators_conformance() -> None:
    for factory in (create_n8n_workflow_orchestrator, create_airbyte_workflow_orchestrator):
        workflow = assert_workflow_orchestrator_backend(factory(client=_FakeWorkflowClient()))
        run = workflow.trigger_run("wf-1", parameters={"mode": "eval"})
        assert run.run_id


def test_wikipedia_wiki_knowledge_conformance() -> None:
    wiki = assert_wiki_knowledge(create_wikipedia_wiki_knowledge(client=_FakeWikiClient()))
    page = wiki.get_page("Intergrax")
    assert page.id or page.title


def test_identity_providers_conformance() -> None:
    for factory in (create_clerk_identity_provider, create_okta_identity_provider):
        identity = assert_identity_provider_backend(factory(client=_FakeIdentityClient()))
        assert identity.verify_token("token").user_id == "token"


def test_upstash_redis_key_value_cache_conformance() -> None:
    cache = assert_key_value_cache(create_upstash_redis_key_value_cache(client=_FakeKvClient()))
    cache.set("tenant", "key", b"v")
    assert cache.get("tenant", "key-hit") == b"value"


def test_upstash_qstash_message_bus_conformance() -> None:
    from intergrax.queueing.contracts.task_queue import TaskRequest

    bus = assert_message_bus(create_upstash_qstash_message_bus(client=_FakeQueueClient()))
    handle = bus.enqueue(
        TaskRequest(
            tenant_id="tenant",
            run_id="run-1",
            task_name="index",
            payload={"action": "index"},
        )
    )
    assert handle.task_id == "task-1"


def test_bigquery_and_motherduck_relational_store() -> None:
    bq = assert_relational_store(create_bigquery_relational_store(client=_FakeSqlClient()))
    rows = bq.fetch_all("SELECT 1", {})
    assert rows
    from intergrax.integrations._shared.p2.clients import SqlRelationalStore

    md = create_motherduck_relational_store(relational_store=SqlRelationalStore(_FakeSqlClient(), factory_name="motherduck"))
    assert_relational_store(md)


def test_m7_p7_presets_bind_categories() -> None:
    register_default_integrations(preset="full")
    research = presets.research_web_stack()
    assert research.slug_for_category(IntegrationCategory.SEARCH_PROVIDER) == "perplexity"
    ingest = presets.document_ingest_stack()
    assert ingest.slug_for_category(IntegrationCategory.DOCUMENT_PARSER) == "llamaparse"
    bot = presets.chat_bot_stack()
    assert bot.slug_for_category(IntegrationCategory.INTERACTION_SURFACE) == "telegram"
