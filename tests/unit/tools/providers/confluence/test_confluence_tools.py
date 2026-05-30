# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord, WikiSearchResult
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.confluence.bundle import register_confluence_tools
from intergrax.tools.providers.confluence.contracts import ConfluenceGetPageInput, ConfluenceSearchPagesInput
from intergrax.tools.providers.confluence.service import confluence_get_page, confluence_search_pages
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class FakeWikiKnowledge:
    def get_page(self, page_id: str) -> WikiPageRecord:
        return WikiPageRecord(
            id=page_id,
            title="Architecture overview",
            space_key="ENG",
            body="System design notes…",
            url=f"https://wiki.example/pages/{page_id}",
            version=3,
        )

    def search_pages(self, query: str, *, limit: int = 25) -> WikiSearchResult:
        return WikiSearchResult(
            pages=[
                WikiPageRecord(
                    id="p-1",
                    title=f"Result for {query}",
                    space_key="ENG",
                    body="Matching content",
                    url="https://wiki.example/pages/p-1",
                )
            ],
            total=1,
        )


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_confluence_get_page() -> None:
    ctx = ToolWiringContext(wiki_knowledge=FakeWikiKnowledge())
    out = confluence_get_page(ctx, ConfluenceGetPageInput(page_id="12345"))
    assert out.id == "12345"
    assert out.title == "Architecture overview"
    assert out.space_key == "ENG"


def test_confluence_search_pages() -> None:
    ctx = ToolWiringContext(wiki_knowledge=FakeWikiKnowledge())
    out = confluence_search_pages(ctx, ConfluenceSearchPagesInput(query="deployment", limit=5))
    assert out.total == 1
    assert out.pages[0].title == "Result for deployment"


def test_confluence_wiki_not_configured() -> None:
    with pytest.raises(RuntimeError, match="wiki_knowledge_not_configured"):
        confluence_get_page(ToolWiringContext(), ConfluenceGetPageInput(page_id="1"))


def test_confluence_tools_registered_in_catalog() -> None:
    register_default_tools()
    assert "confluence.get_page" in list_catalog_tool_ids()
    assert "confluence.search_pages" in list_catalog_tool_ids()
    assert get_bundle("confluence").tool_ids == ("confluence.get_page", "confluence.search_pages")


def test_confluence_get_page_via_runtime_invoker() -> None:
    ctx = ToolWiringContext(wiki_knowledge=FakeWikiKnowledge())
    registry = ToolRegistry()
    register_confluence_tools(registry, ctx)

    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="wiki_run")
    request = ToolExecutionRequest(
        run_id="wiki_run",
        step_id="step/1",
        tool_id="confluence.get_page",
        input=ConfluenceGetPageInput(page_id="999"),
    )

    result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.title == "Architecture overview"


def test_build_registry_enables_confluence_bundle() -> None:
    register_default_tools()
    ctx = ToolWiringContext(wiki_knowledge=FakeWikiKnowledge())
    registry = build_registry_from_profile(ToolProfile(enabled_bundles=["confluence"]), ctx=ctx)
    assert registry.has("confluence.search_pages")
