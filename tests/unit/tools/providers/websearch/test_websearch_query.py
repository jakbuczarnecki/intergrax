# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pytest

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.websearch.bundle import register_websearch_tools, websearch_query_contract
from intergrax.tools.providers.websearch.contracts import WebsearchQueryInput
from intergrax.tools.providers.websearch.service import perform_websearch_query
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.websearch.schemas.search_hit import SearchHit
from intergrax.websearch.schemas.web_search_result import WebSearchResult
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


@dataclass
class FakeWebSearchExecutor:
    results: List[WebSearchResult]

    def search_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        locale: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        safe_search: Optional[bool] = None,
        top_n_fetch: Optional[int] = None,
    ) -> List[WebSearchResult]:
        limit = top_k or len(self.results)
        return self.results[:limit]


class FakeSearchProvider:
    def __init__(self, hits: List[SearchHit]) -> None:
        self._hits = hits

    def search(self, query: str, *, limit: int = 10):
        return self._hits[:limit]


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def _sample_web_result() -> WebSearchResult:
    return WebSearchResult(
        provider="fake",
        rank=1,
        source_rank=1,
        quality_score=0.9,
        title="Intergrax docs",
        url="https://example.com/intergrax",
        snippet="Agent runtime overview",
        description=None,
        lang="en",
        domain="example.com",
        published_at=None,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        text="Intergrax is an agent OS.",
        document=None,  # type: ignore[arg-type]
    )


def test_websearch_query_via_executor() -> None:
    ctx = ToolWiringContext(websearch_executor=FakeWebSearchExecutor([_sample_web_result()]))
    out = perform_websearch_query(ctx, WebsearchQueryInput(query="intergrax agent", limit=3))

    assert out.used is True
    assert len(out.results) == 1
    assert out.results[0].url == "https://example.com/intergrax"
    assert "Intergrax" in out.context_text
    assert out.reason == "ok"


def test_websearch_query_via_search_provider() -> None:
    hit = SearchHit(
        provider="bing",
        query_issued="intergrax",
        rank=1,
        title="Intergrax",
        url="https://example.org/page",
        snippet="Snippet text",
    )
    ctx = ToolWiringContext(search_provider=FakeSearchProvider([hit]))
    out = perform_websearch_query(ctx, WebsearchQueryInput(query="intergrax"))

    assert out.used is True
    assert out.results[0].provider == "bing"
    assert "Snippet text" in out.context_text


def test_websearch_query_not_configured() -> None:
    out = perform_websearch_query(ToolWiringContext(), WebsearchQueryInput(query="test"))
    assert out.used is False
    assert out.reason == "websearch_not_configured"


def test_websearch_tool_registered_in_catalog() -> None:
    register_default_tools()
    assert "websearch.query" in list_catalog_tool_ids()
    assert get_bundle("websearch").tool_ids == (
        "websearch.query",
        "websearch.read_url",
        "websearch.fetch_batch",
        "websearch.invalidate_cache",
    )


def test_websearch_query_via_runtime_invoker() -> None:
    ctx = ToolWiringContext(websearch_executor=FakeWebSearchExecutor([_sample_web_result()]))
    registry = ToolRegistry()
    register_websearch_tools(registry, ctx)

    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="ws_run")
    request = ToolExecutionRequest(
        run_id="ws_run",
        step_id="step/1",
        tool_id="websearch.query",
        input=WebsearchQueryInput(query="agent runtime", limit=5),
    )

    result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    assert result.output is not None
    assert result.output.used is True

    contract = websearch_query_contract()
    assert contract.injects_context is True


def test_build_registry_enables_websearch_tool() -> None:
    register_default_tools()
    ctx = ToolWiringContext(websearch_executor=FakeWebSearchExecutor([]))
    registry = build_registry_from_profile(ToolProfile(enabled=["websearch.query"]), ctx=ctx)
    assert registry.has("websearch.query")
