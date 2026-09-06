# © Artur Czarnecki. All rights reserved.

"""Tool-domain keyword overlap search primitive tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import BaseModel

from intergrax.tools.search.keyword_ranking import (
    ToolKeywordSearchDocument,
    score_tool_keyword_document,
    tokenize_tool_search_query,
)
from testing_support.builder import tools_agent_make_contract

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


def test_tokenize_lowercases_and_ignores_short_tokens() -> None:
    assert tokenize_tool_search_query("Catalog SEARCH tool") == ("catalog", "search", "tool")


def test_tokenize_empty_query_returns_empty_tuple() -> None:
    assert tokenize_tool_search_query("") == ()
    assert tokenize_tool_search_query("  a  bb  ") == ()


def test_tokenize_is_deterministic() -> None:
    query = "catalog search echo"
    assert tokenize_tool_search_query(query) == tokenize_tool_search_query(query)


def test_score_matches_identifier() -> None:
    document = ToolKeywordSearchDocument(tool_id="tools.echo.ping")
    tokens = tokenize_tool_search_query("echo")
    assert score_tool_keyword_document(document, tokens) == 1


def test_score_matches_additional_text_parts() -> None:
    document = ToolKeywordSearchDocument(
        tool_id="tools.search.catalog",
        text_parts=("Catalog Search",),
    )
    tokens = tokenize_tool_search_query("catalog search")
    assert score_tool_keyword_document(document, tokens) == 2


def test_score_allows_absent_optional_text_parts() -> None:
    document = ToolKeywordSearchDocument(tool_id="tools.echo.ping")
    tokens = tokenize_tool_search_query("unrelated")
    assert score_tool_keyword_document(document, tokens) == 0


def test_score_empty_query_tokens_returns_zero() -> None:
    document = ToolKeywordSearchDocument(tool_id="tools.echo.ping", text_parts=("echo",))
    assert score_tool_keyword_document(document, ()) == 0


def test_score_is_deterministic() -> None:
    document = ToolKeywordSearchDocument(
        tool_id="tools.search.catalog",
        text_parts=("Catalog Search",),
    )
    tokens = tokenize_tool_search_query("catalog search")
    first = score_tool_keyword_document(document, tokens)
    second = score_tool_keyword_document(document, tokens)
    assert first == second == 2


def test_tool_eng5_projection_uses_shared_score_primitive() -> None:
    contract = replace(
        tools_agent_make_contract("tools.search.catalog", _In, _Out),
        description="Catalog Search",
    )
    tool_eng_document = ToolKeywordSearchDocument(
        tool_id=contract.tool_id,
        text_parts=tuple(
            part
            for part in (
                contract.description,
                contract.description_short,
                " ".join(contract.tags),
                contract.category,
            )
            if part
        ),
    )
    minimal_document = ToolKeywordSearchDocument(
        tool_id="tools.search.catalog",
        text_parts=("Catalog Search",),
    )
    tokens = tokenize_tool_search_query("catalog search")
    assert score_tool_keyword_document(tool_eng_document, tokens) == (
        score_tool_keyword_document(minimal_document, tokens)
    )
