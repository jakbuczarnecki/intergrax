# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool-domain keyword overlap search primitive (scoring only, no selection)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolKeywordSearchDocument:
    """Minimal searchable projection for Tool keyword overlap scoring."""

    tool_id: str
    text_parts: tuple[str, ...] = ()


def tokenize_tool_search_query(query: str) -> tuple[str, ...]:
    """Lowercase query tokens longer than two characters; deterministic order."""
    return tuple(token for token in query.lower().split() if len(token) > 2)


def score_tool_keyword_document(
    document: ToolKeywordSearchDocument,
    query_tokens: Sequence[str],
) -> int:
    """Count query tokens present in the tool identifier and optional text parts."""
    if not query_tokens:
        return 0
    haystack = " ".join(
        part for part in (document.tool_id, *document.text_parts) if part
    ).lower()
    return sum(1 for token in query_tokens if token in haystack)
