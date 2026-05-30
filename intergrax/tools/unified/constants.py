# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical catalog tool ids for unified tool model (Phase O.5)."""

from __future__ import annotations

RAG_RETRIEVE_TOOL_ID = "rag.retrieve"
WEBSEARCH_QUERY_TOOL_ID = "websearch.query"

RAG_TOOL_ALIASES: frozenset[str] = frozenset({"rag", RAG_RETRIEVE_TOOL_ID, "nexus.rag"})
WEBSEARCH_TOOL_ALIASES: frozenset[str] = frozenset(
    {"websearch", WEBSEARCH_QUERY_TOOL_ID, "nexus.websearch"},
)
