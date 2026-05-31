# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared REST search helpers for Brave / SerpAPI catalog providers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from intergrax.websearch.schemas.search_hit import SearchHit


def hits_from_brave_payload(query: str, payload: Mapping[str, Any], *, limit: int) -> Sequence[SearchHit]:
    web = payload.get("web") if isinstance(payload.get("web"), dict) else payload
    results = web.get("results") if isinstance(web, dict) else []
    hits: list[SearchHit] = []
    for idx, row in enumerate(results or []):
        if not isinstance(row, dict):
            continue
        hits.append(
            SearchHit(
                provider="brave",
                query_issued=query,
                rank=idx + 1,
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("description") or row.get("snippet") or ""),
            )
        )
        if len(hits) >= limit:
            break
    return hits


def hits_from_tavily_payload(query: str, payload: Mapping[str, Any], *, limit: int) -> Sequence[SearchHit]:
    results = payload.get("results") or []
    hits: list[SearchHit] = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        hits.append(
            SearchHit(
                provider="tavily",
                query_issued=query,
                rank=idx + 1,
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("content") or row.get("snippet") or ""),
            )
        )
        if len(hits) >= limit:
            break
    return hits


def hits_from_exa_payload(query: str, payload: Mapping[str, Any], *, limit: int) -> Sequence[SearchHit]:
    results = payload.get("results") or payload.get("data") or []
    hits: list[SearchHit] = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        hits.append(
            SearchHit(
                provider="exa",
                query_issued=query,
                rank=idx + 1,
                title=str(row.get("title") or ""),
                url=str(row.get("url") or row.get("link") or ""),
                snippet=str(row.get("text") or row.get("snippet") or ""),
            )
        )
        if len(hits) >= limit:
            break
    return hits


def hits_from_serpapi_payload(query: str, payload: Mapping[str, Any], *, limit: int) -> Sequence[SearchHit]:
    results = payload.get("organic_results") or payload.get("news_results") or []
    hits: list[SearchHit] = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        hits.append(
            SearchHit(
                provider="serpapi",
                query_issued=query,
                rank=idx + 1,
                title=str(row.get("title") or ""),
                url=str(row.get("link") or row.get("url") or ""),
                snippet=str(row.get("snippet") or row.get("description") or ""),
            )
        )
        if len(hits) >= limit:
            break
    return hits
