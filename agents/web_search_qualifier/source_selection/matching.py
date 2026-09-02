# © Artur Czarnecki. All rights reserved.

"""URL matching helpers for source selection."""

from __future__ import annotations

import re

from web_search_qualifier.source_selection.url_normalization import normalize_url_identity
from web_search_qualifier.web_search import WebSearchCandidate


def candidate_url_set(candidates: tuple[WebSearchCandidate, ...]) -> dict[str, str]:
    """Map normalized URL identity to canonical candidate URL."""
    mapping: dict[str, str] = {}
    for candidate in candidates:
        normalized = normalize_url_identity(candidate.url)
        if normalized not in mapping:
            mapping[normalized] = candidate.url
    return mapping


def resolve_candidate_url(
    url: str,
    candidates: tuple[WebSearchCandidate, ...],
) -> str | None:
    normalized = normalize_url_identity(url)
    for candidate in candidates:
        if normalize_url_identity(candidate.url) == normalized:
            return candidate.url
    return None


def match_url_from_response(
    response: str,
    candidates: tuple[WebSearchCandidate, ...],
) -> str | None:
    text = response.strip()
    for candidate in candidates:
        if candidate.url in text:
            resolved = resolve_candidate_url(candidate.url, candidates)
            if resolved is not None:
                return resolved
    match = re.search(r"https?://\S+", text)
    if match:
        raw = match.group(0).rstrip(").,]")
        return resolve_candidate_url(raw, candidates)
    return None


__all__ = [
    "candidate_url_set",
    "match_url_from_response",
    "resolve_candidate_url",
]
