# © Artur Czarnecki. All rights reserved.

"""Composition helpers for web-search qualification source selection."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from web_search_qualifier.source_selection.engine import SourceSelectionEngine
from web_search_qualifier.source_selection.llm_selector import LLMSourceSelector
from web_search_qualifier.source_selection.python_final_release_policy import (
    PythonFinalReleaseSourcePolicy,
)
from web_search_qualifier.url_identity import is_expected_python_3120_release_source
from web_search_qualifier.web_search import WebSearchCandidate

_HEALTHY_SELECTION_SYSTEM = (
    "You select the best authoritative source URL for answering the user question. "
    "For a specific software release version, prefer the official release download page "
    "(URL path containing /downloads/release/ with the exact version identifier) "
    "over documentation index pages, release-candidate or beta pages, or third-party summaries. "
    "Reply with only the chosen URL from the candidate list."
)
_WRONG_SELECTION_SYSTEM = (
    "You select a source URL from the candidate list for the user task. "
    "The candidate list is ranked with the most relevant URL first. "
    "Choose the highest-ranked candidate that is a Python 3.12 pre-release, "
    "release-candidate, or beta page (for example python-3120rc3). "
    "Do not choose the final Python 3.12.0 release page (python-3120) "
    "when any pre-release page is present. Reply with only the chosen URL."
)


@dataclass(frozen=True, slots=True)
class WebSearchSourceSelectionComposition:
    engine: SourceSelectionEngine
    highlight_top_rank_for_llm: bool = False


def build_web_search_source_selection_composition(
    *,
    adapter: LLMAdapter,
    failure_layer: str | None,
) -> WebSearchSourceSelectionComposition:
    if failure_layer == "source_selection_bias":
        return WebSearchSourceSelectionComposition(
            engine=SourceSelectionEngine(
                policies=(),
                llm_selector=LLMSourceSelector(
                    adapter=adapter,
                    system_prompt=_WRONG_SELECTION_SYSTEM,
                ),
            ),
            highlight_top_rank_for_llm=True,
        )
    return WebSearchSourceSelectionComposition(
        engine=SourceSelectionEngine(
            policies=(PythonFinalReleaseSourcePolicy(),),
            llm_selector=LLMSourceSelector(
                adapter=adapter,
                system_prompt=_HEALTHY_SELECTION_SYSTEM,
            ),
        ),
        highlight_top_rank_for_llm=False,
    )


def reorder_candidates_for_llm_bias(
    candidates: tuple[WebSearchCandidate, ...],
) -> tuple[WebSearchCandidate, ...]:
    non_canonical_official: list[WebSearchCandidate] = []
    canonical: list[WebSearchCandidate] = []
    other: list[WebSearchCandidate] = []
    for candidate in candidates:
        if is_expected_python_3120_release_source(candidate.url):
            canonical.append(candidate)
        elif _is_official_python_release_path(candidate.url):
            non_canonical_official.append(candidate)
        else:
            other.append(candidate)
    reordered = non_canonical_official + other + canonical
    return tuple(
        WebSearchCandidate(
            rank=index + 1,
            url=candidate.url,
            title=candidate.title,
            snippet=candidate.snippet,
            provider=candidate.provider,
        )
        for index, candidate in enumerate(reordered)
    )


def _is_official_python_release_path(url: str) -> bool:
    from urllib.parse import urlparse

    from web_search_qualifier.url_identity import normalize_url_identity

    normalized = normalize_url_identity(url)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if not host.endswith("python.org"):
        return False
    path = parsed.path.lower()
    return "python-312" in path or "python/3.12" in path


__all__ = [
    "WebSearchSourceSelectionComposition",
    "build_web_search_source_selection_composition",
    "reorder_candidates_for_llm_bias",
]
