# © Artur Czarnecki. All rights reserved.

"""Unit tests for PythonFinalReleaseSourcePolicy."""

from __future__ import annotations

import pytest

from web_search_qualifier.source_selection.contracts import SourceSelectionContext, SourceSelectionOutcome
from web_search_qualifier.source_selection.python_final_release_policy import PythonFinalReleaseSourcePolicy
from web_search_qualifier.web_search import WebSearchCandidate

pytestmark = pytest.mark.unit

_RC3_URL = "https://www.python.org/downloads/release/python-3120rc3"
_FINAL_URL = "https://www.python.org/downloads/release/python-3120/"
_DOCS_URL = "https://www.python.org/download/releases/3.12.0/"
_THIRD_PARTY = "https://example.com/python-3120-release"


def _candidate(url: str, *, rank: int) -> WebSearchCandidate:
    return WebSearchCandidate(
        rank=rank,
        url=url,
        title=url,
        snippet="snippet",
        provider="test",
    )


def _context(
    *,
    task: str = "When was Python 3.12.0 released?",
    candidates: tuple[WebSearchCandidate, ...],
) -> SourceSelectionContext:
    return SourceSelectionContext(task_message=task, candidates=candidates)


def test_final_release_selected_over_rc() -> None:
    policy = PythonFinalReleaseSourcePolicy()
    decision = policy.evaluate(
        _context(
            candidates=(
                _candidate(_RC3_URL, rank=1),
                _candidate(_FINAL_URL, rank=2),
            ),
        ),
    )
    assert decision.outcome is SourceSelectionOutcome.SELECT
    assert decision.selected_url == _FINAL_URL


def test_final_selected_over_docs_index() -> None:
    policy = PythonFinalReleaseSourcePolicy()
    decision = policy.evaluate(
        _context(
            candidates=(
                _candidate(_DOCS_URL, rank=1),
                _candidate(_FINAL_URL, rank=2),
            ),
        ),
    )
    assert decision.outcome is SourceSelectionOutcome.SELECT
    assert decision.selected_url == _FINAL_URL


def test_rc_only_abstains() -> None:
    policy = PythonFinalReleaseSourcePolicy()
    decision = policy.evaluate(_context(candidates=(_candidate(_RC3_URL, rank=1),)))
    assert decision.outcome is SourceSelectionOutcome.ABSTAIN


def test_third_party_plus_official_final_selects_final() -> None:
    policy = PythonFinalReleaseSourcePolicy()
    decision = policy.evaluate(
        _context(
            candidates=(
                _candidate(_THIRD_PARTY, rank=1),
                _candidate(_FINAL_URL, rank=2),
            ),
        ),
    )
    assert decision.outcome is SourceSelectionOutcome.SELECT
    assert decision.selected_url == _FINAL_URL


def test_irrelevant_query_abstains_without_python_release_candidates() -> None:
    policy = PythonFinalReleaseSourcePolicy()
    decision = policy.evaluate(
        SourceSelectionContext(
            task_message="What is the weather in Warsaw?",
            candidates=(_candidate("https://weather.example.com/", rank=1),),
        ),
    )
    assert decision.outcome is SourceSelectionOutcome.ABSTAIN
