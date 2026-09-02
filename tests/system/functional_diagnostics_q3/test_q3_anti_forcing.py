# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q3-R2 static and behavioral anti-forcing gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from web_search_qualifier.steps.web_search_job import (
    _extract_fact,
    _match_url_from_response,
    _select_source,
)
from web_search_qualifier.web_search import WebSearchCandidate

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_PATH = _REPO_ROOT / "agents" / "web_search_qualifier" / "steps" / "web_search_job.py"

_RC3_URL = "https://www.python.org/downloads/release/python-3120rc3"
_FINAL_URL = "https://www.python.org/downloads/release/python-3120/"


class _PromptAwareStubAdapter:
    def generate_messages(self, messages, *, temperature: float, run_id: str) -> object:
        del temperature, run_id
        system = messages[0].content.lower()
        user = messages[1].content
        if "pre-release" in system or "release-candidate" in system:
            return type("R", (), {"content": _RC3_URL})()
        if "select" in system or "source url" in system:
            return type("R", (), {"content": _FINAL_URL})()
        if "2023-10-01" in user or "October 1, 2023" in user:
            return type("R", (), {"content": "2023-10-01"})()
        return type("R", (), {"content": "2023-10-02"})()


def _candidates() -> tuple[WebSearchCandidate, ...]:
    return (
        WebSearchCandidate(
            rank=1,
            url=_RC3_URL,
            title="Python 3.12.0rc3",
            snippet="release candidate",
            provider="tavily",
        ),
        WebSearchCandidate(
            rank=2,
            url=_FINAL_URL,
            title="Python 3.12.0",
            snippet="Released Oct. 2, 2023",
            provider="tavily",
        ),
    )


def _function_body_names(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    raise AssertionError(f"{function_name} not found in {path}")


def test_select_source_has_no_post_decision_canonical_discard() -> None:
    source = _JOB_PATH.read_text(encoding="utf-8")
    assert "is_expected_python_3120_release_source(selected)" not in source


def test_extract_fact_has_no_post_decision_date_replacement() -> None:
    names = _function_body_names(_JOB_PATH, "_extract_fact")
    assert "_looks_like_correct_python_3120_release_date" not in names


def test_source_selection_bias_respects_llm_canonical_choice() -> None:
    class _CanonicalOnlyAdapter:
        def generate_messages(self, messages, *, temperature: float, run_id: str) -> object:
            del temperature, run_id, messages
            return type("R", (), {"content": _FINAL_URL})()

    decision = _select_source(
        adapter=_CanonicalOnlyAdapter(),
        run_id="run-test",
        task_message="When was Python 3.12.0 released?",
        candidates=_candidates(),
        failure_layer="source_selection_bias",
    )
    assert decision.selected_url == _FINAL_URL
    assert _match_url_from_response(decision.raw_response, _candidates()) == _FINAL_URL


def test_source_selection_bias_induces_wrong_source_via_prompt() -> None:
    class _BiasFollowingAdapter:
        def generate_messages(self, messages, *, temperature: float, run_id: str) -> object:
            del temperature, run_id
            return type("R", (), {"content": _RC3_URL})()

    decision = _select_source(
        adapter=_BiasFollowingAdapter(),
        run_id="run-test",
        task_message="When was Python 3.12.0 released?",
        candidates=_candidates(),
        failure_layer="source_selection_bias",
    )
    assert decision.selected_url == _RC3_URL
    assert decision.raw_response.strip() == _RC3_URL


def test_extraction_bias_respects_llm_correct_date() -> None:
    class _HealthyExtractorAdapter:
        def generate_messages(self, messages, *, temperature: float, run_id: str) -> object:
            del temperature, run_id, messages
            return type("R", (), {"content": "2023-10-02"})()

    decision = _extract_fact(
        adapter=_HealthyExtractorAdapter(),
        run_id="run-test",
        selected_url=_FINAL_URL,
        snippet="Released Oct. 2, 2023",
        failure_layer="extraction_bias",
    )
    assert decision.fact == "2023-10-02"
    assert decision.raw_response == "2023-10-02"


def test_extraction_bias_induces_wrong_date_via_prompt() -> None:
    decision = _extract_fact(
        adapter=_PromptAwareStubAdapter(),
        run_id="run-test",
        selected_url=_FINAL_URL,
        snippet="Released Oct. 2, 2023",
        failure_layer="extraction_bias",
    )
    assert decision.fact == "2023-10-01"
    assert decision.raw_response == "2023-10-01"
    assert decision.extractor_input_modified is True
