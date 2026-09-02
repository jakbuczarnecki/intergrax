# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q3-R3 extraction-input injection boundary gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from web_search_qualifier.steps.web_search_job import (
    _build_extractor_input_context,
    _extract_fact,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_PATH = _REPO_ROOT / "agents" / "web_search_qualifier" / "steps" / "web_search_job.py"
_FINAL_URL = "https://www.python.org/downloads/release/python-3120/"


class _ContextFollowingExtractorAdapter:
    def generate_messages(self, messages, *, temperature: float, run_id: str) -> object:
        del temperature, run_id
        user = messages[1].content
        if "2023-10-01" in user or "October 1, 2023" in user:
            return type("R", (), {"content": "2023-10-01"})()
        return type("R", (), {"content": "2023-10-02"})()


def test_extraction_bias_modifies_extractor_input_not_provider_snippet() -> None:
    provider_snippet = "Released Oct. 2, 2023 — Python 3.12.0 final release."
    extractor_input, bounded_snippet, modified = _build_extractor_input_context(
        selected_url=_FINAL_URL,
        snippet=provider_snippet,
        failure_layer="extraction_bias",
    )
    assert modified is True
    assert bounded_snippet == provider_snippet
    assert "Oct. 2, 2023" not in extractor_input
    assert "2023-10-01" in extractor_input
    assert _FINAL_URL in extractor_input


def test_healthy_extraction_leaves_extractor_input_unmodified() -> None:
    provider_snippet = "Released Oct. 2, 2023"
    extractor_input, bounded_snippet, modified = _build_extractor_input_context(
        selected_url=_FINAL_URL,
        snippet=provider_snippet,
        failure_layer=None,
    )
    assert modified is False
    assert bounded_snippet == provider_snippet
    assert provider_snippet in extractor_input


def test_extraction_bias_emits_raw_llm_output_without_post_override() -> None:
    decision = _extract_fact(
        adapter=_ContextFollowingExtractorAdapter(),
        run_id="run-test",
        selected_url=_FINAL_URL,
        snippet="Released Oct. 2, 2023",
        failure_layer="extraction_bias",
    )
    assert decision.extractor_input_modified is True
    assert decision.fact == "2023-10-01"
    assert decision.raw_response == "2023-10-01"
    assert "Oct. 2, 2023" in decision.provider_source_snippet


def test_extract_fact_has_no_post_decision_output_replacement() -> None:
    source = _JOB_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_fact":
            body_source = ast.get_source_segment(source, node) or ""
            assert body_source.count("fact = _llm_text") == 1
            after_llm = body_source.split("fact = _llm_text", 1)[1]
            assert "fact =" not in after_llm.split("return", 1)[0]
            assert "_looks_like_correct_python_3120_release_date" not in body_source
            return
    raise AssertionError("_extract_fact not found")
