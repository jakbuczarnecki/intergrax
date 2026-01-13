# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for JSON parsing helpers inside LLMAdapter.

We test two production-critical helper methods:
- _strip_code_fences(): removes Markdown code fences (```json ... ```) safely.
- _extract_json_object(): extracts the outermost JSON object from model output,
  tolerating extra text around it.

Why this matters:
LLMs frequently wrap JSON in Markdown fences or include pre/post text.
These helpers are the last line of defense for structured output reliability.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter


pytestmark = pytest.mark.unit


class _DummyAdapter(LLMAdapter):
    """
    Minimal concrete adapter for testing non-public helper behavior.
    """

    provider = "fake"

    @property
    def context_window_tokens(self) -> int:
        return 1

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("Not needed for these unit tests.")


def test_strip_code_fences_removes_json_fences() -> None:
    """
    _strip_code_fences() must remove ```json ... ``` wrapper and return inner content.
    """
    a = _DummyAdapter()

    text = "```json\n{\n  \"x\": 1\n}\n```"
    assert a._strip_code_fences(text) == "{\n  \"x\": 1\n}"


def test_strip_code_fences_removes_plain_fences() -> None:
    """
    _strip_code_fences() must remove ``` ... ``` wrapper even without language tag.
    """
    a = _DummyAdapter()

    text = "```\n{\"x\": 1}\n```"
    assert a._strip_code_fences(text) == "{\"x\": 1}"


def test_strip_code_fences_returns_input_when_not_fenced() -> None:
    """
    If there are no outer code fences, _strip_code_fences() must return text unchanged.
    """
    a = _DummyAdapter()

    text = "  {\"x\": 1}  "
    assert a._strip_code_fences(text) == text


def test_extract_json_object_from_fenced_json() -> None:
    """
    _extract_json_object() must:
    - strip fences,
    - then return the outermost {...} object.
    """
    a = _DummyAdapter()

    text = "```json\n{\"x\": 1}\n```"
    assert a._extract_json_object(text) == "{\"x\": 1}"


def test_extract_json_object_tolerates_prefix_and_suffix_text() -> None:
    """
    Model outputs often include extra text around JSON.
    _extract_json_object() must extract the first '{' to the last '}'.
    """
    a = _DummyAdapter()

    text = "Here is the result:\n\n{\"a\": 1, \"b\": 2}\n\nThanks!"
    assert a._extract_json_object(text) == "{\"a\": 1, \"b\": 2}"


def test_extract_json_object_returns_empty_string_when_no_object_found() -> None:
    """
    If no JSON object braces are present, _extract_json_object() must return ''.
    """
    a = _DummyAdapter()

    assert a._extract_json_object("") == ""
    assert a._extract_json_object("no json here") == ""


def test_extract_json_object_returns_empty_string_when_braces_incomplete() -> None:
    """
    Incomplete JSON (missing '{' or '}' or wrong order) must return ''.
    """
    a = _DummyAdapter()

    assert a._extract_json_object("{") == ""
    assert a._extract_json_object("}") == ""
    assert a._extract_json_object("}{") == ""
