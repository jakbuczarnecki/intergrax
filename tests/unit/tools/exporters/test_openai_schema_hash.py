# © Artur Czarnecki. All rights reserved.

"""Generic OpenAI tool-schema hashing tests."""

from __future__ import annotations

import pytest

from intergrax.tools.exporters.openai import compute_openai_tools_schema_hash

pytestmark = pytest.mark.unit


def test_compute_openai_tools_schema_hash_empty() -> None:
    first = compute_openai_tools_schema_hash([])
    second = compute_openai_tools_schema_hash(())
    assert first == second


def test_compute_openai_tools_schema_hash_sorts_by_function_name() -> None:
    schema_a = [
        {
            "type": "function",
            "function": {"name": "beta.tool", "description": "b", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "alpha.tool", "description": "a", "parameters": {}},
        },
    ]
    schema_b = list(reversed(schema_a))
    assert compute_openai_tools_schema_hash(schema_a) == compute_openai_tools_schema_hash(schema_b)
