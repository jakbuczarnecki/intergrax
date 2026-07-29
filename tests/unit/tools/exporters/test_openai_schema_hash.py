# © Artur Czarnecki. All rights reserved.

"""Generic OpenAI tool-schema hashing tests."""

from __future__ import annotations

import copy

import pytest

from intergrax.tools.exporters.openai import compute_openai_tools_schema_hash

pytestmark = pytest.mark.unit


def _alpha_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {"name": "alpha.tool", "description": "a", "parameters": {}},
    }


def _beta_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {"name": "beta.tool", "description": "b", "parameters": {}},
    }


def test_compute_openai_tools_schema_hash_empty() -> None:
    first = compute_openai_tools_schema_hash([])
    second = compute_openai_tools_schema_hash(())
    assert first == second


def test_compute_openai_tools_schema_hash_identical_ordered_schema() -> None:
    schema = [_alpha_tool(), _beta_tool()]
    first = compute_openai_tools_schema_hash(schema)
    second = compute_openai_tools_schema_hash(list(schema))
    assert first == second


def test_compute_openai_tools_schema_hash_dictionary_key_order_irrelevant() -> None:
    schema_a = [
        {
            "type": "function",
            "function": {
                "name": "alpha.tool",
                "description": "a",
                "parameters": {"type": "object", "properties": {"z": {"type": "string"}}},
            },
        }
    ]
    schema_b = [
        {
            "function": {
                "parameters": {"properties": {"z": {"type": "string"}}, "type": "object"},
                "description": "a",
                "name": "alpha.tool",
            },
            "type": "function",
        }
    ]
    assert compute_openai_tools_schema_hash(schema_a) == compute_openai_tools_schema_hash(schema_b)


def test_compute_openai_tools_schema_hash_description_change() -> None:
    schema_a = [_alpha_tool()]
    schema_b = [
        {
            "type": "function",
            "function": {"name": "alpha.tool", "description": "changed", "parameters": {}},
        }
    ]
    assert compute_openai_tools_schema_hash(schema_a) != compute_openai_tools_schema_hash(schema_b)


def test_compute_openai_tools_schema_hash_parameters_change() -> None:
    schema_a = [
        {
            "type": "function",
            "function": {
                "name": "alpha.tool",
                "description": "a",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
    ]
    schema_b = [
        {
            "type": "function",
            "function": {
                "name": "alpha.tool",
                "description": "a",
                "parameters": {"type": "object", "properties": {"y": {"type": "integer"}}},
            },
        }
    ]
    assert compute_openai_tools_schema_hash(schema_a) != compute_openai_tools_schema_hash(schema_b)


def test_compute_openai_tools_schema_hash_add_or_remove_tool() -> None:
    one_tool = [_alpha_tool()]
    two_tools = [_alpha_tool(), _beta_tool()]
    assert compute_openai_tools_schema_hash(one_tool) != compute_openai_tools_schema_hash(two_tools)


def test_compute_openai_tools_schema_hash_reversed_order_differs() -> None:
    alpha_beta = [_alpha_tool(), _beta_tool()]
    beta_alpha = [_beta_tool(), _alpha_tool()]
    assert compute_openai_tools_schema_hash(alpha_beta) != compute_openai_tools_schema_hash(
        beta_alpha
    )


def test_compute_openai_tools_schema_hash_does_not_mutate_input() -> None:
    schema = [_alpha_tool(), _beta_tool()]
    snapshot = copy.deepcopy(schema)
    _ = compute_openai_tools_schema_hash(schema)
    assert schema == snapshot
