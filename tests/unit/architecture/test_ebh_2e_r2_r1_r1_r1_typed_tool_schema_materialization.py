# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R2-R1-R1-R1 — typed tool schema materialization without cast."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    ToolArgumentConformance,
    StrictWireProjectionKind,
)
from intergrax.llm_adapters.contracts.strict_tool_call_validation import (
    StrictToolContractValidationError,
)
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    build_atomic_planner_round_tool_definition,
)
from intergrax.runtime.nexus.tools.canonical_tool_dispatch import (
    materialize_canonical_tool_definitions_for_llm_dispatch,
)
from testing_support.atomic_planner_round_transport import poc_business_tool_schemas

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATERIALIZER_PATH = _REPO_ROOT / "intergrax/runtime/nexus/tools/canonical_tool_dispatch.py"


def _materializer_source() -> str:
    return _MATERIALIZER_PATH.read_text(encoding="utf-8-sig")


def _materializer_ast() -> ast.Module:
    return ast.parse(_materializer_source())


def _ast_uses_name(module: ast.Module, name: str) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == name:
                    return True
    return False


def _ast_has_cast_call(module: ast.Module) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "cast":
                return True
    return False


def _valid_nested_wire_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "demo.nested",
            "description": "nested json",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["a", "b"]},
                    "alt": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "integer"},
                        ]
                    },
                },
                "required": ["tags"],
                "additionalProperties": False,
            },
        },
    }


def test_ebh_2e_r2_r1_r1_r1_materializer_module_has_no_cast() -> None:
    tree = _materializer_ast()
    assert not _ast_has_cast_call(tree)
    assert "from typing import cast" not in _materializer_source()
    assert re.search(r"\bcast\s*\(", _materializer_source()) is None


def test_ebh_2e_r2_r1_r1_r1_materializer_module_has_no_any() -> None:
    tree = _materializer_ast()
    assert not _ast_uses_name(tree, "Any")


def test_ebh_2e_r2_r1_r1_r1_materializer_module_has_no_type_ignore() -> None:
    assert "type: ignore" not in _materializer_source()


def test_ebh_2e_r2_r1_r1_r1_rejects_nested_object_instance() -> None:
    schema = _valid_nested_wire_schema()
    params = schema["function"]["parameters"]
    assert isinstance(params, dict)
    props = params["properties"]
    assert isinstance(props, dict)
    props["bag"] = {"x": object()}
    with pytest.raises(ValueError, match=r"tools\[0\]"):
        materialize_canonical_tool_definitions_for_llm_dispatch([schema])


def test_ebh_2e_r2_r1_r1_r1_rejects_callback() -> None:
    schema = _valid_nested_wire_schema()
    params = schema["function"]["parameters"]
    assert isinstance(params, dict)
    props = params["properties"]
    assert isinstance(props, dict)
    props["cb"] = {"fn": (lambda: None)}
    with pytest.raises(ValueError, match=r"tools\[0\]"):
        materialize_canonical_tool_definitions_for_llm_dispatch([schema])


def test_ebh_2e_r2_r1_r1_r1_rejects_custom_class_instance() -> None:
    class _Payload:
        pass

    schema = _valid_nested_wire_schema()
    params = schema["function"]["parameters"]
    assert isinstance(params, dict)
    props = params["properties"]
    assert isinstance(props, dict)
    props["custom"] = _Payload()
    with pytest.raises(ValueError, match=r"tools\[0\]"):
        materialize_canonical_tool_definitions_for_llm_dispatch([schema])


def test_ebh_2e_r2_r1_r1_r1_rejects_non_string_top_level_key() -> None:
    bad: dict[object, object] = {
        "type": "function",
        "function": {"name": "demo", "parameters": {"type": "object"}},
    }
    bad[1] = "leak"
    with pytest.raises(ValueError, match=r"keys must be strings"):
        materialize_canonical_tool_definitions_for_llm_dispatch([bad])


def test_ebh_2e_r2_r1_r1_r1_rejects_non_string_nested_key() -> None:
    nested: dict[object, object] = {"type": "object", "properties": {}}
    nested_props = nested["properties"]
    assert isinstance(nested_props, dict)
    nested_props[("not", "a", "str")] = {"type": "string"}
    schema = {
        "type": "function",
        "function": {"name": "demo", "parameters": nested},
    }
    with pytest.raises(ValueError, match=r"keys must be strings"):
        materialize_canonical_tool_definitions_for_llm_dispatch([schema])


def test_ebh_2e_r2_r1_r1_r1_accepts_valid_nested_json() -> None:
    (definition,) = materialize_canonical_tool_definitions_for_llm_dispatch(
        [_valid_nested_wire_schema()]
    )
    assert definition.wire_schema["function"]["name"] == "demo.nested"


def test_ebh_2e_r2_r1_r1_r1_canonical_definition_passthrough_preserves_metadata() -> None:
    atomic = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    guidance = atomic.argument_guidance_text
    assert guidance is not None
    (out,) = materialize_canonical_tool_definitions_for_llm_dispatch([atomic])
    assert out is atomic
    assert out.dispatch_requirements == atomic.dispatch_requirements
    assert out.argument_guidance_text == guidance
    assert out.wire_schema == atomic.wire_schema


def test_ebh_2e_r2_r1_r1_r1_atomic_strict_metadata_not_degraded() -> None:
    atomic = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    (out,) = materialize_canonical_tool_definitions_for_llm_dispatch([atomic])
    assert out.requires_strict_argument_conformance is True
    assert out.dispatch_requirements.argument_conformance is ToolArgumentConformance.STRICT
    assert (
        out.dispatch_requirements.strict_wire_projection
        is StrictWireProjectionKind.ATOMIC_PLANNER_DISCRIMINATED_ACTIONS
    )


def test_ebh_2e_r2_r1_r1_r1_preserves_tool_order() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": f"tool.{index}", "parameters": {"type": "object"}},
        }
        for index in range(3)
    ]
    names = [
        d.wire_schema["function"]["name"]
        for d in materialize_canonical_tool_definitions_for_llm_dispatch(tools)
    ]
    assert names == ["tool.0", "tool.1", "tool.2"]


def test_ebh_2e_r2_r1_r1_r1_duplicate_names_still_rejected() -> None:
    duplicate = {
        "type": "function",
        "function": {"name": "same", "parameters": {"type": "object"}},
    }
    with pytest.raises(StrictToolContractValidationError, match="duplicate"):
        materialize_canonical_tool_definitions_for_llm_dispatch([duplicate, duplicate])


def test_ebh_2e_r2_r1_r1_r1_does_not_mutate_caller_raw_mapping() -> None:
    schema = _valid_nested_wire_schema()
    before = schema["function"]["parameters"]
    materialize_canonical_tool_definitions_for_llm_dispatch([schema])
    assert schema["function"]["parameters"] is before
