# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R2-R1-R1-R1-R1 — pure JSON value validation ownership (no Knowledge policy leakage)."""

from __future__ import annotations

import ast
import importlib
import math
import re
from pathlib import Path

import pytest

from intergrax.contracts.structured_json_value import (
    JsonValue,
    validate_json_value_structure,
    validate_structured_json_value,
)
from intergrax.knowledge.contracts.validation import assert_safe_mapping, validate_json_value
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.runtime.nexus.tools.canonical_tool_dispatch import (
    materialize_canonical_tool_definitions_for_llm_dispatch,
)
from intergrax.tools.exporters.schema import pydantic_parameters_schema
from intergrax.tools.providers.identity.contracts import IdentityVerifyTokenInput

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STRUCTURED_JSON = _REPO_ROOT / "intergrax/contracts/structured_json_value.py"
_MATERIALIZER = _REPO_ROOT / "intergrax/runtime/nexus/tools/canonical_tool_dispatch.py"
_SERIALIZED_VALUE = _REPO_ROOT / "intergrax/llm_adapters/contracts/serialized_value.py"
_KNOWLEDGE_VALIDATION = _REPO_ROOT / "intergrax/knowledge/contracts/validation.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _imports_knowledge_module(source: str, module_suffix: str) -> bool:
    pattern = rf"from\s+intergrax\.knowledge\.{module_suffix}\s+import"
    return re.search(pattern, source) is not None


def _wire_tool_with_property(field_name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": f"demo.{field_name}",
            "description": "schema property name regression",
            "parameters": {
                "type": "object",
                "properties": {field_name: {"type": "string"}},
                "required": [field_name],
                "additionalProperties": False,
            },
        },
    }


def test_ebh_2e_r2_r1_r1_r1_r1_canonical_json_types_defined_once() -> None:
    structured_src = _read(_STRUCTURED_JSON)
    knowledge_src = _read(_KNOWLEDGE_VALIDATION)
    assert "type JsonValue" in structured_src or "type JsonValue =" in structured_src
    assert "type JsonValue =" not in knowledge_src
    assert "type JsonPrimitive" not in knowledge_src
    assert "type JsonObject" not in knowledge_src


def test_ebh_2e_r2_r1_r1_r1_r1_neutral_validator_has_no_domain_policy_tokens() -> None:
    src = _read(_STRUCTURED_JSON)
    forbidden = (
        "SecretSafeValidationPolicy",
        "is_secret_like_key",
        "validate_safe_url",
        "KNOWLEDGE_SECRET_POLICY",
        "intergrax.knowledge",
        "intergrax.llm_adapters",
        "intergrax.runtime",
    )
    for token in forbidden:
        assert token not in src


def test_ebh_2e_r2_r1_r1_r1_r1_materializer_uses_neutral_validator_not_knowledge() -> None:
    src = _read(_MATERIALIZER)
    assert "validate_json_value_structure" in src
    assert not _imports_knowledge_module(src, "contracts.validation")
    assert "validate_json_value(" not in src


def test_ebh_2e_r2_r1_r1_r1_r1_serialized_value_has_no_knowledge_import() -> None:
    src = _read(_SERIALIZED_VALUE)
    assert not _imports_knowledge_module(src, "contracts.validation")
    assert "intergrax.contracts.structured_json_value" in src


def test_ebh_2e_r2_r1_r1_r1_r1_knowledge_still_enforces_secret_policy() -> None:
    with pytest.raises(ValueError, match="secret-bearing key"):
        assert_safe_mapping({"token": "not-a-secret-value"}, field_name="metadata")


@pytest.mark.parametrize(
    "field_name",
    ["token", "password", "authorization", "api_key", "secret"],
)
def test_ebh_2e_r2_r1_r1_r1_r1_tool_schema_security_field_names_pass(
    field_name: str,
) -> None:
    (definition,) = materialize_canonical_tool_definitions_for_llm_dispatch(
        [_wire_tool_with_property(field_name)]
    )
    assert isinstance(definition, CanonicalFunctionToolDefinition)
    params = definition.wire_schema["function"]["parameters"]
    assert isinstance(params, dict)
    props = params.get("properties")
    assert isinstance(props, dict)
    assert field_name in props


def test_ebh_2e_r2_r1_r1_r1_r1_identity_verify_token_input_end_to_end() -> None:
    wire = {
        "type": "function",
        "function": {
            "name": "identity.verify_token",
            "description": "Verify bearer token",
            "parameters": pydantic_parameters_schema(IdentityVerifyTokenInput),
        },
    }
    (definition,) = materialize_canonical_tool_definitions_for_llm_dispatch([wire])
    params = definition.wire_schema["function"]["parameters"]
    assert isinstance(params, dict)
    props = params.get("properties")
    assert isinstance(props, dict)
    assert "token" in props


def test_ebh_2e_r2_r1_r1_r1_r1_pure_validator_rejects_custom_object() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_json_value_structure({"x": object()}, field_name="tools[0]")


def test_ebh_2e_r2_r1_r1_r1_r1_pure_validator_rejects_non_string_key() -> None:
    bad: dict[object, object] = {"ok": 1}
    bad[1] = "x"
    with pytest.raises(ValueError, match="keys must be strings"):
        validate_json_value_structure(bad, field_name="tools[0]")


def test_ebh_2e_r2_r1_r1_r1_r1_pure_validator_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_json_value_structure(math.nan, field_name="v")


def test_ebh_2e_r2_r1_r1_r1_r1_dual_proof_token_tool_pass_knowledge_fail() -> None:
    schema = _wire_tool_with_property("token")
    materialize_canonical_tool_definitions_for_llm_dispatch([schema])
    with pytest.raises(ValueError, match="secret-bearing key"):
        validate_json_value({"token": "x"}, field_name="metadata")


def test_ebh_2e_r2_r1_r1_r1_r1_json_value_import_authority() -> None:
    mod = importlib.import_module("intergrax.contracts.structured_json_value")
    assert hasattr(mod, "JsonValue")
    knowledge = importlib.import_module("intergrax.knowledge.contracts.validation")
    assert knowledge.JsonValue == mod.JsonValue


def test_ebh_2e_r2_r1_r1_r1_r1_knowledge_validation_ast_still_has_policy() -> None:
    tree = ast.parse(_read(_KNOWLEDGE_VALIDATION))
    source = _read(_KNOWLEDGE_VALIDATION)
    assert "KNOWLEDGE_SECRET_POLICY" in source
    assert "validate_safe_url" in source
    assert "_enforce_knowledge_metadata_policies" in source
    assert "validate_structured_json_value" in source
    assert not _ast_defines_json_value_alias(tree)


def _ast_defines_json_value_alias(module: ast.Module) -> bool:
    for node in module.body:
        if isinstance(node, ast.TypeAlias) and getattr(node, "name", None) == "JsonValue":
            return True
    return False


def test_ebh_2e_r2_r1_r1_r1_r1_pure_structure_allows_secret_key_names() -> None:
    result = validate_json_value_structure(
        {"token": "schema-metadata-only"},
        field_name="tools[0]",
    )
    assert result == {"token": "schema-metadata-only"}


def test_ebh_2e_r2_r1_r1_r1_r1_validate_json_value_structure_matches_structured() -> None:
    sample: JsonValue = {"a": [1, {"b": True}]}
    assert validate_json_value_structure(sample, field_name="v") == sample
    assert validate_structured_json_value(sample, field_name="v") == sample
