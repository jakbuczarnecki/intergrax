# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R2-R1-R1-R1-R1-R1-R1 — Knowledge JSON primitive return semantics preservation."""

from __future__ import annotations

import ast
import math
import re
from enum import Enum
from pathlib import Path

import pytest

from intergrax.contracts.structured_json_value import (
    validate_json_value_structure,
    validate_structured_json_value,
)
from intergrax.knowledge.contracts.validation import (
    _enforce_knowledge_metadata_policies,
    assert_safe_mapping,
    validate_json_value,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KNOWLEDGE_VALIDATION = _REPO_ROOT / "intergrax/knowledge/contracts/validation.py"
_STRUCTURED_JSON = _REPO_ROOT / "intergrax/contracts/structured_json_value.py"
_MATERIALIZER = _REPO_ROOT / "intergrax/runtime/nexus/tools/canonical_tool_dispatch.py"
_SERIALIZED_VALUE = _REPO_ROOT / "intergrax/llm_adapters/contracts/serialized_value.py"
_LLM_PROFILE = _REPO_ROOT / "intergrax/llm_adapters/contracts/llm_profile.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


class _SampleEnum(Enum):
    A = "a"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (False, False),
        (42, 42),
        (-7, -7),
        (0, 0),
        (1.25, 1.25),
        ("plain", "plain"),
        ("https://example.test/item?page=1", "https://example.test/item?page=1"),
    ],
)
def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_primitive_return_matrix(
    value: object,
    expected: object,
) -> None:
    result = validate_json_value(value, field_name="v")
    assert result == expected
    if value is None:
        assert result is None
    elif isinstance(value, bool):
        assert result is value
        assert type(result) is bool
    elif isinstance(value, int) and not isinstance(value, bool):
        assert type(result) is int
    elif isinstance(value, float):
        assert type(result) is float


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_bool_not_coerced_to_int() -> None:
    assert validate_json_value(True, field_name="v") is True
    assert type(validate_json_value(True, field_name="v")) is bool
    assert validate_json_value(1, field_name="v") == 1
    assert type(validate_json_value(1, field_name="v")) is int


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_list_and_dict_return_semantics() -> None:
    assert validate_json_value([1, "x", True], field_name="v") == [1, "x", True]
    nested = {"a": [1, {"b": True}]}
    assert validate_json_value(nested, field_name="v") == nested


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_credential_ref_still_passes() -> None:
    result = assert_safe_mapping(
        {"credential_ref": "vault://item/1", "region": "eu"},
        field_name="metadata",
    )
    assert result["credential_ref"] == "vault://item/1"


@pytest.mark.parametrize(
    "value",
    [
        (1, 2),
        {"x": (1, 2)},
        [{"nested": (1,)}],
    ],
)
def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_error_matrix_tuple(value: object) -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_json_value(value, field_name="v")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_error_matrix_enum() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_json_value(_SampleEnum.A, field_name="v")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_error_matrix_custom_object() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_json_value({"x": object()}, field_name="v")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_error_matrix_non_string_key() -> None:
    bad: dict[object, object] = {"ok": 1}
    bad[1] = "x"
    with pytest.raises(ValueError, match="keys must be strings"):
        validate_json_value(bad, field_name="v")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_error_matrix_non_finite_float(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_json_value(value, field_name="v")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_error_matrix_secret_key() -> None:
    with pytest.raises(ValueError, match="secret-bearing key"):
        validate_json_value({"token": "x"}, field_name="v")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_error_matrix_unsafe_url() -> None:
    with pytest.raises(ValueError, match="must not embed credentials"):
        validate_json_value(
            "https://user:pass@example.test/item",
            field_name="v",
        )


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_domain_separation_tuple() -> None:
    assert validate_structured_json_value((1, 2), field_name="v") == [1, 2]
    assert validate_json_value_structure((1, 2), field_name="v") == [1, 2]
    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_json_value((1, 2), field_name="v")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_domain_separation_secret_key_name() -> None:
    payload = {"token": "schema-metadata-only"}
    assert validate_json_value_structure(payload, field_name="tools[0]") == payload
    with pytest.raises(ValueError, match="secret-bearing key"):
        validate_json_value(payload, field_name="metadata")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_policy_wrapper_returns_json_value_for_valid_samples() -> None:
    samples: list[object] = [
        None,
        True,
        False,
        0,
        42,
        -3,
        1.5,
        "plain",
        "https://example.test/safe",
        [1, "x"],
        {"k": 1},
    ]
    for sample in samples:
        structured = validate_structured_json_value(sample, field_name="v")
        policy_result = _enforce_knowledge_metadata_policies(
            structured,
            field_name="v",
        )
        assert policy_result == structured


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_enforce_policies_ast_final_return_branch() -> None:
    tree = ast.parse(_read(_KNOWLEDGE_VALIDATION))
    func_def: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_enforce_knowledge_metadata_policies":
            func_def = node
            break
    assert func_def is not None
    body_stmts = func_def.body
    assert body_stmts
    last = body_stmts[-1]
    assert isinstance(last, ast.Return)
    assert isinstance(last.value, ast.Name)
    assert last.value.id == "value"


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_policy_wrapper_no_optional_return_annotation() -> None:
    source = _read(_KNOWLEDGE_VALIDATION)
    match = re.search(
        r"def _enforce_knowledge_metadata_policies\([^)]*\)\s*->\s*([^\n:]+)",
        source,
    )
    assert match is not None
    annotation = match.group(1).strip()
    assert "None" not in annotation
    assert "Optional" not in annotation


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_no_platform_owner_change_in_task() -> None:
    assert "validate_structured_json_value" in _read(_STRUCTURED_JSON)


def test_ebh_2e_r2_r1_r1_r1_r1_r1_r1_no_llm_runtime_import_knowledge_validation() -> None:
    for path in (_MATERIALIZER, _SERIALIZED_VALUE, _LLM_PROFILE):
        src = _read(path)
        assert "intergrax.knowledge.contracts.validation" not in src
        assert "validate_json_value(" not in src
