# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R2-R1-R1-R1-R1-R1 — Knowledge JSON validation backward compatibility (tuple admission)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from intergrax.contracts.structured_json_value import (
    validate_json_value_structure,
    validate_structured_json_value,
)
from intergrax.knowledge.contracts.validation import validate_json_value

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KNOWLEDGE_VALIDATION = _REPO_ROOT / "intergrax/knowledge/contracts/validation.py"
_STRUCTURED_JSON = _REPO_ROOT / "intergrax/contracts/structured_json_value.py"
_MATERIALIZER = _REPO_ROOT / "intergrax/runtime/nexus/tools/canonical_tool_dispatch.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _imports_knowledge_module(source: str, module_suffix: str) -> bool:
    pattern = rf"from\s+intergrax\.knowledge\.{module_suffix}\s+import"
    return re.search(pattern, source) is not None


@pytest.mark.parametrize(
    "value",
    [
        (1, 2),
        {"x": (1, 2)},
        [{"x": (1, 2)}],
    ],
)
def test_ebh_2e_r2_r1_r1_r1_r1_r1_knowledge_rejects_tuple_containers(value: object) -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_json_value(value, field_name="metadata")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_knowledge_accepts_list_containers() -> None:
    assert validate_json_value([1, 2], field_name="metadata") == [1, 2]
    assert validate_json_value({"x": [1, 2]}, field_name="metadata") == {"x": [1, 2]}


def test_ebh_2e_r2_r1_r1_r1_r1_r1_platform_tuple_normalized_to_list() -> None:
    assert validate_structured_json_value((1, 2), field_name="v") == [1, 2]
    assert validate_json_value_structure((1, 2), field_name="v") == [1, 2]


def test_ebh_2e_r2_r1_r1_r1_r1_r1_dual_proof_tuple_knowledge_fail_platform_pass() -> None:
    assert validate_json_value_structure((1, 2), field_name="metadata") == [1, 2]
    with pytest.raises(ValueError, match="JSON-compatible"):
        validate_json_value((1, 2), field_name="metadata")


def test_ebh_2e_r2_r1_r1_r1_r1_r1_knowledge_no_duplicate_structural_json_validator() -> None:
    source = _read(_KNOWLEDGE_VALIDATION)
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "validate_structured_json_value" not in defined
    assert "validate_json_value_structure" not in defined
    assert "if isinstance(value, Enum)" not in source
    assert "math.isfinite" not in source
    assert "_reject_knowledge_legacy_tuple_containers" in source


def test_ebh_2e_r2_r1_r1_r1_r1_r1_knowledge_imports_neutral_json_contract() -> None:
    source = _read(_KNOWLEDGE_VALIDATION)
    assert "from intergrax.contracts.structured_json_value import" in source
    assert "validate_structured_json_value" in source


def test_ebh_2e_r2_r1_r1_r1_r1_r1_runtime_materializer_does_not_import_knowledge() -> None:
    src = _read(_MATERIALIZER)
    assert not _imports_knowledge_module(src, "contracts.validation")
    assert "validate_json_value(" not in src


def test_ebh_2e_r2_r1_r1_r1_r1_r1_structured_json_owner_unchanged_tuple_support() -> None:
    src = _read(_STRUCTURED_JSON)
    assert "list | tuple" in src
