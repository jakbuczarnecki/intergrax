# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-EC1 — strong typing gates for canonical OBS integration contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.integrations.contracts.observability_backend import TraceRecord

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OBS_BACKEND_CONTRACT = _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "observability_backend.py"


def test_trace_record_metadata_is_typed_observability_attributes() -> None:
    from intergrax.contracts.application_observability_attributes import ObservabilityAttributeValue

    field = TraceRecord.model_fields["metadata"]
    assert field.annotation == dict[str, ObservabilityAttributeValue]


def test_observability_backend_contract_no_raw_any_metadata_annotation() -> None:
    tree = ast.parse(_OBS_BACKEND_CONTRACT.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "TraceRecord":
            continue
        for item in node.body:
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id == "metadata"
                and item.annotation is not None
            ):
                src = ast.unparse(item.annotation)
                assert "Any" not in src
                return
    pytest.fail("TraceRecord.metadata annotation not found")
