# © Artur Czarnecki. All rights reserved.

"""Regression: EE ``application_graph_spec_to_plan`` must remain a real implementation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "execution"
    / "application_graph_spec_to_plan.py"
)


def test_application_graph_spec_to_plan_is_not_self_import_shim() -> None:
    source = _MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == (
            "intergrax.runtime.execution.application_graph_spec_to_plan"
        ):
            raise AssertionError(
                "application_graph_spec_to_plan must not import itself; "
                "restore implementation in this module"
            )
    assert "def application_graph_spec_to_nexus_plan" in source
    assert "def should_seed_plan_from_graph_spec" in source
