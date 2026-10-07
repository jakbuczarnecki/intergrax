# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R3 closed-world model-call surface discovery gates."""

from __future__ import annotations

import ast
import subprocess

import pytest

from tests.qualification.trace_x._trace_x_p4_model_surface_registry import MODEL_CALL_SURFACE_REGISTRY
from tests.qualification.trace_x._trace_x_p4_support import (
    MODEL_CALL_SURFACE_INVENTORY,
    TRACE_X_P4_R3_START_HEAD,
    compare_model_call_surfaces_to_registry,
    discovered_model_call_surface_keys,
    discover_model_call_surfaces_ast,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_txp4r3_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R3_START_HEAD, "HEAD"],
    )


def test_txp4r3_q02_model_call_surfaces_closed_world_classified() -> None:
    discovered = discover_model_call_surfaces_ast()
    result = compare_model_call_surfaces_to_registry(
        discovered_model_call_surface_keys(),
        MODEL_CALL_SURFACE_REGISTRY,
    )
    assert not result.unknown, f"unclassified model-call surfaces: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan registry entries: {sorted(result.orphan)}"
    assert len(MODEL_CALL_SURFACE_INVENTORY) == len(discovered)


def test_txp4r3_ast_scanner_finds_synthetic_representative_call() -> None:
    source = "class _Probe:\n    def run(self, adapter):\n        adapter.generate_messages([])\n"
    tree = ast.parse(source)
    methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            methods.add(node.func.attr)
    assert "generate_messages" in methods
