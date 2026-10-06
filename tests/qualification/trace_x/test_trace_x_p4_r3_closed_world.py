# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R3 closed-world model-call surface discovery gates."""

from __future__ import annotations

import ast
import subprocess

import pytest

from tests.qualification.trace_x._trace_x_p4_support import (
    MODEL_CALL_SURFACE_INVENTORY,
    TRACE_X_P4_R3_START_HEAD,
    classified_model_call_surfaces,
    discover_model_call_surfaces_ast,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_txp4r3_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R3_START_HEAD, "HEAD"],
    )


def test_txp4r3_q02_model_call_surfaces_closed_world_classified() -> None:
    discovered = discover_model_call_surfaces_ast()
    classified = classified_model_call_surfaces()
    discovered_keys = frozenset((surface.path, surface.method) for surface in discovered)
    unknown = discovered_keys - classified
    orphans = classified - discovered_keys
    assert not unknown, f"unclassified model-call surfaces: {sorted(unknown)}"
    assert not orphans, f"orphan registry entries: {sorted(orphans)}"
    assert len(MODEL_CALL_SURFACE_INVENTORY) == len(discovered_keys)


def test_txp4r3_ast_scanner_finds_synthetic_representative_call() -> None:
    source = "class _Probe:\n    def run(self, adapter):\n        adapter.generate_messages([])\n"
    tree = ast.parse(source)
    methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            methods.add(node.func.attr)
    assert "generate_messages" in methods
