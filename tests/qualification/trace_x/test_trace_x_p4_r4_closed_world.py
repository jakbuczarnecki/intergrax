# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R4 independent closed-world registry and negative-sensitivity gates."""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p4_model_surface_registry import MODEL_CALL_SURFACE_REGISTRY
from tests.qualification.trace_x._trace_x_p4_context_surface_registry import CONTEXT_SURFACE_REGISTRY
from tests.qualification.trace_x._trace_x_p4_registry_types import ModelCallSurfaceClassification
from tests.qualification.trace_x._trace_x_p4_support import (
    TRACE_X_P4_R4_START_HEAD,
    compare_context_surfaces_to_registry,
    compare_model_call_surfaces_to_registry,
    discovered_context_surface_keys,
    discovered_model_call_surface_keys,
)
from tests.qualification.trace_x import _trace_x_p4_support as p4_support

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_txp4r4_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R4_START_HEAD, "HEAD"],
    )


def test_txp4r4_q02_model_call_surfaces_closed_world_parity() -> None:
    result = compare_model_call_surfaces_to_registry(
        discovered_model_call_surface_keys(),
        MODEL_CALL_SURFACE_REGISTRY,
    )
    assert result.duplicate_registry_keys == frozenset()
    assert not result.unknown, f"unknown model-call surfaces: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan model registry rows: {sorted(result.orphan)}"
    assert result.ok


def test_txp4r4_q03_model_call_negative_sensitivity_unregistered_surface() -> None:
    baseline = compare_model_call_surfaces_to_registry(
        discovered_model_call_surface_keys(),
        MODEL_CALL_SURFACE_REGISTRY,
    )
    assert baseline.ok
    synthetic_key = (
        "intergrax/runtime/example_new_model_path.py",
        "generate_messages",
    )
    augmented = frozenset(discovered_model_call_surface_keys() | {synthetic_key})
    failed = compare_model_call_surfaces_to_registry(augmented, MODEL_CALL_SURFACE_REGISTRY)
    assert synthetic_key in failed.unknown
    assert not failed.ok


def test_txp4r4_q04_context_surfaces_closed_world_parity() -> None:
    result = compare_context_surfaces_to_registry(
        discovered_context_surface_keys(),
        CONTEXT_SURFACE_REGISTRY,
    )
    assert result.duplicate_registry_keys == frozenset()
    assert not result.unknown, f"unknown context surfaces: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan context registry rows: {sorted(result.orphan)}"
    assert result.ok


def test_txp4r4_q05_context_negative_sensitivity_unregistered_surface() -> None:
    baseline = compare_context_surfaces_to_registry(
        discovered_context_surface_keys(),
        CONTEXT_SURFACE_REGISTRY,
    )
    assert baseline.ok
    synthetic_key = (
        "intergrax/runtime/example_new_context_producer.py",
        "emit:runtime_event_context_assembled",
    )
    augmented = frozenset(discovered_context_surface_keys() | {synthetic_key})
    failed = compare_context_surfaces_to_registry(augmented, CONTEXT_SURFACE_REGISTRY)
    assert synthetic_key in failed.unknown
    assert not failed.ok


def test_txp4r4_q06_model_registry_is_static_not_discovery_derived() -> None:
    registry_source = inspect.getsource(
        __import__(
            "tests.qualification.trace_x._trace_x_p4_model_surface_registry",
            fromlist=["MODEL_CALL_SURFACE_REGISTRY"],
        ),
    )
    assert "discover_model_call_surfaces_ast" not in registry_source
    assert "build_model_call_surface_inventory" not in registry_source


def test_txp4r4_q07_no_permissive_model_call_classification_fallback() -> None:
    support_source = inspect.getsource(p4_support)
    assert "_classify_model_call_surface" not in support_source
    assert "build_model_call_surface_inventory" not in support_source


def test_txp4r4_q08_not_llm_adapter_registry_rows_under_llm_adapters() -> None:
    for row in MODEL_CALL_SURFACE_REGISTRY:
        if row.classification != ModelCallSurfaceClassification.NOT_LLM_ADAPTER_CALL:
            continue
        assert row.path.startswith("intergrax/llm_adapters/"), row.path


def test_txp4r4_q09_production_composition_roots_attribution_seam() -> None:
    bridge = (_REPO_ROOT / "intergrax/applications/_shared/runtime_config_bridge.py").read_text(
        encoding="utf-8",
    )
    assert "wrap_model_call_runtime_evidence" in bridge
    roots = (
        _REPO_ROOT / "intergrax/applications/_shared/runtime_config_bridge.py",
        _REPO_ROOT / "intergrax/runtime/execution/inference.py",
    )
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found_wrap = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "wrap_model_call_runtime_evidence":
                    found_wrap = True
                    break
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "wrap_model_call_runtime_evidence":
                    found_wrap = True
                    break
        if path.name == "runtime_config_bridge.py":
            assert found_wrap
    primary_rows = [
        row
        for row in MODEL_CALL_SURFACE_REGISTRY
        if row.classification == ModelCallSurfaceClassification.CANONICAL_PRIMARY
    ]
    assert primary_rows
    assert all(row.path == "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py" for row in primary_rows)
