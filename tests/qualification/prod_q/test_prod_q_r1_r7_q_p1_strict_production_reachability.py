# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P1 — strict production reachability regression gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.prod_q._prod_q_r1_r7_q_p1_strict_reachability import (
    BOUNDARY_MODULE_RELPATHS,
    STRICT_PRODUCTION_ROOTS,
    analyze_module_snippet,
    analyze_strict_production_reachability,
    assert_startup_chain_opener,
    boundary_module_paths,
    production_lifecycle_rejection_import_is_guard_only,
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_REACHABLE_FROM_STRICT_OPENER = frozenset(
    {
        (
            "intergrax/applications/_shared/strict_production_serving_launcher.py",
            "open_strict_production_serving_session",
        ),
        (
            "intergrax/applications/_shared/production_process_lifecycle.py",
            "open_production_process_lifecycle",
        ),
        (
            "intergrax/applications/_shared/production_lifecycle.py",
            "reject_reference_runtime_deployment_adapter",
        ),
        (
            "intergrax/applications/_shared/production_lifecycle.py",
            "wire_production_lifecycle_services",
        ),
    }
)


def test_p1_g01_one_strict_production_root_chain() -> None:
    assert_startup_chain_opener()
    report = analyze_strict_production_reachability()
    missing = _EXPECTED_REACHABLE_FROM_STRICT_OPENER - report.reachable_functions
    assert not missing, f"strict opener chain missing reachable functions: {missing}"


def test_p1_g02_no_prohibited_execution_on_strict_reachable_graph() -> None:
    report = analyze_strict_production_reachability()
    assert not report.prohibited, "\n".join(
        f"{v.module_relpath}:{v.lineno} {v.qualname} -> {v.detail}" for v in report.prohibited
    )


def test_p1_g03_reference_lifecycle_import_without_execution_in_strict_modules() -> None:
    rel = "intergrax/applications/_shared/strict_production_serving_launcher.py"
    path = repo_root() / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    text = path.read_text(encoding="utf-8")
    assert "open_reference_production_process_lifecycle" not in text
    report = analyze_strict_production_reachability()
    ref_calls = [v for v in report.prohibited if "open_reference" in v.detail]
    assert not ref_calls


def test_p1_g04_gca_strict_factory_does_not_construct_deployment_adapters() -> None:
    report = analyze_strict_production_reachability()
    assert not report.gca_deployment_constructions


def test_p1_g05_strict_asgi_admission_symbols_present() -> None:
    main = repo_root() / "applications/governed_contractor_application/host/main.py"
    strict = (
        repo_root()
        / "applications/governed_contractor_application/host/strict_production_serving.py"
    )
    main_text = main.read_text(encoding="utf-8")
    strict_text = strict.read_text(encoding="utf-8")
    assert "StrictProductionAsgiPlaceholder" in main_text
    assert "ready_for_traffic" in strict_text
    assert "create_governed_contractor_process_app" in strict_text


def test_p1_g06_closed_world_inventory_complete() -> None:
    missing = [rel for rel, path in boundary_module_paths().items() if not path.is_file()]
    assert not missing, f"boundary module missing: {missing}"
    assert len(BOUNDARY_MODULE_RELPATHS) == len(boundary_module_paths())


def test_p1_production_lifecycle_fake_import_is_rejection_guard_only() -> None:
    production_lifecycle_rejection_import_is_guard_only()
    proc = repo_root() / "intergrax/applications/_shared/production_process_lifecycle.py"
    assert "FakeInMemoryRuntimeDeploymentAdapter" not in proc.read_text(encoding="utf-8")


def test_p1_critical_authority_calls_resolve_without_unresolved_edges() -> None:
    report = analyze_strict_production_reachability()
    assert not report.unresolved_calls, "\n".join(
        f"{c.module_relpath}:{c.lineno} unresolved {c.resolved_target}" for c in report.unresolved_calls
    )


def test_p1_negative_fixture_detects_fake_adapter_instantiation() -> None:
    snippet = """
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter

def evil():
    return FakeInMemoryRuntimeDeploymentAdapter()
"""
    violations = analyze_module_snippet(snippet)
    assert violations


def test_p1_negative_fixture_allows_isinstance_rejection_guard() -> None:
    snippet = """
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter

def reject_reference_runtime_deployment_adapter(adapter):
    if isinstance(adapter, FakeInMemoryRuntimeDeploymentAdapter):
        raise ValueError("forbidden")
"""
    violations = analyze_module_snippet(snippet)
    assert not violations


def test_p1_negative_fixture_detects_open_reference_lifecycle_call() -> None:
    snippet = """
from intergrax.applications._shared.reference_production_process_lifecycle import (
    open_reference_production_process_lifecycle,
)

def bypass():
    open_reference_production_process_lifecycle()
"""
    violations = analyze_module_snippet(snippet)
    assert violations


def test_p1_trusted_roots_documented() -> None:
    assert len(STRICT_PRODUCTION_ROOTS) >= 4
    roots = {r[1] for r in STRICT_PRODUCTION_ROOTS}
    assert "open_strict_production_serving_session" in roots
    assert "run" in roots


def test_p1_coverage_boundary_is_static_callsites_only() -> None:
    """Documented limitation: no dynamic dispatch or attribute call graph."""
    report = analyze_strict_production_reachability()
    assert report.reachable_functions, "expected non-empty bounded reachability set"
