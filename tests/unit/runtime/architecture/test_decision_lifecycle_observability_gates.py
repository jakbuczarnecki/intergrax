# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _module_source(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _module_imports(relative_path: str) -> set[str]:
    tree = ast.parse(_module_source(relative_path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_decision_contracts_do_not_import_runtime_events() -> None:
    contracts_root = _REPO_ROOT / "intergrax" / "contracts"
    forbidden = "intergrax.runtime.events"
    for path in contracts_root.rglob("decision*.py"):
        source = path.read_text(encoding="utf-8")
        assert forbidden not in source, f"{path} imports runtime.events"


def test_decision_contracts_do_not_import_diagnostics() -> None:
    contracts_root = _REPO_ROOT / "intergrax" / "contracts"
    forbidden = "intergrax.runtime.diagnostics"
    for path in contracts_root.rglob("decision*.py"):
        source = path.read_text(encoding="utf-8")
        assert forbidden not in source, f"{path} imports diagnostics"


def test_lifecycle_host_does_not_import_diagnostic_projector() -> None:
    imports = _module_imports("intergrax/runtime/execution/decision_lifecycle_host.py")
    assert "intergrax.runtime.diagnostics.decision_lifecycle_projection" not in imports


def test_diagnostic_projector_does_not_import_lifecycle_host() -> None:
    imports = _module_imports("intergrax/runtime/diagnostics/decision_lifecycle_projection.py")
    assert "intergrax.runtime.execution.decision_lifecycle_host" not in imports
    assert "transition_decision_lifecycle" not in _module_source(
        "intergrax/runtime/diagnostics/decision_lifecycle_projection.py",
    )


def test_no_decision_event_bus_types_in_repo() -> None:
    forbidden_names = (
        "DecisionObservabilityLifecycle",
        "DiagnosticDecisionLifecycle",
        "DecisionAuditRuntime",
        "DecisionEventRuntime",
        "DecisionDiagnosticsSystem",
        "DecisionDiagnosticRuntime",
    )
    runtime_root = _REPO_ROOT / "intergrax" / "runtime"
    for path in runtime_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden_names:
            assert name not in source, f"{path} defines forbidden type {name}"


def test_lifecycle_observability_uses_emit_domain_signal() -> None:
    source = _module_source("intergrax/runtime/decision_lifecycle_observability.py")
    assert "emit_domain_signal" in source
    assert "logger." not in source
