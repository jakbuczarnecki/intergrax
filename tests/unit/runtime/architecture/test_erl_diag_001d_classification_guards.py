# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001D architecture guards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CLASSIFICATION_CONTRACTS = (
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "enterprise_reliability"
    / "diagnostics"
    / "classification.py"
)
_RELIABILITY_PKG = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "reliability"
_BRIDGE_MAP = _RELIABILITY_PKG / "observation_to_problem_signal.py"
_PLUGIN_SEVERITY = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "erl_diagnostics_plugins"
    / "critical_truth_unavailable_severity_strategy.py"
)
_DEFAULT_SEVERITY = _RELIABILITY_PKG / "conservative_reliability_severity_strategy.py"
_DEFAULT_RECOMMENDATION = _RELIABILITY_PKG / "conservative_reliability_recommendation_strategy.py"

_FORBIDDEN_IMPORT_PREFIXES = (
    "platform_proofs",
    "applications",
    "intergrax.runtime.governance",
    "intergrax.runtime.enterprise_reliability",
    "intergrax.runtime.recovery",
)


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_classification_contracts_do_not_import_runtime() -> None:
    for module in _collect_imports(_CLASSIFICATION_CONTRACTS):
        assert not module.startswith("intergrax.runtime"), module


def test_classification_runtime_imports_public_contracts() -> None:
    service = _RELIABILITY_PKG / "reliability_diagnostic_classification_service.py"
    imports = _collect_imports(service)
    assert "intergrax.contracts.enterprise_reliability.diagnostics.classification" in imports


def test_reliability_pkg_has_no_governance_or_scenario_imports() -> None:
    violations: list[str] = []
    for path in sorted(_RELIABILITY_PKG.rglob("*.py")):
        for module in _collect_imports(path):
            for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(f"{path.relative_to(_REPO_ROOT)}: {module}")
    assert violations == []


def test_default_strategies_have_no_payment_terms() -> None:
    for path in (_DEFAULT_SEVERITY, _DEFAULT_RECOMMENDATION):
        lowered = path.read_text(encoding="utf-8").lower()
        assert "payment" not in lowered
        assert "charge" not in lowered


def test_bridge_has_no_severity_or_recommendation_mapping() -> None:
    source = _BRIDGE_MAP.read_text(encoding="utf-8").lower()
    assert "severity=" not in source.replace("platform_problem_signal", "")
    assert "recommendation" not in source


def test_external_plugin_imports_only_public_contracts() -> None:
    imports = _collect_imports(_PLUGIN_SEVERITY)
    intergrax_imports = [module for module in imports if module.startswith("intergrax.")]
    assert intergrax_imports
    for module in intergrax_imports:
        assert module.startswith("intergrax.contracts.enterprise_reliability.diagnostics"), module
        assert not module.startswith("intergrax.runtime"), module


def test_classification_service_has_no_getattr() -> None:
    source = (
        _RELIABILITY_PKG / "reliability_diagnostic_classification_service.py"
    ).read_text(encoding="utf-8")
    assert "getattr" not in source
    assert "setattr" not in source
