# © Artur Czarnecki. All rights reserved.

"""Architecture guards for ERL-QUAL-004 full execution composition."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_ROOT = _REPO_ROOT / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"
_EXECUTION_ROOT = _SCENARIO_ROOT / "application/execution"
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"

_FORBIDDEN_ENGINE_NAMES = frozenset(
    {
        "PaymentScenarioEngine",
        "CustomReliabilityEngine",
        "CustomWorkflowEngine",
    }
)

_ALLOWED_ERL_IMPORT_PREFIXES = (
    "intergrax.runtime.enterprise_reliability",
    "intergrax.contracts.enterprise_reliability",
)


def _python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if path.is_file()]


def test_intergrax_does_not_import_scenario_package() -> None:
    needle = "enterprise_payment_uncertainty_recovery"
    violations: list[str] = []
    for path in _python_files(_INTERGRAX_ROOT):
        text = path.read_text(encoding="utf-8")
        if needle in text:
            violations.append(str(path.relative_to(_REPO_ROOT)))
    assert not violations, violations


def test_execution_layer_has_no_forbidden_engine_types() -> None:
    violations: list[str] = []
    for path in _python_files(_EXECUTION_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in _FORBIDDEN_ENGINE_NAMES:
                violations.append(f"{path.name}:{node.name}")
    assert not violations, violations


def test_execution_composition_imports_only_platform_erl_boundaries() -> None:
    violations: list[str] = []
    for path in _python_files(_EXECUTION_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                "intergrax.",
            ):
                if not node.module.startswith(_ALLOWED_ERL_IMPORT_PREFIXES):
                    violations.append(f"{path.name}: {node.module}")
    assert not violations, violations


def test_execution_runner_has_no_variant_label_branching() -> None:
    runner = _EXECUTION_ROOT / "runner.py"
    text = runner.read_text(encoding="utf-8").lower()
    forbidden_tokens = (
        "payment_completed_after_unknown",
        "payment_failed_after_unknown",
        "payment_truth_unavailable",
        "variant a",
        "variant b",
        "variant c",
    )
    violations = [token for token in forbidden_tokens if token in text]
    assert not violations, f"runner branches on variant labels: {violations}"
