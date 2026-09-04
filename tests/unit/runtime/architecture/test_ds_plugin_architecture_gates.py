# © Artur Czarnecki. All rights reserved.

"""DS-PLUGIN architecture gates — no second Decision plugin lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PRODUCTION_ROOTS = (
    _REPO_ROOT / "intergrax" / "contracts",
    _REPO_ROOT / "intergrax" / "runtime",
)

_FORBIDDEN_CLASS_NAMES = (
    "DecisionPluginSystem",
    "DecisionPluginDiscovery",
    "DecisionPluginManager",
    "DecisionPluginLoader",
    "DecisionPluginRegistry",
)

_CONTRACT_MODULES = (
    "intergrax/contracts/decision_strategy.py",
    "intergrax/contracts/decision_verification_stage.py",
    "intergrax/contracts/decision_artifact_registry.py",
)

_FORBIDDEN_CONTRACT_IMPORTS = (
    "intergrax.core.plugins.discovery",
    "importlib.metadata",
    "setuptools",
)


def _iter_production_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return files


@pytest.mark.parametrize("forbidden_name", _FORBIDDEN_CLASS_NAMES)
def test_no_forbidden_decision_plugin_framework_classes(forbidden_name: str) -> None:
    for path in _iter_production_py_files():
        source = path.read_text(encoding="utf-8")
        assert forbidden_name not in source, f"{path} must not define {forbidden_name}"


@pytest.mark.parametrize("module_path", _CONTRACT_MODULES)
def test_decision_contracts_do_not_import_discovery(module_path: str) -> None:
    source = (_REPO_ROOT / module_path).read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_CONTRACT_IMPORTS:
        assert fragment not in source, f"{module_path} must not import {fragment}"


def test_decision_registries_remain_in_contracts() -> None:
    strategy_source = (
        _REPO_ROOT / "intergrax/contracts/decision_strategy.py"
    ).read_text(encoding="utf-8")
    verification_source = (
        _REPO_ROOT / "intergrax/contracts/decision_verification_stage.py"
    ).read_text(encoding="utf-8")
    artifact_source = (
        _REPO_ROOT / "intergrax/contracts/decision_artifact_registry.py"
    ).read_text(encoding="utf-8")
    assert "class DecisionStrategyRegistry" in strategy_source
    assert "class VerificationStageRegistry" in verification_source
    assert "class DecisionArtifactKindRegistry" in artifact_source
