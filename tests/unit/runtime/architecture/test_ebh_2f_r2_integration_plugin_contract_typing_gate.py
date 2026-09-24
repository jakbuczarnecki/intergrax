# © Artur Czarnecki. All rights reserved.

"""EBH-2F-R2 — integration plugin public contract typing and registration purity gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_PLUGIN_CONTRACT = _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "plugin.py"
_PLUGIN_REGISTER = _REPO_ROOT / "intergrax" / "integrations" / "registry" / "plugin_register.py"
_CATALOG_FACTORY = _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "catalog_factory.py"
_BASE_TYPES = _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "base.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path))


def test_integration_plugin_protocol_has_no_any_semantic_boundary() -> None:
    source = _read(_PLUGIN_CONTRACT)
    assert "-> Any" not in source
    assert "**kwargs: Any" not in source
    assert "from typing import Any" not in source


def test_plugin_register_has_no_contract_specs_reflection() -> None:
    source = _read(_PLUGIN_REGISTER)
    assert "CONTRACT_SPECS" not in source
    assert "getattr(" not in source
    assert "hasattr(" not in source
    tree = _parse(_PLUGIN_REGISTER)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "hasattr"}:
                pytest.fail("plugin_register must not use reflection for plugin semantic dispatch")


def test_catalog_factory_has_no_any_result_boundary() -> None:
    source = _read(_CATALOG_FACTORY)
    assert "Callable[..., Any]" not in source
    assert "-> Any" not in source
    assert "PlatformIntegrationContract" in source


def test_base_integration_factory_aliases_canonical_catalog_factory() -> None:
    source = _read(_BASE_TYPES)
    assert "Callable[..., Any]" not in source
    assert "catalog_factory import IntegrationFactory" in source


def test_registry_v2_reuses_canonical_integration_factory_type() -> None:
    source = _read(_REPO_ROOT / "intergrax" / "runtime" / "integrations" / "registry_v2.py")
    assert "Callable[..., PlatformIntegrationContract]" not in source
    assert "catalog_factory import IntegrationFactory" in source
