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
_RESOLVER = _REPO_ROOT / "intergrax" / "integrations" / "registry" / "factory.py"
_CONTRACT_SPEC = _REPO_ROOT / "intergrax" / "integrations" / "registry" / "contract_spec.py"
_INTEGRATION_PROFILE = _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "integration_profile.py"
_CONTRACT_METADATA = _REPO_ROOT / "intergrax" / "runtime" / "integrations" / "contract_metadata.py"


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


def test_canonical_resolver_has_no_any_semantic_result() -> None:
    source = _read(_RESOLVER)
    assert "-> Any" not in source
    assert "CategoryIntegrationInstance" in source
    assert "_require_platform_integration_contract" not in source


def test_contract_spec_factory_aliases_catalog_factory_without_any() -> None:
    source = _read(_CONTRACT_SPEC)
    assert "Callable[..., Any]" not in source
    assert "IntegrationContractFactory = IntegrationFactory" in source
    assert "contract_class: type[PlatformIntegrationContract]" in source
    assert "security_posture: PlatformIntegrationSecurityPosture" in source


def test_instance_for_category_uses_canonical_contract_for_category() -> None:
    source = _read(_INTEGRATION_PROFILE)
    assert "contract_for_category" in source
    assert "isinstance(instance, expected_contract)" in source
    assert "PROVIDER_CATEGORY_CONTRACT_REGISTRY" not in source
    assert "if category ==" not in source
    assert "expected PlatformIntegrationContract" not in source


def test_factory_materialization_validates_category_contract() -> None:
    source = _read(_RESOLVER)
    assert "contract_for_category" in source
    assert "expected_contract" in source
    assert "expected a PlatformIntegrationContract" not in source


def test_canonical_category_contract_resolver_supports_di_only_categories() -> None:
    source = _read(_CONTRACT_METADATA)
    assert "DI_ONLY_CATEGORY_CONTRACT_REGISTRY" in source
    assert '"external_work"' in source
    assert "PROVIDER_CATEGORY_CONTRACT_REGISTRY.get" in source
    assert "DI_ONLY_CATEGORY_CONTRACT_REGISTRY.get" in source
    assert "CategoryIntegrationInstance" in source
    profile_source = _read(_INTEGRATION_PROFILE)
    assert "IntegrationCategory.EXTERNAL_WORK" not in profile_source
    assert "contract_for_category" in profile_source


def _function_return_annotation(module: ast.Module, name: str) -> str:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if node.returns is None:
                return ""
            return ast.unparse(node.returns)
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    if child.returns is None:
                        return ""
                    return ast.unparse(child.returns)
    pytest.fail(f"function {name!r} not found for return annotation gate")


def test_instance_for_category_declares_category_integration_instance() -> None:
    tree = _parse(_INTEGRATION_PROFILE)
    annotation = _function_return_annotation(tree, "instance_for_category")
    assert "CategoryIntegrationInstance" in annotation
    assert "PlatformIntegrationContract" not in annotation


def test_resolve_from_profile_declares_category_integration_instance() -> None:
    tree = _parse(_RESOLVER)
    annotation = _function_return_annotation(tree, "resolve_from_profile")
    assert "CategoryIntegrationInstance" in annotation
    assert "PlatformIntegrationContract" not in annotation


def test_external_work_integration_not_subclass_of_platform_contract() -> None:
    external_work_path = (
        _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "external_work.py"
    )
    source = _read(external_work_path)
    assert "PlatformIntegrationContract" not in source


def test_category_resolution_surfaces_pyright_clean() -> None:
    import subprocess

    targets = [str(_RESOLVER)]
    result = subprocess.run(
        ["uv", "run", "pyright", *targets],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
