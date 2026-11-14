# © Artur Czarnecki. All rights reserved.

"""PDC-D1-R020 — one owner for plugin discovery (core/plugins/discovery.py)."""

from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path

import pytest
from intergrax.core.plugins import discovery
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge import contribution_catalog
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP,
    VendorKnowledgePluginConflict,
    VendorKnowledgePluginLoadError,
    discover_vendor_knowledge_contributions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OWNER = "intergrax/core/plugins/discovery.py"
_PRODUCT_ROOTS = ("intergrax", "agents", "applications")
_OWNER_API = frozenset(
    {
        "select_entry_points",
        "iter_entry_point_specs",
        "load_entry_point_targets",
        "load_entry_point_value",
    }
)


def _product_files() -> list[Path]:
    return sorted(
        path
        for root in _PRODUCT_ROOTS
        for path in (_REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    )


def _scans_entry_points(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib.metadata":
            if any(alias.name == "entry_points" for alias in node.names):
                return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "entry_points":
                return True
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "entry_points"
                and isinstance(func.value, (ast.Name, ast.Attribute))
                and ast.unparse(func.value).endswith("metadata")
            ):
                return True
    return False


def test_r020_only_the_discovery_owner_scans_setuptools_entry_points() -> None:
    scanners: list[str] = []
    for path in _product_files():
        text = path.read_text(encoding="utf-8")
        if "entry_points" not in text:
            continue
        if _scans_entry_points(ast.parse(text)):
            scanners.append(path.relative_to(_REPO_ROOT).as_posix())
    assert scanners == [_OWNER]


def test_r020_discovery_primitives_are_defined_only_by_the_owner() -> None:
    definitions: dict[str, list[str]] = {name: [] for name in _OWNER_API}
    for path in _product_files():
        text = path.read_text(encoding="utf-8")
        if not any(name in text for name in _OWNER_API):
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name in definitions:
                definitions[node.name].append(path.relative_to(_REPO_ROOT).as_posix())
    assert definitions == {name: [_OWNER] for name in _OWNER_API}


def test_r020_rag_entry_point_loader_delegates_to_the_owner() -> None:
    path = _REPO_ROOT / "intergrax/rag/bootstrap/entry_point_load.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name: node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert imports["register_plugins_with_report"] == "intergrax.core.plugins.discovery"
    assert not any(
        isinstance(node, ast.Import) and any(a.name.startswith("importlib.metadata") for a in node.names)
        for node in ast.walk(tree)
    )


class _EntryPoint:
    def __init__(self, name: str, target: object) -> None:
        self.name = name
        self.value = f"fixture:{name}"
        self._target = target

    def load(self) -> object:
        return self._target


class _EntryPoints:
    def __init__(self, entries: dict[str, tuple[_EntryPoint, ...]]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> tuple[_EntryPoint, ...]:
        return self._entries.get(group, ())


def _contribution(provider_id: str) -> VendorKnowledgeProviderContribution:
    return VendorKnowledgeProviderContribution(
        provider_id=provider_id,
        integration_category=IntegrationCategory.ISSUE_TRACKER,
    )


def test_r020_vendor_knowledge_discovery_enumerates_through_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contribution = _contribution("fixture-b")
    group = VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP
    fake = _EntryPoints(
        {
            group: (
                _EntryPoint("zz-last", _contribution("fixture-z")),
                _EntryPoint("aa-first", contribution),
            ),
            "intergrax.tools": (_EntryPoint("unrelated", object()),),
        }
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: fake)
    calls: list[str] = []
    original = discovery.select_entry_points

    def _spy(requested_group: str) -> tuple[object, ...]:
        calls.append(requested_group)
        return original(requested_group)

    monkeypatch.setattr(contribution_catalog, "select_entry_points", _spy)

    loaded = discover_vendor_knowledge_contributions()

    assert calls == [group]  # one enumeration, owned by core/plugins/discovery.py
    assert [item.provider_id for item in loaded] == ["fixture-b", "fixture-z"]  # deterministic order
    discovery.reset_entry_point_spec_cache_for_tests()
    try:
        specs = discovery.iter_entry_point_specs(group)
    finally:
        discovery.reset_entry_point_spec_cache_for_tests()
    assert [spec.name for spec in specs] == ["aa-first", "zz-last"]  # same order as the owner's specs


def test_r020_vendor_knowledge_keeps_its_domain_checks_on_top_of_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP

    class _Dup(_EntryPoints):
        def select(self, *, group: str) -> tuple[_EntryPoint, ...]:
            return (
                _EntryPoint("same", _contribution("one")),
                _EntryPoint("same", _contribution("two")),
            )

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _Dup({}))
    with pytest.raises(VendorKnowledgePluginConflict, match="duplicate_entry_point_name"):
        discover_vendor_knowledge_contributions()

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints({group: (_EntryPoint("bad", object()),)}),
    )
    with pytest.raises(VendorKnowledgePluginLoadError, match="external_contribution_invalid"):
        discover_vendor_knowledge_contributions()


def test_r020_enumeration_failure_fails_closed_without_leaking_the_raw_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> object:
        raise RuntimeError("secret-token-in-message")

    monkeypatch.setattr(importlib.metadata, "entry_points", _boom)
    with pytest.raises(VendorKnowledgePluginLoadError) as error:
        discover_vendor_knowledge_contributions()
    assert "secret-token-in-message" not in str(error.value)
    assert str(error.value) == "entry_point_discovery_failed"
