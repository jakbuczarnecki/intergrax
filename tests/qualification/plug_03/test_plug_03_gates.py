# © Artur Czarnecki. All rights reserved.

"""PLUG-03 — external replacement qualification gates and targeted proofs."""

from __future__ import annotations

import ast
import importlib.metadata
from collections.abc import Iterator
from pathlib import Path

import pytest

from intergrax.context.budget.compaction import (
    ContextCompactionInput,
    ContextCompactionStrategy,
    NoOpContextCompactionStrategy,
)
from intergrax.context.registry import ContextPluginRegistry
from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.catalog_snapshot import snapshot_catalogs
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.examples.custom_memory_kv.plugin import CustomMemoryKvPlugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.tools.tool_invocation_pattern import resolve_invocation_pattern
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.fixtures.plugin_packages.intergrax_catalog_fixture.src.intergrax_catalog_fixture.tool import (
    FIXTURE_ECHO_TOOL_ID,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.usefixtures("catalog_fixture_installed")]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PUBLIC_PLUGIN_ROOTS = (
    _REPO_ROOT / "examples" / "platform_plugins",
    _REPO_ROOT / "tests" / "fixtures" / "plugin_packages",
)

_NEXUS_ALLOWLIST_RELATIVE: frozenset[str] = frozenset()


def _module_imports_nexus(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime.nexus"):
                hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime.nexus"):
                    hits.append(alias.name)
    return hits


def _iter_public_plugin_python_files() -> Iterator[Path]:
    for root in _PUBLIC_PLUGIN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in _NEXUS_ALLOWLIST_RELATIVE:
                continue
            yield path


@pytest.fixture(autouse=True)
def _reset_catalog_state() -> Iterator[None]:
    clear_catalog()
    clear_tool_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    yield
    clear_catalog()
    clear_tool_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()


def test_public_external_plugin_packages_do_not_import_nexus() -> None:
    violations: list[str] = []
    for path in _iter_public_plugin_python_files():
        hits = _module_imports_nexus(path)
        if hits:
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {hits}")
    assert violations == [], "public plugin packages must not import Nexus:\n" + "\n".join(violations)


def test_memory_session_storage_fixture_does_not_import_nexus() -> None:
    plugin_root = (
        _REPO_ROOT
        / "tests"
        / "fixtures"
        / "plugin_packages"
        / "memory_store_plugin"
        / "memory_store_plugin"
    )
    violations: list[str] = []
    for path in plugin_root.glob("*.py"):
        hits = _module_imports_nexus(path)
        if hits:
            violations.append(f"{path.name}: {hits}")
    assert violations == []


def test_tools_discovered_but_unselected_not_in_execution_registry() -> None:
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    snap = snapshot_catalogs()
    assert "fixture_ep" in snap.tool_bundle_ids

    registry = build_registry_from_profile(ToolProfile.lab(), ctx=ToolWiringContext())
    assert not registry.has(FIXTURE_ECHO_TOOL_ID)


def test_tools_profile_selection_executes_custom_not_catalog_default() -> None:
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    registry = build_registry_from_profile(
        ToolProfile(enabled_bundles=["fixture_ep"]),
        ctx=ToolWiringContext(),
    )
    assert registry.has(FIXTURE_ECHO_TOOL_ID)


def test_integration_discovered_but_unselected_keeps_default_binding() -> None:
    bootstrap_catalogs(
        register_shipped=False,
        discover_entry_points=True,
        integration_plugins=(CustomMemoryKvPlugin,),
    )
    snap = snapshot_catalogs()
    assert "fixture_ep_kv" in snap.integration_slugs

    profile = IntegrationProfile(key_value_cache="custom_memory_kv")
    assert profile.key_value_cache is not None
    assert profile.key_value_cache.resolved_slug() == "custom_memory_kv"
    assert profile.key_value_cache.resolved_slug() != "fixture_ep_kv"
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    cache.set("tenant-a", "selected", b"custom_memory_kv")
    assert cache.get("tenant-a", "selected") == b"custom_memory_kv"


def test_integration_explicit_slug_activates_fixture_provider() -> None:
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    profile = IntegrationProfile(key_value_cache="fixture_ep_kv")
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    cache.set("tenant-a", "proof-key", b"custom")
    assert cache.get("tenant-a", "proof-key") == b"custom"


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _CustomMarkerPattern:
    marker = "plug-03-custom-pattern"

    @property
    def pattern_id(self) -> str:
        return "plug_03_custom_pattern"

    def execute(self, **_kwargs: object) -> object:
        raise AssertionError("execute should not run during resolver qualification")


def test_canonical_resolver_selects_custom_pattern_without_shipped_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "plug_03_custom_pattern",
                f"{__name__}:_CustomMarkerPattern",
                "intergrax.tool_invocation_patterns",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    def _forbidden_shipped_default(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shipped default pattern must not be instantiated")

    monkeypatch.setattr(
        "intergrax.runtime.nexus.tools.tool_invocation_pattern.pattern_for_mode",
        _forbidden_shipped_default,
    )

    resolved = resolve_invocation_pattern(
        mode=ToolInvocationMode.SINGLE_PASS,
        max_iterations=1,
        entry_point_pattern_id="plug_03_custom_pattern",
    )
    assert resolved.pattern_id == "plug_03_custom_pattern"


class _CustomMarkerCompaction(ContextCompactionStrategy):
    @property
    def strategy_id(self) -> str:
        return "plug_03_custom_compaction"

    def compact(self, item: ContextCompactionInput) -> None:
        _ = item
        return None


def test_context_custom_compaction_default_strategy_not_invoked() -> None:
    registry = ContextPluginRegistry()
    registry.set_compaction_strategy(_CustomMarkerCompaction())
    strategy = registry.compaction_strategy
    assert strategy.strategy_id == "plug_03_custom_compaction"
    assert not isinstance(strategy, NoOpContextCompactionStrategy)
