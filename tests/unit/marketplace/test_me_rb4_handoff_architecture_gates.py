# © Artur Czarnecki. All rights reserved.

"""ME-RB4 — lifecycle handoff architecture import gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_HANDOFF_CORE_PREFIX = "intergrax.marketplace.handoff"
_ADAPTER_PREFIX = "intergrax.marketplace.handoff.adapters"

_FORBIDDEN_CORE_PREFIXES = (
    "intergrax.agent_distribution",
    "intergrax.tools.registry.runtime",
    "intergrax.skills.registry.runtime",
    "intergrax.runtime.execution",
    "intergrax.runtime.nexus",
    "intergrax.nexus",
)


def _package_root(module_name: str) -> Path:
    package = importlib.import_module(module_name)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_handoff_core_does_not_import_domain_implementations() -> None:
    root = _package_root("intergrax.marketplace.handoff")
    adapter_root = root / "adapters"
    for path in _iter_py_files(root):
        if adapter_root in path.parents or path.parent == adapter_root:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_CORE_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"handoff core {path.relative_to(root)} imports forbidden: {imported}",
                    )


def test_handoff_adapters_do_not_import_runtime_registries() -> None:
    root = _package_root("intergrax.marketplace.handoff.adapters")
    forbidden = (
        "intergrax.tools.registry.runtime",
        "intergrax.skills.registry.runtime",
        "intergrax.runtime.nexus",
        "intergrax.nexus",
    )
    for path in _iter_py_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in forbidden:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"adapter {path.relative_to(root)} imports forbidden: {imported}",
                    )


_DOMAIN_CONTRACT_PACKAGES = (
    "intergrax.contracts.tools",
    "intergrax.contracts.skills",
)


def test_domain_lifecycle_contracts_do_not_import_marketplace() -> None:
    marketplace = "intergrax.contracts.marketplace"
    for module_name in _DOMAIN_CONTRACT_PACKAGES:
        root = _package_root(module_name)
        for path in _iter_py_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for imported in _collect_imports(tree):
                if imported == marketplace or imported.startswith(f"{marketplace}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports marketplace contracts: {imported}",
                    )


def test_handoff_adapters_may_import_domain_lifecycle_contracts() -> None:
    root = _package_root("intergrax.marketplace.handoff.adapters")
    allowed_domain_contract_prefixes = (
        "intergrax.contracts.tools",
        "intergrax.contracts.skills",
        "intergrax.contracts.agent_distribution",
        "intergrax.contracts.lifecycle_handoff",
    )
    for path in _iter_py_files(root):
        if path.name == "agent_distribution_bridge.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            if imported.startswith("intergrax.tools.") or imported.startswith(
                "intergrax.skills.",
            ):
                raise AssertionError(
                    f"{path.relative_to(root)} imports tool/skill runtime: {imported}",
                )
            if imported.startswith("intergrax.agent_distribution.") and imported not in (
                "intergrax.agent_distribution.dynamic_acquisition",
            ):
                raise AssertionError(
                    f"{path.relative_to(root)} imports agent_distribution impl: {imported}",
                )
            if imported.startswith("intergrax.contracts.") and not any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in allowed_domain_contract_prefixes
            ):
                if imported.startswith(
                    (
                        "intergrax.contracts.capability_catalog",
                        "intergrax.contracts.marketplace",
                    ),
                ):
                    continue
                if imported.startswith("intergrax.contracts."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports unexpected contract: {imported}",
                    )


def test_marketplace_contracts_do_not_own_tool_skill_lifecycle_ports() -> None:
    init_path = _package_root("intergrax.contracts.marketplace") / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    forbidden_exports = (
        "ToolMarketplaceLifecycleDomainPort",
        "SkillMarketplaceLifecycleDomainPort",
        "AgentMarketplaceLifecycleDomainPort",
        "ToolMarketplaceLifecycleHandoffPort",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in forbidden_exports:
            raise AssertionError(f"marketplace contracts export domain port: {node.value}")
        if isinstance(node, ast.Name) and node.id in forbidden_exports:
            raise AssertionError(f"marketplace contracts export domain port: {node.id}")


def test_agent_bridge_imports_agent_distribution_acquisition_contract_only() -> None:
    bridge = _package_root("intergrax.marketplace.handoff.adapters") / "agent_distribution_bridge.py"
    tree = ast.parse(bridge.read_text(encoding="utf-8"))
    agent_distribution_imports = [
        imported
        for imported in _collect_imports(tree)
        if imported == "intergrax.agent_distribution"
        or imported.startswith("intergrax.agent_distribution.")
    ]
    assert agent_distribution_imports == [
        "intergrax.agent_distribution.dynamic_acquisition",
        "intergrax.agent_distribution.task_scoped_agents",
    ]
