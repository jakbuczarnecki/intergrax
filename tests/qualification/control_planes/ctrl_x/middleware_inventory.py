# © Artur Czarnecki. All rights reserved.

"""Evidence-only closed-world inventory of production RuntimeMiddleware (CTRL-X-R3-R1)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[4]

_PRODUCTION_SCAN_ROOTS: Final[tuple[Path, ...]] = (
    _REPO_ROOT / "intergrax/runtime",
    _REPO_ROOT / "intergrax/applications/_shared",
    _REPO_ROOT / "intergrax/harness",
)


class MiddlewareExposure(StrEnum):
    CROSS_LAYER = "CROSS-LAYER"
    INTERNAL_ONLY = "INTERNAL-ONLY"


@dataclass(frozen=True, slots=True)
class MiddlewareInventoryEntry:
    module_path: str
    class_name: str
    exposure: MiddlewareExposure


# SSOT classification — must cover every discovered production RuntimeMiddleware subclass.
MIDDLEWARE_INVENTORY: Final[tuple[MiddlewareInventoryEntry, ...]] = (
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/application_security_wiring.py",
        "PromptDefenseMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/application_security_wiring.py",
        "ToolInjectionDefenseMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/application_security_wiring.py",
        "TenantSecurityMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/application_guardrail_middleware.py",
        "LlmGuardrailMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/autonomy_middleware.py",
        "AutonomyGovernanceMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/runtime/security/defense_plugin.py",
        "PluginSecurityDefenseMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/runtime/security/encryption_middleware.py",
        "EncryptionEnforcementMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/runtime/middleware/trace_middleware.py",
        "TraceEmittingMiddleware",
        MiddlewareExposure.CROSS_LAYER,
    ),
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/application_environment_state_middleware.py",
        "ApplicationEnvironmentStateMiddleware",
        MiddlewareExposure.INTERNAL_ONLY,
    ),
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/environment_snapshot_middleware.py",
        "EnvironmentSnapshotMiddleware",
        MiddlewareExposure.INTERNAL_ONLY,
    ),
    MiddlewareInventoryEntry(
        "intergrax/applications/_shared/capability_alias_middleware.py",
        "CapabilityAliasMiddleware",
        MiddlewareExposure.INTERNAL_ONLY,
    ),
    MiddlewareInventoryEntry(
        "intergrax/harness/hooks.py",
        "ApplicationHostMiddleware",
        MiddlewareExposure.INTERNAL_ONLY,
    ),
)


def _normalize_module_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _runtime_middleware_bases(tree: ast.Module) -> set[str]:
    bases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.middleware.base":
            for alias in node.names:
                if alias.name == "RuntimeMiddleware":
                    bases.add(alias.asname or alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "intergrax.runtime.middleware.base":
                    bases.add("RuntimeMiddleware")
    return bases


def discover_production_runtime_middleware_classes() -> frozenset[tuple[str, str]]:
    """AST discovery of classes inheriting RuntimeMiddleware under production roots."""
    discovered: set[tuple[str, str]] = set()
    for root in _PRODUCTION_SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            runtime_bases = _runtime_middleware_bases(tree)
            if not runtime_bases:
                continue
            module_path = _normalize_module_path(path)
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for base in node.bases:
                    base_name: str | None = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute) and base.attr == "RuntimeMiddleware":
                        base_name = "RuntimeMiddleware"
                    if base_name and base_name in runtime_bases | {"RuntimeMiddleware"}:
                        discovered.add((module_path, node.name))
                        break
    return frozenset(discovered)


def inventory_index() -> dict[tuple[str, str], MiddlewareExposure]:
    return {(entry.module_path, entry.class_name): entry.exposure for entry in MIDDLEWARE_INVENTORY}


def cross_layer_middleware_source_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for entry in MIDDLEWARE_INVENTORY:
        if entry.exposure == MiddlewareExposure.CROSS_LAYER:
            paths.append(_REPO_ROOT / entry.module_path)
    return tuple(paths)
