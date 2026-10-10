# © Artur Czarnecki. All rights reserved.

"""Mechanical configuration-authority owner discovery (CONFIG-X)."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "agents",
)

_EXCLUDE_PATH_PARTS = (
    "/tests/",
    "\\tests\\",
    "/qualification/",
    "\\qualification\\",
    "/examples/",
    "\\examples\\",
    "/scaffold/",
    "\\scaffold\\",
    "docker/runtime-context",
)

CONFIG_X_OWNER_EXPECTATIONS: Final[dict[str, frozenset[str]]] = {
    "integration_provider_selection": frozenset(
        {
            "intergrax/integrations/registry/factory.py",
        },
    ),
    "integration_typed_resolution_delegate": frozenset(
        {
            "intergrax/integrations/registry/resolve_typed.py",
        },
    ),
    "llm_provider_selection": frozenset(
        {
            "intergrax/llm_adapters/llm_provider_registry.py",
        },
    ),
    "execution_bound_integration_resolution": frozenset(
        {
            "intergrax/integrations/execution_bound_integration_resolution.py",
        },
    ),
    "existing_capability_configuration_realization": frozenset(
        {
            "intergrax/integrations/existing_capability_configuration_service.py",
        },
    ),
    "plugin_integration_catalog": frozenset(
        {
            "intergrax/integrations/registry/catalog.py",
        },
    ),
}

_OWNER_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "integration_provider_selection": ("def resolve_from_profile", "def resolve_slug"),
    "integration_typed_resolution_delegate": (
        "def resolve_contract",
        "Typed helpers for ``IntegrationProfile.resolve``",
    ),
    "llm_provider_selection": ("class LLMAdapterRegistry", "def create(cls, provider"),
    "execution_bound_integration_resolution": ("class ExecutionBoundIntegrationResolution",),
    "existing_capability_configuration_realization": (
        "class ExistingCapabilityConfigurationRealizationService",
    ),
    "plugin_integration_catalog": ("def get_entry",),
}


def _normalize_repo_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_discovery_candidate(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    for part in _EXCLUDE_PATH_PARTS:
        if part.replace("/", "\\") in normalized or part in normalized:
            return False
    return True


@lru_cache(maxsize=1)
def discover_owner_paths(concern: str) -> frozenset[str]:
    markers = _OWNER_MARKERS.get(concern)
    if markers is None:
        raise KeyError(f"unknown CONFIG-X owner concern: {concern}")
    found: set[str] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            rel = _normalize_repo_path(path)
            if concern == "plugin_integration_catalog" and rel != (
                "intergrax/integrations/registry/catalog.py"
            ):
                continue
            if concern == "integration_typed_resolution_delegate" and rel != (
                "intergrax/integrations/registry/resolve_typed.py"
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if all(marker in text for marker in markers):
                found.add(rel)
    return frozenset(found)


def compare_owner_gate(concern: str) -> tuple[frozenset[str], frozenset[str]]:
    discovered = discover_owner_paths(concern)
    expected = CONFIG_X_OWNER_EXPECTATIONS[concern]
    return discovered, expected
