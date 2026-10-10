# © Artur Czarnecki. All rights reserved.

"""Path extraction and glob resolution for CONFIG-X mechanical evidence."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]

_REPO_PATH_TOKEN = re.compile(
    r"(?P<path>(?:intergrax|applications|agents)/[A-Za-z0-9_./\-*]+)",
)

CONFIG_X_OWNER_EXPECTATIONS_PATHS: Final[tuple[str, ...]] = (
    "intergrax/integrations/registry/factory.py",
    "intergrax/integrations/registry/resolve_typed.py",
    "intergrax/llm_adapters/llm_provider_registry.py",
    "intergrax/integrations/execution_bound_integration_resolution.py",
    "intergrax/integrations/existing_capability_configuration_service.py",
    "intergrax/integrations/registry/catalog.py",
)


def repo_root() -> Path:
    return _REPO_ROOT


def primary_repo_paths_from_text(field: str) -> tuple[str, ...]:
    """Extract repo-relative path tokens from inventory metadata strings."""
    seen: list[str] = []
    for match in _REPO_PATH_TOKEN.finditer(field):
        raw = match.group("path").rstrip(").,;")
        if "*" in raw:
            continue
        if raw not in seen:
            seen.append(raw)
    return tuple(seen)


def expand_inventory_glob_pattern(pattern: str) -> frozenset[str]:
    """Expand repo-relative glob (forward slashes) to files."""
    normalized = pattern.strip().replace("\\", "/")
    if not normalized or "*" not in normalized:
        return frozenset()
    matches: set[str] = set()
    for path in _REPO_ROOT.glob(normalized):
        if path.is_file():
            matches.add(path.relative_to(_REPO_ROOT).as_posix())
        elif path.is_dir():
            for child in path.rglob("*.py"):
                if child.is_file():
                    matches.add(child.relative_to(_REPO_ROOT).as_posix())
    return frozenset(matches)


def expand_provider_surface_glob(pattern: str) -> frozenset[str]:
    """Resolve provider_surface glob to existing file paths (repo-relative)."""
    if not pattern.strip():
        return frozenset()
    normalized = pattern.strip().replace("\\", "/")
    if "*" not in normalized:
        path = _REPO_ROOT / normalized
        if path.is_file():
            return frozenset({normalized})
        if path.is_dir():
            return frozenset(
                p.relative_to(_REPO_ROOT).as_posix()
                for p in path.rglob("*.py")
                if p.is_file()
            )
        return frozenset()

    # Legacy inventory pattern: providers/*/{category} → providers/{category}/*
    legacy = re.fullmatch(
        r"intergrax/integrations/providers/\*/(?P<cat>[a-z_]+)",
        normalized,
    )
    if legacy is not None:
        cat = legacy.group("cat")
        normalized = f"intergrax/integrations/providers/{cat}/*"

    matches: set[str] = set()
    for path in _REPO_ROOT.glob(normalized):
        if path.is_file() and path.suffix == ".py":
            matches.add(path.relative_to(_REPO_ROOT).as_posix())
        elif path.is_dir():
            for child in path.rglob("*.py"):
                if child.is_file():
                    matches.add(child.relative_to(_REPO_ROOT).as_posix())
    return frozenset(matches)


@lru_cache(maxsize=1)
def sanctioned_composition_owner_paths_from_inventory() -> frozenset[str]:
    from tests.qualification.config_x._config_x_concern_inventory import (
        CONFIG_X_CONCERN_INVENTORY,
    )

    paths: set[str] = set()
    for row in CONFIG_X_CONCERN_INVENTORY:
        paths.update(primary_repo_paths_from_text(row.composition_owner))
        paths.update(primary_repo_paths_from_text(row.effective_resolution_owner))
    paths.update(CONFIG_X_OWNER_EXPECTATIONS_PATHS)
    return frozenset(sorted(paths))
