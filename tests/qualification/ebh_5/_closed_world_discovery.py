# © Artur Czarnecki. All rights reserved.

"""Closed-world discovery helpers for EBH-5."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from intergrax.integrations.contracts.base import IntegrationCategory

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

COMPOSITION_ROOT_HINTS: Final[tuple[str, ...]] = (
    "intergrax/integrations/registry/factory.py",
    "intergrax/integrations/execution_bound_integration_resolution.py",
    "intergrax/llm_adapters/registry/profile.py",
    "intergrax/applications/_shared/memory_wiring.py",
    "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
    "intergrax/tools/registry/wiring.py",
    "intergrax/rag/embedding/registry/profile.py",
    "intergrax/integrations/existing_capability_configuration_facade.py",
)


@lru_cache(maxsize=1)
def discover_composition_root_paths() -> frozenset[str]:
    found: set[str] = set()
    for hint in COMPOSITION_ROOT_HINTS:
        if (_REPO_ROOT / hint).is_file():
            found.add(hint.replace("\\", "/"))
    return frozenset(found)


@lru_cache(maxsize=1)
def discover_sqlite_connect_paths_under_intergrax() -> frozenset[str]:
    root = _REPO_ROOT / "intergrax"
    paths: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "sqlite3.connect" in text:
            rel = path.relative_to(_REPO_ROOT).as_posix()
            paths.add(rel)
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_integration_category_enum_size() -> int:
    return len(tuple(IntegrationCategory))
