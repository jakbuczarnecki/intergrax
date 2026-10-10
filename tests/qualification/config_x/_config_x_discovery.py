# © Artur Czarnecki. All rights reserved.

"""CONFIG-X closed-world discovery (fail-closed parity)."""

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

_EXCLUDE_PATH_PARTS: Final[tuple[str, ...]] = (
    "/tests/",
    "\\tests\\",
    "/test_",
    "\\test_",
    "/qualification/",
    "\\qualification\\",
    "/examples/",
    "\\examples\\",
    "/scaffold/",
    "\\scaffold\\",
    "docker/runtime-context",
)

_COMPOSITION_ROOT_SUFFIXES: Final[tuple[str, ...]] = (
    "integration_wiring.py",
    "tool_wiring.py",
    "host/factory.py",
)

_BLOCKER_PATH_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "intergrax/tools/providers/observability/resolve.py": (
        "resolve_observability_backend",
        "observability_role_backend_not_configured",
    ),
    "intergrax/tokenizers/registry/tokenizer_registry.py": (
        "class TokenizerRegistry",
        "_default_tokenizer_id",
    ),
    "intergrax/applications/_shared/harness_task_routes.py": (
        "tenant_id_required",
        "HarnessAsyncRunRequest",
    ),
    "intergrax/applications/_shared/trace_explorer_routes.py": (
        "Query(..., min_length=1)",
        "create_trace_explorer_router",
    ),
    "intergrax/multimedia/image_smart_loader.py": (
        "tenant_id: str,",
        "ImageSmartLoader",
    ),
    "intergrax/integrations/_shared/p3/configs.py": (
        "require_tenant_id",
        "class VectorIntegrationConfig",
    ),
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
def discover_composition_root_paths() -> frozenset[str]:
    paths: set[str] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            rel = _normalize_repo_path(path)
            if any(rel.endswith(suffix) for suffix in _COMPOSITION_ROOT_SUFFIXES):
                paths.add(rel)
            elif "/integrations/registry/" in rel or rel.endswith("execution_bound_integration_resolution.py"):
                paths.add(rel)
            elif "/llm_adapters/registry/" in rel and rel.endswith(".py"):
                paths.add(rel)
    return frozenset(paths)


def discover_blocker_path_keys() -> frozenset[str]:
    keys: set[str] = set()
    for rel_path, markers in _BLOCKER_PATH_MARKERS.items():
        path = _REPO_ROOT / rel_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if all(marker in text for marker in markers):
            keys.add(rel_path)
    return frozenset(keys)


@lru_cache(maxsize=1)
def discover_integration_category_enum_size() -> int:
    from intergrax.integrations.contracts.base import IntegrationCategory

    return len(tuple(IntegrationCategory))


def classify_synthetic_unknown_surface(module_text: str) -> str | None:
    """Return classification letter or None when surface must fail closed as unclassified."""
    if "CONFIG_X_SYNTHETIC_UNCLASSIFIED" in module_text:
        return None
    return "H"


@lru_cache(maxsize=2048)
def _parse_module(repo_relative_path: str) -> ast.Module | None:
    path = _REPO_ROOT / repo_relative_path
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def discover_class_names_defining_resolver() -> frozenset[str]:
    """Independent discovery of *Resolver classes (duplicate-authority probe input)."""
    names: set[str] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            tree = _parse_module(_normalize_repo_path(path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.endswith("Resolver"):
                    names.add(node.name)
    return frozenset(names)
