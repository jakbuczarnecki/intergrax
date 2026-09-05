# © Artur Czarnecki. All rights reserved.

"""Architecture gates for reference production process composition (AC-4 Phase 9)."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_DIR = REPO_ROOT / "intergrax" / "applications" / "_shared"

_PRODUCTION_COMPOSITION_MODULES = (
    "production_process_composition.py",
    "production_agent_capability_runtime.py",
    "production_delegated_subtask_plans.py",
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "tests.",
    "testing_support.",
    "applications.",
    "agents.",
)

_FORBIDDEN_FRAGMENTS = (
    "StaticAgentDiscoveryStrategy",
    "get_global",
    "singleton",
    "_UNLIMITED_LEDGER",
)


def _module_paths() -> tuple[Path, ...]:
    return tuple(SHARED_DIR / name for name in _PRODUCTION_COMPOSITION_MODULES)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_production_composition_modules_avoid_forbidden_import_roots() -> None:
    violations: list[str] = []
    for path in _module_paths():
        for root in sorted(_imported_roots(path)):
            if any(
                root == prefix.rstrip(".") or root.startswith(prefix)
                for prefix in _FORBIDDEN_IMPORT_PREFIXES
            ):
                violations.append(f"{path.name} imports forbidden root {root!r}")
    assert not violations, "\n".join(violations)


def test_production_composition_modules_avoid_test_harness_fragments() -> None:
    violations: list[str] = []
    for path in _module_paths():
        text = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            if fragment in text:
                violations.append(
                    f"{path.name} contains forbidden fragment {fragment!r}"
                )
    assert not violations, "\n".join(violations)


def test_production_process_composition_wires_capability_runtime_field() -> None:
    path = SHARED_DIR / "production_process_composition.py"
    text = path.read_text(encoding="utf-8")
    assert "agent_capability_runtime" in text
    assert "build_production_agent_capability_runtime" in text
