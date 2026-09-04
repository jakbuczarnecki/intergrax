# © Artur Czarnecki. All rights reserved.

"""DS-MIG-04 — legacy Critic runtime must be physically deleted."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"
_PLATFORM_PROOFS_ROOT = _REPO_ROOT / "platform_proofs"

_FORBIDDEN_MODULE_PREFIX = "intergrax.runtime.critic"

_FORBIDDEN_SYMBOLS = (
    "CriticOrchestrator",
    "CriticGraphHooks",
    "CriticHookConfig",
    "CriticVerdict",
    "CriticAction",
    "L0Gateway",
    "L1Gateway",
    "EvaluatorLoopExecutor",
    "validate_final_with_critic",
    "validate_node_with_critic",
    "validate_uaep_step_with_critic",
)

_PRODUCTION_ROOTS = (
    _INTERGRAX_ROOT,
    _APPLICATIONS_ROOT,
    _PLATFORM_PROOFS_ROOT,
)

_SKIP_PATH_MARKERS = (
    "/docker/runtime-context/",
    "/__pycache__/",
)


def _should_scan(path: Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    if not rel.endswith(".py"):
        return False
    return not any(marker in f"/{rel}/" for marker in _SKIP_PATH_MARKERS)


def test_critic_runtime_directory_deleted() -> None:
    assert not (_REPO_ROOT / "intergrax/runtime/critic").exists()


def _collect_import_violations(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return []
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _FORBIDDEN_MODULE_PREFIX or alias.name.startswith(
                    f"{_FORBIDDEN_MODULE_PREFIX}.",
                ):
                    violations.append(f"{rel}:{node.lineno}: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == _FORBIDDEN_MODULE_PREFIX or node.module.startswith(
                f"{_FORBIDDEN_MODULE_PREFIX}.",
            ):
                violations.append(f"{rel}:{node.lineno}: {node.module}")
    return violations


def test_production_code_has_no_critic_runtime_imports() -> None:
    violations: list[str] = []
    for root in _PRODUCTION_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _should_scan(path):
                continue
            violations.extend(_collect_import_violations(path))
    assert violations == [], "Forbidden critic runtime imports: " + ", ".join(violations)


def test_production_code_has_no_live_critic_symbols() -> None:
    violations: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if not _should_scan(path):
            continue
        if path.parts[-2:] == ("migration", "legacy_critic_contracts.py"):
            continue
        if path.parts[-2:] == ("migration", "legacy_critic_trace.py"):
            continue
        if path.parts[-2:] == ("migration", "decision_critic_parity.py"):
            continue
        source = path.read_text(encoding="utf-8-sig")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for symbol in _FORBIDDEN_SYMBOLS:
            if symbol in source:
                violations.append(f"{rel}: {symbol}")
    assert violations == [], "Forbidden live critic symbols: " + ", ".join(violations)
