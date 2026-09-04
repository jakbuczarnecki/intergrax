# © Artur Czarnecki. All rights reserved.

"""DS-MIG-05 — retired Critic policy surface must not exist in active production."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"

_FORBIDDEN_TOKENS = (
    "CriticPolicyContext",
    "evaluate_critic_verdict",
    "critic.l2_escalation",
    "critic.require_on_completion",
    "critic.allow",
)

_LEGACY_MIGRATION_ALLOWLIST = (
    "intergrax/applications/contracts/environment_profile/decision_profile_legacy.py",
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


def _collect_forbidden_imports(path: Path) -> list[str]:
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
                if alias.name == "CriticPolicyContext":
                    violations.append(f"{rel}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "CriticPolicyContext":
                    violations.append(f"{rel}:{node.lineno}: from {node.module} import {alias.name}")
    return violations


def test_active_production_has_no_critic_policy_context_imports() -> None:
    violations: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if not _should_scan(path):
            continue
        violations.extend(_collect_forbidden_imports(path))
    assert violations == [], "Forbidden CriticPolicyContext imports: " + ", ".join(violations)


def test_active_production_has_no_critic_policy_tokens() -> None:
    violations: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if not _should_scan(path):
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _LEGACY_MIGRATION_ALLOWLIST:
            continue
        source = path.read_text(encoding="utf-8-sig")
        for token in _FORBIDDEN_TOKENS:
            if token in source:
                violations.append(f"{rel}: {token}")
    assert violations == [], "Forbidden critic policy tokens: " + ", ".join(violations)
