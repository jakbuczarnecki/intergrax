# © Artur Czarnecki. All rights reserved.

"""DS-MIG-02 — production modules must not import legacy Critic orchestration wiring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"

_FORBIDDEN_MODULES = frozenset(
    {
        "intergrax.runtime.critic.critic_orchestrator",
        "intergrax.runtime.critic.critic_wiring",
    },
)

_DECISION_AUTHORITY_PATH_PREFIXES = (
    "intergrax/applications/_shared/decision_wiring.py",
    "intergrax/applications/_shared/harness_host_runtime.py",
    "intergrax/applications/_shared/scenario_runtime_baseline.py",
    "intergrax/applications/_shared/nexus_factory.py",
    "intergrax/runtime/decision_flow.py",
    "intergrax/runtime/decision_flow_host.py",
)

_LEGACY_CONFIG_PATH_PREFIXES = (
    "intergrax/applications/_shared/runtime_config_bridge.py",
    "intergrax/applications/contracts/environment_profile/",
)

_CRITIC_AUTHORITY_PATTERNS = (
    "critic_profile.evaluator_loop_max_iterations",
    "critic_profile.scopes",
    "resolve_critic_wiring_options",
    "evaluator_loop_max_iterations - 1",
    "wire_application_critic",
    "critic_runtime_bridge",
    "critic_wiring",
)

_ALLOWED_PATHS = (
    "intergrax/runtime/migration/",
    "intergrax/applications/contracts/environment_profile/decision_profile_legacy.py",
)


def _is_allowed_path(path: Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return rel.startswith(_ALLOWED_PATHS)


def _module_from_import(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in _FORBIDDEN_MODULES:
                return alias.name
        return None
    if node.module in _FORBIDDEN_MODULES:
        return node.module
    return None


def _collect_forbidden_imports(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        forbidden = _module_from_import(node)
        if forbidden is not None:
            violations.append(f"{rel}:{node.lineno}: {forbidden}")
    return violations


def _production_forbidden_import_violations() -> list[str]:
    violations: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if _is_allowed_path(path):
            continue
        violations.extend(_collect_forbidden_imports(path))
    return violations


def test_production_modules_do_not_import_critic_orchestrator_wiring() -> None:
    violations = _production_forbidden_import_violations()
    assert violations == [], (
        "Production intergrax modules must not import critic orchestrator wiring: "
        + ", ".join(violations)
    )


def _decision_authority_critic_pattern_violations() -> list[str]:
    violations: list[str] = []
    for rel_prefix in _DECISION_AUTHORITY_PATH_PREFIXES:
        path = _REPO_ROOT / rel_prefix
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8-sig")
        for pattern in _CRITIC_AUTHORITY_PATTERNS:
            if pattern in source:
                violations.append(f"{rel_prefix}: {pattern}")
    return violations


def test_decision_authority_modules_do_not_consume_critic_profile_fields() -> None:
    violations = _decision_authority_critic_pattern_violations()
    assert violations == [], (
        "Decision authority modules must not derive behavior from CriticProfile: "
        + ", ".join(violations)
    )
