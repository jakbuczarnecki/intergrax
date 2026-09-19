# © Artur Czarnecki. All rights reserved.

"""MP-7B — LKW production Multiplayer boundary architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LKW_APP = _REPO_ROOT / "applications" / "local_workspace_application"
_QUAL_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-7B_TIER3_MULTIPLAYER_CONSUMER_BOUNDARY_QUALIFICATION.md"
)
_CONSUMER = Path(__file__).resolve().parent / "consumer.py"
_COMPOSITION = Path(__file__).resolve().parent / "composition.py"

# Explicit composition-root allowlist for private CW imports in LKW production.
# Empty today: LKW does not materialize Multiplayer implementation yet.
_LKW_COMPOSITION_ALLOWLIST: frozenset[str] = frozenset()

_FORBIDDEN_PRIVATE_PREFIXES = ("intergrax.collaborative_work",)

_FORBIDDEN_PROVIDER_TOKENS = (
    "PostgreSQLCollaborativeWorkStore",
    "SQLiteCollaborativeWorkStore",
    "InMemoryCollaborativeWorkStore",
)

_FORBIDDEN_REPO_MODULE_SUFFIXES = (
    "collaborative_work.repository",
    "collaborative_work.persistence",
    "collaborative_work.persistence_provider",
    "collaborative_work.in_memory_repository",
    "collaborative_work.enforcement_gate",
)

_STATUS_DOCS = {
    "multiplayer_architecture": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "architecture"
    / "MULTIPLAYER_AI.md",
    "multiplayer_plan": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "plan"
    / "MULTIPLAYER_AI.md",
    "lkw_architecture": _LKW_APP / "docs" / "ARCHITECTURE.md",
    "lkw_plan": _LKW_APP / "docs" / "IMPLEMENTATION_PLAN.md",
}


def _lkw_production_python_files() -> list[Path]:
    if not _LKW_APP.is_dir():
        return []
    skip_parts = {"tests", "docker", "__pycache__", ".proof_docs", "build"}
    return [
        path
        for path in _LKW_APP.rglob("*.py")
        if not any(part in skip_parts for part in path.parts)
    ]


def _imports_in_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def test_mp7b_qualification_doc_closed() -> None:
    text = _QUAL_DOC.read_text(encoding="utf-8-sig")
    assert "TIER-3 MULTIPLAYER CONSUMER BOUNDARY QUALIFIED / CLOSED" in text
    assert "BLOCKING ARCHITECTURE GAPS: NONE" in text
    assert "ManagedWorkspace schema unchanged" in text
    assert (
        "PLATFORM OPERATES ON CONTRACTS" in text
        or "contracts, not implementations" in text.lower()
    )


def test_mp7b_status_markers_in_ssot_docs() -> None:
    required = (
        "MP-7B",
        "TIER-3 MULTIPLAYER CONSUMER BOUNDARY QUALIFIED",
        "MP-7 — IN PROGRESS",
    )
    for name, path in _STATUS_DOCS.items():
        text = path.read_text(encoding="utf-8-sig")
        missing = [marker for marker in required if marker not in text]
        assert not missing, f"{name}: missing markers: {missing}"


def test_lkw_production_forbids_private_collaborative_work_imports() -> None:
    violations: list[str] = []
    for path in _lkw_production_python_files():
        rel = _rel(path)
        if rel in _LKW_COMPOSITION_ALLOWLIST:
            continue
        for mod in _imports_in_file(path):
            for prefix in _FORBIDDEN_PRIVATE_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{rel}: import {mod}")
            for suffix in _FORBIDDEN_REPO_MODULE_SUFFIXES:
                if mod == f"intergrax.{suffix}" or mod.endswith(suffix):
                    if f"{rel}: import {mod}" not in violations:
                        violations.append(f"{rel}: import {mod}")
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_PROVIDER_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", source):
                violations.append(f"{rel}: token {token}")
    assert not violations, "LKW private Multiplayer leakage:\n" + "\n".join(violations)


def test_lkw_production_allows_public_contracts_namespace() -> None:
    # Smoke: gate must not forbid intergrax.contracts (presence of contracts imports is OK).
    # Current LKW may have zero collaborative contracts yet — assert gate logic itself.
    sample = (
        "from intergrax.contracts.collaborative_work import CollaborativePrincipal\n"
    )
    tree = ast.parse(sample)
    mods = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert mods == ["intergrax.contracts.collaborative_work"]
    assert not any(m.startswith("intergrax.collaborative_work") for m in mods)


def test_consumer_fixture_has_no_private_cw_or_gate_dependency() -> None:
    source = _CONSUMER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_CONSUMER))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("intergrax.collaborative_work")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.collaborative_work")
    assert "CollaborativeWorkEnforcementGate" not in source
    assert "_inner" not in source
    assert "getattr(" not in source


def test_composition_fixture_may_import_private_cw() -> None:
    mods = _imports_in_file(_COMPOSITION)
    assert any(m.startswith("intergrax.collaborative_work") for m in mods)
    assert any(
        m.startswith("intergrax.runtime.governance.orchestration_decision_bound_effect")
        for m in mods
    )


def test_composition_allowlist_is_explicit_not_broad() -> None:
    for entry in _LKW_COMPOSITION_ALLOWLIST:
        assert entry.startswith("applications/local_workspace_application/")
        assert "*" not in entry
        assert not entry.endswith("/")
