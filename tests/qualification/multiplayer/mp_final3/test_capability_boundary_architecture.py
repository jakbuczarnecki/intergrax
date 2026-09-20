# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-3 — architecture gates for capability-wide backend E2E composition."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MP_FINAL3 = Path(__file__).resolve().parent
_CONSUMER_FILES = (
    _MP_FINAL3 / "scenario.py",
    _MP_FINAL3 / "test_capability_wide_e2e.py",
)
_HOST = _MP_FINAL3 / "host.py"
_SCAN_FILES = (_HOST, *_CONSUMER_FILES)
_ALL_QUAL_FILES = tuple(sorted(p for p in _MP_FINAL3.glob("*.py") if p.name != "test_capability_boundary_architecture.py"))

_FORBIDDEN_ORCHESTRATORS = (
    "MultiplayerOrchestrator",
    "MultiplayerWorkflowEngine",
    "MultiplayerE2EOrchestrator",
)

_PRIVATE_REACH_THROUGH = re.compile(
    r"\.(?:_repo|_store|_items|_by_id|_state|_inner|_queue)\b",
)

_LKW_PRODUCTION = _REPO_ROOT / "applications" / "local_workspace_application"


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_mega_orchestrator_introduced() -> None:
    for path in _SCAN_FILES:
        text = path.read_text(encoding="utf-8")
        for name in _FORBIDDEN_ORCHESTRATORS:
            assert name not in text, f"{path.name}: forbidden {name}"


def test_consumer_avoids_private_reach_through_and_reflection() -> None:
    for path in _CONSUMER_FILES:
        text = path.read_text(encoding="utf-8")
        assert _PRIVATE_REACH_THROUGH.search(text) is None, path.name
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"getattr", "hasattr", "setattr"}, path.name
            if isinstance(node, ast.Name) and node.id == "Any":
                raise AssertionError(f"{path.name}: Any bridge forbidden")


def test_consumer_no_provider_isinstance_branching() -> None:
    for path in _CONSUMER_FILES:
        text = path.read_text(encoding="utf-8")
        assert "SQLite" not in text
        assert "PostgreSQL" not in text
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id != "isinstance":
                    continue
                for arg in node.args[1:]:
                    name = ""
                    if isinstance(arg, ast.Name):
                        name = arg.id
                    elif isinstance(arg, ast.Attribute):
                        name = arg.attr
                    assert "SQLite" not in name and "PostgreSQL" not in name, path.name


def test_consumer_no_semantic_monkeypatch() -> None:
    for path in _CONSUMER_FILES:
        text = path.read_text(encoding="utf-8")
        assert "monkeypatch" not in text
        assert "unittest.mock.patch" not in text
        assert "MagicMock" not in text
        assert "patch(" not in text


def test_no_second_decision_store_in_qualification() -> None:
    for path in _SCAN_FILES:
        text = path.read_text(encoding="utf-8")
        assert "class DecisionStore" not in text
        assert "InMemoryDecisionRepository" not in text


def test_context_view_does_not_embed_memory_rag_truth() -> None:
    scenario = (_MP_FINAL3 / "scenario.py").read_text(encoding="utf-8")
    host = (_MP_FINAL3 / "host.py").read_text(encoding="utf-8")
    assert "UserProfileMemoryEntry" not in scenario
    assert "RetrievalChunk" not in scenario
    assert "InMemoryUserProfileStore" not in host
    assert "DefaultMemoryContextSource" not in host


def test_activity_source_does_not_bypass_append_store_in_consumer() -> None:
    for path in _CONSUMER_FILES:
        text = path.read_text(encoding="utf-8")
        assert "append_store.append" not in text
        assert "activity_read_port.query" not in text


def test_no_lkw_production_changes_in_scope() -> None:
    # Gate: this qualification package must not import LKW production modules.
    for path in _ALL_QUAL_FILES:
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "local_workspace_application" not in node.module, path.name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "local_workspace_application" not in alias.name, path.name
    assert _LKW_PRODUCTION.is_dir()


def test_host_is_composition_root_not_consumer_leak() -> None:
    host_text = (_MP_FINAL3 / "host.py").read_text(encoding="utf-8")
    assert "open_sqlite_collaborative_work_repositories" in host_text
    assert "wire_collaborative_work_service_with_activity_publication" in host_text
    consumer = (_MP_FINAL3 / "test_capability_wide_e2e.py").read_text(encoding="utf-8")
    assert "open_sqlite_collaborative_work_repositories" not in consumer
    assert "InMemory" not in consumer
