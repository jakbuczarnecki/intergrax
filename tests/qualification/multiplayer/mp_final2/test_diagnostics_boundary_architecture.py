# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-2 — architecture gates for Multiplayer diagnostics / operability boundaries."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CW_ROOT = _REPO_ROOT / "intergrax" / "collaborative_work"
_MP_FINAL2 = Path(__file__).resolve().parent
_E2E = _MP_FINAL2 / "test_diagnostics_operability_e2e.py"
_HOST = _MP_FINAL2 / "host_operability.py"

_FORBIDDEN_ENGINE_NAMES = (
    "MultiplayerDiagnosticEngine",
    "CollaborativeWorkDiagnosticEngine",
    "MultiplayerProblemStore",
)

_FORBIDDEN_DIAG_IMPL_IMPORTS = (
    "intergrax.runtime.diagnostics.diagnostic_orchestrator",
    "intergrax.runtime.diagnostics.in_memory_problem_persistence",
    "intergrax.runtime.diagnostics.problem_lifecycle",
    "intergrax.runtime.diagnostics.document_store_problem_persistence",
    "intergrax.runtime.diagnostics.diagnostic_read_service",
)

_PRIVATE_REACH_THROUGH = re.compile(
    r"\.(?:_queue|_events|_problems|_inner|_by_id|_store)\b",
)


def _iter_cw_python() -> list[Path]:
    return sorted(p for p in _CW_ROOT.rglob("*.py") if p.is_file())


def test_no_multiplayer_diagnostic_engine_in_production() -> None:
    for path in _iter_cw_python():
        text = path.read_text(encoding="utf-8")
        for name in _FORBIDDEN_ENGINE_NAMES:
            assert name not in text, f"{path}: forbidden {name}"


def test_collaborative_work_does_not_import_concrete_diagnostic_implementations() -> None:
    for path in _iter_cw_python():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in _FORBIDDEN_DIAG_IMPL_IMPORTS:
                    assert not node.module.startswith(forbidden), (
                        f"{path}: imports concrete diagnostics {node.module}"
                    )
                assert node.module != "intergrax.runtime.diagnostics", (
                    f"{path}: must not import diagnostics package root"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("intergrax.runtime.diagnostics"), (
                        f"{path}: forbidden import {alias.name}"
                    )


def test_collaborative_work_publishes_evidence_contract_not_problem_store() -> None:
    evidence_modules = (
        _CW_ROOT / "decision_binding_application.py",
        _CW_ROOT / "decision_binding_evidence.py",
        _CW_ROOT / "functional_evidence_projection.py",
    )
    for path in evidence_modules:
        text = path.read_text(encoding="utf-8")
        assert "FunctionalEvidencePersistence" in text or "PlatformFunctionalEvidence" in text
        assert "ProblemPersistence" not in text
        assert "DiagnosticOrchestrator" not in text
        assert "InMemoryProblemPersistence" not in text


def test_activity_is_not_runtime_event_alias() -> None:
    activity_path = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_activity.py"
    activity_contract = activity_path.read_text(encoding="utf-8-sig")
    assert "class CollaborativeActivity" in activity_contract
    tree = ast.parse(activity_contract, filename=str(activity_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CollaborativeActivity":
            base_names = [
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in node.bases
            ]
            assert "RuntimeEvent" not in base_names
            assert "BaseModel" in base_names


def test_e2e_avoids_private_store_reach_through_and_semantic_monkeypatch() -> None:
    e2e_text = _E2E.read_text(encoding="utf-8")
    assert "unittest.mock" not in e2e_text
    assert "monkeypatch" not in e2e_text
    assert "patch(" not in e2e_text
    assert "getattr(" not in e2e_text
    assert "hasattr(" not in e2e_text
    assert "setattr(" not in e2e_text
    assert not _PRIVATE_REACH_THROUGH.search(e2e_text), "private reach-through in E2E"
    assert "list_problems(" in e2e_text
    assert "query_evidence(" in e2e_text
    assert "interpret_binding_create_operability" in e2e_text


def test_host_composition_selects_implementations_without_service_locator() -> None:
    host_text = _HOST.read_text(encoding="utf-8")
    assert "get_global" not in host_text
    assert "service_locator" not in host_text.lower()
    assert "registry.get(" not in host_text
    assert "FunctionalEvidencePersistence" in host_text
    assert "FunctionalDiagnosticAnalyzer" in host_text
    assert "FunctionalOperatorProjector" in host_text


def test_no_any_authority_bypass_in_qualification_slice() -> None:
    for path in (_E2E, _HOST, _MP_FINAL2 / "test_diagnostics_pluginability.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                raise AssertionError(f"{path}: Any authority/typing bypass forbidden")
            if isinstance(node, ast.Attribute) and node.attr == "Any":
                raise AssertionError(f"{path}: typing.Any forbidden")
