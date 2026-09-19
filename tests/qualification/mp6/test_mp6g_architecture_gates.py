# © Artur Czarnecki. All rights reserved.

"""MP-6G architecture gates — no source/store bypass in qualification harness."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_MP6G_DIR = _REPO / "tests" / "qualification" / "mp6"
_EVIDENCE = _REPO / "docs" / "project" / "maintainers" / "qualification" / (
    "MP-6G_E2E_ISOLATION_IDEMPOTENCY_QUALIFICATION.md"
)

_REQUIRED_WIRING_MARKERS = (
    "wire_collaborative_work_service_with_activity_publication",
    "wire_collaborative_work_artifact_service_with_activity_publication",
    "wire_collaborative_decision_binding_service_with_activity_publication",
    "wire_context_view_composer_with_activity_publication",
    "build_collaborative_activity_ingestion_service",
    "build_collaborative_activity_read_service",
    "VerifiedCollaborativeActivityPublisherIdentity",
)

_SOURCE_COVERAGE_MARKERS = (
    "run_mp6g_shared_work_source_coverage",
    "run_mp6g_artifact_source_coverage",
    "run_mp6g_decision_source_coverage",
    "run_mp6g_context_view_source_coverage",
)

_FORBIDDEN_HAPPY_PATH_MARKERS = (
    "append_idempotent(",
    "append_store.append",
    "INSERT INTO collaborative_activity",
)

_FORBIDDEN_HARNESS_ANNOTATION_NAMES = frozenset({"object", "Any"})

_FORBIDDEN_PRIVATE_ACCESS_FRAGMENTS = (
    "._inner",
    "._enforcement_gate",
)


def _read_mp6g_sources() -> str:
    chunks: list[str] = []
    for path in sorted(_MP6G_DIR.glob("*.py")):
        if path.name == "test_mp6g_architecture_gates.py":
            continue
        chunks.append(path.read_text(encoding="utf-8-sig"))
    return "\n".join(chunks)


def _mp6g_python_paths() -> list[Path]:
    return [
        path
        for path in sorted(_MP6G_DIR.glob("*.py"))
        if path.name != "test_mp6g_architecture_gates.py"
    ]


def _annotation_name(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_name(node.left)
        right = _annotation_name(node.right)
        if left and right:
            return left
        return left or right
    return None


def test_mp6g_harness_uses_production_composition_roots() -> None:
    source = _read_mp6g_sources()
    for marker in _REQUIRED_WIRING_MARKERS:
        assert marker in source, f"missing production composition marker: {marker}"


def test_mp6g_harness_no_direct_append_bypass_in_contract() -> None:
    contract = (_MP6G_DIR / "mp6g_e2e_contract.py").read_text(encoding="utf-8-sig")
    for marker in _FORBIDDEN_HAPPY_PATH_MARKERS:
        assert marker not in contract, f"forbidden bypass marker in contract: {marker}"


def test_mp6g_evidence_artifact_exists() -> None:
    assert _EVIDENCE.is_file(), "MP-6G qualification evidence artifact missing"


def test_mp6g_harness_no_object_any_contract_bypass() -> None:
    harness_path = _MP6G_DIR / "mp6g_harness.py"
    tree = ast.parse(harness_path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        name = _annotation_name(node.annotation)
        if name in _FORBIDDEN_HARNESS_ANNOTATION_NAMES:
            raise AssertionError(f"forbidden harness annotation {name!r} in mp6g_harness.py")


def test_mp6g_harness_no_type_ignore() -> None:
    for path in _mp6g_python_paths():
        text = path.read_text(encoding="utf-8-sig")
        assert "type: ignore" not in text, f"type: ignore forbidden in {path.name}"


def test_mp6g_harness_no_private_source_service_introspection() -> None:
    contract = (_MP6G_DIR / "mp6g_e2e_contract.py").read_text(encoding="utf-8-sig")
    for fragment in _FORBIDDEN_PRIVATE_ACCESS_FRAGMENTS:
        assert fragment not in contract, f"forbidden private access {fragment!r} in contract"


def test_mp6g_source_coverage_markers() -> None:
    contract = (_MP6G_DIR / "mp6g_e2e_contract.py").read_text(encoding="utf-8-sig")
    for marker in _SOURCE_COVERAGE_MARKERS:
        assert marker in contract, f"missing source coverage scenario: {marker}"
