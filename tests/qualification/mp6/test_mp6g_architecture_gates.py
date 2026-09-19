# © Artur Czarnecki. All rights reserved.

"""MP-6G architecture gates — no source/store bypass in qualification harness."""

from __future__ import annotations

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
    "build_collaborative_activity_ingestion_service",
    "build_collaborative_activity_read_service",
    "VerifiedCollaborativeActivityPublisherIdentity",
)

_FORBIDDEN_HAPPY_PATH_MARKERS = (
    "append_idempotent(",
    "append_store.append",
    "INSERT INTO collaborative_activity",
)


def _read_mp6g_sources() -> str:
    chunks: list[str] = []
    for path in sorted(_MP6G_DIR.glob("*.py")):
        if path.name == "test_mp6g_architecture_gates.py":
            continue
        chunks.append(path.read_text(encoding="utf-8-sig"))
    return "\n".join(chunks)


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
