# © Artur Czarnecki. All rights reserved.

"""MP-6D-Q1 — documentation gates for PostgreSQL provider qualification evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_EVIDENCE = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-6D-Q1_POSTGRESQL_PROVIDER_QUALIFICATION.md"
)

_REQUIRED_MARKERS = (
    "MP-6D-Q1-E1",
    "implementation_sha",
    "131b5e8fac9e6c05bcc0415677dee91288691500",
    "qualification_sha",
    "provider",
    "PostgreSQL",
    "16.6",
    "passed: 3",
    "skipped: 0",
    "xfailed: 0",
    "failed: 0",
    "bundle A",
    "bundle B",
    "open_postgresql_collaborative_work_repositories",
    "Per-instance RLock was not shared",
    "MP-6D — CLOSED / RECERTIFIED",
    "BLOCKING FINDINGS: NONE",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp6d_q1_postgresql_qualification_evidence_artifact_exists() -> None:
    assert _EVIDENCE.is_file(), "missing MP-6D-Q1 PostgreSQL qualification evidence artifact"


def test_mp6d_q1_postgresql_qualification_evidence_markers() -> None:
    text = _read(_EVIDENCE)
    missing = [marker for marker in _REQUIRED_MARKERS if marker not in text]
    assert not missing, f"MP-6D-Q1 evidence: missing markers: {missing}"
