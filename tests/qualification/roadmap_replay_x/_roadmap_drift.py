# © Artur Czarnecki. All rights reserved.

"""Semantic roadmap consistency for ROADMAP-REPLAY-X C0 / W10 closure."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROADMAP = _REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"

_FORBIDDEN_DRIFT_SUBSTRINGS: tuple[str, ...] = (
    "OD-DUPX-08 OPEN",
    "OPERATOR CLOSURE PENDING",
    "must not start until `DUP-X = CLOSED`",
    "Parent NOT CLOSED until **OD-DUPX-08**",
    "ROADMAP-REPLAY-X has not started",
    "ROADMAP-REPLAY-X **not** started",
    "READY FOR INDEPENDENT AUDIT** (implementation wave)",
    "B6 **OPEN** — external Protocol 03",
)


def roadmap_drift_violations() -> list[str]:
    text = _ROADMAP.read_text(encoding="utf-8")
    hits: list[str] = []
    for needle in _FORBIDDEN_DRIFT_SUBSTRINGS:
        if needle in text:
            hits.append(f"forbidden:{needle}")

    table_closed = re.search(
        r"\| \*\*ROADMAP-REPLAY-X\*\* \|[^\n]+\| \*\*\[x\] CLOSED — INDEPENDENTLY ACCEPTED \(SCOPED\)\*\*",
        text,
    )
    if not table_closed:
        hits.append("§3_table:ROADMAP-REPLAY-X not marked CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)")

    dossier = _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PARENT_IMPLEMENTATION_DOSSIER.md"
    if dossier.is_file():
        body = dossier.read_text(encoding="utf-8")
        if "CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)" not in body:
            hits.append("qual:parent dossier missing CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)")
        if "NOT CLOSED" in body or "B6 OPEN" in body:
            hits.append("qual:parent dossier still asserts NOT CLOSED or B6 OPEN")

    reg = _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_OPERATOR_DECISION_REGISTER.md"
    if reg.is_file():
        reg_text = reg.read_text(encoding="utf-8")
        if "OD-REPLAYX-08" not in reg_text or "APPROVED" not in reg_text:
            hits.append("qual:OD-REPLAYX-08 not APPROVED in operator register")

    pass_rec = (
        _REPO_ROOT
        / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md"
    )
    if pass_rec.is_file() and "ISSUED — PASS (SCOPED)" not in pass_rec.read_text(encoding="utf-8"):
        hits.append("qual:Protocol 03 pass record missing ISSUED — PASS (SCOPED)")

    return hits


def assert_roadmap_drift_zero() -> None:
    violations = roadmap_drift_violations()
    assert not violations, violations
