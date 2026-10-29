# © Artur Czarnecki. All rights reserved.

"""Semantic roadmap consistency for ROADMAP-REPLAY-X C0."""

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
)


def roadmap_drift_violations() -> list[str]:
    text = _ROADMAP.read_text(encoding="utf-8")
    hits: list[str] = []
    for needle in _FORBIDDEN_DRIFT_SUBSTRINGS:
        if needle in text:
            hits.append(f"forbidden:{needle}")

    # §3 stage table vs narrative: REPLAY-X must not claim CLOSED without independent B6.
    table_match = re.search(
        r"\| \*\*ROADMAP-REPLAY-X\*\* \|[^\n]+\| \*\*\[x\] CLOSED",
        text,
    )
    if table_match:
        hits.append("§3_table:ROADMAP-REPLAY-X marked CLOSED in checkbox column")

    if "**NOT CLOSED**" not in text and "ROADMAP-REPLAY-X" in text:
        dossier = (
            _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PARENT_IMPLEMENTATION_DOSSIER.md"
        )
        if dossier.is_file() and "NOT CLOSED" not in dossier.read_text(encoding="utf-8"):
            hits.append("qual:parent dossier missing NOT CLOSED")

    # §3 table REPLAY-X must indicate audit-ready / in progress, not blank FINAL only.
    replay_row = re.search(r"\| \*\*ROADMAP-REPLAY-X\*\* \|[^\n]+\| \*\*\[ \]", text)
    if not replay_row:
        hits.append("§3_table:ROADMAP-REPLAY-X row missing or malformed")

    replay_section = re.search(
        r"\| \*\*ROADMAP-REPLAY-X\*\* \|[^\n]+READY FOR INDEPENDENT AUDIT",
        text,
    )
    if not replay_section:
        hits.append("§3_table:ROADMAP-REPLAY-X missing READY FOR INDEPENDENT AUDIT wording")

    return hits


def assert_roadmap_drift_zero() -> None:
    violations = roadmap_drift_violations()
    assert not violations, violations
