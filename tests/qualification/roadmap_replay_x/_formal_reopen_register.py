# © Artur Czarnecki. All rights reserved.

"""Formal parent reopen records + §3 status cross-check (B6-02)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.qualification.roadmap_replay_x._ledger_classification import _CATEGORY_A_STAGE_IDS

ReopenStatus = Literal["OPEN", "CLOSED"]

_REPO = Path(__file__).resolve().parents[3]
_ROADMAP = _REPO / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"


@dataclass(frozen=True, slots=True)
class FormalReopenRecord:
    reopen_id: str
    parent_stage_id: str
    status: ReopenStatus
    authority: str
    evidence_ref: str
    remediation_sha: str | None = None


# SSOT: no signed reopen register @ handoff.
FORMAL_REOPEN_REGISTER: tuple[FormalReopenRecord, ...] = ()


def section3_checkbox_status(stage_id: str, roadmap_text: str | None = None) -> str | None:
    text = roadmap_text if roadmap_text is not None else _ROADMAP.read_text(encoding="utf-8")
    escaped = re.escape(stage_id)
    pattern = rf"^\|\s*\*?\*?{escaped}\*?\*?\s*\|[^\n]+\|\s*(\[[^\]]+\][^\|]*)\|\s*$"
    for line in text.splitlines():
        match = re.match(pattern, line)
        if match:
            return match.group(1).strip()
    bold_pattern = rf"^\|\s*\*\*{escaped}\*\*\s*\|[^\n]+\|\s*(\[[^\]]+\][^\|]*)\|\s*$"
    for line in text.splitlines():
        match = re.match(bold_pattern, line)
        if match:
            return match.group(1).strip()
    return None


def formal_reopen_register_violations(
    register: tuple[FormalReopenRecord, ...] | None = None,
    roadmap_text: str | None = None,
) -> tuple[str, ...]:
    rows = FORMAL_REOPEN_REGISTER if register is None else register
    text = roadmap_text if roadmap_text is not None else _ROADMAP.read_text(encoding="utf-8")
    violations: list[str] = []
    seen: set[str] = set()
    for rec in rows:
        if not rec.reopen_id.strip():
            violations.append("missing:reopen_id")
            continue
        if rec.reopen_id in seen:
            violations.append(f"duplicate:{rec.reopen_id}")
        seen.add(rec.reopen_id)
        if rec.parent_stage_id not in _CATEGORY_A_STAGE_IDS:
            violations.append(f"parent_not_category_a:{rec.reopen_id}:{rec.parent_stage_id}")
        checkbox = section3_checkbox_status(rec.parent_stage_id, text)
        if checkbox is None:
            violations.append(f"§3_row_missing:{rec.parent_stage_id}")
        elif rec.status == "OPEN":
            upper = checkbox.upper()
            if "REOPEN" not in upper and "BLOCKED" not in upper and "NOT CLOSED" not in upper:
                violations.append(f"open_reopen_parent_not_reopened:{rec.reopen_id}")
        elif rec.status == "CLOSED":
            if not (rec.remediation_sha or "").strip():
                violations.append(f"closed_reopen_without_sha:{rec.reopen_id}")
            if checkbox and "REOPEN" in checkbox.upper() and "CLOSED" not in checkbox.upper():
                violations.append(f"closed_reopen_parent_still_reopen:{rec.reopen_id}")
        if not rec.authority.strip() or not rec.evidence_ref.strip():
            violations.append(f"missing_reopen_metadata:{rec.reopen_id}")
    return tuple(violations)


def count_reopened_historical_stages(
    register: tuple[FormalReopenRecord, ...] | None = None,
    roadmap_text: str | None = None,
) -> int:
    violations = formal_reopen_register_violations(register, roadmap_text)
    rows = FORMAL_REOPEN_REGISTER if register is None else register
    open_rows = sum(1 for row in rows if row.status == "OPEN")
    return open_rows + len(violations)
