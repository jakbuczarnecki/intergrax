# © Artur Czarnecki. All rights reserved.

"""Deterministic ROADMAP-REPLAY-X BLOCKER register (B6-01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BlockerStatus = Literal["OPEN", "CLOSED"]


@dataclass(frozen=True, slots=True)
class ReplayBlockerRecord:
    blocker_id: str
    source_ref: str
    status: BlockerStatus
    authority: str
    evidence_ref: str
    closed_task: str | None = None


# SSOT: empty @ handoff — increment only via explicit OPEN rows or register integrity failures.
REPLAY_BLOCKER_REGISTER: tuple[ReplayBlockerRecord, ...] = ()


def replay_blocker_register_violations(
    register: tuple[ReplayBlockerRecord, ...] | None = None,
) -> tuple[str, ...]:
    rows = REPLAY_BLOCKER_REGISTER if register is None else register
    violations: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.blocker_id.strip():
            violations.append("missing:blocker_id")
            continue
        if row.blocker_id in seen:
            violations.append(f"duplicate:{row.blocker_id}")
        seen.add(row.blocker_id)
        if not row.source_ref.strip():
            violations.append(f"missing_source_ref:{row.blocker_id}")
        if not row.authority.strip():
            violations.append(f"missing_authority:{row.blocker_id}")
        if not row.evidence_ref.strip():
            violations.append(f"missing_evidence_ref:{row.blocker_id}")
        if row.status == "CLOSED" and not (row.closed_task or "").strip():
            violations.append(f"closed_without_task:{row.blocker_id}")
        if row.status == "OPEN" and row.closed_task:
            violations.append(f"open_with_closed_task:{row.blocker_id}")
    return tuple(violations)


def count_unresolved_replay_blockers(
    register: tuple[ReplayBlockerRecord, ...] | None = None,
) -> int:
    violations = replay_blocker_register_violations(register)
    if violations:
        return len(violations)
    rows = REPLAY_BLOCKER_REGISTER if register is None else register
    return sum(1 for row in rows if row.status == "OPEN")
