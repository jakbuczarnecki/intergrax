# © Artur Czarnecki. All rights reserved.

"""Six ROADMAP-REPLAY-X exit counters — deterministic @ AUDITED_HEAD worktree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.qualification.roadmap_replay_x._formal_reopen_register import count_reopened_historical_stages
from tests.qualification.roadmap_replay_x._ledger_classification import classify_ledger_row
from tests.qualification.roadmap_replay_x._prod_delta_disposition import historical_evidence_invalidation_violations
from tests.qualification.roadmap_replay_x._replay_blocker_register import (
    REPLAY_BLOCKER_REGISTER,
    count_unresolved_replay_blockers,
)
from tests.qualification.roadmap_replay_x._replay_finding_ledger import count_unclassified_replay_findings
from tests.qualification.roadmap_replay_x._roadmap_drift import roadmap_drift_violations
from tests.qualification.roadmap_replay_x._vbn_replay_counter import count_vendor_neutrality_replay_failures

_REPO = Path(__file__).resolve().parents[3]
_ROADMAP = _REPO / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"


@dataclass(frozen=True, slots=True)
class CounterResult:
    name: str
    value: int
    denominator: str
    source: str
    failure_mode: str


def _parse_section5_rows() -> list[tuple[str, str, str]]:
    text = _ROADMAP.read_text(encoding="utf-8")
    in_ledger = False
    raw: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if line.startswith("## 5. Closure evidence ledger"):
            in_ledger = True
            continue
        if in_ledger and line.startswith("## ") and not line.startswith("## 5."):
            break
        if not in_ledger or not line.startswith("|"):
            continue
        if line.startswith("| Etap") or line.startswith("|---"):
            continue
        cells = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(cells) == 2:
            raw.append((cells[0], "—", cells[1]))
        elif len(cells) >= 3:
            raw.append((cells[0], cells[1], cells[2]))
    return raw


def compute_reopened_historical_stages() -> CounterResult:
    rows = _parse_section5_rows()
    a_count = sum(1 for s, sh, n in rows if classify_ledger_row(s, sh, n).category == "A")
    reopen = count_reopened_historical_stages()
    return CounterResult(
        "reopened_historical_stages",
        reopen,
        f"mandatory_parent_ledger_rows={a_count}",
        "tests/qualification/roadmap_replay_x/_formal_reopen_register.py",
        "non-zero formal OPEN reopen or §3 contradiction",
    )


def compute_unresolved_replay_blockers() -> CounterResult:
    blockers = count_unresolved_replay_blockers()
    return CounterResult(
        "unresolved_replay_blockers",
        blockers,
        f"register_rows={len(REPLAY_BLOCKER_REGISTER)}",
        "tests/qualification/roadmap_replay_x/_replay_blocker_register.py",
        "increment on OPEN BLOCKER or register integrity failure",
    )


def compute_roadmap_drift() -> CounterResult:
    violations = roadmap_drift_violations()
    return CounterResult(
        "roadmap_drift",
        len(violations),
        "§3 table + §3.0.x + §5 active stage strings",
        "tests/qualification/roadmap_replay_x/_roadmap_drift.py",
        "non-zero contradictory current-status strings",
    )


def compute_historical_evidence_invalidated() -> CounterResult:
    violations = historical_evidence_invalidation_violations(_REPO)
    return CounterResult(
        "historical_evidence_invalidated_by_later_changes",
        len(violations),
        "prod paths since DUP-X acceptance minus replay L2 scoped dispositions",
        "tests/qualification/roadmap_replay_x/_prod_delta_disposition.py",
        "non-zero unscoped prod delta or invalid scoped disposition",
    )


def compute_unclassified_replay_findings() -> CounterResult:
    rows = _parse_section5_rows()
    unclassified = count_unclassified_replay_findings()
    return CounterResult(
        "unclassified_replay_findings",
        unclassified,
        f"§5 ledger rows={len(rows)} + ECAP/VBN/arch/qual inventories",
        "tests/qualification/roadmap_replay_x/_replay_finding_ledger.py",
        "inventory finding without ledger row or §5 RULE-UNKNOWN",
    )


def compute_vendor_neutrality_replay_failures() -> CounterResult:
    failures = count_vendor_neutrality_replay_failures(_REPO)
    return CounterResult(
        "vendor_neutrality_replay_failures",
        failures,
        "7 VBN artifacts + ebh_6/_findings_register.py @ HEAD",
        "tests/qualification/roadmap_replay_x/_vbn_replay_counter.py",
        "non-zero PROHIBITED unresolved or VBN artifact/register mismatch",
    )


def all_counters() -> tuple[CounterResult, ...]:
    return (
        compute_reopened_historical_stages(),
        compute_unresolved_replay_blockers(),
        compute_roadmap_drift(),
        compute_historical_evidence_invalidated(),
        compute_unclassified_replay_findings(),
        compute_vendor_neutrality_replay_failures(),
    )
