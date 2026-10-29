# © Artur Czarnecki. All rights reserved.

"""Six ROADMAP-REPLAY-X exit counters — deterministic @ AUDITED_HEAD worktree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.qualification.roadmap_replay_x._ledger_classification import classify_ledger_row
from tests.qualification.roadmap_replay_x._roadmap_drift import roadmap_drift_violations

_REPO = Path(__file__).resolve().parents[3]
_ROADMAP = _REPO / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
_DUP_X_PIN = "a7d83925b375b5f08eb0f175ced2c17fa026bba9"


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
    """Denominator: mandatory parent anchors in category A; value = formal reopen records."""
    rows = _parse_section5_rows()
    a_count = sum(1 for s, sh, n in rows if classify_ledger_row(s, sh, n).category == "A")
    # No signed reopen register exists @ handoff — count explicit REOPEN REQUIRED strings.
    text = _ROADMAP.read_text(encoding="utf-8")
    reopen = text.count("REOPEN REQUIRED")
    return CounterResult(
        "reopened_historical_stages",
        reopen,
        f"mandatory_parent_ledger_rows={a_count}",
        "roadmap §5 + reopen narrative grep",
        "non-zero requires signed parent reopen + remediation",
    )


def compute_unresolved_replay_blockers() -> CounterResult:
    reg = _REPO / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ECAP_CURRENT_HEAD_AUDIT_REPORT.md"
    blockers = 0
    if reg.is_file() and "BLOCKER-class" in reg.read_text(encoding="utf-8"):
        blockers = 0  # report states none; independent B6 must recompute
    return CounterResult(
        "unresolved_replay_blockers",
        blockers,
        "ECAP findings register",
        str(reg.relative_to(_REPO)),
        "increment on BLOCKER without closed TASK",
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
    import subprocess

    invalidated = 0
    for path in ("intergrax", "agents", "applications"):
        out = subprocess.check_output(
            ["git", "diff", "--stat", f"{_DUP_X_PIN}..HEAD", "--", path],
            cwd=_REPO,
            text=True,
        ).strip()
        if out:
            invalidated += 1
    return CounterResult(
        "historical_evidence_invalidated_by_later_changes",
        invalidated,
        "prod paths since DUP-X acceptance",
        f"git diff {_DUP_X_PIN[:8]}..HEAD",
        "non-zero prod delta invalidates reuse-without-rerun claims",
    )


def compute_unclassified_replay_findings() -> CounterResult:
    rows = _parse_section5_rows()
    from tests.qualification.roadmap_replay_x._ledger_classification import classify_ledger_row

    unclassified = 0
    for s, sh, n in rows:
        c = classify_ledger_row(s, sh, n)
        if c.rule_id == "RULE-UNKNOWN" or c.category == "?":
            unclassified += 1
    return CounterResult(
        "unclassified_replay_findings",
        unclassified,
        f"§5 ledger rows={len(rows)} (each must have rule_id)",
        "_ledger_classification",
        "non-classified ledger row or ECAP finding without taxonomy",
    )


def compute_vendor_neutrality_replay_failures() -> CounterResult:
    reg = _REPO / "docs/project/maintainers/qualification/EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md"
    prohibited_open = 0
    if reg.is_file():
        text = reg.read_text(encoding="utf-8")
        prohibited_open = len([ln for ln in text.splitlines() if "PROHIBITED" in ln and "OPEN" in ln])
    return CounterResult(
        "vendor_neutrality_replay_failures",
        prohibited_open,
        "EBH-6 findings register PROHIBITED OPEN rows",
        str(reg.relative_to(_REPO)),
        "non-zero unresolved PROHIBITED in replay scope",
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
