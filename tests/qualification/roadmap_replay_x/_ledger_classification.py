# © Artur Czarnecki. All rights reserved.

"""Deterministic §5 ledger row classification for ROADMAP-REPLAY-X."""

from __future__ import annotations

from dataclasses import dataclass

# Category A — mandatory parent replay anchors (exact §5 stage labels @ a2ed11f).
_CATEGORY_A_STAGE_IDS: frozenset[str] = frozenset(
    {
        "EBH-2F",
        "EBH-2G",
        "EBH-2H",
        "EBH-2I",
        "HARNESS-QINF-01",
        "HARNESS-W4",
        "HARNESS-W5",
        "HARNESS-W6",
        "GOV-X1-FINAL",
        "INT-EXTCOMP-X",
        "INT-CONFIG-REAL-X",
        "AW-7C",
        "EBH-3",
        "EBH-4",
        "GOV-X2",
        "CTRL-X",
        "STATE-X-FINAL-R1-R1",
        "TRACE-X",
        "CONFIG-X",
        "COMPAT-X",
        "TENANT-X",
        "QUAL-X",
        "SKILL-X",
        "TOOL-LIFE-X",
        "DIST-SCALE-X",
        "EBH-5",
        "EBH-6",
        "DUP-X",
    }
)

# Category C — rejected independent audit attempts (not accepted parent rows).
_CATEGORY_C_STAGE_IDS: frozenset[str] = frozenset(
    {
        "EBH-2F-R2",
        "EBH-2F-R2-R1",
        "EBH-2F-R2-R1-R1",
        "EBH-2F-R2-R1-R1-R1",
        "EBH-2F-R2-R1-R1-R1-R1",
    }
)

_PLACEHOLDER_STAGE_IDS: frozenset[str] = frozenset({"remaining mandatory stages"})

_FUTURE_STAGE_IDS: frozenset[str] = frozenset(
    {"EVOL-X", "ENT-AUDIT-X", "ARCH-FREEZE", "SCENARIO-READINESS-X", "SCENARIO-GATE"}
)


@dataclass(frozen=True, slots=True)
class Classification:
    category: str
    rule_id: str
    replay_action: str


def classify_ledger_row(stage_id: str, closure_sha: str, notes: str) -> Classification:
    sid = stage_id.strip()
    nl = notes.lower()
    sha = closure_sha.strip()

    if sid in _PLACEHOLDER_STAGE_IDS:
        return Classification("P", "RULE-PLACEHOLDER-ROW", "NOT_EVIDENCE_ROW")

    if sid in _FUTURE_STAGE_IDS or (sha in {"—", "-", ""} and "not started" in nl):
        return Classification("G", "RULE-FUTURE-MANDATORY", "NOT_REPLAY_CLOSURE_TARGET")

    if sid in _CATEGORY_C_STAGE_IDS and "independent audit rejected" in nl:
        return Classification("C", "RULE-REJECTED-AUDIT", "REJECTED_LINEAGE_ONLY")

    if sid == "EBH-2F" and "independent audit rejected" in nl and "accepted parent closure" not in nl:
        return Classification("C", "RULE-EBH2F-BLOCKED", "REJECTED_LINEAGE_ONLY")

    if sid.startswith("PROD-Q") or sid == "PROD-Q-R1-R7":
        return Classification("E", "RULE-PROD-Q-DIM-B", "PATH_B_NA_WITH_EVIDENCE")

    if "wl-7" in nl and ("14400" in nl or "4h" in nl or "deferred" in nl):
        return Classification("F", "RULE-DIST-WL7-DEFERRED", "SCOPED_DEFERRED_VERIFY")

    if "not yet qualified" in nl or "51 not yet" in nl:
        return Classification("F", "RULE-EBH5-NOT-YET", "SCOPED_DEFERRED_VERIFY")

    if "superseded" in nl or "historical cursor" in nl:
        return Classification("D", "RULE-SUPERSEDED-NARRATIVE", "MAP_TO_SUPERSEDING_SHA")

    if sid in _CATEGORY_A_STAGE_IDS and ("closed" in nl or "accepted" in nl):
        return Classification("A", "RULE-MANDATORY-PARENT", "L2_PARENT_REPLAY")

    if "independent audit rejected" in nl:
        return Classification("C", "RULE-REJECTED-NOTES", "REJECTED_LINEAGE_ONLY")

    if "blocked pending" in nl and "accepted" not in nl[:80].lower():
        return Classification("C", "RULE-BLOCKED-PENDING", "REJECTED_LINEAGE_ONLY")

    if "closed" in nl or "accepted" in nl or "pass" in nl:
        return Classification("B", "RULE-ACCEPTED-CHILD", "L2_IMPACT_VIA_PARENT")

    return Classification("I", "RULE-OTHER", "CLASSIFY_OTHER_ACCEPTED")


def category_totals(rows: list[tuple[str, str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sid, sha, notes in rows:
        cat = classify_ledger_row(sid, sha, notes).category
        counts[cat] = counts.get(cat, 0) + 1
    return counts
