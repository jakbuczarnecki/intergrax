# © Artur Czarnecki. All rights reserved.

"""ECAP / VBN / arch / qual finding inventories vs replay ledger (B6-04)."""

from __future__ import annotations

from tests.qualification.ebh_6._family_catalog import (
    EBH6_FAMILY_CATALOG,
    FamilyIndependentDisposition,
)
from tests.qualification.ebh_6._findings_register import all_hardcoding_findings
from tests.qualification.roadmap_replay_x._frz_rows import FRZ_REPLAY_ROWS
from tests.qualification.roadmap_replay_x._ledger_classification import classify_ledger_row
from tests.qualification.roadmap_replay_x._replay_blocker_register import REPLAY_BLOCKER_REGISTER


def _parse_section5_rows() -> list[tuple[str, str, str]]:
    from tests.qualification.roadmap_replay_x._closure_counters import _parse_section5_rows as parse_rows

    return parse_rows()


def replay_finding_ledger() -> dict[str, str]:
    ledger: dict[str, str] = {}
    for row in all_hardcoding_findings():
        ledger[row.finding_id] = "RULE-VBN-EBH6-REGISTER"
    for family in EBH6_FAMILY_CATALOG:
        key = f"ARCH-FAMILY-{family.family_id}"
        if family.independent_disposition == FamilyIndependentDisposition.PENDING:
            ledger[key] = "RULE-ARCH-FAMILY-L9-PENDING"
        else:
            ledger[key] = "RULE-ARCH-FAMILY-RECERT"
    for frz_id, _status, _owner, _cat, _action, _blocker in FRZ_REPLAY_ROWS:
        ledger[f"FRZ-{frz_id}"] = "RULE-FRZ-REPLAY-ROW"
    for blocker in REPLAY_BLOCKER_REGISTER:
        ledger[f"BLOCKER-{blocker.blocker_id}"] = "RULE-ECAP-BLOCKER"
    return ledger


def replay_finding_inventories() -> dict[str, frozenset[str]]:
    vbn = frozenset(row.finding_id for row in all_hardcoding_findings())
    arch = frozenset(f"ARCH-FAMILY-{family.family_id}" for family in EBH6_FAMILY_CATALOG)
    qual = frozenset(f"FRZ-{row[0]}" for row in FRZ_REPLAY_ROWS)
    ecap = frozenset(f"BLOCKER-{row.blocker_id}" for row in REPLAY_BLOCKER_REGISTER)
    return {
        "vbn_ebh6_register": vbn,
        "architecture_family_matrix": arch,
        "qual_frz_replay_rows": qual,
        "ecap_blocker_register": ecap,
    }


def unclassified_replay_finding_violations(
    extra_inventory_ids: frozenset[str] | None = None,
) -> tuple[str, ...]:
    violations: list[str] = []
    ledger = replay_finding_ledger()
    inventories = replay_finding_inventories()
    union_ids: set[str] = set()
    for source, ids in inventories.items():
        union_ids.update(ids)
    if extra_inventory_ids:
        union_ids.update(extra_inventory_ids)
    for finding_id in sorted(union_ids):
        if finding_id not in ledger:
            violations.append(f"inventory_without_ledger:{finding_id}")
    for stage_id, _sha, notes in _parse_section5_rows():
        classification = classify_ledger_row(stage_id, _sha, notes)
        if classification.rule_id == "RULE-UNKNOWN" or classification.category == "?":
            violations.append(f"§5_unclassified:{stage_id}")
    return tuple(violations)


def count_unclassified_replay_findings(
    extra_inventory_ids: frozenset[str] | None = None,
) -> int:
    return len(unclassified_replay_finding_violations(extra_inventory_ids))
