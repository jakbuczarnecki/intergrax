# © Artur Czarnecki. All rights reserved.

"""Bidirectional reconciliation: discovered source claims ↔ Python SSOT registers (B6 R2-01/02)."""

from __future__ import annotations

from tests.qualification.roadmap_replay_x._closure_counters import all_counters
from tests.qualification.roadmap_replay_x._formal_reopen_register import (
    FORMAL_REOPEN_REGISTER,
    FormalReopenRecord,
)
from tests.qualification.roadmap_replay_x._replay_blocker_register import (
    REPLAY_BLOCKER_REGISTER,
    ReplayBlockerRecord,
)
from tests.qualification.roadmap_replay_x._replay_finding_ledger import (
    replay_finding_inventories,
    replay_finding_ledger,
)
from tests.qualification.roadmap_replay_x._replay_source_adapters import (
    discover_closure_matrix_blocker_claims,
    discover_ecap_counter_claims,
    discover_ecap_explicit_blocker_ids,
    discover_qual_frz_inventory_ids,
    discover_roadmap_category_a_reopen_claims,
    discover_vbn_finding_inventory_ids,
)


def replay_blocker_bidirectional_violations(
    register: tuple[ReplayBlockerRecord, ...] | None = None,
    *,
    closure_text: str | None = None,
    ecap_text: str | None = None,
) -> tuple[str, ...]:
    rows = REPLAY_BLOCKER_REGISTER if register is None else register
    register_ids = {row.blocker_id for row in rows}
    violations: list[str] = []

    for claim in discover_closure_matrix_blocker_claims(closure_text):
        if claim.blocker_id not in register_ids:
            violations.append(f"discovered_blocker_without_register:{claim.blocker_id}:{claim.source}")

    for ecap_id in discover_ecap_explicit_blocker_ids(ecap_text):
        if ecap_id not in register_ids:
            violations.append(f"ecap_blocker_without_register:{ecap_id}")

    for reg_id in register_ids:
        if reg_id not in discover_ecap_explicit_blocker_ids(ecap_text):
            if not any(c.blocker_id == reg_id for c in discover_closure_matrix_blocker_claims(closure_text)):
                violations.append(f"register_blocker_without_source_discovery:{reg_id}")

    for row in rows:
        if row.status == "CLOSED" and not (row.closed_task or "").strip():
            violations.append(f"closed_without_task:{row.blocker_id}")
        if row.status == "OPEN" and not row.evidence_ref.strip():
            violations.append(f"open_without_evidence:{row.blocker_id}")

    return tuple(violations)


def formal_reopen_bidirectional_violations(
    register: tuple[FormalReopenRecord, ...] | None = None,
    *,
    roadmap_text: str | None = None,
) -> tuple[str, ...]:
    rows = FORMAL_REOPEN_REGISTER if register is None else register
    by_parent = {row.parent_stage_id: row for row in rows}
    violations: list[str] = []

    for claim in discover_roadmap_category_a_reopen_claims(roadmap_text):
        rec = by_parent.get(claim.parent_stage_id)
        if rec is None or rec.status != "OPEN":
            violations.append(f"roadmap_reopen_without_formal_record:{claim.parent_stage_id}")

    open_parents = {row.parent_stage_id for row in rows if row.status == "OPEN"}
    for claim in discover_roadmap_category_a_reopen_claims(roadmap_text):
        open_parents.discard(claim.parent_stage_id)
    for orphan_parent in sorted(open_parents):
        violations.append(f"formal_reopen_without_roadmap_signal:{orphan_parent}")

    return tuple(violations)


def ecap_counter_bidirectional_violations(*, ecap_text: str | None = None) -> tuple[str, ...]:
    claims = discover_ecap_counter_claims(ecap_text)
    mechanical = {c.name: c.value for c in all_counters()}
    violations: list[str] = []
    for name, mech_value in mechanical.items():
        if name not in claims:
            violations.append(f"ecap_missing_counter_row:{name}")
            continue
        if claims[name] != mech_value:
            violations.append(f"ecap_counter_vs_mechanical:{name}:{claims[name]}!={mech_value}")
    return tuple(violations)


def finding_inventory_bidirectional_violations() -> tuple[str, ...]:
    ledger = replay_finding_ledger()
    inventories = replay_finding_inventories()
    violations: list[str] = []
    for source, ids in inventories.items():
        for finding_id in ids:
            if finding_id not in ledger:
                violations.append(f"inventory_without_ledger:{source}:{finding_id}")

    for finding_id in discover_vbn_finding_inventory_ids():
        if finding_id.startswith("FRZ-"):
            continue
        if finding_id not in ledger:
            violations.append(f"vbn_source_without_ledger:{finding_id}")

    for frz_id in discover_qual_frz_inventory_ids():
        if frz_id not in ledger:
            violations.append(f"qual_source_without_ledger:{frz_id}")

    return tuple(violations)


def replay_r2_reconciliation_violations(
    *,
    roadmap_text: str | None = None,
    ecap_text: str | None = None,
    closure_text: str | None = None,
) -> tuple[str, ...]:
    parts = (
        replay_blocker_bidirectional_violations(closure_text=closure_text, ecap_text=ecap_text),
        formal_reopen_bidirectional_violations(roadmap_text=roadmap_text),
        ecap_counter_bidirectional_violations(ecap_text=ecap_text),
        finding_inventory_bidirectional_violations(),
    )
    merged: list[str] = []
    for chunk in parts:
        merged.extend(chunk)
    return tuple(merged)
