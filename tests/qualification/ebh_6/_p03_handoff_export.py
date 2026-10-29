# © Artur Czarnecki. All rights reserved.

"""Protocol 03 handoff exports @ CERTIFICATION_HANDOFF_SHA (T-P03-01 / T-FAM-01)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.qualification.ebh_6._evidence_manifest import evidence_git_head_sha
from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG, FamilyIndependentDisposition
from tests.qualification.ebh_6._family_l9_evidence import FAMILY_L9_EVIDENCE
from tests.qualification.ebh_6._findings_register import FindingStatus, all_hardcoding_findings
from tests.qualification.ebh_6._l3_aj_group_reconciliation import (
    aj_group_by_layer_class,
    count_evidence_required_groups,
)
from tests.qualification.ebh_6._tier_audit import tier_census


def _p03_independent_acceptance_blockers() -> list[str]:
    from tests.qualification.ebh_6._l3_aj_group_reconciliation import AjControlOutcome, aj_group_by_layer_class

    blockers: list[str] = []
    reg = (
        Path(__file__).resolve().parents[3]
        / "docs/project/maintainers/qualification/EBH_6_OPERATOR_DECISION_REGISTER.md"
    )
    text = reg.read_text(encoding="utf-8")
    q2_row = text.split("| OD-EBH6-Q2D2 |", 1)[-1].split("\n", 1)[0]
    if "OPEN" in q2_row and "APPROVED" not in q2_row:
        blockers.append("OD-EBH6-Q2D2 OPEN")
    group = aj_group_by_layer_class()[("semantic_core", "F")]
    if group.outcome == AjControlOutcome.PASS_CANDIDATE:
        blockers.append("semantic_core×F PASS_CANDIDATE — E01-PRE-01 / Protocol 03 pending")
    elif group.outcome == AjControlOutcome.FINDING:
        blockers.append("semantic_core×F FINDING unresolved")
    blockers.append("EBH-6 independent acceptance not recorded (OD-EBH6-08 OPEN)")
    blockers.append("E02-AUD-01 PENDING (E02-OP-01 operator ratified — not auditor ACCEPT)")
    return blockers


def certification_banner_status() -> str:
    census = tier_census()
    t1 = census.get("T1", 0)
    blockers: list[str] = []
    if t1:
        blockers.append(f"T1={t1}")
    blockers.extend(_p03_independent_acceptance_blockers())
    from tests.qualification.ebh_6._findings_register import count_prohibited_unresolved

    if count_prohibited_unresolved():
        blockers.append("prohibited unresolved")
    if blockers:
        return f"PARTIAL / EXACT BLOCKERS ({'; '.join(blockers)})"
    return "READY FOR INDEPENDENT PROTOCOL 03"


def export_l9_blocker_replay_bundle() -> dict[str, dict[str, object]]:
    blockers = (
        "EBH6-BLK-L9-OWN-01",
        "EBH6-BLK-L9-EXE-01",
        "EBH6-BLK-L9-GOV-01",
        "EBH6-BLK-L9-REC-01",
        "EBH6-BLK-L9-SEC-01",
        "EBH6-BLK-L9-REL-01",
        "EBH6-BLK-L9-HRN-01",
        "EBH6-BLK-L9-TEN-01",
    )
    out: dict[str, dict[str, object]] = {}
    for bundle in FAMILY_L9_EVIDENCE:
        if bundle.blocker_id in blockers:
            out[bundle.blocker_id] = {
                "family_id": bundle.family_id,
                "gate_test_ids": list(bundle.gate_test_ids),
                "scope_limitation": bundle.scope_limitation,
                "invariant_under_test": bundle.invariant_under_test,
            }
    return out


def export_l9_blocker_replay_map() -> dict[str, tuple[str, ...]]:
    legacy: dict[str, tuple[str, ...]] = {}
    for bid, payload in export_l9_blocker_replay_bundle().items():
        legacy[bid] = tuple(payload["gate_test_ids"])  # type: ignore[index]
    return legacy


def assert_family_gate_sync_complete() -> None:
    """T-FAM-SYNC — catalog gate nodeids match L9 bundles with behavioral/arch tests."""
    from pathlib import Path

    from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG

    repo = Path(__file__).resolve().parents[3]
    bundles = {b.family_id: b for b in FAMILY_L9_EVIDENCE}
    assert len(EBH6_FAMILY_CATALOG) == 23
    for row in EBH6_FAMILY_CATALOG:
        bundle = bundles[row.family_id]
        gates = bundle.gate_test_ids or row.gate_test_ids
        assert gates, row.family_id
        for node in gates:
            assert "::test_" in node, node
            assert (repo / node.split("::", 1)[0]).is_file(), node


def export_aj_group_dispositions() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for (layer, cls), group in sorted(aj_group_by_layer_class().items()):
        rows.append(
            {
                "group_id": group.group_id,
                "layer": layer,
                "detection_class": cls,
                "outcome": group.outcome.value,
                "invariant_proves": group.invariant_proves,
                "evidence_refs": list(group.evidence_refs),
                "scope_limitation": group.scope_limitation,
                "mandatory_blocker": group.mandatory_blocker,
            },
        )
    return tuple(rows)


def export_family_dispositions() -> tuple[dict[str, Any], ...]:
    bundles = {b.family_id: b for b in FAMILY_L9_EVIDENCE}
    rows: list[dict[str, Any]] = []
    for row in EBH6_FAMILY_CATALOG:
        bundle = bundles[row.family_id]
        rows.append(
            {
                "family_id": row.family_id,
                "frz_family": row.frz_family,
                "independent_disposition": row.independent_disposition.value,
                "implementer_proposed": row.implementer_proposed_disposition.value,
                "l9_disposition": bundle.l9_evidence_disposition.value,
                "gate_test_ids": list(row.gate_test_ids),
                "blocker_id": bundle.blocker_id,
            },
        )
    return tuple(rows)


def export_layer_register_snapshot() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for row in all_hardcoding_findings():
        if not row.finding_id.startswith("EBH6-LAYER-"):
            continue
        rows.append(
            {
                "finding_id": row.finding_id,
                "layer": row.layer,
                "status": row.status.value,
                "evidence_sha": row.evidence_sha,
            },
        )
    return tuple(rows)


def p03_handoff_bundle() -> dict[str, Any]:
    sha = evidence_git_head_sha()
    assert_family_gate_sync_complete()
    census = tier_census()
    return {
        "certification_handoff_sha": sha,
        "certification_banner": certification_banner_status(),
        "tier_census": census,
        "aj_group_count": len(export_aj_group_dispositions()),
        "family_count": len(export_family_dispositions()),
        "l9_blocker_count": len(export_l9_blocker_replay_map()),
        "layer_candidate_resolved_count": sum(
            1 for row in export_layer_register_snapshot() if row["status"] == FindingStatus.CANDIDATE_RESOLVED.value
        ),
        "layer_l9_evidence_submitted_count": sum(
            1 for row in export_layer_register_snapshot() if row["status"] == FindingStatus.L9_EVIDENCE_SUBMITTED.value
        ),
        "aj_evidence_required_count": count_evidence_required_groups(),
        "independent_disposition_pending": sum(
            1
            for row in export_family_dispositions()
            if row["independent_disposition"] == FamilyIndependentDisposition.PENDING.value
        ),
        "aj_groups": export_aj_group_dispositions(),
        "families": export_family_dispositions(),
        "l9_blockers": export_l9_blocker_replay_map(),
        "l9_blocker_replay": export_l9_blocker_replay_bundle(),
        "layers": export_layer_register_snapshot(),
    }


def p03_handoff_json() -> str:
    return json.dumps(p03_handoff_bundle(), indent=2, sort_keys=True)
