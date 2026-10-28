# © Artur Czarnecki. All rights reserved.

"""Protocol 03 handoff exports @ CERTIFICATION_HANDOFF_SHA (T-P03-01 / T-FAM-01)."""

from __future__ import annotations

import json
from typing import Any

from tests.qualification.ebh_6._evidence_manifest import evidence_git_head_sha
from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG, FamilyIndependentDisposition
from tests.qualification.ebh_6._family_l9_evidence import FAMILY_L9_EVIDENCE
from tests.qualification.ebh_6._findings_register import FindingStatus, all_hardcoding_findings
from tests.qualification.ebh_6._l3_aj_group_reconciliation import aj_group_by_layer_class


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


def export_l9_blocker_replay_map() -> dict[str, tuple[str, ...]]:
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
    out: dict[str, tuple[str, ...]] = {}
    for bundle in FAMILY_L9_EVIDENCE:
        if bundle.blocker_id in blockers:
            out[bundle.blocker_id] = bundle.gate_test_ids
    return out


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
    return {
        "certification_handoff_sha": sha,
        "aj_group_count": len(export_aj_group_dispositions()),
        "family_count": len(export_family_dispositions()),
        "l9_blocker_count": len(export_l9_blocker_replay_map()),
        "layer_candidate_resolved_count": sum(
            1 for row in export_layer_register_snapshot() if row["status"] == FindingStatus.CANDIDATE_RESOLVED.value
        ),
        "independent_disposition_pending": sum(
            1
            for row in export_family_dispositions()
            if row["independent_disposition"] == FamilyIndependentDisposition.PENDING.value
        ),
        "aj_groups": export_aj_group_dispositions(),
        "families": export_family_dispositions(),
        "l9_blockers": export_l9_blocker_replay_map(),
        "layers": export_layer_register_snapshot(),
    }


def p03_handoff_json() -> str:
    return json.dumps(p03_handoff_bundle(), indent=2, sort_keys=True)
