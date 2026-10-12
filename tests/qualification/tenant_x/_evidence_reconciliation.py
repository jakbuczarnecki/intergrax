# © Artur Czarnecki. All rights reserved.

"""Reconcile historical tenant-scoped evidence vs current HEAD (supporting only)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._types import HistoricalEvidenceItem, TENANT_X_P0_BASELINE_SHA

HISTORICAL_TENANT_EVIDENCE: Final[tuple[HistoricalEvidenceItem, ...]] = (
    HistoricalEvidenceItem(
        stage_id="GOV-X2",
        accepted_sha="e2d7a40b291c658677cc6d981183e20957c7cf0e",
        domain="governance_execution",
        frz_ten_criteria=("FRZ-TEN-01", "FRZ-TEN-02", "FRZ-TEN-03", "FRZ-TEN-07", "FRZ-TEN-09", "FRZ-TEN-10"),
        historical_status="CLOSED_LOCAL_TENANT_AUDIT",
        current_head_replay_required=True,
        replay_note="Scoped GOV-X2 chain; not global FRZ-TEN PASS",
    ),
    HistoricalEvidenceItem(
        stage_id="CTRL-X",
        accepted_sha="0e8d4025da1e9bfc089b92fc7f11635dcd30dc59",
        domain="control_plane",
        frz_ten_criteria=("FRZ-TEN-04", "FRZ-TEN-07"),
        historical_status="CLOSED_WITH_TRACKED_DEBT",
        current_head_replay_required=True,
        replay_note="run_id scoping explicitly not tenant isolation (evaluation plane)",
    ),
    HistoricalEvidenceItem(
        stage_id="STATE-X-R3-R2",
        accepted_sha="d5979a531e41f9a45a3a00b0b150765ee34bb0f8",
        domain="state_recovery",
        frz_ten_criteria=("FRZ-TEN-04", "FRZ-TEN-08"),
        historical_status="CLOSED_SCOPED",
        current_head_replay_required=True,
        replay_note="Does not prove every checkpoint/restore tenant path",
    ),
    HistoricalEvidenceItem(
        stage_id="TRACE-X-CERT",
        accepted_sha="69d8b111e8d2f6717b75d866d3738a0106d00304",
        domain="traceability",
        frz_ten_criteria=("FRZ-TEN-07",),
        historical_status="CLOSED_LOCAL_TENANT_AUDIT",
        current_head_replay_required=True,
        replay_note="TRACE-X P0 historically PARTIAL before remediation — revalidate on HEAD",
    ),
    HistoricalEvidenceItem(
        stage_id="CONFIG-X",
        accepted_sha="fac9a43f3b128fde63541d2b56ebd8f4a751e8fb",
        domain="configuration",
        frz_ten_criteria=("FRZ-TEN-05",),
        historical_status="CLOSED_LOCAL_TENANT_AUDIT",
        current_head_replay_required=True,
        replay_note="CONFIG-X local tenant audit; not global TENANT-X",
    ),
    HistoricalEvidenceItem(
        stage_id="COMPAT-X",
        accepted_sha="84b8739b8b6e62c8e580e26ca3011f6aa46d457a",
        domain="compatibility",
        frz_ten_criteria=("FRZ-TEN-11",),
        historical_status="CLOSED",
        current_head_replay_required=True,
        replay_note="Contract evolution only; tenant isolation not primary closure",
    ),
    HistoricalEvidenceItem(
        stage_id="INT-CONFIG-REAL-X-CERT",
        accepted_sha="a59744517b92847f55def1db22826d17d89ee155",
        domain="configuration_realization",
        frz_ten_criteria=("FRZ-TEN-01", "FRZ-TEN-02", "FRZ-TEN-05", "FRZ-TEN-07", "FRZ-TEN-11"),
        historical_status="CLOSED_ADVERSARIAL",
        current_head_replay_required=True,
        replay_note="INT-CONFIG scoped continuity proofs",
    ),
    HistoricalEvidenceItem(
        stage_id="CE-01",
        accepted_sha="a8d47a6396c3e5e73a979be74429ed59a93e1551",
        domain="context_engineering",
        frz_ten_criteria=("FRZ-TEN-05",),
        historical_status="CLOSED_LOCAL",
        current_head_replay_required=True,
        replay_note="Local CE tenant evidence; not global TENANT-X",
    ),
    HistoricalEvidenceItem(
        stage_id="INT-EXTCOMP-X-CERT",
        accepted_sha="234d3c03dce5708766f8e068afbd2ea496132002",
        domain="external_compatibility",
        frz_ten_criteria=("FRZ-TEN-01", "FRZ-TEN-02", "FRZ-TEN-07"),
        historical_status="CLOSED_ADVERSARIAL",
        current_head_replay_required=True,
        replay_note="Adversarial tenant matrix in extcomp certification tests",
    ),
)


def evidence_items_requiring_current_head_replay() -> tuple[HistoricalEvidenceItem, ...]:
    return tuple(item for item in HISTORICAL_TENANT_EVIDENCE if item.current_head_replay_required)


def p0_baseline_matches_head(head_sha: str) -> bool:
    return head_sha == TENANT_X_P0_BASELINE_SHA
