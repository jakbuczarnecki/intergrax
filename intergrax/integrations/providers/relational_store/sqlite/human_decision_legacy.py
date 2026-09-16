# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite-only legacy human decision inventory and history read (MP-4R6)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import ValidationError

from intergrax.contracts.human_decision_legacy_disposition import (
    HumanDecisionApproverRecoverySource,
    HumanDecisionLegacyDataAssessment,
    LegacyHumanDecisionArchiveRecord,
    LegacyHumanDecisionProvenanceStatus,
)
from intergrax.contracts.human_approver import HumanApproverEvidence

__all__ = [
    "assess_sqlite_human_decision_legacy_rows",
    "list_sqlite_legacy_human_decision_archive_records",
    "classify_sqlite_approver_provenance",
]


def _connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='human_decisions'"
    ).fetchone()
    return row is not None


def classify_sqlite_approver_provenance(
    approver_raw: str | None,
    *,
    tenant_id: str,
) -> LegacyHumanDecisionProvenanceStatus:
    if not approver_raw or not str(approver_raw).strip():
        return LegacyHumanDecisionProvenanceStatus.MISSING
    try:
        approver = HumanApproverEvidence.model_validate_json(approver_raw)
    except ValidationError:
        return LegacyHumanDecisionProvenanceStatus.MALFORMED
    if approver.tenant_id != tenant_id:
        return LegacyHumanDecisionProvenanceStatus.MALFORMED
    return LegacyHumanDecisionProvenanceStatus.VALID


def assess_sqlite_human_decision_legacy_rows(
    db_path: Path,
    *,
    recovery_source: HumanDecisionApproverRecoverySource | None = None,
) -> HumanDecisionLegacyDataAssessment:
    with _connection(db_path) as conn:
        if not _table_exists(conn):
            return HumanDecisionLegacyDataAssessment(
                rows_total=0,
                rows_with_approver_json=0,
                rows_missing_approver_json=0,
                rows_malformed_approver_json=0,
                rows_recoverable=0,
                rows_unrecoverable=0,
            )
        rows = conn.execute(
            """
            SELECT decision_id, tenant_id, approver_json
            FROM human_decisions
            """
        ).fetchall()

    total = len(rows)
    with_proof = 0
    missing = 0
    malformed = 0
    recoverable = 0
    unrecoverable = 0

    for row in rows:
        status = classify_sqlite_approver_provenance(
            row["approver_json"],
            tenant_id=str(row["tenant_id"]),
        )
        if status is LegacyHumanDecisionProvenanceStatus.VALID:
            with_proof += 1
            continue
        if status is LegacyHumanDecisionProvenanceStatus.MISSING:
            missing += 1
        else:
            malformed += 1
        if recovery_source is None:
            unrecoverable += 1
            continue
        proof = recovery_source.recover_approver(
            decision_id=str(row["decision_id"]),
            tenant_id=str(row["tenant_id"]),
        )
        if proof is None or proof.tenant_id != str(row["tenant_id"]):
            unrecoverable += 1
        else:
            recoverable += 1

    return HumanDecisionLegacyDataAssessment(
        rows_total=total,
        rows_with_approver_json=with_proof,
        rows_missing_approver_json=missing,
        rows_malformed_approver_json=malformed,
        rows_recoverable=recoverable,
        rows_unrecoverable=unrecoverable,
    )


def list_sqlite_legacy_human_decision_archive_records(
    db_path: Path,
) -> list[LegacyHumanDecisionArchiveRecord]:
    """History read path — rows lacking valid approver proof only."""
    with _connection(db_path) as conn:
        if not _table_exists(conn):
            return []
        rows = conn.execute(
            """
            SELECT
                decision_id, task_id, tenant_id, user_id, human_request_id,
                verdict, response_text, escalation_level, escalation_target,
                agent_id, run_id, notes, created_at_utc, approver_json
            FROM human_decisions
            ORDER BY created_at_utc ASC
            """
        ).fetchall()

    archive: list[LegacyHumanDecisionArchiveRecord] = []
    for row in rows:
        status = classify_sqlite_approver_provenance(
            row["approver_json"],
            tenant_id=str(row["tenant_id"]),
        )
        if status is LegacyHumanDecisionProvenanceStatus.VALID:
            continue
        archive.append(
            LegacyHumanDecisionArchiveRecord(
                decision_id=str(row["decision_id"]),
                tenant_id=str(row["tenant_id"]),
                task_id=str(row["task_id"]),
                user_id=str(row["user_id"]),
                human_request_id=str(row["human_request_id"]),
                verdict=str(row["verdict"]),
                response_text=str(row["response_text"]),
                escalation_level=int(row["escalation_level"]),
                escalation_target=row["escalation_target"],
                agent_id=row["agent_id"],
                run_id=row["run_id"],
                notes=str(row["notes"]),
                created_at_utc=str(row["created_at_utc"]),
                provenance_status=status,
            )
        )
    return archive
