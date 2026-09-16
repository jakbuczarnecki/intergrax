# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Offline legacy human decision disposition (MP-4R6) — admin/migration only."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from intergrax.contracts.human_decision_legacy_disposition import (
    HumanDecisionApproverRecoverySource,
    HumanDecisionLegacyDispositionReport,
    HumanDecisionLegacyDispositionStrategy,
    LegacyHumanDecisionProvenanceStatus,
    serialize_legacy_human_decision_archive_record,
)
from intergrax.integrations.providers.relational_store.sqlite.human_decision_legacy import (
    assess_sqlite_human_decision_legacy_rows,
    classify_sqlite_approver_provenance,
    list_sqlite_legacy_human_decision_archive_records,
)

__all__ = [
    "run_sqlite_human_decision_legacy_disposition",
    "export_sqlite_legacy_human_decision_archive_json",
]


def export_sqlite_legacy_human_decision_archive_json(db_path: Path) -> str:
    import json

    records = list_sqlite_legacy_human_decision_archive_records(db_path)
    payload = [serialize_legacy_human_decision_archive_record(record) for record in records]
    return json.dumps(payload, indent=2, sort_keys=True)


def run_sqlite_human_decision_legacy_disposition(
    db_path: Path,
    *,
    strategy: HumanDecisionLegacyDispositionStrategy,
    dry_run: bool = True,
    recovery_source: HumanDecisionApproverRecoverySource | None = None,
    allow_delete: bool = False,
) -> HumanDecisionLegacyDispositionReport:
    """
    Idempotent offline disposition for SQLite human decision rows.

    Default ``dry_run=True`` — no mutations unless explicitly disabled.
    """
    if strategy is HumanDecisionLegacyDispositionStrategy.CONTROLLED_DELETE and not allow_delete:
        raise ValueError("controlled delete requires allow_delete=True")

    assessment = assess_sqlite_human_decision_legacy_rows(
        db_path,
        recovery_source=recovery_source,
    )

    rows_scanned = assessment.rows_total
    rows_valid = assessment.rows_with_approver_json
    rows_missing = assessment.rows_missing_approver_json
    rows_malformed = assessment.rows_malformed_approver_json
    rows_recoverable = assessment.rows_recoverable
    rows_unrecoverable = assessment.rows_unrecoverable

    rows_recovered = 0
    rows_deleted = 0
    rows_quarantined = 0
    rows_unchanged = rows_valid

    if rows_scanned == 0 and strategy is not HumanDecisionLegacyDispositionStrategy.NO_DATA_PRESENT:
        strategy = HumanDecisionLegacyDispositionStrategy.NO_DATA_PRESENT

    if strategy is HumanDecisionLegacyDispositionStrategy.NO_DATA_PRESENT:
        return HumanDecisionLegacyDispositionReport(
            assessment=assessment,
            dry_run=dry_run,
            rows_scanned=rows_scanned,
            rows_valid=rows_valid,
            rows_missing_provenance=rows_missing,
            rows_malformed_provenance=rows_malformed,
            rows_recoverable=rows_recoverable,
            rows_unrecoverable=rows_unrecoverable,
            rows_recovered=0,
            rows_unchanged=rows_unchanged,
            rows_deleted=0,
            rows_quarantined=0,
        )

    if strategy is HumanDecisionLegacyDispositionStrategy.HISTORY_ONLY_QUARANTINE:
        rows_quarantined = rows_missing + rows_malformed
        rows_unchanged = rows_valid + rows_quarantined
        return HumanDecisionLegacyDispositionReport(
            assessment=assessment,
            dry_run=dry_run,
            rows_scanned=rows_scanned,
            rows_valid=rows_valid,
            rows_missing_provenance=rows_missing,
            rows_malformed_provenance=rows_malformed,
            rows_recoverable=rows_recoverable,
            rows_unrecoverable=rows_unrecoverable,
            rows_recovered=0,
            rows_unchanged=rows_unchanged,
            rows_deleted=0,
            rows_quarantined=rows_quarantined,
        )

    if strategy is HumanDecisionLegacyDispositionStrategy.PROVENANCE_RECOVERY:
        if recovery_source is None:
            raise ValueError("provenance recovery requires recovery_source")
        rows_recovered, rows_unchanged = _apply_exact_provenance_recovery(
            db_path,
            recovery_source=recovery_source,
            dry_run=dry_run,
        )
        rows_quarantined = max(0, rows_missing + rows_malformed - rows_recovered)
        return HumanDecisionLegacyDispositionReport(
            assessment=assess_sqlite_human_decision_legacy_rows(
                db_path,
                recovery_source=recovery_source,
            ),
            dry_run=dry_run,
            rows_scanned=rows_scanned,
            rows_valid=rows_valid,
            rows_missing_provenance=rows_missing,
            rows_malformed_provenance=rows_malformed,
            rows_recoverable=rows_recoverable,
            rows_unrecoverable=rows_unrecoverable,
            rows_recovered=rows_recovered,
            rows_unchanged=rows_unchanged,
            rows_deleted=0,
            rows_quarantined=rows_quarantined,
        )

    if strategy is HumanDecisionLegacyDispositionStrategy.CONTROLLED_DELETE:
        rows_deleted, rows_unchanged = _apply_controlled_delete(
            db_path,
            dry_run=dry_run,
        )
        return HumanDecisionLegacyDispositionReport(
            assessment=assess_sqlite_human_decision_legacy_rows(db_path),
            dry_run=dry_run,
            rows_scanned=rows_scanned,
            rows_valid=rows_valid,
            rows_missing_provenance=rows_missing,
            rows_malformed_provenance=rows_malformed,
            rows_recoverable=0,
            rows_unrecoverable=rows_missing + rows_malformed,
            rows_recovered=0,
            rows_unchanged=rows_unchanged,
            rows_deleted=rows_deleted,
            rows_quarantined=0,
        )

    raise ValueError(f"unsupported disposition strategy: {strategy.value!r}")


def _apply_exact_provenance_recovery(
    db_path: Path,
    *,
    recovery_source: HumanDecisionApproverRecoverySource,
    dry_run: bool,
) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = "DEFERRED"
    try:
        rows = conn.execute(
            "SELECT decision_id, tenant_id, approver_json FROM human_decisions"
        ).fetchall()
        recovered = 0
        unchanged = 0
        try:
            for row in rows:
                tenant_id = str(row["tenant_id"])
                status = classify_sqlite_approver_provenance(row["approver_json"], tenant_id=tenant_id)
                if status is LegacyHumanDecisionProvenanceStatus.VALID:
                    unchanged += 1
                    continue
                proof = recovery_source.recover_approver(
                    decision_id=str(row["decision_id"]),
                    tenant_id=tenant_id,
                )
                if proof is None or proof.tenant_id != tenant_id:
                    unchanged += 1
                    continue
                if not dry_run:
                    conn.execute(
                        """
                        UPDATE human_decisions
                        SET approver_json = ?
                        WHERE decision_id = ? AND tenant_id = ?
                        """,
                        (proof.model_dump_json(), str(row["decision_id"]), tenant_id),
                    )
                recovered += 1
            if dry_run:
                conn.rollback()
            else:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        return recovered, unchanged
    finally:
        conn.close()


def _apply_controlled_delete(db_path: Path, *, dry_run: bool) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = "DEFERRED"
    try:
        rows = conn.execute(
            "SELECT decision_id, tenant_id, approver_json FROM human_decisions"
        ).fetchall()
        deleted = 0
        unchanged = 0
        try:
            for row in rows:
                tenant_id = str(row["tenant_id"])
                status = classify_sqlite_approver_provenance(row["approver_json"], tenant_id=tenant_id)
                if status is LegacyHumanDecisionProvenanceStatus.VALID:
                    unchanged += 1
                    continue
                if not dry_run:
                    conn.execute(
                        """
                        DELETE FROM human_decisions
                        WHERE decision_id = ? AND tenant_id = ?
                        """,
                        (str(row["decision_id"]), tenant_id),
                    )
                deleted += 1
            if dry_run:
                conn.rollback()
            else:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        return deleted, unchanged
    finally:
        conn.close()
