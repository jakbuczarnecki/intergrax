# © Artur Czarnecki. All rights reserved.

"""MP-4R6 — offline legacy human decision disposition."""

from __future__ import annotations

import json

import pytest

from intergrax.applications._shared.harness_principal import (
    HarnessAuthenticatedPrincipal,
    harness_principal_to_approver_evidence,
)
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.human_decision_legacy_disposition import (
    HumanDecisionLegacyDispositionStrategy,
    LegacyHumanDecisionProvenanceStatus,
)
from intergrax.contracts.human_approver import HumanApproverEvidence, local_development_approver_evidence
from intergrax.integrations.providers.relational_store.sqlite.human_decision_legacy import (
    assess_sqlite_human_decision_legacy_rows,
    list_sqlite_legacy_human_decision_archive_records,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.persistence_errors import HumanDecisionApproverProvenanceError
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.migration.human_decision_legacy_disposition import (
    export_sqlite_legacy_human_decision_archive_json,
    run_sqlite_human_decision_legacy_disposition,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TASK = "task-legacy"
_DECISION = "hdec_legacy_missing_proof"


def _identity_approver() -> HumanApproverEvidence:
    principal = HarnessAuthenticatedPrincipal(
        tenant_id=_TENANT,
        user_id="idp-approver-1",
        principal_type=PrincipalType.USER,
        auth_subject="idp-subject-1",
        auth_mode="identity_provider",
    )
    return harness_principal_to_approver_evidence(principal)


def _insert_row(db_path, *, approver_json: str | None, decision_id: str = _DECISION) -> None:
    store = SQLiteHumanDecisionStore(db_path=db_path)
    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO human_decisions (
                decision_id, task_id, tenant_id, user_id, human_request_id,
                verdict, response_text, escalation_level, escalation_target,
                agent_id, run_id, notes, created_at_utc, approver_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                _TASK,
                _TENANT,
                "legacy-user",
                "",
                HumanResponseVerdict.APPROVE.value,
                "ok",
                0,
                None,
                None,
                "run-1",
                "",
                "2020-01-01T00:00:00+00:00",
                approver_json,
            ),
        )


class _StaticRecovery:
    def __init__(self, mapping: dict[tuple[str, str], HumanApproverEvidence]) -> None:
        self._mapping = mapping

    def recover_approver(self, *, decision_id: str, tenant_id: str) -> HumanApproverEvidence | None:
        return self._mapping.get((decision_id, tenant_id))


def test_assessment_empty_db(tmp_path) -> None:
    db = tmp_path / "human.db"
    SQLiteHumanDecisionStore(db_path=db)
    assessment = assess_sqlite_human_decision_legacy_rows(db)
    assert assessment.rows_total == 0


def test_assessment_counts_missing_and_valid(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_row(db, approver_json=None)
    _insert_row(db, approver_json=_identity_approver().model_dump_json(), decision_id="modern")
    assessment = assess_sqlite_human_decision_legacy_rows(db)
    assert assessment.rows_total == 2
    assert assessment.rows_with_approver_json == 1
    assert assessment.rows_missing_approver_json == 1


def test_archive_record_is_not_authoritative_input(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_row(db, approver_json=None)
    records = list_sqlite_legacy_human_decision_archive_records(db)
    assert len(records) == 1
    record = records[0]
    assert record.provenance_status is LegacyHumanDecisionProvenanceStatus.MISSING
    assert not hasattr(record, "approver")
    store = SQLiteHumanDecisionStore(db_path=db)
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision(_DECISION, _TENANT)


def test_quarantine_dry_run_zero_mutations(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_row(db, approver_json=None)
    report = run_sqlite_human_decision_legacy_disposition(
        db,
        strategy=HumanDecisionLegacyDispositionStrategy.HISTORY_ONLY_QUARANTINE,
        dry_run=True,
    )
    assert report.rows_quarantined == 1
    assert report.rows_recovered == 0
    store = SQLiteHumanDecisionStore(db_path=db)
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision(_DECISION, _TENANT)


def test_recovery_writes_exact_proof(tmp_path) -> None:
    db = tmp_path / "human.db"
    approver = _identity_approver()
    _insert_row(db, approver_json=None)
    source = _StaticRecovery({(_DECISION, _TENANT): approver})
    report = run_sqlite_human_decision_legacy_disposition(
        db,
        strategy=HumanDecisionLegacyDispositionStrategy.PROVENANCE_RECOVERY,
        dry_run=False,
        recovery_source=source,
    )
    assert report.rows_recovered == 1
    store = SQLiteHumanDecisionStore(db_path=db)
    loaded = store.get_decision(_DECISION, _TENANT)
    assert loaded is not None
    assert loaded.approver == approver


def test_recovery_tenant_mismatch_fails_closed(tmp_path) -> None:
    db = tmp_path / "human.db"
    wrong = local_development_approver_evidence(tenant_id="other", actor_id="x")
    _insert_row(db, approver_json=None)
    source = _StaticRecovery({(_DECISION, _TENANT): wrong})
    report = run_sqlite_human_decision_legacy_disposition(
        db,
        strategy=HumanDecisionLegacyDispositionStrategy.PROVENANCE_RECOVERY,
        dry_run=False,
        recovery_source=source,
    )
    assert report.rows_recovered == 0
    store = SQLiteHumanDecisionStore(db_path=db)
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision(_DECISION, _TENANT)


def test_recovery_idempotent_second_run(tmp_path) -> None:
    db = tmp_path / "human.db"
    approver = _identity_approver()
    _insert_row(db, approver_json=None)
    source = _StaticRecovery({(_DECISION, _TENANT): approver})
    run_sqlite_human_decision_legacy_disposition(
        db,
        strategy=HumanDecisionLegacyDispositionStrategy.PROVENANCE_RECOVERY,
        dry_run=False,
        recovery_source=source,
    )
    second = run_sqlite_human_decision_legacy_disposition(
        db,
        strategy=HumanDecisionLegacyDispositionStrategy.PROVENANCE_RECOVERY,
        dry_run=False,
        recovery_source=source,
    )
    assert second.rows_recovered == 0
    assert second.assessment.rows_with_approver_json == 1


def test_valid_modern_row_unchanged_by_recovery(tmp_path) -> None:
    db = tmp_path / "human.db"
    approver = _identity_approver()
    _insert_row(db, approver_json=approver.model_dump_json(), decision_id="modern")
    source = _StaticRecovery({("modern", _TENANT): approver})
    report = run_sqlite_human_decision_legacy_disposition(
        db,
        strategy=HumanDecisionLegacyDispositionStrategy.PROVENANCE_RECOVERY,
        dry_run=False,
        recovery_source=source,
    )
    assert report.rows_recovered == 0
    store = SQLiteHumanDecisionStore(db_path=db)
    loaded = store.get_decision("modern", _TENANT)
    assert loaded is not None
    assert loaded.approver == approver


def test_malformed_json_not_auto_overwritten_without_proof(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_row(db, approver_json="{bad")
    report = run_sqlite_human_decision_legacy_disposition(
        db,
        strategy=HumanDecisionLegacyDispositionStrategy.PROVENANCE_RECOVERY,
        dry_run=False,
        recovery_source=_StaticRecovery({}),
    )
    assert report.rows_recovered == 0
    assert report.assessment.rows_malformed_approver_json == 1


def test_controlled_delete_requires_flag(tmp_path) -> None:
    db = tmp_path / "human.db"
    with pytest.raises(ValueError, match="allow_delete"):
        run_sqlite_human_decision_legacy_disposition(
            db,
            strategy=HumanDecisionLegacyDispositionStrategy.CONTROLLED_DELETE,
            dry_run=False,
        )


def test_archive_export_json(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_row(db, approver_json=None)
    payload = json.loads(export_sqlite_legacy_human_decision_archive_json(db))
    assert len(payload) == 1
    assert payload[0]["provenance_status"] == LegacyHumanDecisionProvenanceStatus.MISSING.value
