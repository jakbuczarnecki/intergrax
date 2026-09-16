# © Artur Czarnecki. All rights reserved.

"""MP-4R6 — SQLite human decision store must not fabricate approver provenance on read."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.harness_principal import (
    HarnessAuthenticatedPrincipal,
    harness_principal_to_approver_evidence,
)
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.human_approver import (
    HumanApproverAuthMode,
    HumanApproverEvidence,
    local_development_approver_evidence,
)
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.runtime.codecraft.ownership import (
    CodeCraftSessionOwnership,
    codecraft_exec_hitl_notes,
    resolve_codecraft_exec_authorization,
)
from intergrax.runtime.human.models import (
    EscalationTarget,
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.runtime.human.persistence_errors import HumanDecisionApproverProvenanceError
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TASK = "task-legacy"
_DECISION = "hdec_legacy_missing_proof"


def _identity_provider_approver() -> HumanApproverEvidence:
    principal = HarnessAuthenticatedPrincipal(
        tenant_id=_TENANT,
        user_id="idp-approver-1",
        principal_type=PrincipalType.USER,
        auth_subject="idp-subject-1",
        auth_mode="identity_provider",
    )
    return harness_principal_to_approver_evidence(principal)


def _api_key_approver() -> HumanApproverEvidence:
    principal = HarnessAuthenticatedPrincipal(
        tenant_id=_TENANT,
        user_id="svc-key",
        principal_type=PrincipalType.SERVICE,
        auth_subject="api-key-subject",
        auth_mode="api_key",
    )
    return harness_principal_to_approver_evidence(principal)


def _insert_legacy_row(
    db_path,
    *,
    user_id: str,
    approver_json: str | None,
    decision_id: str = _DECISION,
) -> None:
    store = SQLiteHumanDecisionStore(db_path=db_path)
    with store._connection() as conn:  # noqa: SLF001 — persistence contract test fixture
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
                user_id,
                "",
                HumanResponseVerdict.APPROVE.value,
                "ok",
                0,
                None,
                None,
                "run-1",
                codecraft_exec_hitl_notes("craft-1"),
                "2020-01-01T00:00:00+00:00",
                approver_json,
            ),
        )


def test_legacy_row_without_approver_json_fails_closed(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_legacy_row(db, user_id="someone", approver_json=None)
    store = SQLiteHumanDecisionStore(db_path=db)

    with pytest.raises(HumanDecisionApproverProvenanceError) as exc_info:
        store.get_decision(_DECISION, _TENANT)

    err = exc_info.value
    assert err.decision_id == _DECISION
    assert err.tenant_id == _TENANT


def test_legacy_row_empty_user_id_without_approver_fails_closed(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_legacy_row(db, user_id="", approver_json=None)
    store = SQLiteHumanDecisionStore(db_path=db)

    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.list_for_task(_TASK, _TENANT)


def test_malformed_approver_json_fails_closed(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_legacy_row(db, user_id="u1", approver_json="{not-json")
    store = SQLiteHumanDecisionStore(db_path=db)

    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision(_DECISION, _TENANT)


def test_identity_provider_approver_round_trip(tmp_path) -> None:
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    approver = _identity_provider_approver()
    record = build_human_decision_record(
        task_id=_TASK,
        tenant_id=_TENANT,
        approver=approver,
        verdict=HumanResponseVerdict.APPROVE,
        response_text="yes",
        task_subject_user_id="task-subject-not-approver",
    )
    store.record(record)
    loaded = store.get_decision(record.decision_id, _TENANT)
    assert loaded is not None
    assert loaded.approver == approver
    assert loaded.approver.auth_mode is HumanApproverAuthMode.IDENTITY_PROVIDER
    assert loaded.approver.auth_subject == "idp-subject-1"
    assert loaded.approver.principal_type is PrincipalType.USER
    assert loaded.user_id == "task-subject-not-approver"


def test_api_key_approver_round_trip(tmp_path) -> None:
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    approver = _api_key_approver()
    record = build_human_decision_record(
        task_id=_TASK,
        tenant_id=_TENANT,
        approver=approver,
        verdict=HumanResponseVerdict.APPROVE,
        response_text="yes",
    )
    store.record(record)
    loaded = store.list_for_task(_TASK, _TENANT)[0]
    assert loaded.approver == approver
    assert loaded.approver.auth_mode is HumanApproverAuthMode.API_KEY


def test_explicit_local_development_persisted_round_trip(tmp_path) -> None:
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    approver = local_development_approver_evidence(tenant_id=_TENANT, actor_id="explicit-dev-op")
    record = build_human_decision_record(
        task_id=_TASK,
        tenant_id=_TENANT,
        approver=approver,
        verdict=HumanResponseVerdict.APPROVE,
        response_text="dev",
    )
    store.record(record)
    loaded = store.get_decision(record.decision_id, _TENANT)
    assert loaded is not None
    assert loaded.approver == approver
    assert loaded.approver.auth_mode is HumanApproverAuthMode.LOCAL_DEVELOPMENT


def test_persisted_approver_json_wins_over_legacy_user_id(tmp_path) -> None:
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    approver = _identity_provider_approver()
    record = build_human_decision_record(
        task_id=_TASK,
        tenant_id=_TENANT,
        approver=approver,
        verdict=HumanResponseVerdict.APPROVE,
        response_text="yes",
        task_subject_user_id="legacy-user-id-field",
    )
    store.record(record)

    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE human_decisions SET user_id = ? WHERE decision_id = ?",
            ("different-legacy-user", record.decision_id),
        )

    loaded = store.get_decision(record.decision_id, _TENANT)
    assert loaded is not None
    assert loaded.approver == approver
    assert loaded.user_id == "different-legacy-user"


def test_cross_tenant_persisted_approver_fails_closed(tmp_path) -> None:
    db = tmp_path / "human.db"
    other_tenant_approver = local_development_approver_evidence(tenant_id="other-tenant", actor_id="x")
    _insert_legacy_row(
        db,
        user_id="u",
        approver_json=other_tenant_approver.model_dump_json(),
    )
    store = SQLiteHumanDecisionStore(db_path=db)
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision(_DECISION, _TENANT)


def test_codecraft_authorization_cannot_use_legacy_missing_provenance_row(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_legacy_row(db, user_id="someone", approver_json=None)
    store = SQLiteHumanDecisionStore(db_path=db)
    ctx = ToolWiringContext(human_decision_store=store)
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True)
    ownership = CodeCraftSessionOwnership(tenant_id=_TENANT, task_id=_TASK, run_id="run-1")

    with pytest.raises(HumanDecisionApproverProvenanceError):
        resolve_codecraft_exec_authorization(
            ctx,
            profile=profile,
            ownership=ownership,
            craft_id="craft-1",
        )


def test_summarize_queue_counts_legacy_rows_without_materializing_approver(tmp_path) -> None:
    db = tmp_path / "human.db"
    _insert_legacy_row(db, user_id="someone", approver_json=None)
    store = SQLiteHumanDecisionStore(db_path=db)
    counts = store.summarize_queue(_TENANT)
    assert counts.get(HumanResponseVerdict.APPROVE.value) == 1
