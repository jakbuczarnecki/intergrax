# © Artur Czarnecki. All rights reserved.

"""MP-6D — atomic Collaborative Activity append store qualification."""

from __future__ import annotations

import ast
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.collaborative_work.collaborative_activity_append_store import (
    CollaborativeActivityAppendIntegrityError,
    CollaborativeActivityAppendPersistenceError,
    SQLiteCollaborativeActivityAppendStore,
    materialize_collaborative_activity_from_intent,
)
from intergrax.collaborative_work.persistence import (
    collaborative_activity_append_store_from_sqlite_bundle,
    open_sqlite_collaborative_work_repositories,
    sqlite_collaborative_activity_append_store,
)
from intergrax.collaborative_work.serialization import (
    collaborative_activity_from_json,
    collaborative_activity_to_json,
)
from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    ApprovalActivityProvenanceRef,
    ApprovalActivityTargetRef,
    ArtifactVersionActivityProvenanceRef,
    AssignmentActivityTargetRef,
    CollaborativeActivityAppendIntent,
    CollaborativeActivityAppendStore,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityPublication,
    ContextViewActivityProvenanceRef,
    ContextViewActivityTargetRef,
    DecisionActivityProvenanceRef,
    DecisionActivityTargetRef,
    ExecutionActivityProvenanceRef,
    GovernanceEvidenceActivityProvenanceRef,
    ProofReceiptActivityProvenanceRef,
    WorkArtifactActivityTargetRef,
    WorkArtifactVersionActivityTargetRef,
    WorkItemActivityTargetRef,
)
from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.execution_provenance import (
    AttemptId,
    ExecutionId,
    ExecutionProvenanceRef,
    RunId,
    TaskId,
)
from intergrax.contracts.governed_proof import GovernanceEvidenceRef
from tests.unit.collaborative_work.collaborative_activity_append_store_contract import (
    fixed_recorded_at,
    make_intent,
    make_publication,
    run_append_store_contract_suite,
    run_concurrent_distinct_contract,
    run_concurrent_duplicate_contract,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STORE_MODULE = _REPO_ROOT / "intergrax" / "collaborative_work" / "collaborative_activity_append_store.py"

_FORBIDDEN_IMPORT_MARKERS = (
    "collaborative_activity_ingestion",
    "collaborative_activity_publisher",
    "policy_source",
    "openai",
    "agents.",
    "applications.",
)


def _sqlite_store(tmp_path: Path) -> SQLiteCollaborativeActivityAppendStore:
    return sqlite_collaborative_activity_append_store(
        str(tmp_path / "mp6d.sqlite"),
        utc_now=fixed_recorded_at,
    )


def _second_sqlite_store(tmp_path: Path, db_name: str = "mp6d.sqlite") -> SQLiteCollaborativeActivityAppendStore:
    return sqlite_collaborative_activity_append_store(
        str(tmp_path / db_name),
        utc_now=fixed_recorded_at,
    )


def test_mp6d_store_no_forbidden_imports() -> None:
    text = _STORE_MODULE.read_text(encoding="utf-8-sig").lower()
    for marker in _FORBIDDEN_IMPORT_MARKERS:
        assert marker not in text, marker


def test_mp6d_sqlite_append_store_contract_suite(tmp_path: Path) -> None:
    run_append_store_contract_suite(lambda: _sqlite_store(tmp_path))


def test_mp6d_sqlite_concurrent_duplicate(tmp_path: Path) -> None:
    run_concurrent_duplicate_contract(
        lambda: _sqlite_store(tmp_path),
        open_second_connection=lambda: _second_sqlite_store(tmp_path),
    )


def test_mp6d_sqlite_concurrent_distinct(tmp_path: Path) -> None:
    run_concurrent_distinct_contract(
        lambda: _sqlite_store(tmp_path),
        open_second_connection=lambda: _second_sqlite_store(tmp_path),
    )


def test_mp6d_bundle_factory_uses_shared_sqlite_store(tmp_path: Path) -> None:
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "bundle.sqlite"))
    try:
        store = collaborative_activity_append_store_from_sqlite_bundle(
            bundle,
            utc_now=fixed_recorded_at,
        )
        activity = store.append_idempotent(make_intent())
        assert activity.append_position == 1
    finally:
        bundle.close()


def test_mp6d_custom_in_memory_store_satisfies_protocol() -> None:
    class _RecordingStore:
        def __init__(self) -> None:
            self.saved: CollaborativeActivityAppendIntent | None = None
            self._by_key: dict[tuple[str, str, str, str, str], object] = {}

        def append_idempotent(self, intent: CollaborativeActivityAppendIntent):
            key = intent.publication.idempotency_key
            token = (
                key.tenant_id,
                key.workspace_id,
                key.source.qualified_id,
                key.source_stable_id,
                key.activity_type.qualified_id,
            )
            if token in self._by_key:
                return self._by_key[token]
            activity = materialize_collaborative_activity_from_intent(
                intent,
                append_position=len(self._by_key) + 1,
                recorded_at=fixed_recorded_at(),
            )
            self._by_key[token] = activity
            self.saved = intent
            return activity

        def get_by_idempotency_key(self, key: ActivityIdempotencyKey):
            token = (
                key.tenant_id,
                key.workspace_id,
                key.source.qualified_id,
                key.source_stable_id,
                key.activity_type.qualified_id,
            )
            return self._by_key.get(token)

    custom: CollaborativeActivityAppendStore = _RecordingStore()
    activity = custom.append_idempotent(make_intent())
    assert custom.append_idempotent(make_intent()) is activity


def test_mp6d_serialization_roundtrip_target_and_provenance_variants() -> None:
    targets = (
        WorkItemActivityTargetRef(work_item_id="wi-1"),
        AssignmentActivityTargetRef(assignment_id="asg-1", work_item_id="wi-1"),
        WorkArtifactActivityTargetRef(work_artifact_id="art-1", work_item_id="wi-1"),
        WorkArtifactVersionActivityTargetRef(
            version_ref=WorkArtifactVersionRef(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                work_item_id="wi-1",
                work_artifact_id="art-1",
                work_artifact_version_id="v1",
            )
        ),
        ContextViewActivityTargetRef(view_id="view-1"),
        DecisionActivityTargetRef(decision_id="dec-1"),
        ApprovalActivityTargetRef(approval_id="apr-1"),
    )
    provenance = (
        ExecutionActivityProvenanceRef(
            execution=ExecutionProvenanceRef(
                task_id=TaskId("task_" + "a" * 32),
                run_id=RunId("run_" + "b" * 32),
                attempt_id=AttemptId("attempt_" + "c" * 32),
                execution_id=ExecutionId("exec_" + "d" * 32),
            )
        ),
        GovernanceEvidenceActivityProvenanceRef(
            evidence=GovernanceEvidenceRef(kind="hitl", evidence_id="ev-1"),
        ),
        ProofReceiptActivityProvenanceRef(proof_id="proof-1"),
    )
    for target in targets:
        pub = make_publication(stable_id=f"target-{target.schema_version}").model_copy(
            update={"target": target, "provenance_refs": provenance[:1]}
        )
        activity = materialize_collaborative_activity_from_intent(
            make_intent(pub),
            append_position=1,
            recorded_at=fixed_recorded_at(),
        )
        restored = collaborative_activity_from_json(collaborative_activity_to_json(activity))
        assert restored == activity


def test_mp6d_transaction_rollback_leaves_no_partial_row(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    intent = make_intent(make_publication(stable_id="rollback"))

    def boom(_conn, _activity):
        raise ValueError("simulated persistence failure")

    with patch.object(store, "_insert_activity", side_effect=boom):
        with pytest.raises(ValueError, match="simulated persistence failure"):
            store.append_idempotent(intent)

    assert store.get_by_idempotency_key(intent.publication.idempotency_key) is None
    retry = store.append_idempotent(intent)
    assert retry.append_position == 1


def test_mp6d_corrupt_row_fails_closed(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    intent = make_intent(make_publication(stable_id="corrupt"))
    store.append_idempotent(intent)
    conn = store._store.transaction()
    conn.execute(
        "UPDATE collaborative_activities SET record_json = ? WHERE activity_id = ?",
        ("not-json", materialize_collaborative_activity_from_intent(
            intent, append_position=1, recorded_at=fixed_recorded_at()
        ).activity_id),
    )
    conn.commit()
    with pytest.raises(CollaborativeActivityAppendIntegrityError):
        store.get_by_idempotency_key(intent.publication.idempotency_key)
