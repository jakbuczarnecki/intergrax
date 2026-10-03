# © Artur Czarnecki. All rights reserved.

"""STATE-X-R2 — decision snapshot CAS, attempt lifecycle, lineage concurrency closure."""

from __future__ import annotations

import ast
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.decision_checkpoint import decision_checkpoint_state
from intergrax.contracts.decision_event_append import StaleDecisionEventAppendError
from intergrax.contracts.decision_finalization import (
    decision_finalization_key,
    guard_decision_finalization,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageIntegrityError,
    build_execution_lineage_attempt_scope,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryEligibilityRequest,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    MaterializedDecisionCheckpoint,
    StaleDecisionCheckpointWriteError,
    load_materialized_decision_checkpoint,
    save_decision_checkpoint,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.decision_recovery import (
    DecisionCheckpointCorruptionError,
    persist_terminal_decision_state,
    resume_decision_from_durable_state,
)
from intergrax.runtime.execution.in_memory_decision_checkpoint_persistence import (
    InMemoryDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.in_memory_decision_finalization_persistence import (
    InMemoryDecisionFinalizationPersistence,
)
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
)
from intergrax.runtime.execution.sqlite_decision_checkpoint_persistence import (
    SQLiteDecisionCheckpointPersistence,
)
from testing_support.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"
_DECISION_CHECKPOINT_PROTOCOL = (
    _REPO_ROOT / "intergrax/runtime/execution/decision_checkpoint_persistence.py"
)
_LINEAGE_ROOT = _REPO_ROOT / "intergrax/runtime/execution/lineage"
_RETRY_SERVICE = _REPO_ROOT / "intergrax/runtime/execution/retry/service.py"
_CHECKPOINT_PROVIDER_PATHS = (
    _REPO_ROOT / "intergrax/runtime/execution/in_memory_decision_checkpoint_persistence.py",
    _REPO_ROOT / "intergrax/runtime/execution/sqlite_decision_checkpoint_persistence.py",
)

_BYPASS_SAVE_EXCLUDE_SUFFIXES = (
    "in_memory_decision_checkpoint_persistence.py",
    "sqlite_decision_checkpoint_persistence.py",
    "decision_checkpoint_persistence.py",
)


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(tenant_id: str = "tenant-a") -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


def _lifecycle_at_finalization(identity: DecisionIdentity) -> object:
    state = initial_decision_lifecycle_state(identity)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.VERIFICATION)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.RESOLUTION)
    return transition_decision_lifecycle(state, DecisionLifecycleStage.FINALIZATION)


def _accepted(identity: DecisionIdentity) -> AuthoritativeAcceptedDecision:
    return AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="rollback"),
        ),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )


def _checkpoint_for_identity(identity: DecisionIdentity) -> object:
    guard = guard_decision_finalization(
        initial_decision_finalize_guard(decision_finalization_key(identity)),
        _accepted(identity),
    ).state
    return decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=guard,
    )


def _checkpoint_store_factory(kind: str, tmp_path: Path) -> InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload] | SQLiteDecisionCheckpointPersistence:
    if kind == "memory":
        return InMemoryDecisionCheckpointPersistence()
    return SQLiteDecisionCheckpointPersistence(
        db_path=tmp_path / "r2-checkpoint.db",
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )


def _retry_request() -> ExecutionRetryEligibilityRequest:
    return ExecutionRetryEligibilityRequest(
        classification=classify_execution_failure(
            kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
        ),
        attempt_number=1,
        max_attempts=3,
        cancelled=False,
        terminal_outcome=None,
        global_deadline_monotonic=None,
        now_monotonic=None,
        proposed_backoff_seconds=0.0,
        side_effect_idempotency_guaranteed=False,
    )


def test_r2_r1_q01_materialized_checkpoint_typed_envelope() -> None:
    source = _DECISION_CHECKPOINT_PROTOCOL.read_text(encoding="utf-8")
    assert "class MaterializedDecisionCheckpoint" in source
    assert "snapshot_revision" in source
    assert "def load_materialized(" in source
    tree = ast.parse(source, filename=str(_DECISION_CHECKPOINT_PROTOCOL))
    class_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MaterializedDecisionCheckpoint"
    ]
    assert len(class_defs) == 1
    cls = class_defs[0]
    assert cls.decorator_list
    field_names = {item.target.id for item in cls.body if isinstance(item, ast.AnnAssign)}
    assert field_names == {"key", "checkpoint", "snapshot_revision"}


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_r1_q02_materialized_load_absent(store_kind: str, tmp_path: Path) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    key = decision_finalization_key(identity)
    assert load_materialized_decision_checkpoint(store, key=key) is None


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_q02_decision_initial_cas(store_kind: str, tmp_path: Path) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    key = decision_finalization_key(identity)
    assert load_materialized_decision_checkpoint(store, key=key) is None
    save_decision_checkpoint(
        store,
        checkpoint=_checkpoint_for_identity(identity),
        expected_revision=0,
    )
    materialized = load_materialized_decision_checkpoint(store, key=key)
    assert materialized is not None
    assert materialized.snapshot_revision == 1


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_r1_q03_materialized_revision_one_after_first_save(
    store_kind: str,
    tmp_path: Path,
) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    key = decision_finalization_key(identity)
    checkpoint = _checkpoint_for_identity(identity)
    save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=0)
    materialized = load_materialized_decision_checkpoint(store, key=key)
    assert materialized is not None
    assert materialized.checkpoint is not None
    assert materialized.snapshot_revision == 1


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_q03_decision_update_cas(store_kind: str, tmp_path: Path) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    key = decision_finalization_key(identity)
    checkpoint = _checkpoint_for_identity(identity)
    save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=0)
    save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=1)
    materialized = load_materialized_decision_checkpoint(store, key=key)
    assert materialized is not None
    assert materialized.snapshot_revision == 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_r1_q04_materialized_revision_advances_with_cas(
    store_kind: str,
    tmp_path: Path,
) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    key = decision_finalization_key(identity)
    checkpoint = _checkpoint_for_identity(identity)
    save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=0)
    updated = _accepted(identity)
    updated_artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("incident_resolution"),
        content=IncidentDecisionPayload(recommendation="escalate"),
    )
    updated_accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=updated_artifact,
        lineage=updated.lineage,
    )
    guard = guard_decision_finalization(
        initial_decision_finalize_guard(key),
        updated_accepted,
    ).state
    checkpoint_v2 = decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=guard,
    )
    save_decision_checkpoint(store, checkpoint=checkpoint_v2, expected_revision=1)
    materialized = load_materialized_decision_checkpoint(store, key=key)
    assert materialized is not None
    assert materialized.snapshot_revision == 2
    assert (
        materialized.checkpoint.finalization.authoritative_outcome is not None
        and materialized.checkpoint.finalization.authoritative_outcome.artifact.content.recommendation
        == "escalate"
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_q04_stale_decision_snapshot(store_kind: str, tmp_path: Path) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    checkpoint = _checkpoint_for_identity(identity)
    save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=0)
    barrier = threading.Barrier(2)
    results: list[str] = []

    def writer() -> str:
        barrier.wait()
        try:
            save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=1)
            return "ok"
        except StaleDecisionCheckpointWriteError:
            return "stale"

    t1 = threading.Thread(target=lambda: results.append(writer()))
    t2 = threading.Thread(target=lambda: results.append(writer()))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results.count("ok") == 1
    assert results.count("stale") == 1
    materialized = load_materialized_decision_checkpoint(
        store,
        key=decision_finalization_key(identity),
    )
    assert materialized is not None
    assert materialized.snapshot_revision == 2


def test_r2_q05_no_production_blind_decision_write() -> None:
    violations: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name: str | None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue
            if name != "save_decision_checkpoint":
                continue
            if rel.endswith("decision_checkpoint_persistence.py") and isinstance(
                func,
                ast.Name,
            ):
                continue
            has_expected = any(
                kw.arg == "expected_revision" for kw in node.keywords if kw.arg
            )
            if not has_expected:
                violations.append(f"{rel}:{node.lineno}")
    assert violations == []


def test_r2_q06_finalization_dominates_snapshot() -> None:
    identity = _identity()
    accepted = _accepted(identity)
    from intergrax.contracts.decision_resolution import (
        AuthoritativeResolutionRecord,
        DecisionResolution,
    )

    rejected = AuthoritativeResolutionRecord(
        identity=identity,
        resolution=DecisionResolution.REJECTED,
    )
    durable_guard = guard_decision_finalization(
        initial_decision_finalize_guard(decision_finalization_key(identity)),
        accepted,
    ).state
    checkpoint = decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=guard_decision_finalization(
            initial_decision_finalize_guard(decision_finalization_key(identity)),
            rejected,
        ).state,
    )
    checkpoint_store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    save_decision_checkpoint(
        checkpoint_store,
        checkpoint=checkpoint,
        expected_revision=0,
    )
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    finalization_store.commit_authoritative_outcome(
        key=decision_finalization_key(identity),
        requested_outcome=accepted,
    )
    with pytest.raises(DecisionCheckpointCorruptionError):
        resume_decision_from_durable_state(
            checkpoint_persistence=checkpoint_store,
            finalization_persistence=finalization_store,
            key=decision_finalization_key(identity),
        )


class _ConcurrentSnapshotAdvancePersistence(
    InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload],
):
    def save(
        self,
        *,
        checkpoint: object,
        expected_revision: int | None = None,
    ) -> None:
        if checkpoint.lifecycle.stage is DecisionLifecycleStage.TERMINAL:  # type: ignore[attr-defined]
            if expected_revision is not None:
                key = checkpoint.finalization.key  # type: ignore[attr-defined]
                with self._lock:
                    self._revisions[key] = expected_revision + 1
        super().save(checkpoint=checkpoint, expected_revision=expected_revision)


def test_r2_q07_cas_loss_after_finalization_commit() -> None:
    identity = _identity()
    checkpoint = _checkpoint_for_identity(identity)
    key = decision_finalization_key(identity)
    checkpoint_store = _ConcurrentSnapshotAdvancePersistence()
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint, expected_revision=0)
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    with pytest.raises(StaleDecisionCheckpointWriteError):
        persist_terminal_decision_state(
            checkpoint_persistence=checkpoint_store,
            finalization_persistence=finalization_store,
            checkpoint=checkpoint,
            expected_snapshot_revision=1,
        )
    loaded = finalization_store.load_guard_state(key=key)
    assert loaded is not None
    assert loaded.authoritative_outcome is not None


def test_r2_q08_event_append_vs_snapshot_conflict_distinct() -> None:
    assert StaleDecisionEventAppendError is not StaleDecisionCheckpointWriteError
    assert StaleDecisionEventAppendError.__name__ != StaleDecisionCheckpointWriteError.__name__


@pytest.mark.parametrize("tenant_a,tenant_b", [("tenant-a", "tenant-b")])
def test_r2_q09_decision_tenant_separation(tenant_a: str, tenant_b: str) -> None:
    id_a = _identity(tenant_id=tenant_a)
    id_b = replace(
        _identity(tenant_id=tenant_b),
        decision_id=id_a.decision_id,
        scope=id_a.scope,
    )
    store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    save_decision_checkpoint(store, checkpoint=_checkpoint_for_identity(id_a), expected_revision=0)
    assert store.load(key=decision_finalization_key(id_a)) is not None
    assert store.load(key=decision_finalization_key(id_b)) is None
    assert load_materialized_decision_checkpoint(store, key=decision_finalization_key(id_b)) is None


def test_r2_r1_q05_sqlite_consistent_row_read() -> None:
    source = (
        _REPO_ROOT / "intergrax/runtime/execution/sqlite_decision_checkpoint_persistence.py"
    ).read_text(encoding="utf-8")
    fetch_block = source.split("def _fetch_materialized_row", 1)[1].split("def load_materialized", 1)[0]
    assert "SELECT checkpoint_blob, snapshot_revision" in fetch_block
    load_block = source.split("def load_materialized", 1)[1].split("def load", 1)[0]
    assert "_fetch_materialized_row" in load_block
    assert "SELECT checkpoint_blob" not in load_block.replace("_fetch_materialized_row", "")


def test_r2_r1_q06_memory_consistent_read_under_lock() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/runtime/execution/in_memory_decision_checkpoint_persistence.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "load_materialized":
            continue
        body_source = ast.get_source_segment(source, node) or ""
        assert "with self._lock" in body_source
        assert "_store" in body_source
        assert "_revisions" in body_source


def test_r2_r1_q07_standalone_decision_revision_api_removed() -> None:
    for rel in (
        "intergrax/runtime/execution/decision_checkpoint_persistence.py",
        "intergrax/runtime/execution/in_memory_decision_checkpoint_persistence.py",
        "intergrax/runtime/execution/sqlite_decision_checkpoint_persistence.py",
        "intergrax/runtime/execution/decision_recovery.py",
    ):
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "def materialized_revision(" not in text
        assert ".materialized_revision(" not in text


def _checkpoint_with_recommendation(
    identity: DecisionIdentity,
    recommendation: str,
) -> object:
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation=recommendation),
        ),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    guard = guard_decision_finalization(
        initial_decision_finalize_guard(decision_finalization_key(identity)),
        accepted,
    ).state
    return decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=guard,
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_r1_q08_stale_read_adversarial_before_terminal(
    store_kind: str,
    tmp_path: Path,
) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    key = decision_finalization_key(identity)
    checkpoint_a = _checkpoint_with_recommendation(identity, "rollback")
    save_decision_checkpoint(store, checkpoint=checkpoint_a, expected_revision=0)
    loaded_a = load_materialized_decision_checkpoint(store, key=key)
    assert loaded_a is not None
    checkpoint_b = _checkpoint_with_recommendation(identity, "contain")
    save_decision_checkpoint(store, checkpoint=checkpoint_b, expected_revision=1)
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    with pytest.raises(StaleDecisionCheckpointWriteError):
        persist_terminal_decision_state(
            checkpoint_persistence=store,
            finalization_persistence=finalization_store,
            checkpoint=checkpoint_a,
            expected_snapshot_revision=1,
            materialized_checkpoint=loaded_a,
        )
    current = load_materialized_decision_checkpoint(store, key=key)
    assert current is not None
    outcome = current.checkpoint.finalization.authoritative_outcome
    assert outcome is not None
    assert outcome.artifact.content.recommendation == "contain"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_r1_q09_stale_writer_does_not_commit_finalization(
    store_kind: str,
    tmp_path: Path,
) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    key = decision_finalization_key(identity)
    checkpoint_a = _checkpoint_with_recommendation(identity, "rollback")
    save_decision_checkpoint(store, checkpoint=checkpoint_a, expected_revision=0)
    loaded_a = load_materialized_decision_checkpoint(store, key=key)
    assert loaded_a is not None
    checkpoint_b = _checkpoint_with_recommendation(identity, "contain")
    save_decision_checkpoint(store, checkpoint=checkpoint_b, expected_revision=1)
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    with pytest.raises(StaleDecisionCheckpointWriteError):
        persist_terminal_decision_state(
            checkpoint_persistence=store,
            finalization_persistence=finalization_store,
            checkpoint=checkpoint_a,
            expected_snapshot_revision=1,
            materialized_checkpoint=loaded_a,
        )
    assert finalization_store.load_guard_state(key=key) is None


def test_r2_r1_q10_race_after_finalization_commit_preserved() -> None:
    test_r2_q07_cas_loss_after_finalization_commit()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_r1_q11_initial_absence_token(store_kind: str, tmp_path: Path) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    checkpoint = _checkpoint_for_identity(identity)
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    terminal = persist_terminal_decision_state(
        checkpoint_persistence=store,
        finalization_persistence=finalization_store,
        checkpoint=checkpoint,
        expected_snapshot_revision=0,
    )
    assert terminal.lifecycle.stage is DecisionLifecycleStage.TERMINAL
    identity_b = _identity()
    checkpoint_b = _checkpoint_for_identity(identity_b)
    store_b = _checkpoint_store_factory(store_kind, tmp_path)
    save_decision_checkpoint(store_b, checkpoint=checkpoint_b, expected_revision=0)
    finalization_b = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    with pytest.raises(StaleDecisionCheckpointWriteError):
        persist_terminal_decision_state(
            checkpoint_persistence=store_b,
            finalization_persistence=finalization_b,
            checkpoint=checkpoint_b,
            expected_snapshot_revision=0,
        )


def test_r2_r1_q12_token_key_binding_rejects_mismatch() -> None:
    id_a = _identity(tenant_id="tenant-a")
    id_b = _identity(tenant_id="tenant-b")
    store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    key_a = decision_finalization_key(id_a)
    save_decision_checkpoint(
        store,
        checkpoint=_checkpoint_for_identity(id_a),
        expected_revision=0,
    )
    materialized_a = load_materialized_decision_checkpoint(store, key=key_a)
    assert materialized_a is not None
    checkpoint_b = _checkpoint_for_identity(id_b)
    finalization_store = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    with pytest.raises((StaleDecisionCheckpointWriteError, ValueError)):
        persist_terminal_decision_state(
            checkpoint_persistence=store,
            finalization_persistence=finalization_store,
            checkpoint=checkpoint_b,
            expected_snapshot_revision=materialized_a.snapshot_revision,
            materialized_checkpoint=materialized_a,
        )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_r2_r1_first_write_absence_race(store_kind: str, tmp_path: Path) -> None:
    identity = _identity()
    store = _checkpoint_store_factory(store_kind, tmp_path)
    checkpoint = _checkpoint_for_identity(identity)
    barrier = threading.Barrier(2)
    results: list[str] = []

    def writer() -> str:
        barrier.wait()
        fin = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
        try:
            persist_terminal_decision_state(
                checkpoint_persistence=store,
                finalization_persistence=fin,
                checkpoint=checkpoint,
                expected_snapshot_revision=0,
            )
            return "ok"
        except StaleDecisionCheckpointWriteError:
            return "stale"

    t1 = threading.Thread(target=lambda: results.append(writer()))
    t2 = threading.Thread(target=lambda: results.append(writer()))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results.count("ok") == 1
    assert results.count("stale") == 1


def test_r2_r1_token_laundering_checkpoint_mismatch() -> None:
    id_a = _identity()
    id_b = _identity()
    store = InMemoryDecisionCheckpointPersistence[IncidentDecisionPayload]()
    save_decision_checkpoint(
        store,
        checkpoint=_checkpoint_for_identity(id_a),
        expected_revision=0,
    )
    mat_a = load_materialized_decision_checkpoint(
        store,
        key=decision_finalization_key(id_a),
    )
    assert mat_a is not None
    save_decision_checkpoint(
        store,
        checkpoint=_checkpoint_for_identity(id_b),
        expected_revision=0,
    )
    mat_b = load_materialized_decision_checkpoint(
        store,
        key=decision_finalization_key(id_b),
    )
    assert mat_b is not None
    fin = InMemoryDecisionFinalizationPersistence[IncidentDecisionPayload]()
    with pytest.raises(ValueError):
        persist_terminal_decision_state(
            checkpoint_persistence=store,
            finalization_persistence=fin,
            checkpoint=_checkpoint_for_identity(id_a),
            expected_snapshot_revision=mat_b.snapshot_revision,
            materialized_checkpoint=mat_b,
        )


def test_r2_q10_attempt_initial_creation() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt = mint_attempt_id()
    state = service.record_initial_attempt(
        tenant_id="tenant-a",
        run_id=run_id,
        attempt_id=attempt,
    )
    assert state.generation == 1
    assert service.get_active_attempt_id(tenant_id="tenant-a", run_id=run_id) == attempt


def test_r2_q11_attempt_successful_retry_transition() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    service.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    result = service.transition_to_next_attempt(
        tenant_id="tenant-a",
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        reason=AttemptTransitionReason.RETRY,
    )
    assert result.run_id == run_id
    assert result.previous_attempt_id == attempt_a1
    assert result.active_attempt_id != attempt_a1
    assert result.generation == 2


def test_r2_q12_stale_attempt_transition() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = service.transition_to_next_attempt(
        tenant_id="tenant-a",
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id
    with pytest.raises(StaleClaimError):
        service.transition_to_next_attempt(
            tenant_id="tenant-a",
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            reason=AttemptTransitionReason.RETRY,
        )
    assert service.get_active_attempt_id(tenant_id="tenant-a", run_id=run_id) == attempt_a2


def test_r2_q13_concurrent_attempt_transition() -> None:
    store = InMemoryAttemptLifecycleStore()
    service = AttemptLifecycleService(store)
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    service.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    barrier = threading.Barrier(2)
    winners: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        barrier.wait()
        try:
            result = service.transition_to_next_attempt(
                tenant_id="tenant-a",
                run_id=run_id,
                expected_attempt_id=attempt_a1,
                reason=AttemptTransitionReason.RETRY,
            )
            winners.append(str(result.active_attempt_id))
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(winners) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], StaleClaimError)


def test_r2_q14_failed_cas_does_not_rebind_active_identity() -> None:
    from unittest.mock import MagicMock

    from intergrax.contracts.attempt_lifecycle import AttemptLifecycleError

    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a1)
    service.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    service._store.compare_and_swap = MagicMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    try:
        with pytest.raises(AttemptLifecycleError):
            service.transition_to_next_attempt(
                tenant_id="tenant-a",
                run_id=run_id,
                expected_attempt_id=attempt_a1,
                reason=AttemptTransitionReason.RETRY,
            )
        assert peek_active_execution_identity() == (run_id, attempt_a1)
    finally:
        reset_active_execution_identity(token)


def test_r2_q15_attempt_tenant_isolation() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    service.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a)
    assert service.get_active_attempt_id(tenant_id="tenant-b", run_id=run_id) is None
    assert service.get_active_attempt_id(tenant_id="tenant-a", run_id=run_id) == attempt_a


def test_r2_q16_retry_seals_old_lineage() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    persistence = InMemoryExecutionLineagePersistence()
    service = ExecutionAttemptRetryService(lifecycle, lineage_persistence=persistence)
    tenant_id = "tenant-a"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    register_v1_attempt(persistence, scope)
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_retry_request(),
    )
    assert transition is not None
    seal = persistence.read_seal(scope)
    assert seal is not None
    assert seal.closure_kind is ExecutionLineageAttemptClosureKind.RETRY_SUPERSEDED


def test_r2_q17_failed_attempt_transition_does_not_seal_lineage() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    persistence = InMemoryExecutionLineagePersistence()
    service = ExecutionAttemptRetryService(lifecycle, lineage_persistence=persistence)
    tenant_id = "tenant-a"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    register_v1_attempt(persistence, scope)
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    wrong = mint_attempt_id()
    assert (
        service.transition_for_retry(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            expected_attempt_id=wrong,
            request=_retry_request(),
        )
        is None
    )
    assert persistence.read_seal(scope) is None


@pytest.mark.parametrize(
    "persistence_factory",
    [
        InMemoryExecutionLineagePersistence,
        lambda: DocumentStoreExecutionLineagePersistence(InMemoryDocumentStore()),
    ],
    ids=["memory", "document_store"],
)
def test_r2_q18_sealed_lineage_blocks_writes(persistence_factory: object) -> None:
    persistence = persistence_factory()  # type: ignore[operator]
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    with pytest.raises(ExecutionLineageIntegrityError):
        persistence.admit_child(scope, root, mint_execution_id(), root)


@pytest.mark.parametrize(
    "persistence_factory",
    [
        InMemoryExecutionLineagePersistence,
        lambda: DocumentStoreExecutionLineagePersistence(InMemoryDocumentStore()),
    ],
    ids=["memory", "document_store"],
)
def test_r2_q19_conflicting_lineage_seal(persistence_factory: object) -> None:
    persistence = persistence_factory()  # type: ignore[operator]
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    root = mint_execution_id()
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    with pytest.raises(ExecutionLineageIntegrityError, match="conflicting attempt seal"):
        persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.FAILED)


def test_r2_q20_lineage_tenant_isolation() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    scope_a = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    scope_b = build_execution_lineage_attempt_scope(
        tenant_id="tenant-b",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    root = mint_execution_id()
    register_v1_attempt(persistence, scope_a)
    persistence.open_segment(scope_a, root)
    persistence.admit_root(scope_a, root, root)
    assert persistence.read_attempt_lineage_state(scope_b) is None


def test_r2_q21_lineage_does_not_mint_attempts() -> None:
    violations: list[str] = []
    for path in _LINEAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        if "mint_attempt_id" in text or "mint_retry_attempt_id" in text:
            violations.append(rel)
    assert violations == []


def test_r2_q22_discovery_does_not_define_active_attempt() -> None:
    retry_source = _RETRY_SERVICE.read_text(encoding="utf-8")
    assert "AttemptLifecycleService" in retry_source
    assert "transition_to_next_attempt" in retry_source
    assert "register_attempt_for_run" not in retry_source


def test_r2_q24_retry_is_not_fork() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_retry_request(),
    )
    assert transition is not None
    assert transition.run_id == run_id
    assert transition.active_attempt_id != attempt_a1


def test_r2_production_direct_checkpoint_save_bypass_gate() -> None:
    violations: list[str] = []
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if path.name in _BYPASS_SAVE_EXCLUDE_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "save":
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            receiver = node.func.value.id
            if receiver not in {
                "checkpoint_persistence",
                "checkpoint_store",
                "persistence",
            }:
                continue
            if "decision" not in rel:
                continue
            if any(kw.arg == "expected_revision" for kw in node.keywords if kw.arg):
                continue
            if "DecisionCheckpoint" in path.read_text(encoding="utf-8"):
                violations.append(f"{rel}:{node.lineno}")
    assert violations == []
