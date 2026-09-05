# © Artur Czarnecki. All rights reserved.

"""DS-REC-01 — durable atomic finalization persistence conformance."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from intergrax.contracts.decision_finalization import decision_finalization_key
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
    assert_concurrent_finalization_race,
    assert_concurrent_idempotent_replay,
    assert_decision_finalization_persistence_conformance,
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.in_memory_decision_finalization_persistence import (
    InMemoryDecisionFinalizationPersistence,
)
from intergrax.runtime.execution.sqlite_decision_finalization_persistence import (
    SQLiteDecisionFinalizationPersistence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _accepted_identity() -> DecisionIdentity:
    decision_id = mint_decision_id()
    return DecisionIdentity(
        decision_id=decision_id,
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="release-proof"),
        tenant_id="tenant-release",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def test_in_memory_finalization_conformance() -> None:
    assert_decision_finalization_persistence_conformance(
        InMemoryDecisionFinalizationPersistence,
        label="in_memory",
    )


def test_sqlite_finalization_conformance(tmp_path: Path) -> None:
    db_path = tmp_path / "finalization.db"

    def _factory() -> SQLiteDecisionFinalizationPersistence:
        return SQLiteDecisionFinalizationPersistence(
            db_path=db_path,
            payload_codecs=conformance_artifact_payload_codec_registry(),
        )

    assert_decision_finalization_persistence_conformance(_factory, label="sqlite")


def test_in_memory_concurrent_race() -> None:
    store = InMemoryDecisionFinalizationPersistence()
    assert_concurrent_finalization_race(lambda: store, label="in_memory")


def test_sqlite_concurrent_race(tmp_path: Path) -> None:
    db_path = tmp_path / "finalization-race.db"
    store = SQLiteDecisionFinalizationPersistence(
        db_path=db_path,
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )
    try:
        assert_concurrent_finalization_race(lambda: store, label="sqlite")
    finally:
        store.close()


def test_sqlite_finalization_releases_db_after_close(tmp_path: Path) -> None:
    db_dir = tmp_path / "release"
    db_dir.mkdir()
    db_path = db_dir / "finalization.db"
    identity = _accepted_identity()
    store = SQLiteDecisionFinalizationPersistence(
        db_path=db_path,
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )
    store.commit_authoritative_outcome(
        key=decision_finalization_key(identity),
        requested_outcome=AuthoritativeAcceptedDecision(
            identity=identity,
            artifact=DecisionArtifact(
                kind=validate_decision_artifact_kind("incident_resolution"),
                content=IncidentDecisionPayload(recommendation="release-proof"),
            ),
            lineage=decision_version_lineage(
                current=decision_lineage_ref(identity.version),
            ),
        ),
    )
    store.close()
    released_path = db_dir / "released"
    db_path.rename(released_path)
    shutil.rmtree(released_path.parent)


def test_sqlite_concurrent_race_releases_db_after_close(tmp_path: Path) -> None:
    db_dir = tmp_path / "race-release"
    db_dir.mkdir()
    db_path = db_dir / "finalization.db"
    store = SQLiteDecisionFinalizationPersistence(
        db_path=db_path,
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )
    try:
        assert_concurrent_finalization_race(lambda: store, label="sqlite")
    finally:
        store.close()
    released_path = db_dir / "released"
    db_path.rename(released_path)
    shutil.rmtree(released_path.parent)


def test_in_memory_concurrent_idempotent_replay() -> None:
    store = InMemoryDecisionFinalizationPersistence()
    assert_concurrent_idempotent_replay(lambda: store, label="in_memory")


def test_sqlite_concurrent_idempotent_replay(tmp_path: Path) -> None:
    db_path = tmp_path / "finalization-idempotent.db"
    store = SQLiteDecisionFinalizationPersistence(
        db_path=db_path,
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )
    assert_concurrent_idempotent_replay(lambda: store, label="sqlite")


def test_sqlite_finalization_does_not_use_unconditional_overwrite() -> None:
    source = Path(
        "intergrax/runtime/execution/sqlite_decision_finalization_persistence.py",
    ).read_text(encoding="utf-8")
    assert "ON CONFLICT" not in source
    assert "DO UPDATE" not in source
    assert "INSERT INTO decision_finalizations" in source
