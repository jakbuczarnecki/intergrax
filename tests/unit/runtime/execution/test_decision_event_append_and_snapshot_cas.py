# © Artur Czarnecki. All rights reserved.

"""W3-C2 — decision event compare-and-append and snapshot revision CAS."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.decision_checkpoint import decision_checkpoint_state
from intergrax.contracts.decision_event_append import (
    DuplicateDecisionEventError,
    StaleDecisionEventAppendError,
)
from intergrax.contracts.decision_finalization import (
    DecisionFinalizationKey,
    decision_finalization_key,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.decision_lifecycle import initial_decision_lifecycle_state
from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    StaleDecisionCheckpointWriteError,
    save_decision_checkpoint,
)
from intergrax.runtime.execution.decision_event_payload_codec import (
    decision_event_payload_codec_registry,
)
from intergrax.runtime.execution.decision_event_record import DecisionEventRecord
from intergrax.runtime.execution.decision_finalization_conformance import (
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.in_memory_decision_checkpoint_persistence import (
    InMemoryDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.in_memory_decision_event_append_persistence import (
    InMemoryDecisionEventAppendPersistence,
)
from intergrax.runtime.execution.sqlite_decision_checkpoint_persistence import (
    SQLiteDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.sqlite_decision_event_append_persistence import (
    SQLiteDecisionEventAppendPersistence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class StreamNotePayload:
    note: str


@dataclass(frozen=True, slots=True)
class StreamNotePayloadCodec:
    def payload_type_name(self) -> str:
        return "decision.stream.note.v1"

    def encode(self, payload: object) -> JsonValue:
        if type(payload) is not StreamNotePayload:
            raise TypeError("stream note codec expects StreamNotePayload")
        return {"note": payload.note}

    def decode(self, payload: JsonValue) -> StreamNotePayload:
        if type(payload) is not dict:
            raise TypeError("stream note payload must be a JSON object")
        note = payload.get("note")
        if type(note) is not str:
            raise TypeError("stream note payload note must be str")
        return StreamNotePayload(note=note)


def _event_codec_registry() -> object:
    return decision_event_payload_codec_registry(codecs=(StreamNotePayloadCodec(),))


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-42"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )


def _key(identity: DecisionIdentity) -> DecisionFinalizationKey:
    return decision_finalization_key(identity)


def _event(
    identity: DecisionIdentity,
    *,
    event_id: str,
    note: str,
    sequence: int = 0,
) -> DecisionEventRecord:
    return DecisionEventRecord(
        event_id=event_id,
        decision_id=identity.decision_id,
        event_sequence=sequence,
        event_type="decision.stream.note",
        occurred_at_utc=datetime.now(tz=UTC).isoformat(),
        payload=StreamNotePayload(note=note),
    )


def _checkpoint(identity: DecisionIdentity) -> object:
    lifecycle = initial_decision_lifecycle_state(identity)
    return decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
    )


@pytest.mark.parametrize(
    "store_factory",
    ["memory", "sqlite"],
)
def test_sequential_event_append_assigns_monotonic_sequence(
    store_factory: str,
    tmp_path,
) -> None:
    identity = _identity()
    key = _key(identity)
    store = (
        InMemoryDecisionEventAppendPersistence(payload_codecs=_event_codec_registry())
        if store_factory == "memory"
        else SQLiteDecisionEventAppendPersistence(
            db_path=tmp_path / "events.db",
            payload_codecs=_event_codec_registry(),
        )
    )
    first = store.append(
        key=key,
        event=_event(identity, event_id="evt-1", note="one"),
        expected_last_sequence=0,
    )
    second = store.append(
        key=key,
        event=_event(identity, event_id="evt-2", note="two"),
        expected_last_sequence=1,
    )
    third = store.append(
        key=key,
        event=_event(identity, event_id="evt-3", note="three"),
        expected_last_sequence=2,
    )
    assert (first.event_sequence, second.event_sequence, third.event_sequence) == (1, 2, 3)
    assert store.last_sequence(key=key) == 3


def test_twenty_concurrent_writers_exactly_one_wins(tmp_path) -> None:
    identity = _identity()
    key = _key(identity)
    store = SQLiteDecisionEventAppendPersistence(
        db_path=tmp_path / "race.db",
        payload_codecs=_event_codec_registry(),
    )
    writer_count = 20
    results: list[str] = []

    def attempt(worker_index: int) -> str:
        try:
            store.append(
                key=key,
                event=_event(
                    identity,
                    event_id=f"evt-race-{worker_index}",
                    note=f"worker-{worker_index}",
                ),
                expected_last_sequence=0,
            )
            return "success"
        except StaleDecisionEventAppendError:
            return "stale"

    with ThreadPoolExecutor(max_workers=writer_count) as pool:
        futures = [pool.submit(attempt, index) for index in range(writer_count)]
        for future in as_completed(futures):
            results.append(future.result())

    assert results.count("success") == 1
    assert results.count("stale") == writer_count - 1
    assert store.last_sequence(key=key) == 1


def test_duplicate_event_id_same_payload_is_idempotent(tmp_path) -> None:
    identity = _identity()
    key = _key(identity)
    store = SQLiteDecisionEventAppendPersistence(
        db_path=tmp_path / "dup.db",
        payload_codecs=_event_codec_registry(),
    )
    event = _event(identity, event_id="evt-dup", note="same")
    first = store.append(key=key, event=event, expected_last_sequence=0)
    replay = store.append(key=key, event=event, expected_last_sequence=0)
    assert replay.event_sequence == first.event_sequence == 1
    assert store.last_sequence(key=key) == 1


def test_duplicate_event_id_different_payload_rejected(tmp_path) -> None:
    identity = _identity()
    key = _key(identity)
    store = SQLiteDecisionEventAppendPersistence(
        db_path=tmp_path / "dup-mismatch.db",
        payload_codecs=_event_codec_registry(),
    )
    store.append(
        key=key,
        event=_event(identity, event_id="evt-dup", note="first"),
        expected_last_sequence=0,
    )
    with pytest.raises(DuplicateDecisionEventError):
        store.append(
            key=key,
            event=_event(identity, event_id="evt-dup", note="second"),
            expected_last_sequence=0,
        )


@pytest.mark.parametrize(
    "store_factory",
    ["memory", "sqlite"],
)
def test_snapshot_cas_stale_writer_rejected(store_factory: str, tmp_path) -> None:
    identity = _identity()
    checkpoint = _checkpoint(identity)
    store = (
        InMemoryDecisionCheckpointPersistence()
        if store_factory == "memory"
        else SQLiteDecisionCheckpointPersistence(
            db_path=tmp_path / "checkpoint.db",
            payload_codecs=conformance_artifact_payload_codec_registry(),
        )
    )
    save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=0)
    assert store.materialized_revision(key=_key(identity)) == 1
    save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=1)
    with pytest.raises(StaleDecisionCheckpointWriteError):
        save_decision_checkpoint(store, checkpoint=checkpoint, expected_revision=1)
    assert store.materialized_revision(key=_key(identity)) == 2
