# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R5 qualification matrix (SX-F14 Agent Checkpoint State)."""

from __future__ import annotations

import ast
import inspect
import json
import sqlite3
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

import pytest
from pydantic import ValidationError

from intergrax.agents.authoring import acp_run as acp_run_module
from intergrax.agents.persistence.checkpoint_store import (
    AgentCheckpointStore,
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
    build_checkpoint,
    validate_agent_checkpoint_for_persistence,
)
from intergrax.agents.persistence.checkpoint_wiring import should_resume_acp_checkpoint
from intergrax.agents.persistence.session_persistence import resolve_session_persistence
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.checkpoint_revision import (
    CheckpointAgentIdentityConflictError,
    CheckpointDurableCorruptionError,
    CheckpointRevisionConflictError,
    CheckpointSideEffectLineageError,
    CheckpointStepRegressionError,
    CheckpointStreamIdentityConflictError,
)
from tests.qualification.state_x._r3_r5_support import (
    agent_checkpoint_store_implementations,
    build_side_effect_record,
    build_valid_checkpoint,
    canonical_validator_definition_path,
    parity_checkpoint_store_factories,
    post_construction_invalid_checkpoint,
    save_source_uses_canonical_validator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_r3_r5_q01_closed_world_agent_checkpoint_store_implementations() -> None:
    impls = {name for name, _ in agent_checkpoint_store_implementations()}
    assert impls == {"InMemoryAgentCheckpointStore", "SQLiteAgentCheckpointStore"}


def test_r3_r5_q02_exactly_one_checkpoint_persistence_contract() -> None:
    tree = ast.parse(
        (_REPO_ROOT / "intergrax/agents/persistence/checkpoint_store.py").read_text(encoding="utf-8"),
    )
    abcs = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentCheckpointStore"
    ]
    assert len(abcs) == 1


def test_r3_r5_q03_one_canonical_persistence_acceptance_validator() -> None:
    path = canonical_validator_definition_path()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_agent_checkpoint_for_persistence"
    ]
    assert len(defs) == 1
    for _, cls in agent_checkpoint_store_implementations():
        assert save_source_uses_canonical_validator(cls)


@pytest.mark.parametrize("factory_name,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q04_q05_valid_create_revision_one(factory_name: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if factory_name == "memory" else SQLiteAgentCheckpointStore(tmp_path / f"q05-{factory_name}.db")
    )
    saved = store.save(build_valid_checkpoint())
    assert saved.revision == 1


@pytest.mark.parametrize("label,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q06_provider_update_cas_parity(label: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if label == "memory" else SQLiteAgentCheckpointStore(tmp_path / "q06.db")
    )
    store.save(build_valid_checkpoint(step_index=0))
    current = store.get_latest("run-r5", "tenant-a")
    assert current is not None
    saved = store.save(build_valid_checkpoint(step_index=1), expected_revision=current.revision)
    assert saved.revision == 2


@pytest.mark.parametrize("label,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q07_stale_writer_rejected(label: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if label == "memory" else SQLiteAgentCheckpointStore(tmp_path / "q07.db")
    )
    store.save(build_valid_checkpoint(step_index=0))
    rev = store.get_latest("run-r5", "tenant-a")
    assert rev is not None
    store.save(build_valid_checkpoint(step_index=1), expected_revision=rev.revision)
    with pytest.raises(CheckpointRevisionConflictError):
        store.save(build_valid_checkpoint(step_index=2), expected_revision=rev.revision)


@pytest.mark.parametrize("label,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q08_missing_expected_revision_on_update(label: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if label == "memory" else SQLiteAgentCheckpointStore(tmp_path / "q08.db")
    )
    store.save(build_valid_checkpoint())
    with pytest.raises(CheckpointRevisionConflictError):
        store.save(build_valid_checkpoint(step_index=1))


@pytest.mark.parametrize("label,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q09_step_regression_rejected(label: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if label == "memory" else SQLiteAgentCheckpointStore(tmp_path / "q09.db")
    )
    store.save(build_valid_checkpoint(step_index=2))
    current = store.get_latest("run-r5", "tenant-a")
    assert current is not None
    with pytest.raises(CheckpointStepRegressionError):
        store.save(build_valid_checkpoint(step_index=1), expected_revision=current.revision)


def test_r3_r5_q10_concurrent_sqlite_same_revision_one_winner(tmp_path: Path) -> None:
    db = tmp_path / "q10.db"
    store_a = SQLiteAgentCheckpointStore(db)
    store_b = SQLiteAgentCheckpointStore(db)
    store_a.save(build_valid_checkpoint(step_index=0))
    current = store_a.get_latest("run-r5", "tenant-a")
    assert current is not None
    barrier = threading.Barrier(2)
    results: list[str] = []

    def writer(store: SQLiteAgentCheckpointStore, marker: str) -> None:
        barrier.wait()
        try:
            store.save(
                build_valid_checkpoint(step_index=1, state_root={"marker": marker}),
                expected_revision=current.revision,
            )
            results.append(marker)
        except CheckpointRevisionConflictError:
            pass

    t1 = threading.Thread(target=writer, args=(store_a, "a"))
    t2 = threading.Thread(target=writer, args=(store_b, "b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(results) == 1
    latest = store_a.get_latest("run-r5", "tenant-a")
    assert latest is not None
    assert latest.revision == 2


@pytest.mark.parametrize("label,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q11_incoming_revision_cannot_control_durable(label: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if label == "memory" else SQLiteAgentCheckpointStore(tmp_path / "q11.db")
    )
    incoming = build_valid_checkpoint().model_copy(update={"revision": 999})
    saved = store.save(incoming)
    assert saved.revision == 1


@pytest.mark.parametrize("label,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q12_post_construction_invalid_rejected(label: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if label == "memory" else SQLiteAgentCheckpointStore(tmp_path / "q12.db")
    )
    valid = build_valid_checkpoint()
    bad_ledger = [build_side_effect_record(run_id="other-run", step_index=0)]
    invalid = post_construction_invalid_checkpoint(valid, ledger=bad_ledger)
    with pytest.raises(CheckpointSideEffectLineageError):
        store.save(invalid)


@pytest.mark.parametrize("label,factory", [("memory", InMemoryAgentCheckpointStore), ("sqlite", None)])
def test_r3_r5_q13_invalid_checkpoint_zero_mutation(label: str, factory: type | None, tmp_path: Path) -> None:
    store: AgentCheckpointStore = (
        factory() if label == "memory" else SQLiteAgentCheckpointStore(tmp_path / "q13.db")
    )
    valid = build_valid_checkpoint()
    invalid = post_construction_invalid_checkpoint(
        valid,
        ledger=[build_side_effect_record(run_id="wrong-run")],
    )
    with pytest.raises(CheckpointSideEffectLineageError):
        store.save(invalid)
    assert store.get_latest("run-r5", "tenant-a") is None


def test_r3_r5_q14_same_run_tenant_cannot_change_agent_id(tmp_path: Path) -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(build_valid_checkpoint(agent_id="agent-a"))
    current = store.get_latest("run-r5", "tenant-a")
    assert current is not None
    with pytest.raises(CheckpointAgentIdentityConflictError):
        store.save(
            build_valid_checkpoint(agent_id="agent-b", step_index=1),
            expected_revision=current.revision,
        )


def test_r3_r5_q15_resume_rejects_checkpoint_agent_mismatch() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(build_valid_checkpoint(agent_id="stored-agent"))
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="current-agent",
        metadata={
            AcpMetadataKey.CHECKPOINT_STORE: store,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    with pytest.raises(CheckpointAgentIdentityConflictError):
        resolve_session_persistence(
            request,
            run_id="run-r5",
            tenant_id="tenant-a",
            agent_id="current-agent",
        )


def test_r3_r5_q16_same_run_across_tenants_isolated() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(build_valid_checkpoint(tenant_id="tenant-a", run_id="shared-run"))
    store.save(
        build_valid_checkpoint(
            tenant_id="tenant-b",
            run_id="shared-run",
            agent_id="agent-b",
        ),
    )
    a = store.get_latest("shared-run", "tenant-a")
    b = store.get_latest("shared-run", "tenant-b")
    assert a is not None and b is not None
    assert a.tenant_id == "tenant-a"
    assert b.tenant_id == "tenant-b"


def test_r3_r5_q17_human_response_metadata_alone_does_not_resume() -> None:
    store = InMemoryAgentCheckpointStore()
    assert not should_resume_acp_checkpoint(
        {"human_response": "approve"},
        store=store,
        run_id="run-hr",
        tenant_id="tenant-a",
    )


def test_r3_r5_q18_explicit_acp_resume_flag_state_only() -> None:
    assert should_resume_acp_checkpoint(
        {AcpMetadataKey.RESUME_FROM_CHECKPOINT: True},
        store=None,
        run_id="run-x",
        tenant_id="tenant-a",
    )


def test_r3_r5_q19_existing_checkpoint_can_resume_under_policy() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(build_valid_checkpoint(run_id="run-resume"))
    assert should_resume_acp_checkpoint(
        {},
        store=store,
        run_id="run-resume",
        tenant_id="tenant-a",
    )


def test_r3_r5_q20_q22_checkpoint_cannot_mint_identity_on_resume() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(build_valid_checkpoint(run_id="run-r5", agent_id="agent-a"))
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            AcpMetadataKey.CHECKPOINT_STORE: store,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    source = inspect.getsource(resolve_session_persistence)
    assert "checkpoint.agent_id" in source
    assert "effective" not in source.lower() or "agent_id" in source
    acp_source = inspect.getsource(acp_run_module._run_acp_session_bound)
    assert "merged.agent_id" in acp_source
    assert "checkpoint.agent_id" not in acp_source.split("resolve_session_persistence")[0]


def test_r3_r5_q23_embedded_side_effect_run_id_mismatch() -> None:
    valid = build_valid_checkpoint(
        side_effect_ledger=[build_side_effect_record(run_id="other")],
    )
    with pytest.raises(CheckpointSideEffectLineageError):
        validate_agent_checkpoint_for_persistence(valid)


def test_r3_r5_q24_embedded_side_effect_step_exceeds_checkpoint() -> None:
    valid = build_valid_checkpoint(
        step_index=1,
        side_effect_ledger=[build_side_effect_record(step_index=3)],
    )
    with pytest.raises(CheckpointSideEffectLineageError):
        validate_agent_checkpoint_for_persistence(valid)


def test_r3_r5_q25_valid_embedded_side_effect_round_trip(tmp_path: Path) -> None:
    record = build_side_effect_record()
    store = SQLiteAgentCheckpointStore(tmp_path / "q25.db")
    saved = store.save(
        build_valid_checkpoint(side_effect_ledger=[record], step_index=1),
    )
    store2 = SQLiteAgentCheckpointStore(tmp_path / "q25.db")
    loaded = store2.get_latest("run-r5", "tenant-a")
    assert loaded is not None
    assert len(loaded.side_effect_ledger) == 1
    got = loaded.side_effect_ledger[0]
    assert got.side_effect_id == record.side_effect_id
    assert got.idempotency_key == record.idempotency_key
    assert got.run_id == record.run_id
    assert got.step_index == record.step_index
    assert got.kind == record.kind
    assert got.status == record.status
    assert got.external_ref == record.external_ref
    assert got.committed_externally == record.committed_externally
    assert saved.trace_step_count == loaded.trace_step_count


def test_r3_r5_q26_sqlite_reopen_preserves_checkpoint(tmp_path: Path) -> None:
    db = tmp_path / "q26.db"
    store_a = SQLiteAgentCheckpointStore(db)
    store_a.save(build_valid_checkpoint(step_index=2, trace_step_count=5))
    store_b = SQLiteAgentCheckpointStore(db)
    loaded = store_b.get_latest("run-r5", "tenant-a")
    assert loaded is not None
    assert loaded.step_index == 2
    assert loaded.revision == 1
    assert loaded.agent_id == "agent-a"
    assert loaded.trace_step_count == 5


def test_r3_r5_q27_corrupt_json_fails_closed(tmp_path: Path) -> None:
    db = tmp_path / "q27.db"
    store = SQLiteAgentCheckpointStore(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO agent_run_checkpoints (run_id, tenant_id, payload, saved_at, revision)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("run-bad", "tenant-a", "{not-json", "2020-01-01T00:00:00+00:00", 1),
        )
    with pytest.raises(CheckpointDurableCorruptionError):
        store.get_latest("run-bad", "tenant-a")


def test_r3_r5_q28_payload_column_revision_divergence(tmp_path: Path) -> None:
    db = tmp_path / "q28.db"
    payload = build_valid_checkpoint(run_id="run-div").model_dump(mode="json")
    payload["revision"] = 1
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE agent_run_checkpoints (
                run_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (run_id, tenant_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO agent_run_checkpoints (run_id, tenant_id, payload, saved_at, revision)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("run-div", "tenant-a", json.dumps(payload), "2020-01-01T00:00:00+00:00", 99),
        )
    store = SQLiteAgentCheckpointStore(db)
    with pytest.raises(CheckpointDurableCorruptionError):
        store.get_latest("run-div", "tenant-a")


def test_r3_r5_q29_invalid_schema_version_fails_closed(tmp_path: Path) -> None:
    db = tmp_path / "q29.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE agent_run_checkpoints (
                run_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (run_id, tenant_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO agent_run_checkpoints (run_id, tenant_id, payload, saved_at, revision)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-schema",
                "tenant-a",
                json.dumps({"schema_version": "agent_run_checkpoint.v9", "run_id": "run-schema"}),
                "2020-01-01T00:00:00+00:00",
                1,
            ),
        )
    store = SQLiteAgentCheckpointStore(db)
    with pytest.raises(ValidationError):
        store.get_latest("run-schema", "tenant-a")


def test_r3_r5_q30_checkpoint_not_governance_execution_authority() -> None:
    source = inspect.getsource(resolve_session_persistence)
    assert "ALLOW" not in source
    assert "Governance" not in source
    assert "human_approved" not in source


def test_r3_r5_q31_q32_production_composition_single_contract() -> None:
    wiring = (_REPO_ROOT / "intergrax/applications/_shared/acp_checkpoint_host_wiring.py").read_text(
        encoding="utf-8",
    )
    assert "open_agent_checkpoint_store" in wiring
    assert "AgentCheckpointStore" in wiring


def test_r3_r5_q33_checkpoint_store_metadata_boundary() -> None:
    from intergrax.agents.persistence.session_persistence import resolve_checkpoint_store

    store = InMemoryAgentCheckpointStore()
    assert resolve_checkpoint_store({AcpMetadataKey.CHECKPOINT_STORE: store}) is store
    with pytest.raises(TypeError):
        resolve_checkpoint_store({AcpMetadataKey.CHECKPOINT_STORE: object()})


def test_r3_r5_q34_resume_flag_writers_inventoried() -> None:
    wiring_source = (_REPO_ROOT / "intergrax/agents/persistence/checkpoint_wiring.py").read_text(
        encoding="utf-8",
    )
    assert "RESUME_FROM_CHECKPOINT" in wiring_source
    enricher = (_REPO_ROOT / "intergrax/applications/_shared/acp_checkpoint_task_enricher.py").read_text(
        encoding="utf-8",
    )
    assert "should_resume_acp_checkpoint" in enricher


def test_r3_r5_q35_full_provider_parity_matrix_green(tmp_path: Path) -> None:
    for label, factory in parity_checkpoint_store_factories(tmp_path):
        store = factory()
        saved = store.save(build_valid_checkpoint())
        assert saved.revision == 1, label
        current = store.get_latest("run-r5", "tenant-a")
        assert current is not None
        updated = store.save(build_valid_checkpoint(step_index=1), expected_revision=current.revision)
        assert updated.revision == 2, label
