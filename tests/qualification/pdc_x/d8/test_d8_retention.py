# © Artur Czarnecki. All rights reserved.

"""PDC-D8-R008 (TTL enforced), R009 (evidence delete refused, legal hold respected)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.artifact_ref import ArtifactRetentionClass
from intergrax.contracts.task_artifacts import ArtifactRetentionPolicy
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.nexus.artifacts.governed_access import ArtifactAccessContext
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.retention import (
    ARTIFACT_RETENTION_SWEEP_TASK_NAME,
    ArtifactRetentionSchedule,
    ArtifactRetentionSweeper,
    ArtifactRetentionSweepOutput,
    register_retention_sweep,
    resolve_expiry,
)
from intergrax.runtime.nexus.artifacts.store_base import ArtifactDeletionRefused, ArtifactStore

from ._d8_support import T0, TENANT_A, MutableClock, make_artifact, make_governance, sqlite_store

pytestmark = pytest.mark.unit


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ArtifactStore:
    return InMemoryArtifactStore() if request.param == "memory" else sqlite_store(tmp_path)


def _sweeper(store: ArtifactStore, clock: MutableClock):
    harness = make_governance()
    return ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder), harness


def _system_context() -> ArtifactAccessContext:
    from ._d8_support import make_context

    return make_context(agent_id="artifact_retention_sweep")


# ------------------------------------------------------------------ R008


def test_expiry_is_resolved_from_class_schedule_and_from_the_retention_policy() -> None:
    schedule = ArtifactRetentionSchedule.default()
    assert resolve_expiry(T0, retention_class=ArtifactRetentionClass.EPHEMERAL, schedule=schedule) == (
        T0 + timedelta(hours=1)
    )
    assert resolve_expiry(T0, retention_class=ArtifactRetentionClass.DEFAULT, schedule=schedule) is None
    # AUDIT_EVIDENCE never gets an expiry, even if a policy asks for one.
    assert (
        resolve_expiry(
            T0,
            retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
            schedule=schedule,
            policy=ArtifactRetentionPolicy(retain_hours=1),
        )
        is None
    )
    # The existing ArtifactRetentionPolicy now has a consumer.
    assert resolve_expiry(
        T0, retention_class=ArtifactRetentionClass.EXTENDED, policy=ArtifactRetentionPolicy(retain_hours=48)
    ) == T0 + timedelta(hours=48)


def test_ttl_expiry_is_enforced_by_the_sweep(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(artifact_id="short", expires_at_utc=T0 + timedelta(minutes=5)))
    store.put(make_artifact(artifact_id="long", expires_at_utc=T0 + timedelta(days=30)))
    store.put(make_artifact(artifact_id="forever"))
    sweeper, harness = _sweeper(store, clock)

    before = sweeper.sweep(TENANT_A, context=_system_context())
    assert before.deleted_ids == ()

    clock.advance(minutes=10)
    report = sweeper.sweep(TENANT_A, context=_system_context())
    assert report.deleted_ids == ("short",)
    with pytest.raises(KeyError):
        store.get(TENANT_A, "short")
    assert store.get(TENANT_A, "long") and store.get(TENANT_A, "forever")

    clock.advance(days=60)
    assert sweeper.sweep(TENANT_A, context=_system_context()).deleted_ids == ("long",)
    assert store.get(TENANT_A, "forever")  # no expiry -> retained


def test_sweep_removes_all_versions_of_an_expired_artifact(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(data=b"v1", expires_at_utc=T0 + timedelta(minutes=1)))
    store.put(make_artifact(data=b"v2", expires_at_utc=T0 + timedelta(minutes=1)))
    sweeper, _ = _sweeper(store, clock)
    clock.advance(minutes=2)
    sweeper.sweep(TENANT_A, context=_system_context())
    assert list(store.list_versions(TENANT_A, "art-1")) == []


def test_ttl_deletion_is_recorded_in_evidence(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(expires_at_utc=T0 + timedelta(minutes=1)))
    sweeper, harness = _sweeper(store, clock)
    clock.advance(minutes=5)
    sweeper.sweep(TENANT_A, context=_system_context())
    events = [e for e in harness.sink.events if e.capability == "artifact.delete.outcome"]
    assert len(events) == 1
    assert events[0].decision is ToolAuthorizationDecisionState.ALLOW
    assert "ttl_expired" in events[0].policy_results[0].reason
    assert events[0].tool_id == "artifact:art-1"


# ------------------------------------------------------------------ R009


def test_audit_evidence_is_never_deleted_by_ttl_even_with_an_expiry_set(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(
        make_artifact(
            artifact_id="evidence",
            retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
            expires_at_utc=T0 + timedelta(minutes=1),
        )
    )
    sweeper, harness = _sweeper(store, clock)
    clock.advance(days=3650)
    report = sweeper.sweep(TENANT_A, context=_system_context())
    assert report.deleted_ids == () and report.protected_ids == ("evidence",)
    assert store.get(TENANT_A, "evidence")
    refusals = [
        e
        for e in harness.sink.events
        if e.capability == "artifact.delete.outcome" and e.decision is ToolAuthorizationDecisionState.DENY
    ]
    assert refusals and "audit_evidence" in refusals[0].policy_results[0].reason


def test_direct_deletion_of_evidence_is_refused_by_the_store(store: ArtifactStore) -> None:
    store.put(make_artifact(artifact_id="evidence", retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE))
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_artifact(TENANT_A, "evidence")
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_for_run(TENANT_A, "run-1")
    assert store.get(TENANT_A, "evidence")


def test_legal_hold_blocks_ttl_and_explicit_deletion_until_released(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(artifact_id="held", expires_at_utc=T0 + timedelta(minutes=1)))
    store.place_legal_hold(TENANT_A, "held", placed_by="counsel", reason="case 42")
    sweeper, _ = _sweeper(store, clock)
    clock.advance(days=1)
    report = sweeper.sweep(TENANT_A, context=_system_context())
    assert report.deleted_ids == () and report.protected_ids == ("held",)
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_artifact(TENANT_A, "held")
    assert store.is_under_legal_hold(TENANT_A, "held")

    store.release_legal_hold(TENANT_A, "held")
    assert sweeper.sweep(TENANT_A, context=_system_context()).deleted_ids == ("held",)


def test_delete_for_run_is_atomic_when_a_protected_artifact_is_present(store: ArtifactStore) -> None:
    store.put(make_artifact(artifact_id="plain"))
    store.put(make_artifact(artifact_id="evidence", retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE))
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_for_run(TENANT_A, "run-1")
    assert store.get(TENANT_A, "plain")  # nothing was deleted


def test_deleting_an_unprotected_artifact_works(store: ArtifactStore) -> None:
    store.put(make_artifact(artifact_id="plain"))
    store.delete_artifact(TENANT_A, "plain")
    with pytest.raises(KeyError):
        store.get(TENANT_A, "plain")


# ------------------------------------------------------------------ background admission


def test_sweep_runs_as_canonical_background_work(tmp_path: Path) -> None:
    """The sweep is a handler on the existing TaskExecutionRegistry, run by execute_logical_task."""
    from intergrax.runtime.background_execution.bootstrap import bootstrap_background_execution
    from intergrax.runtime.background_execution.identity_persistence import (
        KvBackgroundExecutionIdentityPersistence,
    )
    from intergrax.runtime.background_execution.transport_ref import (
        BackgroundTransportExecutionRef,
    )

    from ._kv import DictKV

    store = sqlite_store(tmp_path)
    clock = MutableClock()
    store.put(make_artifact(artifact_id="old", expires_at_utc=T0 + timedelta(minutes=1)))
    store.put(
        make_artifact(
            artifact_id="evidence",
            retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
            expires_at_utc=T0 + timedelta(minutes=1),
        )
    )
    harness = make_governance()
    sweeper = ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder)

    registry = TaskExecutionRegistry()
    register_retention_sweep(registry, sweeper)
    assert registry.get_handler(ARTIFACT_RETENTION_SWEEP_TASK_NAME) is not None
    with pytest.raises(ValueError):
        register_retention_sweep(registry, sweeper)  # one canonical handler, no second registry

    identity = bootstrap_background_execution(
        transport_ref=BackgroundTransportExecutionRef(tenant_id=TENANT_A, provider="test", transport_task_id="sweep-1"),
        identity_persistence=KvBackgroundExecutionIdentityPersistence(DictKV()),
    )
    clock.advance(hours=1)
    result = execute_logical_task(
        registry=registry,
        logical_task_name=ARTIFACT_RETENTION_SWEEP_TASK_NAME,
        tenant_id=TENANT_A,
        run_id=str(identity.run_id),
        payload=json.dumps({"limit": 100}).encode(),
        idempotency_key=None,
        idempotency_store=None,
        execution_identity=identity,
    )
    assert result.success
    assert isinstance(result.output, ArtifactRetentionSweepOutput)
    assert result.output.deleted == 1 and result.output.protected_retained == 1
    with pytest.raises(KeyError):
        store.get(TENANT_A, "old")
    assert store.get(TENANT_A, "evidence")
    # the background run identity is what the evidence names
    event = [e for e in harness.sink.events if e.capability == "artifact.delete.outcome"][0]
    assert str(event.run_id) == str(identity.run_id)
    assert event.agent_id == "artifact_retention_sweep"


def test_sweep_module_adds_no_scheduler_or_registry() -> None:
    import ast

    text = (Path(__file__).resolve().parents[4] / "intergrax/runtime/nexus/artifacts/retention.py").read_text(
        encoding="utf-8"
    )
    classes = [n.name for n in ast.walk(ast.parse(text)) if isinstance(n, ast.ClassDef)]
    assert not [c for c in classes if "Scheduler" in c or "Registry" in c]
