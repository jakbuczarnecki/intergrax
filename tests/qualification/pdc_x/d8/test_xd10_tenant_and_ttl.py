# © Artur Czarnecki. All rights reserved.

"""XD-10 slice owned by B-D8: PDC-D8-R005 (cross-tenant read denied), PDC-D6-R022 (TTL deletion honored).

Re-pointed by PDC-P6-01 to the cross-domain integration suite.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.artifact_ref import ArtifactClass, ArtifactRetentionClass
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied,
    ArtifactNotFound,
    ArtifactReadRequest,
)
from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSweeper

from ._d8_support import (
    T0,
    TENANT_A,
    TENANT_B,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    make_port,
    reader_grants,
    sqlite_store,
)

pytestmark = pytest.mark.unit


def _req(tenant_id: str, artifact_id: str = "art-1") -> ArtifactReadRequest:
    return ArtifactReadRequest(
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
        tenant_id=tenant_id,
        artifact_id=artifact_id,
    )


def _two_tenant_port(tmp_path: Path):
    store = sqlite_store(tmp_path)
    store.put(make_artifact(tenant_id=TENANT_A, data=b"tenant-a secret"))
    store.put(make_artifact(tenant_id=TENANT_B, data=b"tenant-b secret"))
    grants = {**reader_grants(tenant_id=TENANT_A), **reader_grants(tenant_id=TENANT_B)}
    harness = make_governance(granted=grants)
    port, _ = make_port(store, harness=harness)
    return store, port, harness


def test_cross_tenant_read_is_denied_even_for_an_authorized_reader(tmp_path: Path) -> None:
    _, port, harness = _two_tenant_port(tmp_path)
    reader_a = make_context(tenant_id=TENANT_A)
    assert port.read(reader_a, _req(TENANT_A)).content == b"tenant-a secret"
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(reader_a, _req(TENANT_B))
    assert exc.value.reason_code == "tenant_mismatch"
    denied = [
        e
        for e in harness.sink.events
        if e.capability == "artifact.read.outcome" and e.decision is ToolAuthorizationDecisionState.DENY
    ]
    assert denied and denied[-1].tenant_id == TENANT_A  # recorded under the reader's tenant


def test_cross_tenant_denial_does_not_reach_the_store(tmp_path: Path) -> None:
    store, port, _ = _two_tenant_port(tmp_path)
    calls: list[tuple[str, str]] = []
    original = store.get

    def spy(tenant_id, artifact_id, *args, **kwargs):
        calls.append((tenant_id, artifact_id))
        return original(tenant_id, artifact_id, *args, **kwargs)

    store.get = spy  # type: ignore[method-assign]
    with pytest.raises(ArtifactAccessDenied):
        port.read(make_context(tenant_id=TENANT_A), _req(TENANT_B))
    assert calls == []


def test_a_foreign_artifact_id_is_not_found_under_the_readers_own_tenant(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    store.put(make_artifact(tenant_id=TENANT_B, artifact_id="only-b"))
    port, _ = make_port(store, harness=make_governance(granted=reader_grants()))
    with pytest.raises(ArtifactNotFound):
        port.read(make_context(tenant_id=TENANT_A), _req(TENANT_A, "only-b"))


def test_ttl_deletion_is_honored_end_to_end_on_a_durable_store(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    store.put(
        make_artifact(
            artifact_id="ephemeral",
            retention_class=ArtifactRetentionClass.EPHEMERAL,
            expires_at_utc=T0 + timedelta(hours=1),
        )
    )
    store.put(
        make_artifact(
            artifact_id="evidence",
            retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
            expires_at_utc=T0 + timedelta(hours=1),
        )
    )
    harness = make_governance()
    sweeper = ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder)
    ctx = make_context(agent_id="artifact_retention_sweep")

    clock.advance(minutes=59)
    assert sweeper.sweep(TENANT_A, context=ctx).deleted_ids == ()
    clock.advance(minutes=2)
    del store  # restart between expiry and sweep: TTL state is durable
    reopened = sqlite_store(tmp_path)
    sweeper = ArtifactRetentionSweeper(reopened, clock=clock, audit=harness.recorder)
    report = sweeper.sweep(TENANT_A, context=ctx)
    assert report.deleted_ids == ("ephemeral",) and report.protected_ids == ("evidence",)
    with pytest.raises(KeyError):
        reopened.get(TENANT_A, "ephemeral")
    assert reopened.get(TENANT_A, "evidence")


def test_sweep_of_one_tenant_never_touches_another_tenant(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    for tenant in (TENANT_A, TENANT_B):
        store.put(make_artifact(tenant_id=tenant, expires_at_utc=T0 + timedelta(minutes=1)))
    harness = make_governance()
    sweeper = ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder)
    clock.advance(hours=1)
    ctx = make_context(agent_id="artifact_retention_sweep")
    assert sweeper.sweep(TENANT_A, context=ctx).deleted_ids == ("art-1",)
    assert store.get(TENANT_B, "art-1")


def test_workspace_scope_is_enforced_when_the_request_names_a_workspace(tmp_path: Path) -> None:
    _, port, harness = _two_tenant_port(tmp_path)  # artifacts are scoped to workspace "ws-1"
    reader = make_context(tenant_id=TENANT_A)
    ok = ArtifactReadRequest(
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
        tenant_id=TENANT_A,
        artifact_id="art-1",
        workspace_id="ws-1",
    )
    assert port.read(reader, ok).content == b"tenant-a secret"
    wrong = ArtifactReadRequest(
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
        tenant_id=TENANT_A,
        artifact_id="art-1",
        workspace_id="ws-2",
    )
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(reader, wrong)
    assert exc.value.reason_code == "workspace_mismatch"
    assert any(e.decision is ToolAuthorizationDecisionState.DENY for e in harness.sink.events)
