# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-H (F-D8-06, F-D8-04): expiry and workspace are enforced at read and on lifecycle operations."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from intergrax.contracts.artifact_ref import ArtifactRetentionClass
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied,
    ArtifactNotFound,
)
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore
from tests.qualification.pdc_x.d8._d8_support import (
    T0,
    TENANT_A,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    reader_grants,
    sqlite_store,
)
from tests.qualification.pdc_x.d8_repair._support import make_clocked_port, request

pytestmark = pytest.mark.unit


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ArtifactStore:
    return InMemoryArtifactStore() if request.param == "memory" else sqlite_store(tmp_path)


def _checks(harness: Any) -> list[str]:
    return [
        result.policy_id
        for event in harness.sink.events
        if event.capability.endswith(".outcome")
        for result in event.policy_results
    ]


# ------------------------------------------------------------------ F-D8-06 expiry at read


def test_an_expired_unprotected_artifact_is_not_found_and_the_denial_is_audited(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(expires_at_utc=T0 + timedelta(minutes=5)))
    port, harness = make_clocked_port(store, clock=clock)
    clock.advance(minutes=10)
    with pytest.raises(ArtifactNotFound):
        port.read(make_context(), request())
    assert "artifact.port.not_found" in _checks(harness)


def test_an_artifact_before_expiry_is_readable_and_the_boundary_counts_as_expired(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(expires_at_utc=T0 + timedelta(minutes=5)))
    port, _ = make_clocked_port(store, clock=clock)
    clock.advance(minutes=5, seconds=-1)
    assert port.read(make_context(), request()).content == b"payload"
    clock.advance(seconds=1)  # now == expires_at_utc, consistent with list_expired (<=)
    with pytest.raises(ArtifactNotFound):
        port.read(make_context(), request())


def test_legal_hold_and_audit_evidence_stay_readable_past_expiry(store: ArtifactStore) -> None:
    clock = MutableClock()
    expiry = T0 + timedelta(minutes=5)
    store.put(make_artifact(artifact_id="held", expires_at_utc=expiry))
    store.place_legal_hold(TENANT_A, "held", placed_by="counsel", reason="case")
    store.put(
        make_artifact(
            artifact_id="evidence",
            retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
            expires_at_utc=expiry,
        )
    )
    port, _ = make_clocked_port(store, clock=clock)
    clock.advance(hours=1)
    assert port.read(make_context(), request("held")).content == b"payload"
    assert port.read(make_context(), request("evidence")).content == b"payload"


def test_a_specific_expired_version_is_not_found_while_a_later_one_is_readable(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(data=b"v1", expires_at_utc=T0 + timedelta(minutes=5)))
    store.put(make_artifact(data=b"v2"))  # never expires
    port, _ = make_clocked_port(store, clock=clock)
    clock.advance(hours=1)
    with pytest.raises(ArtifactNotFound):
        port.read(make_context(), request(version=1))
    assert port.read(make_context(), request(version=2)).content == b"v2"
    assert port.read(make_context(), request()).content == b"v2"


def test_a_naive_expiry_is_treated_as_utc() -> None:
    store = InMemoryArtifactStore()
    clock = MutableClock()
    naive = datetime(2026, 11, 13, 12, 5)  # the instant T0 + 5 minutes, without tzinfo
    store.put(make_artifact(expires_at_utc=naive))
    port, _ = make_clocked_port(store, clock=clock)
    assert port.read(make_context(), request()).content == b"payload"
    clock.advance(minutes=10)
    with pytest.raises(ArtifactNotFound):
        port.read(make_context(), request())


# ------------------------------------------------------------------ F-D8-04 workspace


def test_a_request_without_the_stored_workspace_is_denied_and_audited(store: ArtifactStore) -> None:
    store.put(make_artifact())  # stored in ws-1
    port, harness = make_clocked_port(store)
    with pytest.raises(ArtifactAccessDenied) as missing:
        port.read(make_context(), request(workspace_id=None))
    assert missing.value.reason_code == "workspace_mismatch"
    with pytest.raises(ArtifactAccessDenied) as wrong:
        port.read(make_context(), request(workspace_id="ws-other"))
    assert wrong.value.reason_code == "workspace_mismatch"
    assert _checks(harness).count("artifact.port.workspace_mismatch") == 2
    assert port.read(make_context(), request(workspace_id="ws-1")).content == b"payload"


def test_an_artifact_stored_without_a_workspace_reads_with_or_without_one(store: ArtifactStore) -> None:
    from dataclasses import replace

    store.put(replace(make_artifact(), workspace_id=None))
    port, _ = make_clocked_port(store)
    assert port.read(make_context(), request(workspace_id=None)).content == b"payload"


@pytest.mark.parametrize("operation", ["place_legal_hold", "release_legal_hold", "delete_artifact"])
def test_lifecycle_operations_need_the_stored_workspace(store: ArtifactStore, operation: str) -> None:
    store.put(make_artifact())
    harness = make_governance(granted=reader_grants())
    port, _ = make_clocked_port(store, harness=harness)

    def call(workspace_id: str | None) -> None:
        method = getattr(port, operation)
        if operation == "place_legal_hold":
            method(make_context(), request(workspace_id=workspace_id), reason="case")
        else:
            method(make_context(), request(workspace_id=workspace_id))

    for workspace_id in (None, "ws-other"):
        with pytest.raises(ArtifactAccessDenied) as denied:
            call(workspace_id)
        assert denied.value.reason_code == "workspace_mismatch"
    assert store.get(TENANT_A, "art-1").artifact_id == "art-1"
    assert not store.is_under_legal_hold(TENANT_A, "art-1")
    call("ws-1")  # the matching workspace works
    if operation == "delete_artifact":
        with pytest.raises(KeyError):
            store.get(TENANT_A, "art-1")
