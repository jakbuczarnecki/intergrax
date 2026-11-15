# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-H (F-D8-05, F-D8-09): the retention sweep never deletes a newer version and is never unaudited."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from intergrax.contracts.artifact_ref import ArtifactRetentionClass
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.retention import (
    ArtifactRetentionSweeper,
    ArtifactSweepAuditRequired,
    make_retention_sweep_handler,
)
from intergrax.runtime.nexus.artifacts.store_base import (
    ArtifactDeletionRefused,
    ArtifactStore,
    ArtifactVersionChanged,
)
from tests.qualification.pdc_x.d8._d8_support import (
    T0,
    TENANT_A,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    sqlite_store,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ArtifactStore:
    return InMemoryArtifactStore() if request.param == "memory" else sqlite_store(tmp_path)


def _sweeper(store: Any, clock: MutableClock):
    harness = make_governance()
    return ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder), harness


def _ctx():
    return make_context(agent_id="artifact_retention_sweep")


class _RacingStore:
    """Delegates to a real store; a new version is written right after the expiry selection."""

    def __init__(self, inner: Any, newer: Any) -> None:
        self._inner = inner
        self._newer = newer
        self.raced = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def list_expired(self, tenant_id: str, now: Any, *, limit: int = 1000) -> list[Any]:
        selected = self._inner.list_expired(tenant_id, now, limit=limit)
        if not self.raced:
            self.raced = True
            self._inner.put(self._newer)
        return selected


# ------------------------------------------------------------------ conditional delete


def test_conditional_delete_refuses_when_a_newer_version_exists(store: ArtifactStore) -> None:
    store.put(make_artifact(data=b"v1"))
    store.put(make_artifact(data=b"v2"))
    with pytest.raises(ArtifactVersionChanged):
        store.delete_artifact(TENANT_A, "art-1", expected_version=1)
    assert [a.version for a in store.list_versions(TENANT_A, "art-1")] == [1, 2]


def test_conditional_delete_with_the_observed_version_deletes_everything(store: ArtifactStore) -> None:
    store.put(make_artifact(data=b"v1"))
    store.put(make_artifact(data=b"v2"))
    store.delete_artifact(TENANT_A, "art-1", expected_version=2)
    with pytest.raises(KeyError):
        store.get(TENANT_A, "art-1")


def test_unconditional_delete_is_unchanged_and_still_protects(store: ArtifactStore) -> None:
    store.put(make_artifact(artifact_id="plain", data=b"v1"))
    store.put(make_artifact(artifact_id="plain", data=b"v2"))
    store.delete_artifact(TENANT_A, "plain")
    assert list(store.list_versions(TENANT_A, "plain")) == []
    store.put(make_artifact(artifact_id="ev", retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE))
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_artifact(TENANT_A, "ev", expected_version=1)
    store.put(make_artifact(artifact_id="held"))
    store.place_legal_hold(TENANT_A, "held", placed_by="counsel", reason="case")
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_artifact(TENANT_A, "held", expected_version=1)


def test_conditional_delete_of_an_unknown_artifact_is_a_key_error(store: ArtifactStore) -> None:
    with pytest.raises(KeyError):
        store.delete_artifact(TENANT_A, "nope", expected_version=1)


# ------------------------------------------------------------------ the sweep race


def test_sweep_does_not_destroy_a_version_written_after_selection(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(artifact_id="race", data=b"v1", expires_at_utc=T0 + timedelta(minutes=1)))
    racing = _RacingStore(store, make_artifact(artifact_id="race", data=b"v2"))  # v2 never expires
    clock.advance(hours=1)
    sweeper, _ = _sweeper(racing, clock)
    report = sweeper.sweep(TENANT_A, context=_ctx())
    assert report.deleted_ids == ()
    assert report.changed_ids == ("race",)
    assert [a.version for a in store.list_versions(TENANT_A, "race")] == [1, 2]


def test_a_stale_selection_is_picked_up_by_the_next_sweep_when_the_newest_is_expired(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(artifact_id="race", data=b"v1", expires_at_utc=T0 + timedelta(minutes=1)))
    racing = _RacingStore(
        store, make_artifact(artifact_id="race", data=b"v2", expires_at_utc=T0 + timedelta(minutes=2))
    )
    clock.advance(hours=1)
    sweeper, _ = _sweeper(racing, clock)
    assert sweeper.sweep(TENANT_A, context=_ctx()).deleted_ids == ()
    assert sweeper.sweep(TENANT_A, context=_ctx()).deleted_ids == ("race",)
    with pytest.raises(KeyError):
        store.get(TENANT_A, "race")


def test_an_undisturbed_sweep_still_deletes_the_expired_artifact(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(artifact_id="old", expires_at_utc=T0 + timedelta(minutes=1)))
    clock.advance(hours=1)
    sweeper, _ = _sweeper(store, clock)
    report = sweeper.sweep(TENANT_A, context=_ctx())
    assert report.deleted_ids == ("old",) and report.changed_ids == ()


# ------------------------------------------------------------------ limit (F-D8-09, negative limit only)


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_a_non_positive_limit_is_rejected(store: ArtifactStore, limit: int) -> None:
    sweeper, _ = _sweeper(store, MutableClock())
    with pytest.raises(ValueError, match="limit"):
        sweeper.sweep(TENANT_A, context=_ctx(), limit=limit)


# ------------------------------------------------------------------ audit is mandatory by default


def test_a_sweeper_without_an_audit_recorder_refuses_and_deletes_nothing(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(artifact_id="old", expires_at_utc=T0 + timedelta(minutes=1)))
    clock.advance(hours=1)
    sweeper = ArtifactRetentionSweeper(store, clock=clock)
    with pytest.raises(ArtifactSweepAuditRequired):
        sweeper.sweep(TENANT_A)
    assert store.get(TENANT_A, "old").artifact_id == "old"


def test_an_explicitly_unaudited_sweeper_still_sweeps(store: ArtifactStore) -> None:
    clock = MutableClock()
    store.put(make_artifact(artifact_id="old", expires_at_utc=T0 + timedelta(minutes=1)))
    clock.advance(hours=1)
    sweeper = ArtifactRetentionSweeper(store, clock=clock, allow_unaudited_deletion=True)
    assert sweeper.sweep(TENANT_A).deleted_ids == ("old",)


def test_the_sweep_handler_fails_instead_of_deleting_unaudited(tmp_path: Path) -> None:
    from intergrax.runtime.background_execution.bootstrap import bootstrap_background_execution
    from intergrax.runtime.background_execution.identity_persistence import (
        KvBackgroundExecutionIdentityPersistence,
    )
    from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
    from tests.qualification.pdc_x.d8._kv import DictKV

    store = sqlite_store(tmp_path)
    clock = MutableClock()
    store.put(make_artifact(artifact_id="old", expires_at_utc=T0 + timedelta(minutes=1)))
    clock.advance(hours=1)
    handler = make_retention_sweep_handler(ArtifactRetentionSweeper(store, clock=clock))
    identity = bootstrap_background_execution(
        transport_ref=BackgroundTransportExecutionRef(tenant_id=TENANT_A, provider="test", transport_task_id="s-1"),
        identity_persistence=KvBackgroundExecutionIdentityPersistence(DictKV()),
    )
    result = handler(
        tenant_id=TENANT_A,
        run_id=str(identity.run_id),
        payload=json.dumps({"limit": 10}).encode(),
        idempotency_key=None,
        execution_identity=identity,
    )
    assert not result.success
    assert "audit_recorder_required" in repr(result)
    assert store.get(TENANT_A, "old").artifact_id == "old"
