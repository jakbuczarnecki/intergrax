# © Artur Czarnecki. All rights reserved.

"""PCM-CHECKPOINT-SCHEDULER-INTEGRITY checkpoint CAS tests (PCM-04)."""

from __future__ import annotations

import pytest

from intergrax.agents.persistence.checkpoint_store import (
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
    build_checkpoint,
)
from intergrax.agents.persistence.session_persistence import (
    AgentSessionPersistence,
    make_checkpoint_hook,
)
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.checkpoint_revision import (
    CheckpointRevisionConflictError,
    CheckpointStepRegressionError,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _checkpoint(
    *,
    run_id: str = "run-1",
    step_index: int = 1,
    state_root: dict | None = None,
) -> object:
    return build_checkpoint(
        run_id=run_id,
        tenant_id="tenant-a",
        agent_id="legal",
        step_index=step_index,
        state_root=state_root or {"acp.state.v1": {"_version": step_index}},
        side_effect_ledger=[],
        trace_step_count=step_index + 1,
    )


@pytest.mark.parametrize("store_factory", ["memory", "sqlite"])
def test_initial_revision_is_one(store_factory: str, tmp_path) -> None:
    store = (
        InMemoryAgentCheckpointStore()
        if store_factory == "memory"
        else SQLiteAgentCheckpointStore(tmp_path / "ckpt.db")
    )
    saved = store.save(_checkpoint(step_index=0))
    assert saved.revision == 1
    loaded = store.get_latest("run-1", "tenant-a")
    assert loaded is not None
    assert loaded.revision == 1


@pytest.mark.parametrize("store_factory", ["memory", "sqlite"])
def test_expected_revision_update_increments(store_factory: str, tmp_path) -> None:
    store = (
        InMemoryAgentCheckpointStore()
        if store_factory == "memory"
        else SQLiteAgentCheckpointStore(tmp_path / "ckpt.db")
    )
    store.save(_checkpoint(step_index=0))
    current = store.get_latest("run-1", "tenant-a")
    assert current is not None
    saved = store.save(_checkpoint(step_index=1), expected_revision=current.revision)
    assert saved.revision == 2


@pytest.mark.parametrize("store_factory", ["memory", "sqlite"])
def test_stale_writer_rejected(store_factory: str, tmp_path) -> None:
    store = (
        InMemoryAgentCheckpointStore()
        if store_factory == "memory"
        else SQLiteAgentCheckpointStore(tmp_path / "ckpt.db")
    )
    store.save(_checkpoint(step_index=0, state_root={"marker": "initial"}))
    revision_a = store.get_latest("run-1", "tenant-a")
    revision_b = store.get_latest("run-1", "tenant-a")
    assert revision_a is not None and revision_b is not None

    store.save(
        _checkpoint(step_index=1, state_root={"marker": "winner"}),
        expected_revision=revision_b.revision,
    )
    with pytest.raises(CheckpointRevisionConflictError):
        store.save(
            _checkpoint(step_index=2, state_root={"marker": "stale"}),
            expected_revision=revision_a.revision,
        )

    latest = store.get_latest("run-1", "tenant-a")
    assert latest is not None
    assert latest.revision == 2
    assert latest.state_root["marker"] == "winner"


def test_sqlite_two_connection_race_exactly_one_wins(tmp_path) -> None:
    db_path = tmp_path / "race.db"
    store_a = SQLiteAgentCheckpointStore(db_path)
    store_b = SQLiteAgentCheckpointStore(db_path)
    store_a.save(_checkpoint(step_index=0))

    current = store_a.get_latest("run-1", "tenant-a")
    assert current is not None
    expected = current.revision

    store_a.save(_checkpoint(step_index=1), expected_revision=expected)
    with pytest.raises(CheckpointRevisionConflictError):
        store_b.save(_checkpoint(step_index=2), expected_revision=expected)

    latest = store_a.get_latest("run-1", "tenant-a")
    assert latest is not None
    assert latest.revision == 2
    assert latest.step_index == 1


@pytest.mark.parametrize("store_factory", ["memory", "sqlite"])
def test_step_regression_rejected(store_factory: str, tmp_path) -> None:
    store = (
        InMemoryAgentCheckpointStore()
        if store_factory == "memory"
        else SQLiteAgentCheckpointStore(tmp_path / "ckpt.db")
    )
    store.save(_checkpoint(step_index=5))
    current = store.get_latest("run-1", "tenant-a")
    assert current is not None
    with pytest.raises(CheckpointStepRegressionError):
        store.save(_checkpoint(step_index=4), expected_revision=current.revision)


@pytest.mark.parametrize("store_factory", ["memory", "sqlite"])
def test_stale_rejection_preserves_payload(store_factory: str, tmp_path) -> None:
    store = (
        InMemoryAgentCheckpointStore()
        if store_factory == "memory"
        else SQLiteAgentCheckpointStore(tmp_path / "ckpt.db")
    )
    store.save(_checkpoint(step_index=0))
    current = store.get_latest("run-1", "tenant-a")
    assert current is not None
    store.save(
        _checkpoint(
            step_index=1,
            state_root={
                "acp.state.v1": {"_version": 9},
                "side_effect_root": {"committed": True},
            },
        ),
        expected_revision=current.revision,
    )
    stale_revision = current.revision
    with pytest.raises(CheckpointRevisionConflictError):
        store.save(
            _checkpoint(step_index=2, state_root={"acp.state.v1": {"_version": 1}}),
            expected_revision=stale_revision,
        )
    latest = store.get_latest("run-1", "tenant-a")
    assert latest is not None
    assert latest.revision == 2
    assert latest.state_root["side_effect_root"]["committed"] is True


def _session_persistence(store: InMemoryAgentCheckpointStore, *, resume_enabled: bool = True):
    return AgentSessionPersistence(store, SideEffectLedger(), resume_enabled)


def _hook_kwargs(
    store: InMemoryAgentCheckpointStore,
    *,
    resume_enabled: bool = True,
    run_id: str = "run-hook",
):
    return dict(
        persistence=_session_persistence(store, resume_enabled=resume_enabled),
        run_id=run_id,
        tenant_id="tenant-a",
        agent_id="legal",
        trace_step_count_fn=lambda: 1,
    )


@pytest.mark.asyncio
async def test_new_session_hook_revisions_increment() -> None:
    store = InMemoryAgentCheckpointStore()
    hook = make_checkpoint_hook(**_hook_kwargs(store))
    assert hook is not None
    await hook({"acp.state.v1": {"_version": 0}}, 0)
    await hook({"acp.state.v1": {"_version": 1}}, 1)
    latest = store.get_latest("run-hook", "tenant-a")
    assert latest is not None
    assert latest.revision == 2
    assert latest.step_index == 1


@pytest.mark.asyncio
async def test_resumed_session_hook_expected_revision() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(
        build_checkpoint(
            run_id="run-resume",
            tenant_id="tenant-a",
            agent_id="legal",
            step_index=0,
            state_root={"acp.state.v1": {"_version": 0}},
            side_effect_ledger=[],
            trace_step_count=1,
        ),
    )
    for step in range(1, 4):
        current = store.get_latest("run-resume", "tenant-a")
        assert current is not None
        store.save(
            build_checkpoint(
                run_id="run-resume",
                tenant_id="tenant-a",
                agent_id="legal",
                step_index=step,
                state_root={"acp.state.v1": {"_version": step}},
                side_effect_ledger=[],
                trace_step_count=step + 1,
            ),
            expected_revision=current.revision,
        )
    latest_before = store.get_latest("run-resume", "tenant-a")
    assert latest_before is not None
    assert latest_before.revision == 4

    hook = make_checkpoint_hook(
        **_hook_kwargs(store, run_id="run-resume"),
    )
    assert hook is not None
    await hook({"acp.state.v1": {"_version": 4}}, 4)
    latest = store.get_latest("run-resume", "tenant-a")
    assert latest is not None
    assert latest.revision == 5
    assert latest.step_index == 4


@pytest.mark.asyncio
async def test_stale_session_hook_conflict() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(
        build_checkpoint(
            run_id="run-stale",
            tenant_id="tenant-a",
            agent_id="legal",
            step_index=0,
            state_root={"acp.state.v1": {"_version": 0}},
            side_effect_ledger=[],
            trace_step_count=1,
        ),
    )
    for step in range(1, 4):
        current = store.get_latest("run-stale", "tenant-a")
        assert current is not None
        store.save(
            build_checkpoint(
                run_id="run-stale",
                tenant_id="tenant-a",
                agent_id="legal",
                step_index=step,
                state_root={"acp.state.v1": {"_version": step}},
                side_effect_ledger=[],
                trace_step_count=step + 1,
            ),
            expected_revision=current.revision,
        )

    hook_a = make_checkpoint_hook(**_hook_kwargs(store, run_id="run-stale"))
    hook_b = make_checkpoint_hook(**_hook_kwargs(store, run_id="run-stale"))
    assert hook_a is not None and hook_b is not None
    await hook_b({"acp.state.v1": {"_version": 4}}, 4)
    with pytest.raises(CheckpointRevisionConflictError):
        await hook_a({"acp.state.v1": {"_version": 5}}, 5)
    latest = store.get_latest("run-stale", "tenant-a")
    assert latest is not None
    assert latest.revision == 5
    assert latest.step_index == 4
