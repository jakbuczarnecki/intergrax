# © Artur Czarnecki. All rights reserved.

"""DS-EXEC-02 — Decision checkpoint persistence capability hosted by canonical ExecutionRuntime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from collections.abc import Awaitable, Callable

import pytest

from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    decision_checkpoint_state,
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
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.active_decision_checkpoint_persistence import (
    ActiveDecisionCheckpointPersistenceBinding,
    is_decision_checkpoint_persistence_active,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    require_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
    load_decision_checkpoint,
    save_decision_checkpoint,
)
from intergrax.runtime.execution.decision_lifecycle_host import (
    CanonicalDecisionLifecycleHost,
)
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootExecutionOptions,
    resolve_root_execution_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    value: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    value: str


@dataclass(frozen=True, slots=True)
class ProbeCheckpointPayload:
    value: str


@dataclass(frozen=True, slots=True)
class AlternateCheckpointPayload:
    value: int


_PROBE = ProbeRequest(value="probe")
_ProbeCallback = Callable[[ProbeRequest], Awaitable[ProbeResult]]


class ProbeDelegate:
    __slots__ = ("_callback",)

    def __init__(self, callback: _ProbeCallback) -> None:
        self._callback = callback

    async def execute(self, request: ProbeRequest) -> ProbeResult:
        return await self._callback(request)


class RecordingDecisionCheckpointPersistence(
    DecisionCheckpointPersistence[ProbeCheckpointPayload],
):
    __slots__ = ("load_calls", "save_calls", "_store")

    def __init__(self) -> None:
        self.load_calls = 0
        self.save_calls = 0
        self._store: dict[
            DecisionFinalizationKey,
            DecisionCheckpointState[ProbeCheckpointPayload],
        ] = {}

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[ProbeCheckpointPayload] | None:
        self.load_calls += 1
        return self._store.get(key)

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[ProbeCheckpointPayload],
    ) -> None:
        self.save_calls += 1
        self._store[checkpoint.finalization.key] = checkpoint


class AlternateRecordingDecisionCheckpointPersistence(
    DecisionCheckpointPersistence[AlternateCheckpointPayload],
):
    __slots__ = ("load_calls", "save_calls", "_store")

    def __init__(self) -> None:
        self.load_calls = 0
        self.save_calls = 0
        self._store: dict[
            DecisionFinalizationKey,
            DecisionCheckpointState[AlternateCheckpointPayload],
        ] = {}

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[AlternateCheckpointPayload] | None:
        self.load_calls += 1
        return self._store.get(key)

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[AlternateCheckpointPayload],
    ) -> None:
        self.save_calls += 1
        self._store[checkpoint.finalization.key] = checkpoint


class InvalidReturningDecisionCheckpointPersistence(
    DecisionCheckpointPersistence[ProbeCheckpointPayload],
):
    __slots__ = ("invalid_checkpoint", "load_calls", "save_calls")

    def __init__(
        self,
        *,
        invalid_checkpoint: DecisionCheckpointState[ProbeCheckpointPayload],
    ) -> None:
        self.invalid_checkpoint = invalid_checkpoint
        self.load_calls = 0
        self.save_calls = 0

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[ProbeCheckpointPayload] | None:
        self.load_calls += 1
        return self.invalid_checkpoint

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[ProbeCheckpointPayload],
    ) -> None:
        self.save_calls += 1


def _decision_identity_from_active_execution() -> DecisionIdentity:
    run_id, attempt_id = require_active_execution_identity()
    execution_id = require_active_execution_id()
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="probe", subject="probe-subject"),
        tenant_id="tenant-probe",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
    )


def _root_context() -> RootExecutionContext:
    return resolve_root_execution_context(
        RootExecutionOptions(authority=ParentExecutionAuthority.unrestricted_root()),
    )


@pytest.mark.asyncio
async def test_ordinary_execution_without_persistence_has_no_active_binding() -> None:
    probe_access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(
        RecordingDecisionCheckpointPersistence(),
    )
    observed: list[DecisionCheckpointPersistence[ProbeCheckpointPayload] | None] = []

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        observed.append(probe_access.get_active())
        return ProbeResult(value="ok")

    runtime = ExecutionRuntime(ProbeDelegate(_delegate))
    assert not is_decision_checkpoint_persistence_active()

    result = await runtime.execute(_PROBE, _root_context())

    assert result == ProbeResult(value="ok")
    assert observed == [None]
    assert not is_decision_checkpoint_persistence_active()


@pytest.mark.asyncio
async def test_configured_persistence_visible_inside_delegate() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    observed: list[DecisionCheckpointPersistence[ProbeCheckpointPayload] | None] = []

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        observed.append(access.require_active())
        return ProbeResult(value="ok")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    await runtime.execute(_PROBE, _root_context())

    assert len(observed) == 1
    assert observed[0] is store


@pytest.mark.asyncio
async def test_configured_persistence_unused_does_not_load_or_save() -> None:
    store = RecordingDecisionCheckpointPersistence()

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        return ProbeResult(value="unchanged")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    result = await runtime.execute(_PROBE, _root_context())

    assert result == ProbeResult(value="unchanged")
    assert store.load_calls == 0
    assert store.save_calls == 0


@pytest.mark.asyncio
async def test_save_and_load_canonical_checkpoint() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    saved: DecisionCheckpointState[ProbeCheckpointPayload] | None = None
    loaded: DecisionCheckpointState[ProbeCheckpointPayload] | None = None
    checkpoint_key: DecisionFinalizationKey | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal saved, loaded, checkpoint_key
        identity = _decision_identity_from_active_execution()
        lifecycle = require_active_decision_lifecycle_host().start(identity)
        lifecycle = require_active_decision_lifecycle_host().transition(
            lifecycle,
            DecisionLifecycleStage.VERIFICATION,
        )
        finalization = initial_decision_finalize_guard(decision_finalization_key(identity))
        checkpoint = decision_checkpoint_state(
            lifecycle=lifecycle,
            finalization=finalization,
        )
        checkpoint_key = decision_finalization_key(identity)
        persistence = access.require_active()
        save_decision_checkpoint(persistence, checkpoint=checkpoint)
        saved = checkpoint
        loaded = load_decision_checkpoint(persistence, key=checkpoint_key)
        return ProbeResult(value="checkpointed")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_lifecycle_host=CanonicalDecisionLifecycleHost(),
        decision_checkpoint_persistence=store,
    )
    await runtime.execute(_PROBE, _root_context())

    assert saved is not None
    assert loaded is not None
    assert loaded == saved
    assert loaded.lifecycle.identity == saved.lifecycle.identity
    assert loaded.lifecycle.stage == saved.lifecycle.stage
    assert loaded.lifecycle.transition_index == saved.lifecycle.transition_index
    assert loaded.finalization.key == saved.finalization.key
    assert loaded.finalization.authoritative_outcome == saved.finalization.authoritative_outcome
    assert store.save_calls == 1
    assert store.load_calls == 1


@pytest.mark.asyncio
async def test_load_absent_checkpoint_returns_none() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    loaded: DecisionCheckpointState[ProbeCheckpointPayload] | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal loaded
        identity = _decision_identity_from_active_execution()
        persistence = access.require_active()
        loaded = load_decision_checkpoint(
            persistence,
            key=decision_finalization_key(identity),
        )
        return ProbeResult(value="absent")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    await runtime.execute(_PROBE, _root_context())

    assert loaded is None
    assert store.load_calls == 1
    assert store.save_calls == 0


@pytest.mark.asyncio
async def test_save_rejects_invalid_checkpoint_before_persist() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        identity = _decision_identity_from_active_execution()
        other_identity = DecisionIdentity(
            decision_id=mint_decision_id(),
            version=initial_decision_version(),
            scope=DecisionScope(namespace="probe", subject="other-subject"),
            tenant_id="tenant-other",
            execution=identity.execution,
        )
        lifecycle = DecisionLifecycleState(
            identity=identity,
            stage=DecisionLifecycleStage.VERIFICATION,
            transition_index=1,
        )
        persistence = access.require_active()
        with pytest.raises(ValueError, match="does not match finalization key"):
            save_decision_checkpoint(
                persistence,
                checkpoint=decision_checkpoint_state(
                    lifecycle=lifecycle,
                    finalization=initial_decision_finalize_guard(
                        decision_finalization_key(other_identity),
                    ),
                ),
            )
        return ProbeResult(value="rejected")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    await runtime.execute(_PROBE, _root_context())

    assert store.save_calls == 0


@pytest.mark.asyncio
async def test_load_rejects_invalid_checkpoint_from_storage() -> None:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="probe", subject="subject-a"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    other_identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="probe", subject="subject-b"),
        tenant_id="tenant-b",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    lifecycle = DecisionLifecycleState(
        identity=identity,
        stage=DecisionLifecycleStage.VERIFICATION,
        transition_index=1,
    )
    finalization = initial_decision_finalize_guard(decision_finalization_key(other_identity))

    @dataclass(frozen=True, slots=True)
    class _CorruptDecisionCheckpointState(
        DecisionCheckpointState[ProbeCheckpointPayload],
    ):
        def __post_init__(self) -> None:
            return

    corrupt_checkpoint = _CorruptDecisionCheckpointState(
        lifecycle=lifecycle,
        finalization=finalization,
    )

    class CorruptingDecisionCheckpointPersistence(
        DecisionCheckpointPersistence[ProbeCheckpointPayload],
    ):
        __slots__ = ("_corrupt_checkpoint", "load_calls")

        def __init__(
            self,
            *,
            corrupt_checkpoint: DecisionCheckpointState[ProbeCheckpointPayload],
        ) -> None:
            self._corrupt_checkpoint = corrupt_checkpoint
            self.load_calls = 0

        def load(
            self,
            *,
            key: DecisionFinalizationKey,
        ) -> DecisionCheckpointState[ProbeCheckpointPayload] | None:
            self.load_calls += 1
            return self._corrupt_checkpoint

        def save(
            self,
            *,
            checkpoint: DecisionCheckpointState[ProbeCheckpointPayload],
        ) -> None:
            return None

    store = CorruptingDecisionCheckpointPersistence(corrupt_checkpoint=corrupt_checkpoint)
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        persistence = access.require_active()
        with pytest.raises(TypeError, match="must be DecisionCheckpointState"):
            load_decision_checkpoint(
                persistence,
                key=decision_finalization_key(identity),
            )
        return ProbeResult(value="rejected")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    await runtime.execute(_PROBE, _root_context())

    assert store.load_calls == 1


@pytest.mark.asyncio
async def test_active_persistence_absent_before_during_after_success() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    phases: list[str] = []

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        phases.append("during")
        assert access.require_active() is store
        return ProbeResult(value="ok")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    phases.append("before")
    assert not is_decision_checkpoint_persistence_active()

    await runtime.execute(_PROBE, _root_context())

    phases.append("after")
    assert phases == ["before", "during", "after"]
    assert not is_decision_checkpoint_persistence_active()


@pytest.mark.asyncio
async def test_active_persistence_reset_after_delegate_exception() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        assert access.require_active() is store
        raise RuntimeError("delegate failed")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )

    with pytest.raises(RuntimeError, match="delegate failed"):
        await runtime.execute(_PROBE, _root_context())

    assert not is_decision_checkpoint_persistence_active()


@pytest.mark.asyncio
async def test_concurrent_executions_isolate_active_persistence() -> None:
    store_a = RecordingDecisionCheckpointPersistence()
    store_b = RecordingDecisionCheckpointPersistence()
    access_a = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store_a)
    access_b = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store_b)
    seen_a: list[DecisionCheckpointPersistence[ProbeCheckpointPayload] | None] = []
    seen_b: list[DecisionCheckpointPersistence[ProbeCheckpointPayload] | None] = []
    gate = asyncio.Event()

    async def _delegate_a(_request: ProbeRequest) -> ProbeResult:
        seen_a.append(access_a.require_active())
        gate.set()
        await asyncio.sleep(0.05)
        seen_a.append(access_a.require_active())
        return ProbeResult(value="a")

    async def _delegate_b(_request: ProbeRequest) -> ProbeResult:
        await gate.wait()
        seen_b.append(access_b.require_active())
        await asyncio.sleep(0.05)
        seen_b.append(access_b.require_active())
        return ProbeResult(value="b")

    runtime_a = ExecutionRuntime(
        ProbeDelegate(_delegate_a),
        decision_checkpoint_persistence=store_a,
    )
    runtime_b = ExecutionRuntime(
        ProbeDelegate(_delegate_b),
        decision_checkpoint_persistence=store_b,
    )

    result_a, result_b = await asyncio.gather(
        runtime_a.execute(_PROBE, _root_context()),
        runtime_b.execute(_PROBE, _root_context()),
    )

    assert result_a == ProbeResult(value="a")
    assert result_b == ProbeResult(value="b")
    assert seen_a == [store_a, store_a]
    assert seen_b == [store_b, store_b]
    assert not is_decision_checkpoint_persistence_active()


@pytest.mark.asyncio
async def test_persistence_without_lifecycle_host() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    saved: DecisionCheckpointState[ProbeCheckpointPayload] | None = None
    loaded: DecisionCheckpointState[ProbeCheckpointPayload] | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal saved, loaded
        identity = _decision_identity_from_active_execution()
        lifecycle = DecisionLifecycleState(
            identity=identity,
            stage=DecisionLifecycleStage.PROPOSAL,
            transition_index=0,
        )
        finalization = initial_decision_finalize_guard(decision_finalization_key(identity))
        checkpoint = decision_checkpoint_state(
            lifecycle=lifecycle,
            finalization=finalization,
        )
        persistence = access.require_active()
        save_decision_checkpoint(persistence, checkpoint=checkpoint)
        saved = checkpoint
        loaded = load_decision_checkpoint(
            persistence,
            key=decision_finalization_key(identity),
        )
        return ProbeResult(value="persisted")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    result = await runtime.execute(_PROBE, _root_context())

    assert result == ProbeResult(value="persisted")
    assert saved is not None
    assert loaded == saved


@pytest.mark.asyncio
async def test_lifecycle_host_and_persistence_together() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    host = CanonicalDecisionLifecycleHost()
    captured_run_id: RunId | None = None
    captured_execution_id: ExecutionId | None = None
    restored: DecisionCheckpointState[ProbeCheckpointPayload] | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal captured_run_id, captured_execution_id, restored
        active_host = require_active_decision_lifecycle_host()
        identity = _decision_identity_from_active_execution()
        captured_run_id = identity.execution.run_id
        captured_execution_id = identity.execution.execution_id
        started = active_host.start(identity)
        verified = active_host.transition(started, DecisionLifecycleStage.VERIFICATION)
        finalization = initial_decision_finalize_guard(decision_finalization_key(identity))
        checkpoint = decision_checkpoint_state(
            lifecycle=verified,
            finalization=finalization,
        )
        persistence = access.require_active()
        save_decision_checkpoint(persistence, checkpoint=checkpoint)
        restored = load_decision_checkpoint(
            persistence,
            key=decision_finalization_key(identity),
        )
        return ProbeResult(value="integrated")

    root_context = _root_context()
    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_lifecycle_host=host,
        decision_checkpoint_persistence=store,
    )
    result = await runtime.execute(_PROBE, root_context)

    assert result == ProbeResult(value="integrated")
    assert captured_run_id == root_context.run_id
    assert captured_execution_id == root_context.execution_id
    assert restored is not None
    assert restored.lifecycle.stage is DecisionLifecycleStage.VERIFICATION
    assert restored.lifecycle.transition_index == 1
    assert restored.lifecycle.identity.execution.run_id == root_context.run_id
    assert restored.lifecycle.identity.execution.execution_id == root_context.execution_id
    assert restored.finalization.authoritative_outcome is None


@pytest.mark.asyncio
async def test_type_preservation_through_active_binding() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    inferred_persistence: DecisionCheckpointPersistence[ProbeCheckpointPayload] | None = None
    inferred_loaded: DecisionCheckpointState[ProbeCheckpointPayload] | None = None
    inferred_checkpoint: DecisionCheckpointState[ProbeCheckpointPayload] | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal inferred_persistence, inferred_loaded, inferred_checkpoint
        identity = _decision_identity_from_active_execution()
        lifecycle = DecisionLifecycleState(
            identity=identity,
            stage=DecisionLifecycleStage.PROPOSAL,
            transition_index=0,
        )
        finalization = initial_decision_finalize_guard(decision_finalization_key(identity))
        checkpoint = decision_checkpoint_state(
            lifecycle=lifecycle,
            finalization=finalization,
        )
        persistence = access.require_active()
        inferred_persistence = persistence
        inferred_checkpoint = checkpoint
        save_decision_checkpoint(persistence, checkpoint=checkpoint)
        inferred_loaded = load_decision_checkpoint(
            persistence,
            key=decision_finalization_key(identity),
        )
        return ProbeResult(value="typed")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    await runtime.execute(_PROBE, _root_context())

    assert inferred_persistence is store
    assert inferred_loaded is not None
    assert inferred_checkpoint is not None
    assert inferred_loaded == inferred_checkpoint


@pytest.mark.asyncio
async def test_alternate_payload_type_has_separate_typed_access() -> None:
    probe_store = RecordingDecisionCheckpointPersistence()
    probe_access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(probe_store)
    alternate_store = AlternateRecordingDecisionCheckpointPersistence()
    alternate_access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(
        alternate_store,
    )
    probe_seen: DecisionCheckpointPersistence[ProbeCheckpointPayload] | None = None
    alternate_seen: DecisionCheckpointPersistence[AlternateCheckpointPayload] | None = None

    async def _probe_delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal probe_seen
        probe_seen = probe_access.require_active()
        return ProbeResult(value="probe")

    async def _alternate_delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal alternate_seen
        alternate_seen = alternate_access.require_active()
        return ProbeResult(value="alternate")

    probe_runtime = ExecutionRuntime(
        ProbeDelegate(_probe_delegate),
        decision_checkpoint_persistence=probe_store,
    )
    alternate_runtime = ExecutionRuntime(
        ProbeDelegate(_alternate_delegate),
        decision_checkpoint_persistence=alternate_store,
    )

    await probe_runtime.execute(_PROBE, _root_context())
    await alternate_runtime.execute(_PROBE, _root_context())

    assert probe_seen is probe_store
    assert alternate_seen is alternate_store


@pytest.mark.asyncio
async def test_binding_rejects_active_alternate_persistence_different_payload() -> None:
    store_probe = RecordingDecisionCheckpointPersistence()
    access_probe = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store_probe)
    store_alternate = AlternateRecordingDecisionCheckpointPersistence()
    observed: DecisionCheckpointPersistence[ProbeCheckpointPayload] | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal observed
        observed = access_probe.get_active()
        with pytest.raises(
            RuntimeError,
            match="active decision checkpoint persistence does not match this binding",
        ):
            access_probe.require_active()
        return ProbeResult(value="rejected")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store_alternate,
    )
    await runtime.execute(_PROBE, _root_context())

    assert observed is None


@pytest.mark.asyncio
async def test_binding_rejects_active_same_payload_different_instance() -> None:
    store_a1 = RecordingDecisionCheckpointPersistence()
    store_a2 = RecordingDecisionCheckpointPersistence()
    access_a1 = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store_a1)
    observed: DecisionCheckpointPersistence[ProbeCheckpointPayload] | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal observed
        observed = access_a1.get_active()
        with pytest.raises(
            RuntimeError,
            match="active decision checkpoint persistence does not match this binding",
        ):
            access_a1.require_active()
        return ProbeResult(value="rejected")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store_a2,
    )
    await runtime.execute(_PROBE, _root_context())

    assert observed is None


@pytest.mark.asyncio
async def test_binding_returns_bound_persistence_when_active_matches() -> None:
    store = RecordingDecisionCheckpointPersistence()
    access = ActiveDecisionCheckpointPersistenceBinding.for_persistence(store)
    observed_get: DecisionCheckpointPersistence[ProbeCheckpointPayload] | None = None
    observed_require: DecisionCheckpointPersistence[ProbeCheckpointPayload] | None = None

    async def _delegate(_request: ProbeRequest) -> ProbeResult:
        nonlocal observed_get, observed_require
        observed_get = access.get_active()
        observed_require = access.require_active()
        return ProbeResult(value="ok")

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_checkpoint_persistence=store,
    )
    await runtime.execute(_PROBE, _root_context())

    assert observed_get is store
    assert observed_require is store
