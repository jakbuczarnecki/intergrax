# © Artur Czarnecki. All rights reserved.

"""DS-EXEC-01 — Decision Lifecycle host capability hosted by canonical ExecutionRuntime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from collections.abc import Awaitable, Callable

import pytest

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
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    get_active_decision_lifecycle_host,
    require_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import (
    CanonicalDecisionLifecycleHost,
    DecisionLifecycleHost,
)
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootExecutionOptions,
    resolve_root_execution_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PROBE = object()
_ProbeCallback = Callable[[object], Awaitable[object]]


class ProbeDelegate:
    __slots__ = ("_callback",)

    def __init__(self, callback: _ProbeCallback) -> None:
        self._callback = callback

    async def execute(self, request: object) -> object:
        return await self._callback(request)


@dataclass
class MutableRecordingDecisionLifecycleHost:
    """Mutable test host for counting without forbidden production patterns."""

    start_calls: int = 0
    transition_calls: int = 0

    def start(
        self,
        identity: DecisionIdentity,
    ) -> DecisionLifecycleState:
        self.start_calls += 1
        return initial_decision_lifecycle_state(identity)

    def transition(
        self,
        state: DecisionLifecycleState,
        to_stage: DecisionLifecycleStage,
    ) -> DecisionLifecycleState:
        self.transition_calls += 1
        return transition_decision_lifecycle(state, to_stage)


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
async def test_ordinary_execution_without_host_has_no_active_binding() -> None:
    observed: list[DecisionLifecycleHost | None] = []

    async def _delegate(_request: object) -> str:
        observed.append(get_active_decision_lifecycle_host())
        return "ok"

    runtime = ExecutionRuntime(ProbeDelegate(_delegate))
    assert get_active_decision_lifecycle_host() is None

    result = await runtime.execute(_PROBE, _root_context())

    assert result == "ok"
    assert observed == [None]
    assert get_active_decision_lifecycle_host() is None


@pytest.mark.asyncio
async def test_configured_host_visible_inside_delegate() -> None:
    host = CanonicalDecisionLifecycleHost()
    observed: list[DecisionLifecycleHost | None] = []

    async def _delegate(_request: object) -> str:
        observed.append(get_active_decision_lifecycle_host())
        return "ok"

    runtime = ExecutionRuntime(ProbeDelegate(_delegate), decision_lifecycle_host=host)
    await runtime.execute(_PROBE, _root_context())

    assert len(observed) == 1
    assert observed[0] is host


@pytest.mark.asyncio
async def test_configured_host_unused_does_not_start_lifecycle() -> None:
    recording_host = MutableRecordingDecisionLifecycleHost()

    async def _delegate(_request: object) -> str:
        return "unchanged"

    runtime = ExecutionRuntime(
        ProbeDelegate(_delegate),
        decision_lifecycle_host=recording_host,
    )
    result = await runtime.execute(_PROBE, _root_context())

    assert result == "unchanged"
    assert recording_host.start_calls == 0
    assert recording_host.transition_calls == 0


@pytest.mark.asyncio
async def test_decision_aware_delegate_starts_and_transitions_lifecycle() -> None:
    host = CanonicalDecisionLifecycleHost()
    captured: dict[str, DecisionLifecycleState | object] = {}

    async def _delegate(_request: object) -> str:
        active_host = require_active_decision_lifecycle_host()
        identity = _decision_identity_from_active_execution()
        started = active_host.start(identity)
        captured["started"] = started
        captured["verified"] = active_host.transition(
            started,
            DecisionLifecycleStage.VERIFICATION,
        )
        captured["run_id"] = identity.execution.run_id
        captured["attempt_id"] = identity.execution.attempt_id
        captured["execution_id"] = identity.execution.execution_id
        return "decided"

    runtime = ExecutionRuntime(ProbeDelegate(_delegate), decision_lifecycle_host=host)
    root_context = _root_context()
    await runtime.execute(_PROBE, root_context)

    started = captured["started"]
    verified = captured["verified"]
    assert isinstance(started, DecisionLifecycleState)
    assert isinstance(verified, DecisionLifecycleState)
    assert started.stage is DecisionLifecycleStage.PROPOSAL
    assert started.transition_index == 0
    assert started.identity.execution.run_id == root_context.run_id
    assert started.identity.execution.attempt_id == root_context.attempt_id
    assert started.identity.execution.execution_id == root_context.execution_id
    assert captured["run_id"] == root_context.run_id
    assert captured["attempt_id"] == root_context.attempt_id
    assert captured["execution_id"] == root_context.execution_id
    assert verified.stage is DecisionLifecycleStage.VERIFICATION
    assert verified.transition_index == 1
    assert verified.identity is started.identity


@pytest.mark.asyncio
async def test_active_host_absent_before_during_after_success() -> None:
    host = CanonicalDecisionLifecycleHost()
    phases: list[str] = []

    async def _delegate(_request: object) -> str:
        phases.append("during")
        assert get_active_decision_lifecycle_host() is host
        return "ok"

    runtime = ExecutionRuntime(ProbeDelegate(_delegate), decision_lifecycle_host=host)
    phases.append("before")
    assert get_active_decision_lifecycle_host() is None

    await runtime.execute(_PROBE, _root_context())

    phases.append("after")
    assert phases == ["before", "during", "after"]
    assert get_active_decision_lifecycle_host() is None


@pytest.mark.asyncio
async def test_active_host_reset_after_delegate_exception() -> None:
    host = CanonicalDecisionLifecycleHost()

    async def _delegate(_request: object) -> str:
        assert get_active_decision_lifecycle_host() is host
        raise RuntimeError("delegate failed")

    runtime = ExecutionRuntime(ProbeDelegate(_delegate), decision_lifecycle_host=host)

    with pytest.raises(RuntimeError, match="delegate failed"):
        await runtime.execute(_PROBE, _root_context())

    assert get_active_decision_lifecycle_host() is None


@pytest.mark.asyncio
async def test_concurrent_executions_isolate_active_hosts() -> None:
    host_a = CanonicalDecisionLifecycleHost()
    host_b = CanonicalDecisionLifecycleHost()
    seen_a: list[DecisionLifecycleHost | None] = []
    seen_b: list[DecisionLifecycleHost | None] = []
    gate = asyncio.Event()

    async def _delegate_a(_request: object) -> str:
        seen_a.append(get_active_decision_lifecycle_host())
        gate.set()
        await asyncio.sleep(0.05)
        seen_a.append(get_active_decision_lifecycle_host())
        return "a"

    async def _delegate_b(_request: object) -> str:
        await gate.wait()
        seen_b.append(get_active_decision_lifecycle_host())
        await asyncio.sleep(0.05)
        seen_b.append(get_active_decision_lifecycle_host())
        return "b"

    runtime_a = ExecutionRuntime(ProbeDelegate(_delegate_a), decision_lifecycle_host=host_a)
    runtime_b = ExecutionRuntime(ProbeDelegate(_delegate_b), decision_lifecycle_host=host_b)

    result_a, result_b = await asyncio.gather(
        runtime_a.execute(_PROBE, _root_context()),
        runtime_b.execute(_PROBE, _root_context()),
    )

    assert result_a == "a"
    assert result_b == "b"
    assert seen_a == [host_a, host_a]
    assert seen_b == [host_b, host_b]
    assert get_active_decision_lifecycle_host() is None


def test_canonical_host_start_matches_contract() -> None:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    host = CanonicalDecisionLifecycleHost()

    host_state = host.start(identity)
    contract_state = initial_decision_lifecycle_state(identity)

    assert host_state == contract_state


def test_canonical_host_transition_matches_contract() -> None:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    host = CanonicalDecisionLifecycleHost()
    state = host.start(identity)

    host_next = host.transition(state, DecisionLifecycleStage.VERIFICATION)
    contract_next = transition_decision_lifecycle(
        state,
        DecisionLifecycleStage.VERIFICATION,
    )

    assert host_next == contract_next


def test_canonical_host_rejects_illegal_transition() -> None:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    host = CanonicalDecisionLifecycleHost()
    state = host.start(identity)

    with pytest.raises(ValueError, match="Unsupported lifecycle transition"):
        host.transition(state, DecisionLifecycleStage.TERMINAL)


def test_require_active_host_raises_when_unbound() -> None:
    assert get_active_decision_lifecycle_host() is None

    with pytest.raises(RuntimeError, match="active decision lifecycle host required"):
        require_active_decision_lifecycle_host()
