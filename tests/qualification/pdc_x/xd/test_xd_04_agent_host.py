# © Artur Czarnecki. All rights reserved.

"""XD-04 (Agent <-> Host): configured vs effective roster; restart preserves the active revision and digest.

PDC-D1-R016 (activation continuity across restart) with PDC-D4-R010 (profile digest preserved by the
supervisor). The agent lifecycle (install, bind, enable, build, activate) runs on the real
``AgentPlatformAdminService`` over the durable SQLite agent-distribution stores. The hosted
application runtime reads that state on every start; the real ``HostedApplicationSupervisor``
restarts it, and the host profile is built from the persisted active revision so the profile digest
is bound to it. Fakes: the deployment adapter and projection coordinator (external runtime vendor).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationEventType,
    HostedApplicationExitKind,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
    RestartPolicy,
)
from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.errors import HostedApplicationSupervisorFailureReason
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.d4._support import (
    RecordingPublisher,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)
from tests.qualification.pdc_x.xd._support import build_sqlite_agent_stack

pytestmark = [pytest.mark.integration]

_WAIT_SECONDS = 20.0
_FAST_RESTART = RestartPolicy(
    mode=RestartMode.ON_FAILURE,
    max_attempts=3,
    initial_backoff_seconds=0.05,
    max_backoff_seconds=0.2,
    jitter_ratio=0.0,
)


@dataclass(frozen=True)
class _Observed:
    revision_id: str | None
    configured_binding_ids: tuple[str, ...]
    effective_agent_ids: tuple[str, ...]
    roster_revision_id: str


@dataclass
class _AgentPlatformRuntime:
    """Hosted application runtime that reads the persisted agent platform state on every start."""

    db_path: Path
    observed: list[_Observed] = field(default_factory=list)
    started = False

    async def start(self, context: HostedApplicationContext) -> None:
        self.observed.append(_observe(self.db_path))
        self.started = True

    async def stop(self, context: HostedApplicationContext) -> None:
        self.started = False

    async def ready(self, context: HostedApplicationContext) -> bool:
        return self.started


def _observe(db_path: Path) -> _Observed:
    """A cold read of the persisted state through a freshly composed stack (what a restart sees)."""
    stack = build_sqlite_agent_stack(db_path)
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    bindings = stack.service.list_bindings(application_id=h.APP, application_environment_id=h.ENV)
    roster = stack.service.inspect_effective_roster(application_id=h.APP, application_environment_id=h.ENV)
    return _Observed(
        revision_id=serving.traffic_serving_revision_id,
        configured_binding_ids=tuple(item.application_binding_id for item in bindings.bindings),
        effective_agent_ids=tuple(e.logical_agent_id for e in roster.entries if e.effective_enablement),
        roster_revision_id=str(roster.effective_roster_revision_id),
    )


def _activated_stack(db_path: Path):  # type: ignore[no-untyped-def]
    stack = build_sqlite_agent_stack(db_path)
    first = h.release("1.0.0")
    h.install_bind_enable(stack, first)
    h.build_revision(stack, "rev-1")
    h.activate_revision(stack, "rev-1")
    return stack


def _profile_for(runtime: _AgentPlatformRuntime, revision_id: str | None):  # type: ignore[no-untyped-def]
    # The host profile pins the active revision, so its digest is bound to the agent state.
    return make_profile(runtime=runtime, restart=_FAST_RESTART, metadata={"active_revision": str(revision_id)})  # type: ignore[arg-type]


async def _wait_until(predicate, description: str) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {description}")
        await asyncio.sleep(0.02)


def _ready_count(publisher: RecordingPublisher) -> int:
    return len(publisher.of_type(HostedApplicationEventType.APPLICATION_READY))


def test_xd04_configured_roster_differs_from_effective_roster_and_both_are_durable(tmp_path: Path) -> None:
    db = tmp_path / "agents.sqlite3"
    stack = _activated_stack(db)

    before = _observe(db)
    assert before.revision_id == "rev-1"
    assert before.configured_binding_ids == ("bind-search",)
    assert before.effective_agent_ids == (h.CONTRACT_ID,)

    # Desired state changes: the binding stays configured but the agent leaves the effective roster.
    from intergrax.agent_distribution.admin_models import SetAgentEnablementRequest

    stack.service.disable_binding(
        application_id=h.APP,
        application_environment_id=h.ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-disable", expected_revision=1),
        principal=h.principal(),
    )
    after = _observe(db)  # a cold read over the reopened database
    assert after.configured_binding_ids == ("bind-search",), "configured state is kept"
    assert after.effective_agent_ids == (), "effective state reflects the disablement"
    assert after.roster_revision_id != before.roster_revision_id
    assert after.revision_id == "rev-1", "disabling a binding does not change what is serving"


async def test_xd04_restart_preserves_the_active_revision_and_the_profile_digest(tmp_path: Path) -> None:
    db = tmp_path / "agents.sqlite3"
    _activated_stack(db)
    runtime = _AgentPlatformRuntime(db)
    pinned = _observe(db)
    definition = definition_of(_profile_for(runtime, pinned.revision_id))

    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()
    counter = iter(range(100))
    digests: list[str] = []

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):  # type: ignore[no-untyped-def]
        # Every attempt recomposes the host profile from the persisted active revision.
        fresh = definition_of(_profile_for(runtime, _observe(db).revision_id))
        digests.append(fresh.profile_digest)
        engine, _, _ = build_engine(
            fresh,
            tmp_path / "host",
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
        )
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: f"xd04-attempt-{next(counter)}",
    )
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(lambda: _ready_count(publisher) == 1, "first instance ready")
        control.request_restart("xd04.restart")
        await _wait_until(lambda: _ready_count(publisher) == 2, "second instance ready")
        control.request_shutdown("xd04.done")
        result = await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("xd04.abort")
            await asyncio.gather(run, return_exceptions=True)

    # Both instances saw the very same persisted agent state and ran under the same digest.
    assert len(runtime.observed) == 2 and runtime.observed[0] == runtime.observed[1]
    assert runtime.observed[0].revision_id == "rev-1"
    assert runtime.observed[0].effective_agent_ids == (h.CONTRACT_ID,)
    assert set(digests) == {definition.profile_digest}
    assert result.profile_digest == definition.profile_digest
    assert {a.exit_record.profile_digest for a in result.attempts if a.exit_record} == {definition.profile_digest}


async def test_xd04_a_new_activation_between_restarts_is_rejected_by_the_pinned_digest(tmp_path: Path) -> None:
    db = tmp_path / "agents.sqlite3"
    stack = _activated_stack(db)
    runtime = _AgentPlatformRuntime(db)
    definition = definition_of(_profile_for(runtime, _observe(db).revision_id))

    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()
    launched: list[str] = []
    counter = iter(range(100))

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):  # type: ignore[no-untyped-def]
        launched.append(launch.instance_id)
        fresh = definition_of(_profile_for(runtime, _observe(db).revision_id))
        engine, _, _ = build_engine(
            fresh,
            tmp_path / "host",
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
        )
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: f"xd04-drift-{next(counter)}",
    )
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(lambda: _ready_count(publisher) == 1, "first instance ready")
        # Operator activates another revision while the host is up, then the host restarts.
        h.build_revision(stack, "rev-2")
        h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)
        assert _observe(db).revision_id == "rev-2"
        control.request_restart("xd04.drift")
        result = await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("xd04.abort")
            await asyncio.gather(run, return_exceptions=True)

    # The restarted engine would run a different agent revision: it is rejected before it starts.
    assert _ready_count(publisher) == 1
    assert len(runtime.observed) == 1
    assert launched == ["xd04-drift-0", "xd04-drift-1"]
    assert result.final_exit.exit_kind is HostedApplicationExitKind.SUPERVISOR_ERROR
    failed = [
        event
        for event in publisher.of_type(HostedApplicationEventType.APPLICATION_FAILED)
        if event.instance_id == "xd04-drift-1"
    ]
    assert len(failed) == 1
    assert HostedApplicationSupervisorFailureReason.ENGINE_PROFILE_DIGEST_MISMATCH.value in repr(failed[0].payload)
