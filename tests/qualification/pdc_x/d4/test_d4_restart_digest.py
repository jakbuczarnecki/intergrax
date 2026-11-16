# © Artur Czarnecki. All rights reserved.

"""PDC-D4-R010 (APP-HOST-5D): restart preserves and verifies the profile digest.

Generic, application-agnostic: the proof runs over a plain hosted application, not the LKW
path. The real ``HostedApplicationSupervisor`` re-launches the application; every engine
instance must run under the exact digest of the supervised definition, and an engine built
from different configuration is rejected before it can start.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationEventType,
    HostedApplicationExitKind,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
    RestartPolicy,
)
from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.errors import HostedApplicationSupervisorFailureReason
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from tests.qualification.pdc_x.d4._support import (
    JournalRuntime,
    RecordingPublisher,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

_WAIT_SECONDS = 20.0


class _RecordingGuard:
    """Delegating guard that records the identity (including digest) of every acquisition."""

    def __init__(self, inner: object, identities: list[HostedApplicationInstanceIdentity]) -> None:
        self._inner = inner
        self._identities = identities

    async def acquire(self, identity: HostedApplicationInstanceIdentity):
        self._identities.append(identity)
        return await self._inner.acquire(identity)  # type: ignore[attr-defined]


_FAST_RESTART = RestartPolicy(
    mode=RestartMode.ON_FAILURE,
    max_attempts=3,
    initial_backoff_seconds=0.05,
    max_backoff_seconds=0.2,
    jitter_ratio=0.0,
)


async def _wait_until(predicate, description: str) -> None:
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {description}")
        await asyncio.sleep(0.02)


def _ready_count(publisher: RecordingPublisher) -> int:
    return len(publisher.of_type(HostedApplicationEventType.APPLICATION_READY))


async def test_r010_restart_preserves_the_profile_digest_on_every_instance(tmp_path: Path) -> None:
    profile = make_profile(runtime=JournalRuntime(), restart=_FAST_RESTART)
    definition = definition_of(profile)
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()
    identities: list[HostedApplicationInstanceIdentity] = []
    counter = iter(range(100))

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        engine, _, _ = build_engine(
            launch.definition,
            tmp_path,
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
        )
        engine.instance_guard = _RecordingGuard(engine.instance_guard, identities)  # type: ignore[assignment]
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: f"digest-attempt-{next(counter)}",
    )
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(lambda: _ready_count(publisher) == 1, "first instance ready")
        control.request_restart("test.config_reload")
        await _wait_until(lambda: _ready_count(publisher) == 2, "second instance ready")
        control.request_shutdown("test.done")
        result = await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("test.abort")
            await asyncio.gather(run, return_exceptions=True)

    expected = definition.profile_digest
    assert [identity.instance_id for identity in identities] == ["digest-attempt-0", "digest-attempt-1"]
    assert {identity.profile_digest for identity in identities} == {expected}
    assert result.profile_digest == expected
    assert {attempt.exit_record.profile_digest for attempt in result.attempts if attempt.exit_record} == {expected}
    restart_events = [
        event
        for event in publisher.events
        if event.event_type
        in {
            HostedApplicationEventType.RESTART_REQUESTED,
            HostedApplicationEventType.RESTART_SCHEDULED,
            HostedApplicationEventType.RESTART_STARTED,
        }
    ]
    assert restart_events
    assert {str(event.payload["profile_digest"]) for event in restart_events} == {expected}
    lease_digests = {
        str(event.payload["profile_digest"])
        for event in publisher.of_type(HostedApplicationEventType.INSTANCE_ACQUIRED)
    }
    assert lease_digests == {expected}


async def test_r010_restart_with_changed_configuration_is_rejected_before_start(
    tmp_path: Path,
) -> None:
    original = make_profile(runtime=JournalRuntime(), restart=_FAST_RESTART, metadata={"config": "v1"})
    drifted = make_profile(runtime=JournalRuntime(), restart=_FAST_RESTART, metadata={"config": "v2"})
    definition = definition_of(original)
    drifted_definition = definition_of(drifted)
    assert drifted_definition.profile_digest != definition.profile_digest

    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()
    launched: list[str] = []
    counter = iter(range(100))

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        launched.append(launch.instance_id)
        # The restarted attempt is (wrongly) composed from drifted configuration.
        chosen = definition if launch.attempt_number == 0 else drifted_definition
        engine, _, _ = build_engine(
            chosen,
            tmp_path,
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
        instance_id_generator=lambda: f"drift-attempt-{next(counter)}",
    )
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(lambda: _ready_count(publisher) == 1, "first instance ready")
        control.request_restart("test.config_drift")
        result = await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("test.abort")
            await asyncio.gather(run, return_exceptions=True)

    # The drifted engine never started: no second READY, no second lease.
    assert _ready_count(publisher) == 1
    assert len(publisher.of_type(HostedApplicationEventType.INSTANCE_ACQUIRED)) == 1
    assert launched == ["drift-attempt-0", "drift-attempt-1"]
    assert result.final_exit.exit_kind is HostedApplicationExitKind.SUPERVISOR_ERROR
    assert result.final_exit.retryable is False
    assert result.restart_exhausted is False
    failed = [
        event
        for event in publisher.of_type(HostedApplicationEventType.APPLICATION_FAILED)
        if event.instance_id == "drift-attempt-1"
    ]
    assert len(failed) == 1
    assert HostedApplicationSupervisorFailureReason.ENGINE_PROFILE_DIGEST_MISMATCH.value in repr(failed[0].payload)
