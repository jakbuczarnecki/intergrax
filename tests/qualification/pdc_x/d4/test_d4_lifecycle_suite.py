# © Artur Czarnecki. All rights reserved.

"""PDC-D4-R003/R004/R006: generic APP-HOST lifecycle qualification over the real engine."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationComponentRegistration,
    HostedApplicationEventType,
    HostedApplicationInstanceConflictError,
    HostedApplicationLifecycleState,
    HostedApplicationReadinessService,
    HostedApplicationStartupError,
)
from tests.qualification.pdc_x.d4._support import (
    APP_ID,
    JournalRuntime,
    OrderedComponent,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

_TEST_TIMEOUT_SECONDS = 20.0
_HOLDER_SCRIPT = Path(__file__).with_name("_lock_holder_process.py")
_HOLDER_MAX_SECONDS = 60.0


async def _wait_for(predicate, description: str) -> None:
    deadline = asyncio.get_running_loop().time() + _TEST_TIMEOUT_SECONDS
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {description}")
        await asyncio.sleep(0.02)


def _has_cause(exc: BaseException, expected: type[BaseException]) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, expected):
            return True
        current = current.__cause__
    return False


def _required(component: OrderedComponent, *dependencies: str) -> HostedApplicationComponentRegistration:
    return HostedApplicationComponentRegistration(
        component=component,
        required=True,
        dependencies=tuple(dependencies),
        start_timeout_seconds=5.0,
        stop_timeout_seconds=5.0,
    )


async def test_r003_startup_failure_rolls_back_in_reverse_order(tmp_path: Path) -> None:
    journal: list[str] = []
    runtime = JournalRuntime(journal)
    profile = make_profile(
        runtime=runtime,
        components=(
            _required(OrderedComponent("alpha", journal)),
            _required(OrderedComponent("bravo", journal), "alpha"),
            _required(OrderedComponent("charlie", journal), "bravo"),
            _required(OrderedComponent("delta", journal, fail_start=True), "charlie"),
        ),
    )
    engine, publisher, _ = build_engine(definition_of(profile), tmp_path)

    with pytest.raises(HostedApplicationStartupError):
        await engine.start()

    starts = [item for item in journal if item.startswith("start:")]
    stops = [item for item in journal if item.startswith("stop:")]
    assert starts == ["start:alpha", "start:bravo", "start:charlie", "start:delta"]
    # The failed component never started, so it is not stopped; the rest roll back in reverse.
    assert stops == ["stop:charlie", "stop:bravo", "stop:alpha"]
    assert "start:runtime" not in journal, "runtime must not start after a required component failed"
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
    assert engine.reuse_blocked

    event_types = [event.event_type for event in publisher.events]
    assert HostedApplicationEventType.COMPONENT_FAILED in event_types
    assert HostedApplicationEventType.APPLICATION_FAILED in event_types
    assert HostedApplicationEventType.INSTANCE_RELEASED in event_types

    # The real file lease was released: a fresh engine on the same run directory acquires it.
    journal_two: list[str] = []
    healthy = make_profile(runtime=JournalRuntime(journal_two))
    second, _, _ = build_engine(definition_of(healthy), tmp_path, instance_id="instance-002")
    await second.start()
    try:
        assert second.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
    finally:
        await second.stop(reason_code="test.cleanup")


async def test_r003_runtime_start_failure_rolls_back_components_and_releases_lease(
    tmp_path: Path,
) -> None:
    journal: list[str] = []

    class _FailingRuntime(JournalRuntime):
        async def start(self, context) -> None:  # type: ignore[override]
            self.journal.append("start:runtime")
            raise RuntimeError("runtime start failed")

    runtime = _FailingRuntime(journal)
    profile = make_profile(
        runtime=runtime,
        components=(
            _required(OrderedComponent("one", journal)),
            _required(OrderedComponent("two", journal), "one"),
        ),
    )
    engine, _, _ = build_engine(definition_of(profile), tmp_path)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    assert journal[:3] == ["start:one", "start:two", "start:runtime"]
    # Reverse of start order: the runtime stops before the pre-runtime components it uses.
    assert [item for item in journal if item.startswith("stop:")] == [
        "stop:runtime",
        "stop:two",
        "stop:one",
    ]
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.instance_lease_released
    assert diagnostics.context_closed


async def test_r004_degraded_component_is_reported_and_host_stays_ready(tmp_path: Path) -> None:
    journal: list[str] = []
    optional_failing = OrderedComponent("optional_cache", journal, fail_start=True)
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(
            HostedApplicationComponentRegistration(component=optional_failing),
            _required(OrderedComponent("core", journal)),
        ),
    )
    engine, publisher, _ = build_engine(definition_of(profile), tmp_path)
    await engine.start()
    try:
        health = engine.health_snapshot()
        assert health.ready is True
        assert health.degraded is True
        assert health.degraded_component_ids == ("optional_cache",)
        assert health.accepting_new_work is True
        assert health.blocking_component_ids == ()

        # The same reading is available to the application through the readiness service.
        readiness = engine.context.services.require(HostedApplicationReadinessService)
        assert readiness.snapshot().degraded_component_ids == ("optional_cache",)

        failed = publisher.of_type(HostedApplicationEventType.COMPONENT_FAILED)
        assert [event.payload.get("component_id") for event in failed] == ["optional_cache"]
    finally:
        await engine.stop(reason_code="test.cleanup")


async def test_r004_component_that_degrades_after_ready_is_reported_and_recovers(
    tmp_path: Path,
) -> None:
    journal: list[str] = []
    flaky = OrderedComponent("flaky_index", journal)
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(
            HostedApplicationComponentRegistration(component=flaky),
            _required(OrderedComponent("core", journal)),
        ),
    )
    engine, publisher, _ = build_engine(definition_of(profile), tmp_path)
    await engine.start()
    try:
        assert engine.health_snapshot().degraded is False

        flaky.healthy = False
        await _wait_for(lambda: engine.health_snapshot().degraded, "degradation reported")
        health = engine.health_snapshot()
        assert health.degraded_component_ids == ("flaky_index",)
        # An optional component degrades the host; it does not make it unready.
        assert health.ready is True and health.accepting_new_work is True
        assert health.blocking_component_ids == ()
        assert publisher.of_type(HostedApplicationEventType.COMPONENT_HEALTH_CHANGED)

        flaky.healthy = True
        await _wait_for(lambda: not engine.health_snapshot().degraded, "recovery reported")
        assert engine.health_snapshot().degraded_component_ids == ()
    finally:
        await engine.stop(reason_code="test.cleanup")


async def test_r004_required_component_not_ready_blocks_readiness(tmp_path: Path) -> None:
    journal: list[str] = []
    unhealthy = OrderedComponent("core", journal, healthy=False)
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(_required(unhealthy),),
    )
    engine, _, _ = build_engine(definition_of(profile), tmp_path)
    with pytest.raises(HostedApplicationStartupError):
        await engine.start()
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
    assert engine.health_snapshot().ready is False


async def test_r006_second_instance_is_rejected_by_the_instance_guard(tmp_path: Path) -> None:
    first_journal: list[str] = []
    second_journal: list[str] = []
    first_profile = make_profile(runtime=JournalRuntime(first_journal))
    first, _, _ = build_engine(definition_of(first_profile), tmp_path, instance_id="instance-001")
    await first.start()
    try:
        second_profile = make_profile(
            runtime=JournalRuntime(second_journal),
            components=(_required(OrderedComponent("late", second_journal)),),
        )
        second, second_events, _ = build_engine(definition_of(second_profile), tmp_path, instance_id="instance-002")
        with pytest.raises(HostedApplicationStartupError) as raised:
            await second.start()
        assert _has_cause(raised.value, HostedApplicationInstanceConflictError)
        assert second_journal == [], "a rejected instance must not start components or runtime"
        assert second.lifecycle_snapshot().state is HostedApplicationLifecycleState.CREATED
        assert [event.event_type for event in second_events.events] == [HostedApplicationEventType.INSTANCE_REJECTED]
        # The incumbent is unaffected.
        assert first.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
        assert first.health_snapshot().instance_ownership_valid is True
    finally:
        await first.stop(reason_code="test.cleanup")

    # After the incumbent released the lease, the rejected engine may start (reusable, not failed).
    await second.start()
    try:
        assert second.lifecycle_snapshot().state is HostedApplicationLifecycleState.READY
    finally:
        await second.stop(reason_code="test.cleanup")


async def test_r006_instance_held_by_another_os_process_is_rejected(tmp_path: Path) -> None:
    """The lease is an OS-level lock: a different process holding it rejects this engine."""
    holder = subprocess.Popen(  # noqa: S603 - fixed argv, test-owned script
        [
            sys.executable,
            "-u",
            str(_HOLDER_SCRIPT),
            str(tmp_path / "run"),
            APP_ID,
            str(_HOLDER_MAX_SECONDS),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert holder.stdout is not None
        (tmp_path / "run").mkdir(parents=True, exist_ok=True)
        banner = await asyncio.wait_for(asyncio.to_thread(holder.stdout.readline), timeout=_TEST_TIMEOUT_SECONDS)
        assert banner.strip() == "acquired"
        assert holder.pid != os.getpid()

        journal: list[str] = []
        profile = make_profile(
            runtime=JournalRuntime(journal),
            components=(_required(OrderedComponent("guarded", journal)),),
        )
        engine, events, _ = build_engine(definition_of(profile), tmp_path)
        with pytest.raises(HostedApplicationStartupError) as raised:
            await engine.start()
        assert _has_cause(raised.value, HostedApplicationInstanceConflictError)
        assert journal == []
        assert [event.event_type for event in events.events] == [HostedApplicationEventType.INSTANCE_REJECTED]
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=5.0)
        if holder.stdout is not None:
            holder.stdout.close()
