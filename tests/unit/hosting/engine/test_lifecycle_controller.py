# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

import pytest

from intergrax.hosting import HostedApplicationLifecycleState
from intergrax.hosting.engine.lifecycle import HostedApplicationLifecycleController
from intergrax.hosting.errors import HostedApplicationLifecycleTransitionError
from tests.unit.hosting.engine._fakes import FixedClock

pytestmark = pytest.mark.unit


def test_all_valid_transitions() -> None:
    clock = FixedClock()
    controller = HostedApplicationLifecycleController(clock)
    controller.transition_to(HostedApplicationLifecycleState.STARTING)
    controller.transition_to(HostedApplicationLifecycleState.READY)
    controller.transition_to(HostedApplicationLifecycleState.STOPPING)
    controller.transition_to(HostedApplicationLifecycleState.STOPPED)
    assert controller.state is HostedApplicationLifecycleState.STOPPED


@pytest.mark.parametrize(
    ("from_state", "to_state"),
    [
        (HostedApplicationLifecycleState.CREATED, HostedApplicationLifecycleState.READY),
        (HostedApplicationLifecycleState.READY, HostedApplicationLifecycleState.STARTING),
        (HostedApplicationLifecycleState.STOPPED, HostedApplicationLifecycleState.STARTING),
        (HostedApplicationLifecycleState.FAILED, HostedApplicationLifecycleState.READY),
    ],
)
def test_invalid_transitions(from_state: HostedApplicationLifecycleState, to_state: HostedApplicationLifecycleState) -> None:
    clock = FixedClock()
    controller = HostedApplicationLifecycleController(clock)
    if from_state is not HostedApplicationLifecycleState.CREATED:
        _reach_state(controller, from_state)
    with pytest.raises(HostedApplicationLifecycleTransitionError):
        controller.transition_to(to_state)


def test_terminal_states_block_further_transitions() -> None:
    clock = FixedClock()
    controller = HostedApplicationLifecycleController(clock)
    controller.transition_to(HostedApplicationLifecycleState.STARTING)
    controller.transition_to(HostedApplicationLifecycleState.FAILED)
    with pytest.raises(HostedApplicationLifecycleTransitionError):
        controller.transition_to(HostedApplicationLifecycleState.STOPPING)


def test_transition_history_and_timezone_aware_timestamps() -> None:
    clock = FixedClock()
    controller = HostedApplicationLifecycleController(clock)
    controller.transition_to(HostedApplicationLifecycleState.STARTING, reason_code="boot")
    history = controller.transition_history()
    assert len(history) == 1
    assert history[0].reason_code == "boot"
    assert history[0].occurred_at.tzinfo is not None


def test_accepting_flag_independent_from_ready_state() -> None:
    clock = FixedClock()
    controller = HostedApplicationLifecycleController(clock)
    controller.transition_to(HostedApplicationLifecycleState.STARTING)
    controller.transition_to(HostedApplicationLifecycleState.READY)
    controller.set_accepting_new_work(False)
    snapshot = controller.snapshot()
    assert snapshot.state is HostedApplicationLifecycleState.READY
    assert snapshot.accepting_new_work is False


def test_concurrent_snapshot_reads() -> None:
    clock = FixedClock()
    controller = HostedApplicationLifecycleController(clock)
    controller.transition_to(HostedApplicationLifecycleState.STARTING)
    errors: list[Exception] = []

    def reader() -> None:
        try:
            for _ in range(100):
                controller.snapshot()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    controller.transition_to(HostedApplicationLifecycleState.READY)
    for thread in threads:
        thread.join()
    assert not errors


def _reach_state(controller: HostedApplicationLifecycleController, state: HostedApplicationLifecycleState) -> None:
    if state is HostedApplicationLifecycleState.CREATED:
        return
    if state is HostedApplicationLifecycleState.STARTING:
        controller.transition_to(HostedApplicationLifecycleState.STARTING)
        return
    if state is HostedApplicationLifecycleState.READY:
        controller.transition_to(HostedApplicationLifecycleState.STARTING)
        controller.transition_to(HostedApplicationLifecycleState.READY)
        return
    if state is HostedApplicationLifecycleState.STOPPING:
        _reach_state(controller, HostedApplicationLifecycleState.READY)
        controller.transition_to(HostedApplicationLifecycleState.STOPPING)
        return
    if state is HostedApplicationLifecycleState.STOPPED:
        _reach_state(controller, HostedApplicationLifecycleState.STOPPING)
        controller.transition_to(HostedApplicationLifecycleState.STOPPED)
        return
    if state is HostedApplicationLifecycleState.FAILED:
        controller.transition_to(HostedApplicationLifecycleState.STARTING)
        controller.transition_to(HostedApplicationLifecycleState.FAILED)
