# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from local_workspace_application.host.lifecycle import (
    HostLifecycleState,
    InvalidHostLifecycleTransitionError,
    LocalWorkspaceHostLifecycle,
)

pytestmark = pytest.mark.unit


def test_lifecycle_initial_state() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    assert lifecycle.state is HostLifecycleState.STARTING
    assert lifecycle.accepts_new_work is False


def test_lifecycle_valid_transitions() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    assert lifecycle.state is HostLifecycleState.READY
    assert lifecycle.accepts_new_work is True
    lifecycle.transition_to_stopping()
    assert lifecycle.state is HostLifecycleState.STOPPING
    assert lifecycle.accepts_new_work is False
    lifecycle.transition_to_stopped()
    assert lifecycle.state is HostLifecycleState.STOPPED


def test_lifecycle_invalid_transitions() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.transition_to_ready()
    with pytest.raises(InvalidHostLifecycleTransitionError):
        lifecycle.transition_to(HostLifecycleState.STARTING)
    lifecycle.transition_to_stopping()
    lifecycle.transition_to_stopped()
    with pytest.raises(InvalidHostLifecycleTransitionError):
        lifecycle.transition_to(HostLifecycleState.READY)


def test_readiness_blocks_on_required_unhealthy_component() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.register_component("runtime", enabled=True, required=True, healthy=False)
    lifecycle.transition_to_ready()
    assert lifecycle.is_ready() is False


def test_readiness_ignores_optional_unhealthy_component() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.register_component("runtime", enabled=True, required=True, healthy=True)
    lifecycle.register_component("scheduler", enabled=True, required=False, healthy=False)
    lifecycle.transition_to_ready()
    assert lifecycle.is_ready() is True


def test_readiness_ignores_disabled_optional_component() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.register_component("runtime", enabled=True, required=True, healthy=True)
    lifecycle.register_component(
        "interaction_intake",
        enabled=False,
        required=False,
        healthy=False,
    )
    lifecycle.transition_to_ready()
    assert lifecycle.is_ready() is True


@pytest.mark.parametrize(
    "transition",
    [
        HostLifecycleState.STOPPING,
        HostLifecycleState.STOPPED,
        HostLifecycleState.FAILED,
    ],
)
def test_non_ready_states_reject_work(transition: HostLifecycleState) -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.register_component("runtime", enabled=True, required=True, healthy=True)
    lifecycle.transition_to_ready()
    if transition is HostLifecycleState.STOPPING:
        lifecycle.transition_to_stopping()
    elif transition is HostLifecycleState.STOPPED:
        lifecycle.transition_to_stopping()
        lifecycle.transition_to_stopped()
    else:
        lifecycle.transition_to_failed()
    assert lifecycle.accepts_new_work is False
