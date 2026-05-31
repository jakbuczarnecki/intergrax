# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.bootstrap import bootstrap_runtime_plugins
from intergrax.runtime.plugins.contract import RuntimePlugin
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase

pytestmark = pytest.mark.gate


def test_bootstrap_runtime_plugins_registers_shutdown():
    shutdown_called = {"value": False}

    def _register(_event_bus, _hooks, _policy) -> None:
        return None

    def _shutdown() -> None:
        shutdown_called["value"] = True

    plugin = RuntimePlugin(
        plugin_id="test.plugin",
        version="1.0.0",
        register=_register,
        on_shutdown=_shutdown,
    )
    result = bootstrap_runtime_plugins(
        [plugin],
        event_bus=RuntimeEventBus(record_history=False),
        hook_registry=HookRegistry(),
    )
    assert len(result.shutdown_callbacks) == 1
    result.shutdown_callbacks[0]()
    assert shutdown_called["value"] is True


def test_default_lab_plugins_subscribe_without_error():
    from intergrax.runtime.plugins.default_plugins import default_lab_plugins

    bus = RuntimeEventBus(record_history=False)
    bootstrap_runtime_plugins(
        default_lab_plugins(),
        event_bus=bus,
        hook_registry=HookRegistry(),
    )
    event = RuntimeEvent(
        task_id="t1",
        run_id="t1",
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
    )
    bus.record(event)
