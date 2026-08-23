# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.bootstrap import bootstrap_runtime_plugins
from intergrax.runtime.plugins.compatibility import RuntimePluginCompatibilityError
from intergrax.runtime.plugins.contract import RuntimePlugin
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.schema.registry import RuntimeVersionInfo, current_runtime_version

pytestmark = pytest.mark.gate


def _bootstrap(plugins, *, event_bus=None, hook_registry=None):
    return bootstrap_runtime_plugins(
        plugins,
        event_bus=event_bus or RuntimeEventBus(record_history=False),
        hook_registry=hook_registry or HookRegistry(),
    )


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
    result = _bootstrap([plugin])
    assert len(result.shutdown_callbacks) == 1
    result.shutdown_callbacks[0]()
    assert shutdown_called["value"] is True


def test_default_lab_plugins_subscribe_without_error():
    from intergrax.runtime.plugins.default_plugins import default_lab_plugins

    bus = RuntimeEventBus(record_history=False)
    _bootstrap(default_lab_plugins(), event_bus=bus)
    event = RuntimeEvent(
        task_id="t1",
        run_id="t1",
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
    )
    bus.record(event)


def test_compatible_default_plugin_registers_successfully():
    register_called = {"value": False}

    def _register(_event_bus, _hooks, _policy) -> None:
        register_called["value"] = True

    plugin = RuntimePlugin(
        plugin_id="test.default_compat",
        version="1.0.0",
        register=_register,
    )
    _bootstrap([plugin])
    assert register_called["value"] is True


def test_compatible_schema_subset_registers():
    runtime = current_runtime_version()
    register_called = {"value": False}

    def _register(_event_bus, _hooks, _policy) -> None:
        register_called["value"] = True

    plugin = RuntimePlugin(
        plugin_id="test.schema_subset",
        version="1.0.0",
        compatible_runtime=RuntimeVersionInfo(
            contract_bundle=runtime.contract_bundle,
            supported_schemas=frozenset({"runtime_event.v1"}),
        ),
        register=_register,
    )
    _bootstrap([plugin])
    assert register_called["value"] is True


def test_contract_bundle_mismatch_rejects_before_register():
    runtime = current_runtime_version()
    register_called = {"value": False}

    def _register(_event_bus, _hooks, _policy) -> None:
        register_called["value"] = True

    plugin = RuntimePlugin(
        plugin_id="test.bundle_mismatch",
        version="1.0.0",
        compatible_runtime=RuntimeVersionInfo(
            contract_bundle="uaep-0.9",
            supported_schemas=runtime.supported_schemas,
        ),
        register=_register,
    )
    with pytest.raises(RuntimePluginCompatibilityError) as exc_info:
        _bootstrap([plugin])
    err = exc_info.value
    assert err.plugin_id == "test.bundle_mismatch"
    assert err.plugin_contract_bundle == "uaep-0.9"
    assert err.runtime_contract_bundle == runtime.contract_bundle
    assert register_called["value"] is False


def test_missing_required_schema_rejects_before_register():
    runtime = current_runtime_version()
    register_called = {"value": False}

    def _register(_event_bus, _hooks, _policy) -> None:
        register_called["value"] = True

    plugin = RuntimePlugin(
        plugin_id="test.missing_schema",
        version="1.0.0",
        compatible_runtime=RuntimeVersionInfo(
            contract_bundle=runtime.contract_bundle,
            supported_schemas=frozenset({"runtime_event.v99"}),
        ),
        register=_register,
    )
    with pytest.raises(RuntimePluginCompatibilityError) as exc_info:
        _bootstrap([plugin])
    err = exc_info.value
    assert err.plugin_id == "test.missing_schema"
    assert err.missing_schemas == frozenset({"runtime_event.v99"})
    assert register_called["value"] is False


def test_atomic_preflight_skips_all_register_on_incompatibility():
    runtime = current_runtime_version()
    first_register_called = {"value": False}

    def _first_register(_event_bus, _hooks, _policy) -> None:
        first_register_called["value"] = True

    valid_plugin = RuntimePlugin(
        plugin_id="test.valid_first",
        version="1.0.0",
        register=_first_register,
    )
    incompatible_plugin = RuntimePlugin(
        plugin_id="test.invalid_second",
        version="1.0.0",
        compatible_runtime=RuntimeVersionInfo(
            contract_bundle="uaep-0.9",
            supported_schemas=runtime.supported_schemas,
        ),
        register=lambda _eb, _hk, _pl: None,
    )
    with pytest.raises(RuntimePluginCompatibilityError):
        _bootstrap([valid_plugin, incompatible_plugin])
    assert first_register_called["value"] is False


def test_runtime_semver_difference_alone_does_not_reject():
    runtime = current_runtime_version()
    register_called = {"value": False}

    def _register(_event_bus, _hooks, _policy) -> None:
        register_called["value"] = True

    plugin = RuntimePlugin(
        plugin_id="test.semver_metadata",
        version="1.0.0",
        compatible_runtime=RuntimeVersionInfo(
            runtime_semver="99.88.77",
            contract_bundle=runtime.contract_bundle,
            supported_schemas=frozenset({"runtime_event.v1"}),
        ),
        register=_register,
    )
    _bootstrap([plugin])
    assert register_called["value"] is True
