# © Artur Czarnecki. All rights reserved.

"""Pyright-visible B6-R3 runtime plugin registration structural proofs."""

from __future__ import annotations

from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationRuntimeEventPort,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.plugins.bootstrap import bootstrap_runtime_plugins
from intergrax.runtime.plugins.contract import RuntimePlugin


def _accept_event_port(value: HostOrchestrationRuntimeEventPort) -> None:
    _ = value


def exercise_b6_r3_plugin_registration_structural_proofs() -> None:
    bus = RuntimeEventBus(record_history=False)
    _accept_event_port(bus)

    def _noop(event_bus: HostOrchestrationRuntimeEventPort) -> None:
        _ = event_bus

    bootstrap_runtime_plugins(
        [RuntimePlugin(plugin_id="proof.noop", version="1.0.0", register=_noop)],
        event_bus=bus,
    )
