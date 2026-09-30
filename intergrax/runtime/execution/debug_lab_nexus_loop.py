# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minimal NexusLoop materialization for debug/lab harnesses (EE owner zone)."""

from __future__ import annotations

from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)
from intergrax.contracts.run_trace_store import RunTraceReader
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry


def build_debug_minimal_nexus_loop(
    registry: AgentRegistry,
    *,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    trace_store: RunTraceReader | None = None,
    runtime_event_store: RuntimeEventPersistence | None = None,
) -> HostOrchestrationApplicationWiringTarget:
    return NexusLoop(
        registry,
        checkpoint_store=checkpoint_store,
        trace_store=trace_store,
        runtime_event_store=runtime_event_store,
    )


def build_lab_organization_nexus_loop(
    registry: AgentRegistry,
    *,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    notification_adapter: NotificationAdapter | None = None,
) -> HostOrchestrationApplicationWiringTarget:
    return NexusLoop(
        registry,
        checkpoint_store=checkpoint_store,
        notification_adapter=notification_adapter,
    )


__all__ = ["build_debug_minimal_nexus_loop", "build_lab_organization_nexus_loop"]
