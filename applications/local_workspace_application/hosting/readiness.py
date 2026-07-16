# © Artur Czarnecki. All rights reserved.

"""Hosted LKW readiness bridge onto platform hosting services (APP-HOST-8B)."""

from __future__ import annotations

from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.engine.health import HostedApplicationReadinessService
from local_workspace_application.host.readiness import (
    LocalWorkspaceComponentReadiness,
    LocalWorkspaceReadinessSnapshot,
)

_STATE_MAP: dict[HostedApplicationLifecycleState, str] = {
    HostedApplicationLifecycleState.CREATED: "starting",
    HostedApplicationLifecycleState.STARTING: "starting",
    HostedApplicationLifecycleState.READY: "ready",
    HostedApplicationLifecycleState.STOPPING: "stopping",
    HostedApplicationLifecycleState.STOPPED: "stopped",
    HostedApplicationLifecycleState.FAILED: "failed",
}


class _HostedLocalWorkspaceReadiness:
    """Private adapter projecting platform readiness into the LKW HTTP contract."""

    def __init__(self, context: HostedApplicationContext) -> None:
        self._context = context
        self._readiness = context.services.require(HostedApplicationReadinessService)

    def readiness_snapshot(self) -> LocalWorkspaceReadinessSnapshot:
        lifecycle = self._context.lifecycle.snapshot()
        health = self._readiness.snapshot()

        ready = (
            lifecycle.state is HostedApplicationLifecycleState.READY
            and health.ready
        )
        accepts_new_work = (
            lifecycle.accepting_new_work
            and health.accepting_new_work
        )
        state = _STATE_MAP[lifecycle.state]
        detail = _hosted_readiness_detail(
            ready=ready,
            accepts_new_work=accepts_new_work,
            lifecycle_state=lifecycle.state,
            lifecycle_shutdown_requested=lifecycle.shutdown_requested,
            health_shutdown_requested=health.shutdown_requested,
            health_evaluation_failed=health.health_evaluation_failed,
            runtime_ready=health.runtime_ready,
            instance_ownership_valid=health.instance_ownership_valid,
            blocking_component_ids=health.blocking_component_ids,
        )
        rejection_error_id = _hosted_rejection_error_id(
            accepts_new_work=accepts_new_work,
            lifecycle_state=lifecycle.state,
            lifecycle_shutdown_requested=lifecycle.shutdown_requested,
            health_shutdown_requested=health.shutdown_requested,
        )
        components = tuple(
            LocalWorkspaceComponentReadiness(
                name=component.component_id,
                enabled=component.enabled,
                required=component.required,
                healthy=component.healthy,
                detail=component.safe_message or component.detail_code,
            )
            for component in sorted(
                health.component_snapshots,
                key=lambda item: item.component_id,
            )
        )
        return LocalWorkspaceReadinessSnapshot(
            ready=ready,
            accepts_new_work=accepts_new_work,
            state=state,
            detail=detail,
            rejection_error_id=rejection_error_id,
            components=components,
        )


def _hosted_rejection_error_id(
    *,
    accepts_new_work: bool,
    lifecycle_state: HostedApplicationLifecycleState,
    lifecycle_shutdown_requested: bool,
    health_shutdown_requested: bool,
) -> str:
    if accepts_new_work:
        return ""
    if lifecycle_state is HostedApplicationLifecycleState.STOPPING:
        return "lkw_host_stopping"
    if lifecycle_shutdown_requested or health_shutdown_requested:
        return "lkw_host_stopping"
    return "lkw_host_not_ready"


def _hosted_readiness_detail(
    *,
    ready: bool,
    accepts_new_work: bool,
    lifecycle_state: HostedApplicationLifecycleState,
    lifecycle_shutdown_requested: bool,
    health_shutdown_requested: bool,
    health_evaluation_failed: bool,
    runtime_ready: bool,
    instance_ownership_valid: bool,
    blocking_component_ids: tuple[str, ...],
) -> str:
    if ready and accepts_new_work:
        return "ready"
    if lifecycle_state in {
        HostedApplicationLifecycleState.CREATED,
        HostedApplicationLifecycleState.STARTING,
    }:
        return "host_state=starting"
    if lifecycle_state is HostedApplicationLifecycleState.STOPPING:
        return "host_state=stopping"
    if lifecycle_state is HostedApplicationLifecycleState.STOPPED:
        return "host_state=stopped"
    if lifecycle_state is HostedApplicationLifecycleState.FAILED:
        return "host_state=failed"
    if lifecycle_shutdown_requested or health_shutdown_requested:
        return "shutdown_requested"
    if health_evaluation_failed:
        return "health_evaluation_failed"
    if not runtime_ready:
        return "runtime_not_ready"
    if not instance_ownership_valid:
        return "instance_ownership_invalid"
    if blocking_component_ids:
        return "blocking_components"
    return "not_accepting_new_work"
