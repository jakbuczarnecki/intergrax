# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application aggregate health and readiness (APP-HOST-2E)."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.hosting.contracts.components import HostedApplicationComponentHealth
from intergrax.hosting.contracts.context import HostedApplicationClock
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.engine.lifecycle import HostedApplicationLifecycleController
from intergrax.hosting.engine.ports import HostedApplicationInstanceLeasePort


class HostedApplicationHealthSnapshot(BaseModel):
    """Aggregate hosted application health and readiness snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    live: bool
    ready: bool
    degraded: bool
    accepting_new_work: bool
    runtime_ready: bool
    instance_ownership_valid: bool
    shutdown_requested: bool
    blocking_component_ids: tuple[str, ...] = ()
    degraded_component_ids: tuple[str, ...] = ()
    component_snapshots: tuple[HostedApplicationComponentHealth, ...] = ()
    last_evaluated_at: datetime


@runtime_checkable
class HostedApplicationReadinessService(Protocol):
    """Narrow readiness service exposed through hosted application context."""

    def snapshot(self) -> HostedApplicationHealthSnapshot: ...

    def accepts_new_work(self) -> bool: ...


@dataclass
class HostedApplicationHealthCoordinator:
    """Aggregate health and readiness evaluator for one engine instance."""

    lifecycle: HostedApplicationLifecycleController
    clock: HostedApplicationClock
    _runtime_ready: bool = False
    _lease: HostedApplicationInstanceLeasePort | None = None
    _component_health: dict[str, HostedApplicationComponentHealth] = field(default_factory=dict)
    _mark_not_ready_failed: frozenset[str] = frozenset()
    _degraded_component_ids: frozenset[str] = frozenset()
    _last_snapshot: HostedApplicationHealthSnapshot | None = None
    _poll_task: asyncio.Task[None] | None = None
    _poll_shutdown: asyncio.Event = field(default_factory=asyncio.Event)
    _poll_interval_seconds: float = 5.0
    _poll_sleeper: object | None = None

    def configure_polling(
        self,
        *,
        interval_seconds: float,
        sleeper: object | None = None,
    ) -> None:
        self._poll_interval_seconds = interval_seconds
        self._poll_sleeper = sleeper

    def set_runtime_ready(self, value: bool) -> None:
        self._runtime_ready = value

    def set_lease(self, lease: HostedApplicationInstanceLeasePort | None) -> None:
        self._lease = lease

    def update_component_health(
        self,
        health: dict[str, HostedApplicationComponentHealth],
        *,
        mark_not_ready_failed: frozenset[str],
        degraded_component_ids: frozenset[str],
    ) -> HostedApplicationHealthSnapshot:
        self._component_health = dict(health)
        self._mark_not_ready_failed = mark_not_ready_failed
        self._degraded_component_ids = degraded_component_ids
        return self.refresh_once()

    def refresh_once(self) -> HostedApplicationHealthSnapshot:
        lifecycle = self.lifecycle.snapshot()
        instance_valid = self._lease.is_valid() if self._lease is not None else False
        shutdown_requested = lifecycle.shutdown_requested
        live = lifecycle.state not in {
            HostedApplicationLifecycleState.STOPPED,
            HostedApplicationLifecycleState.FAILED,
        }
        blocking: list[str] = []
        degraded: list[str] = sorted(self._degraded_component_ids)
        for component_id, component_health in sorted(self._component_health.items()):
            registration_required = component_health.required and component_health.enabled
            if registration_required and (not component_health.healthy or not component_health.ready):
                blocking.append(component_id)
            if component_id in self._mark_not_ready_failed:
                blocking.append(component_id)
        blocking_ids = tuple(sorted(set(blocking)))
        ready = (
            lifecycle.state is HostedApplicationLifecycleState.READY
            and self._runtime_ready
            and instance_valid
            and not shutdown_requested
            and not blocking_ids
        )
        accepting = ready
        snapshot = HostedApplicationHealthSnapshot(
            live=live,
            ready=ready,
            degraded=bool(degraded),
            accepting_new_work=accepting,
            runtime_ready=self._runtime_ready,
            instance_ownership_valid=instance_valid,
            shutdown_requested=shutdown_requested,
            blocking_component_ids=blocking_ids,
            degraded_component_ids=tuple(degraded),
            component_snapshots=tuple(
                self._component_health[component_id]
                for component_id in sorted(self._component_health)
            ),
            last_evaluated_at=self.clock.now(),
        )
        self._last_snapshot = snapshot
        self.lifecycle.set_accepting_new_work(accepting)
        return snapshot

    def snapshot(self) -> HostedApplicationHealthSnapshot:
        if self._last_snapshot is None:
            return self.refresh_once()
        return self._last_snapshot

    def accepts_new_work(self) -> bool:
        return self.snapshot().accepting_new_work

    async def start_polling(
        self,
        refresh_callback: Callable[[], Awaitable[None]] | Callable[[], None],
    ) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            return
        self._poll_shutdown.clear()
        self._poll_task = asyncio.create_task(self._poll_loop(refresh_callback))

    async def stop_polling(self) -> None:
        if self._poll_task is None:
            return
        self._poll_shutdown.set()
        await self._poll_task
        self._poll_task = None

    async def _poll_loop(
        self,
        refresh_callback: Callable[[], Awaitable[None]] | Callable[[], None],
    ) -> None:
        while not self._poll_shutdown.is_set():
            result = refresh_callback()
            if inspect.isawaitable(result):
                await result
            if self._poll_sleeper is not None and hasattr(self._poll_sleeper, "sleep"):
                await self._poll_sleeper.sleep(self._poll_interval_seconds)  # type: ignore[attr-defined]
            else:
                try:
                    await asyncio.wait_for(
                        self._poll_shutdown.wait(),
                        timeout=self._poll_interval_seconds,
                    )
                    break
                except TimeoutError:
                    continue
