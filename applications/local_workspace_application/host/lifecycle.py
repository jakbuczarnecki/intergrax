# © Artur Czarnecki. All rights reserved.

"""LKW application host lifecycle and readiness semantics (LKW.6A)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi import FastAPI

from intergrax.applications._shared.fastapi_lifespan import combine_lifespans


class HostLifecycleState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class InvalidHostLifecycleTransitionError(ValueError):
    """Raised when a host lifecycle transition is not allowed."""


_VALID_TRANSITIONS: dict[HostLifecycleState, frozenset[HostLifecycleState]] = {
    HostLifecycleState.STARTING: frozenset({HostLifecycleState.READY, HostLifecycleState.FAILED}),
    HostLifecycleState.READY: frozenset({HostLifecycleState.STOPPING, HostLifecycleState.FAILED}),
    HostLifecycleState.STOPPING: frozenset({HostLifecycleState.STOPPED, HostLifecycleState.FAILED}),
    HostLifecycleState.STOPPED: frozenset(),
    HostLifecycleState.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    enabled: bool
    required: bool
    healthy: bool
    detail: str = ""


@dataclass
class LocalWorkspaceHostLifecycle:
    """Minimal always-on daemon lifecycle for the LKW application host."""

    def __init__(self) -> None:
        self._state = HostLifecycleState.STARTING
        self._components: dict[str, ComponentHealth] = {}
        self._executor_available = False

    @property
    def state(self) -> HostLifecycleState:
        return self._state

    @property
    def accepts_new_work(self) -> bool:
        return self._state is HostLifecycleState.READY and self._executor_available

    def set_executor_available(self, available: bool) -> None:
        self._executor_available = available

    def register_component(
        self,
        name: str,
        *,
        enabled: bool,
        required: bool,
        healthy: bool = True,
        detail: str = "",
    ) -> None:
        self._components[name] = ComponentHealth(
            name=name,
            enabled=enabled,
            required=required,
            healthy=healthy,
            detail=detail,
        )

    def update_component(self, name: str, *, healthy: bool, detail: str = "") -> None:
        current = self._components.get(name)
        if current is None:
            raise KeyError(f"unknown lifecycle component: {name}")
        self._components[name] = ComponentHealth(
            name=current.name,
            enabled=current.enabled,
            required=current.required,
            healthy=healthy,
            detail=detail,
        )

    def component_health(self) -> tuple[ComponentHealth, ...]:
        return tuple(self._components[name] for name in sorted(self._components))

    def transition_to(self, target: HostLifecycleState) -> None:
        allowed = _VALID_TRANSITIONS.get(self._state, frozenset())
        if target not in allowed:
            raise InvalidHostLifecycleTransitionError(
                f"cannot transition host lifecycle from {self._state.value} to {target.value}"
            )
        self._state = target

    def transition_to_starting(self) -> None:
        if self._state is not HostLifecycleState.STARTING:
            raise InvalidHostLifecycleTransitionError("host lifecycle is already past STARTING")

    def transition_to_ready(self) -> None:
        self.transition_to(HostLifecycleState.READY)

    def transition_to_stopping(self) -> None:
        if self._state is HostLifecycleState.READY:
            self.transition_to(HostLifecycleState.STOPPING)
        elif self._state is HostLifecycleState.STARTING:
            self.transition_to(HostLifecycleState.FAILED)

    def transition_to_stopped(self) -> None:
        if self._state is HostLifecycleState.STOPPING:
            self.transition_to(HostLifecycleState.STOPPED)

    def transition_to_failed(self) -> None:
        if self._state in {HostLifecycleState.STOPPED, HostLifecycleState.FAILED}:
            return
        if self._state is HostLifecycleState.STOPPING:
            self._state = HostLifecycleState.FAILED
            return
        if self._state is HostLifecycleState.READY:
            self._state = HostLifecycleState.FAILED
            return
        if self._state is HostLifecycleState.STARTING:
            self._state = HostLifecycleState.FAILED

    def is_ready(self) -> bool:
        if self._state is not HostLifecycleState.READY:
            return False
        if not self._executor_available:
            return False
        for component in self._components.values():
            if component.enabled and component.required and not component.healthy:
                return False
        return True

    def readiness_detail(self) -> str:
        if self._state is not HostLifecycleState.READY:
            return f"host_state={self._state.value}"
        if not self._executor_available:
            return "task_executor_unavailable"
        for component in self._components.values():
            if component.enabled and component.required and not component.healthy:
                return f"component_unhealthy:{component.name}"
        return "ready"


def make_lkw_daemon_lifespan(lifecycle: LocalWorkspaceHostLifecycle) -> Any:
    """FastAPI lifespan that transitions LKW host lifecycle on startup/shutdown."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            lifecycle.transition_to_ready()
            yield
        except BaseException:
            lifecycle.transition_to_failed()
            raise
        finally:
            if lifecycle.state is HostLifecycleState.READY:
                lifecycle.transition_to_stopping()
            if lifecycle.state is HostLifecycleState.STOPPING:
                lifecycle.transition_to_stopped()

    return _lifespan


def apply_lkw_daemon_lifespan(app: FastAPI, lifecycle: LocalWorkspaceHostLifecycle) -> FastAPI:
    """Append LKW daemon lifecycle so STOPPING happens before other lifespans unwind."""
    existing = app.router.lifespan_context
    app.router.lifespan_context = combine_lifespans(existing, make_lkw_daemon_lifespan(lifecycle))
    return app
