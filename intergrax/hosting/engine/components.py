# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application component coordinator (APP-HOST-2D)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from pydantic import JsonValue

from intergrax.hosting.contracts.components import (
    HostedApplicationComponent,
    HostedApplicationComponentHealth,
    HostedApplicationComponentRegistration,
    HostedApplicationComponentState,
)
from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.contracts.events import HostedApplicationEventType
from intergrax.hosting.contracts.policies import ComponentFailureAction, LifecyclePolicy
from intergrax.hosting.engine.definition import HostedApplicationDefinition, ResolvedComponentRegistration
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder, HostedApplicationFailurePhase
from intergrax.hosting.errors import HostedApplicationComponentError


class ComponentStartOutcome(str, Enum):
    STARTED = "started"
    SKIPPED_DEPENDENCY = "skipped_dependency"
    FAILED = "failed"


@dataclass
class ComponentCoordinator:
    """DAG-based component startup and shutdown coordinator."""

    definition: HostedApplicationDefinition
    lifecycle_policy: LifecyclePolicy
    diagnostics: DiagnosticsRecorder
    publish_event: object
    _started: set[str] = field(default_factory=set)
    _health: dict[str, HostedApplicationComponentHealth] = field(default_factory=dict)
    _mark_not_ready_failed: set[str] = field(default_factory=set)
    _degraded: set[str] = field(default_factory=set)
    _startup_fatal: BaseException | None = field(default=None, repr=False)

    @property
    def started_component_ids(self) -> tuple[str, ...]:
        return tuple(
            component_id
            for component_id in self.definition.component_start_order
            if component_id in self._started
        )

    @property
    def startup_fatal_error(self) -> BaseException | None:
        return self._startup_fatal

    def component_health(self) -> dict[str, HostedApplicationComponentHealth]:
        return {
            component_id: self._health[component_id]
            for component_id in self.definition.component_start_order
            if component_id in self._health
        }

    async def start_phase(
        self,
        context: HostedApplicationContext,
        component_ids: tuple[str, ...],
    ) -> None:
        if not component_ids:
            return
        levels = self._levels_for_ids(component_ids)
        for level in levels:
            await self._start_level(context, level)
            if self._startup_fatal is not None:
                raise self._startup_fatal

    async def stop_started(self, context: HostedApplicationContext) -> None:
        stop_ids = [
            component_id
            for component_id in self.definition.component_stop_order
            if component_id in self._started
        ]
        if not stop_ids:
            return
        levels = self._reverse_levels_for_ids(stop_ids)
        for level in levels:
            await self._stop_level(context, level)

    async def _start_level(
        self,
        context: HostedApplicationContext,
        level: tuple[str, ...],
    ) -> None:
        semaphore = asyncio.Semaphore(self.lifecycle_policy.component_startup_concurrency)
        tasks = [
            asyncio.create_task(self._start_one(context, component_id, semaphore))
            for component_id in level
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, HostedApplicationComponentError):
                self._startup_fatal = result
                raise result
            if isinstance(result, Exception):
                error = HostedApplicationComponentError("component startup failed")
                self._startup_fatal = error
                raise error from result

    async def _start_one(
        self,
        context: HostedApplicationContext,
        component_id: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            resolved = self.definition.enabled_components[component_id]
            registration = resolved.registration
            if not self._dependencies_started(resolved):
                await self._handle_dependency_skip(context, component_id, resolved)
                return
            await self._publish_component_event(
                context,
                HostedApplicationEventType.COMPONENT_STARTING,
                component_id,
            )
            component = registration.component
            assert isinstance(component, HostedApplicationComponent)
            try:
                await asyncio.wait_for(
                    component.start(context),
                    timeout=registration.start_timeout_seconds,
                )
            except Exception as exc:
                await self._handle_start_failure(context, component_id, resolved, exc)
                return
            self._started.add(component_id)
            self.diagnostics.mark_component_started(component_id)
            health = await self._collect_health(context, resolved)
            self._health[component_id] = health
            await self._publish_component_event(
                context,
                HostedApplicationEventType.COMPONENT_STARTED,
                component_id,
            )

    async def _handle_dependency_skip(
        self,
        context: HostedApplicationContext,
        component_id: str,
        resolved: ResolvedComponentRegistration,
    ) -> None:
        action = resolved.registration.failure_action or ComponentFailureAction.FAIL_HOST
        required = resolved.registration.required
        await self._publish_component_event(
            context,
            HostedApplicationEventType.COMPONENT_FAILED,
            component_id,
        )
        failed_health = HostedApplicationComponentHealth(
            component_id=component_id,
            enabled=True,
            required=required,
            state=HostedApplicationComponentState.FAILED,
            healthy=False,
            ready=False,
            detail_code="dependency_not_started",
        )
        self._health[component_id] = failed_health

        if action is ComponentFailureAction.IGNORE_WITH_DIAGNOSTIC:
            self.diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.COMPONENT_START,
                source_kind="component",
                source_id=component_id,
                exc=RuntimeError("dependency_not_started"),
                reason_code="dependency_skipped_ignored",
            )
            return

        if action is ComponentFailureAction.MARK_DEGRADED:
            self._degraded.add(component_id)
            self._health[component_id] = failed_health.model_copy(
                update={"state": HostedApplicationComponentState.DEGRADED},
            )
            return

        if action is ComponentFailureAction.MARK_NOT_READY:
            self._mark_not_ready_failed.add(component_id)
            return

        if required or action is ComponentFailureAction.FAIL_HOST:
            exc = HostedApplicationComponentError(
                f"required component skipped due to dependency failure: {component_id}"
            )
            self.diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.COMPONENT_START,
                source_kind="component",
                source_id=component_id,
                exc=exc,
                reason_code="dependency_not_started",
            )
            self._startup_fatal = exc
            raise exc

    async def _handle_start_failure(
        self,
        context: HostedApplicationContext,
        component_id: str,
        resolved: ResolvedComponentRegistration,
        exc: Exception,
    ) -> None:
        action = resolved.registration.failure_action or ComponentFailureAction.FAIL_HOST
        await self._publish_component_event(
            context,
            HostedApplicationEventType.COMPONENT_FAILED,
            component_id,
        )
        if action is ComponentFailureAction.IGNORE_WITH_DIAGNOSTIC:
            self.diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.COMPONENT_START,
                source_kind="component",
                source_id=component_id,
                exc=exc,
                reason_code="component_ignored",
            )
            return
        if action is ComponentFailureAction.MARK_DEGRADED:
            self._degraded.add(component_id)
            self._health[component_id] = _failed_health(
                resolved.registration,
                state=HostedApplicationComponentState.DEGRADED,
            )
            return
        if action is ComponentFailureAction.MARK_NOT_READY:
            self._mark_not_ready_failed.add(component_id)
            self._health[component_id] = _failed_health(resolved.registration)
            return
        self.diagnostics.record_primary_failure(
            phase=HostedApplicationFailurePhase.COMPONENT_START,
            source_kind="component",
            source_id=component_id,
            exc=exc,
            reason_code="component_start_failed",
        )
        error = HostedApplicationComponentError(
            f"required component failed to start: {component_id}"
        )
        self._startup_fatal = error
        raise error from exc

    def _dependencies_started(self, resolved: ResolvedComponentRegistration) -> bool:
        for dependency in resolved.registration.dependencies:
            if dependency not in self._started:
                return False
        return True

    async def _stop_level(
        self,
        context: HostedApplicationContext,
        level: tuple[str, ...],
    ) -> None:
        semaphore = asyncio.Semaphore(self.lifecycle_policy.component_startup_concurrency)
        await asyncio.gather(
            *(
                self._stop_one(context, component_id, semaphore)
                for component_id in level
            ),
            return_exceptions=True,
        )

    async def _stop_one(
        self,
        context: HostedApplicationContext,
        component_id: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            if component_id not in self._started:
                return
            registration = self.definition.enabled_components[component_id].registration
            await self._publish_component_event(
                context,
                HostedApplicationEventType.COMPONENT_STOPPING,
                component_id,
            )
            component = registration.component
            assert isinstance(component, HostedApplicationComponent)
            try:
                await asyncio.wait_for(
                    component.stop(context),
                    timeout=registration.stop_timeout_seconds,
                )
            except Exception as exc:
                self.diagnostics.record_secondary_failure(
                    phase=HostedApplicationFailurePhase.COMPONENT_STOP,
                    source_kind="component",
                    source_id=component_id,
                    exc=exc,
                    reason_code="component_stop_failed",
                )
            else:
                self._health[component_id] = HostedApplicationComponentHealth(
                    component_id=component_id,
                    enabled=True,
                    required=registration.required,
                    state=HostedApplicationComponentState.STOPPED,
                    healthy=True,
                    ready=False,
                )
            self._started.discard(component_id)
            await self._publish_component_event(
                context,
                HostedApplicationEventType.COMPONENT_STOPPED,
                component_id,
            )

    async def refresh_component_health(
        self,
        context: HostedApplicationContext,
    ) -> dict[str, HostedApplicationComponentHealth]:
        updated: dict[str, HostedApplicationComponentHealth] = {}
        for component_id in self.definition.component_start_order:
            if component_id not in self._started:
                continue
            resolved = self.definition.enabled_components[component_id]
            health = await self._collect_health(context, resolved)
            previous = self._health.get(component_id)
            self._health[component_id] = health
            updated[component_id] = health
            if previous is not None and _health_meaningfully_changed(previous, health):
                await self._publish_component_event(
                    context,
                    HostedApplicationEventType.COMPONENT_HEALTH_CHANGED,
                    component_id,
                    payload={"healthy": health.healthy, "ready": health.ready},
                )
        return updated

    @property
    def mark_not_ready_component_ids(self) -> frozenset[str]:
        return frozenset(self._mark_not_ready_failed)

    @property
    def degraded_component_ids(self) -> frozenset[str]:
        return frozenset(self._degraded)

    async def _collect_health(
        self,
        context: HostedApplicationContext,
        resolved: ResolvedComponentRegistration,
    ) -> HostedApplicationComponentHealth:
        registration = resolved.registration
        component = registration.component
        assert isinstance(component, HostedApplicationComponent)
        try:
            raw = await asyncio.wait_for(
                component.health(context),
                timeout=registration.health_timeout_seconds,
            )
        except Exception:
            return _failed_health(registration)
        return _normalize_component_health(raw, registration)

    async def _publish_component_event(
        self,
        context: HostedApplicationContext,
        event_type: HostedApplicationEventType,
        component_id: str,
        *,
        payload: dict[str, JsonValue] | None = None,
    ) -> None:
        publisher = context.event_publisher
        from intergrax.hosting.contracts.events import HostedApplicationEvent

        event = HostedApplicationEvent(
            event_type=event_type,
            application_id=context.application_id,
            instance_id=context.instance_id,
            lifecycle_state=context.lifecycle.snapshot().state,
            payload={"component_id": component_id, **(payload or {})},
        )
        await publisher.publish(event)

    def _levels_for_ids(self, component_ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        id_set = set(component_ids)
        return tuple(
            tuple(component_id for component_id in level if component_id in id_set)
            for level in self.definition.component_dependency_levels
            if any(component_id in id_set for component_id in level)
        )

    def _reverse_levels_for_ids(
        self,
        component_ids: list[str],
    ) -> tuple[tuple[str, ...], ...]:
        id_set = set(component_ids)
        levels = [
            tuple(component_id for component_id in level if component_id in id_set)
            for level in reversed(self.definition.component_dependency_levels)
        ]
        return tuple(level for level in levels if level)


def _normalize_component_health(
    health: HostedApplicationComponentHealth,
    registration: HostedApplicationComponentRegistration,
) -> HostedApplicationComponentHealth:
    component_id = registration.component_id or ""
    return health.model_copy(
        update={
            "component_id": component_id,
            "enabled": registration.enabled,
            "required": registration.required,
        }
    )


def _failed_health(
    registration: object,
    *,
    state: HostedApplicationComponentState = HostedApplicationComponentState.FAILED,
) -> HostedApplicationComponentHealth:
    reg = registration
    if isinstance(registration, ResolvedComponentRegistration):
        reg = registration.registration
    assert isinstance(reg, HostedApplicationComponentRegistration)
    return HostedApplicationComponentHealth(
        component_id=reg.component_id or "",
        enabled=reg.enabled,
        required=reg.required,
        state=state,
        healthy=False,
        ready=False,
        detail_code="component_failed",
    )


def _health_meaningfully_changed(
    previous: HostedApplicationComponentHealth,
    current: HostedApplicationComponentHealth,
) -> bool:
    return (
        previous.healthy != current.healthy
        or previous.ready != current.ready
        or previous.state != current.state
        or previous.detail_code != current.detail_code
    )
