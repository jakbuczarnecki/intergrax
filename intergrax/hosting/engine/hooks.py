# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application hook coordinator (APP-HOST-2C)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.contracts.hooks import (
    HostedApplicationHook,
    HostedApplicationHookMode,
    HostedApplicationHookPoint,
    hook_point_mode,
)
from intergrax.hosting.contracts.policies import LifecyclePolicy
from intergrax.hosting.engine.callbacks import invoke_callback
from intergrax.hosting.engine.definition import HostedApplicationDefinition
from intergrax.hosting.engine.diagnostics import DiagnosticsRecorder, HostedApplicationFailurePhase
from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry
from intergrax.hosting.errors import HostedApplicationHookError


class HookCoordinator:
    """Deterministic hook execution for one hosted application engine instance."""

    def __init__(
        self,
        definition: HostedApplicationDefinition,
        lifecycle_policy: LifecyclePolicy,
        diagnostics: DiagnosticsRecorder,
        observer_tasks: ObserverTaskRegistry,
        publish_event: Callable[[HostedApplicationEvent], Awaitable[None]],
    ) -> None:
        self._definition = definition
        self._lifecycle_policy = lifecycle_policy
        self._diagnostics = diagnostics
        self._observer_tasks = observer_tasks
        self._publish_event = publish_event

    def ordered_hooks(self, point: HostedApplicationHookPoint) -> tuple[HostedApplicationHook, ...]:
        hooks = list(enumerate(self._definition.hook_registrations.get(point, ())))
        hooks.sort(key=lambda item: (item[1].priority, item[1].source_id, item[0]))
        return tuple(hook for _, hook in hooks)

    async def execute_blocking(
        self,
        point: HostedApplicationHookPoint,
        context: HostedApplicationContext,
    ) -> None:
        for hook in self.ordered_hooks(point):
            timeout = hook.timeout_seconds or (
                self._lifecycle_policy.default_blocking_hook_timeout_seconds
                if hook_point_mode(point) is HostedApplicationHookMode.BLOCKING
                else self._lifecycle_policy.default_observer_hook_timeout_seconds
            )
            try:
                await asyncio.wait_for(
                    self._invoke_hook_tracked(hook, point, context),
                    timeout=timeout,
                )
            except Exception as exc:
                phase = _hook_failure_phase(point)
                if point in {
                    HostedApplicationHookPoint.BEFORE_START,
                    HostedApplicationHookPoint.BEFORE_READY,
                }:
                    self._diagnostics.record_primary_failure(
                        phase=phase,
                        source_kind="hook",
                        source_id=hook.hook_id,
                        exc=exc,
                        reason_code="hook_failed",
                    )
                    raise HostedApplicationHookError(
                        f"blocking hook failed: {hook.hook_id}"
                    ) from exc
                self._diagnostics.record_secondary_failure(
                    phase=phase,
                    source_kind="hook",
                    source_id=hook.hook_id,
                    exc=exc,
                    reason_code="hook_failed",
                )

    def schedule_observers(
        self,
        point: HostedApplicationHookPoint,
        context: HostedApplicationContext,
    ) -> None:
        for hook in self.ordered_hooks(point):
            timeout = hook.timeout_seconds or (
                self._lifecycle_policy.default_observer_hook_timeout_seconds
            )
            self._observer_tasks.schedule(
                self._invoke_hook_with_timeout_tracked(hook, point, context, timeout),
                phase=_hook_failure_phase(point),
                source_id=hook.hook_id,
            )

    def schedule_on_failure(
        self,
        context: HostedApplicationContext,
    ) -> None:
        """Schedule on_failure observer hooks without blocking rollback."""
        self.schedule_observers(HostedApplicationHookPoint.ON_FAILURE, context)

    async def _invoke_hook_with_timeout_tracked(
        self,
        hook: HostedApplicationHook,
        point: HostedApplicationHookPoint,
        context: HostedApplicationContext,
        timeout: float,
    ) -> None:
        await asyncio.wait_for(
            self._invoke_hook_tracked(hook, point, context),
            timeout=timeout,
        )

    async def _invoke_hook_tracked(
        self,
        hook: HostedApplicationHook,
        point: HostedApplicationHookPoint,
        context: HostedApplicationContext,
    ) -> None:
        await self._publish_hook_event(
            context,
            HostedApplicationEventType.HOOK_STARTED,
            hook,
            point,
        )
        try:
            await invoke_callback(hook.handler, context)
        except Exception as exc:
            await self._publish_hook_event(
                context,
                HostedApplicationEventType.HOOK_FAILED,
                hook,
                point,
            )
            raise exc
        await self._publish_hook_event(
            context,
            HostedApplicationEventType.HOOK_COMPLETED,
            hook,
            point,
        )

    async def _publish_hook_event(
        self,
        context: HostedApplicationContext,
        event_type: HostedApplicationEventType,
        hook: HostedApplicationHook,
        point: HostedApplicationHookPoint,
    ) -> None:
        try:
            await self._publish_event(
                HostedApplicationEvent(
                    event_type=event_type,
                    application_id=context.application_id,
                    instance_id=context.instance_id,
                    lifecycle_state=context.lifecycle.snapshot().state,
                    payload={
                        "hook_id": hook.hook_id,
                        "hook_point": point.value,
                        "handler_id": hook.handler_id or "",
                        "source_id": hook.source_id,
                        "mode": hook_point_mode(point).value,
                    },
                )
            )
        except Exception as exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.EVENT_PUBLISH,
                source_kind="hook_event",
                source_id=hook.hook_id,
                exc=exc,
                reason_code="hook_event_publish_failed",
            )


def _hook_failure_phase(point: HostedApplicationHookPoint) -> HostedApplicationFailurePhase:
    mapping = {
        HostedApplicationHookPoint.BEFORE_START: HostedApplicationFailurePhase.BEFORE_START_HOOK,
        HostedApplicationHookPoint.BEFORE_READY: HostedApplicationFailurePhase.BEFORE_READY_HOOK,
        HostedApplicationHookPoint.BEFORE_STOP: HostedApplicationFailurePhase.BEFORE_STOP_HOOK,
        HostedApplicationHookPoint.AFTER_START: HostedApplicationFailurePhase.AFTER_START_OBSERVER,
        HostedApplicationHookPoint.AFTER_READY: HostedApplicationFailurePhase.AFTER_START_OBSERVER,
        HostedApplicationHookPoint.AFTER_STOP: HostedApplicationFailurePhase.AFTER_STOP_OBSERVER,
        HostedApplicationHookPoint.ON_FAILURE: HostedApplicationFailurePhase.AFTER_START_OBSERVER,
    }
    return mapping[point]
