# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application hook coordinator (APP-HOST-2C)."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.contracts.hooks import (
    HostedApplicationHook,
    HostedApplicationHookMode,
    HostedApplicationHookPoint,
    hook_point_mode,
)
from intergrax.hosting.contracts.policies import LifecyclePolicy
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
    ) -> None:
        self._definition = definition
        self._lifecycle_policy = lifecycle_policy
        self._diagnostics = diagnostics
        self._observer_tasks = observer_tasks

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
                    self._invoke_hook(hook, context),
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
                self._invoke_hook_with_timeout(hook, context, timeout),
                phase=_hook_failure_phase(point),
                source_id=hook.hook_id,
            )

    async def execute_on_failure(
        self,
        context: HostedApplicationContext,
        *,
        primary_exc: BaseException,
    ) -> None:
        for hook in self.ordered_hooks(HostedApplicationHookPoint.ON_FAILURE):
            timeout = hook.timeout_seconds or (
                self._lifecycle_policy.default_observer_hook_timeout_seconds
            )
            try:
                await asyncio.wait_for(
                    self._invoke_hook(hook, context),
                    timeout=timeout,
                )
            except Exception as exc:
                self._diagnostics.record_secondary_failure(
                    phase=HostedApplicationFailurePhase.AFTER_START_OBSERVER,
                    source_kind="hook",
                    source_id=hook.hook_id,
                    exc=exc,
                    reason_code="on_failure_hook_failed",
                )

    async def _invoke_hook_with_timeout(
        self,
        hook: HostedApplicationHook,
        context: HostedApplicationContext,
        timeout: float,
    ) -> None:
        await asyncio.wait_for(self._invoke_hook(hook, context), timeout=timeout)

    async def _invoke_hook(
        self,
        hook: HostedApplicationHook,
        context: HostedApplicationContext,
    ) -> None:
        result = hook.handler(context)
        if inspect.isawaitable(result):
            await result
            return
        await asyncio.to_thread(self._run_sync_handler, hook.handler, context)

    @staticmethod
    def _run_sync_handler(
        handler: Callable[[HostedApplicationContext], Any],
        context: HostedApplicationContext,
    ) -> None:
        handler(context)


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
