# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application engine orchestration (APP-HOST-2F)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from intergrax.hosting.contracts.context import (
    HostedApplicationClock,
    HostedApplicationContext,
    HostedApplicationEventPublisher,
    HostedApplicationLogger,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.contracts.hooks import HostedApplicationHookPoint
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleState,
    HostedApplicationShutdownCoordinator,
)
from intergrax.hosting.engine.components import ComponentCoordinator
from intergrax.hosting.engine.definition import HostedApplicationDefinition
from intergrax.hosting.engine.diagnostics import (
    DiagnosticsRecorder,
    HostedApplicationDiagnosticSnapshot,
    HostedApplicationEngineTerminalResult,
    HostedApplicationFailurePhase,
)
from intergrax.hosting.engine.health import (
    HostedApplicationHealthCoordinator,
    HostedApplicationHealthSnapshot,
    HostedApplicationReadinessService,
)
from intergrax.hosting.engine.hooks import HookCoordinator
from intergrax.hosting.engine.lifecycle import HostedApplicationLifecycleController
from intergrax.hosting.engine.observer_tasks import ObserverTaskRegistry
from intergrax.hosting.engine.ports import (
    HostedApplicationInstanceGuardPort,
    HostedApplicationInstanceIdentity,
    HostedApplicationInstanceLeasePort,
    HostedApplicationRuntime,
)
from intergrax.hosting.engine.runtime import invoke_application_factory
from intergrax.hosting.errors import (
    HostedApplicationEngineError,
    HostedApplicationStartupError,
)
from intergrax.hosting.eventing import HostingEventDispatcher
from intergrax.hosting.services import HostedApplicationServiceRegistry


class _ReadinessServiceAdapter:
    def __init__(self, coordinator: HostedApplicationHealthCoordinator) -> None:
        self._coordinator = coordinator

    def snapshot(self) -> HostedApplicationHealthSnapshot:
        return self._coordinator.snapshot()

    def accepts_new_work(self) -> bool:
        return self._coordinator.accepts_new_work()


@dataclass
class HostedApplicationEngine:
    """Coordinates one hosted application lifecycle instance."""

    definition: HostedApplicationDefinition
    instance_id: str
    paths: HostedApplicationPaths
    process_identity: HostedApplicationProcessIdentity
    clock: HostedApplicationClock
    logger: HostedApplicationLogger
    shutdown: HostedApplicationShutdownCoordinator
    event_publisher: HostedApplicationEventPublisher
    instance_guard: HostedApplicationInstanceGuardPort
    health_poll_interval_seconds: float = 5.0
    health_poll_sleeper: object | None = None

    def __post_init__(self) -> None:
        self._lifecycle = HostedApplicationLifecycleController(self.clock)
        self._services = HostedApplicationServiceRegistry()
        self._diagnostics = DiagnosticsRecorder(
            clock=self.clock,
            application_id=self.definition.profile.application_id,
            instance_id=self.instance_id,
            profile_digest=self.definition.profile_digest,
            definition_digest=self.definition.definition_digest,
            component_start_order=self.definition.component_start_order,
            hook_ids_by_point={
                point.value: tuple(hook.hook_id for hook in hooks)
                for point, hooks in self.definition.hook_registrations.items()
            },
            event_subscription_ids=tuple(
                subscription.subscription_id
                for subscription in self.definition.event_subscriptions
            ),
        )
        self._observer_tasks = ObserverTaskRegistry(self._diagnostics)
        self._health = HostedApplicationHealthCoordinator(self._lifecycle, self.clock)
        self._health.configure_polling(
            interval_seconds=self.health_poll_interval_seconds,
            sleeper=self.health_poll_sleeper,
        )
        self._readiness = _ReadinessServiceAdapter(self._health)
        self._event_dispatcher = HostingEventDispatcher(
            self.event_publisher,
            self.definition.event_subscriptions,
            self._diagnostics,
            self._observer_tasks,
        )
        self._hooks = HookCoordinator(
            self.definition,
            self.definition.profile.lifecycle,
            self._diagnostics,
            self._observer_tasks,
        )
        self._components = ComponentCoordinator(
            definition=self.definition,
            lifecycle_policy=self.definition.profile.lifecycle,
            diagnostics=self._diagnostics,
            publish_event=self._event_dispatcher,
        )
        self._context: HostedApplicationContext | None = None
        self._runtime: HostedApplicationRuntime | None = None
        self._lease: HostedApplicationInstanceLeasePort | None = None
        self._lease_released = False
        self._context_closed = False
        self._startup_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()

    @property
    def context(self) -> HostedApplicationContext:
        if self._context is None:
            raise HostedApplicationEngineError("hosted application context is not available")
        return self._context

    @property
    def accepts_new_work(self) -> bool:
        return self._health.accepts_new_work()

    def lifecycle_snapshot(self):
        return self._lifecycle.snapshot()

    def health_snapshot(self) -> HostedApplicationHealthSnapshot:
        return self._health.snapshot()

    def diagnostics_snapshot(self) -> HostedApplicationDiagnosticSnapshot:
        self._diagnostics.set_observer_task_count(self._observer_tasks.task_count)
        return self._diagnostics.snapshot(
            lifecycle=self._lifecycle.snapshot(),
            health=self._health.snapshot(),
        )

    async def start(self) -> None:
        async with self._startup_lock:
            if self._lifecycle.state is not HostedApplicationLifecycleState.CREATED:
                if self._lifecycle.is_terminal:
                    raise HostedApplicationStartupError(
                        "cannot start hosted application engine from terminal state"
                    )
                raise HostedApplicationStartupError(
                    "hosted application engine has already started"
                )
            self._context = self._build_context()
            self._services.register(HostedApplicationReadinessService, self._readiness)
            try:
                await self._startup_sequence()
            except Exception as exc:
                await self._startup_failure_cleanup(exc)
                primary = self._diagnostics.primary_exception or exc
                raise HostedApplicationStartupError("hosted application startup failed") from primary

    async def stop(self, *, reason_code: str = "engine.stop") -> HostedApplicationEngineTerminalResult:
        async with self._stop_lock:
            if self._lifecycle.state is HostedApplicationLifecycleState.STOPPED:
                return self._terminal_result(reason_code)
            if self._lifecycle.state is HostedApplicationLifecycleState.FAILED:
                return self._terminal_result(reason_code)
            return await self._shutdown_sequence(reason_code=reason_code)

    async def run_until_stopped(self) -> HostedApplicationEngineTerminalResult:
        if self._lifecycle.state is HostedApplicationLifecycleState.CREATED:
            await self.start()
        request = await self.shutdown.wait_until_requested()
        return await self.stop(reason_code=request.reason_code)

    def _build_context(self) -> HostedApplicationContext:
        return HostedApplicationContext(
            application_id=self.definition.profile.application_id,
            instance_id=self.instance_id,
            profile=self.definition.profile.public_view(),
            profile_digest=self.definition.profile_digest,
            paths=self.paths,
            process_identity=self.process_identity,
            services=self._services,
            clock=self.clock,
            logger=self.logger,
            event_publisher=self._event_dispatcher,
            shutdown=self.shutdown,
            lifecycle=self._lifecycle,
        )

    async def _startup_sequence(self) -> None:
        assert self._context is not None
        identity = HostedApplicationInstanceIdentity(
            application_id=self.definition.profile.application_id,
            instance_id=self.instance_id,
            profile_digest=self.definition.profile_digest,
            process_identity=self.process_identity,
        )
        self._lease = await self.instance_guard.acquire(identity)
        self._health.set_lease(self._lease)
        self._diagnostics.mark_lease_acquired()
        self._lifecycle.transition_to(HostedApplicationLifecycleState.STARTING, reason_code="starting")
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STARTING)
        await self._hooks.execute_blocking(HostedApplicationHookPoint.BEFORE_START, self._context)
        await self._components.start_phase(
            self._context,
            self.definition.pre_runtime_component_ids,
        )
        self._runtime = await invoke_application_factory(
            self.definition.application_factory,
            self._context,
        )
        self._diagnostics.mark_runtime_created()
        await self._runtime.start(self._context)
        self._diagnostics.mark_runtime_started()
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STARTED)
        self._hooks.schedule_observers(HostedApplicationHookPoint.AFTER_START, self._context)
        await self._components.start_phase(
            self._context,
            self.definition.post_runtime_component_ids,
        )
        await self._hooks.execute_blocking(HostedApplicationHookPoint.BEFORE_READY, self._context)
        runtime_ready = await self._runtime.ready(self._context)
        self._health.set_runtime_ready(runtime_ready)
        self._health.update_component_health(
            self._components.component_health(),
            mark_not_ready_failed=self._components.mark_not_ready_component_ids,
            degraded_component_ids=self._components.degraded_component_ids,
        )
        self._services.seal()
        self._lifecycle.transition_to(HostedApplicationLifecycleState.READY, reason_code="ready")
        self._health.refresh_once()
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_READY)
        self._hooks.schedule_observers(HostedApplicationHookPoint.AFTER_READY, self._context)
        await self._health.start_polling(self._refresh_health)

    async def _refresh_health(self) -> None:
        if self._context is None or self._runtime is None:
            return
        runtime_ready = await self._runtime.ready(self._context)
        self._health.set_runtime_ready(runtime_ready)
        await self._components.refresh_component_health(self._context)
        self._health.update_component_health(
            self._components.component_health(),
            mark_not_ready_failed=self._components.mark_not_ready_component_ids,
            degraded_component_ids=self._components.degraded_component_ids,
        )

    async def _startup_failure_cleanup(self, exc: Exception) -> None:
        assert self._context is not None
        if self._diagnostics.primary_exception is None:
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.RUNTIME_START,
                source_kind="engine",
                source_id="startup",
                exc=exc,
                reason_code="startup_failed",
            )
        await self._hooks.execute_on_failure(self._context, primary_exc=exc)
        if self._components.started_component_ids:
            await self._components.stop_started(self._context)
        if self._runtime is not None:
            try:
                await self._runtime.stop(self._context)
            except Exception as stop_exc:
                self._diagnostics.record_secondary_failure(
                    phase=HostedApplicationFailurePhase.ROLLBACK,
                    source_kind="runtime",
                    source_id="rollback_stop",
                    exc=stop_exc,
                    reason_code="runtime_rollback_failed",
                )
        await self._release_lease()
        self._close_context()
        if self._lifecycle.state is HostedApplicationLifecycleState.STARTING:
            self._lifecycle.transition_to(HostedApplicationLifecycleState.FAILED, reason_code="failed")
        try:
            await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_FAILED)
        except Exception as publish_exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.EVENT_PUBLISH,
                source_kind="event_publisher",
                source_id="application_failed",
                exc=publish_exc,
                reason_code="failure_event_publish_failed",
            )

    async def _shutdown_sequence(self, *, reason_code: str) -> HostedApplicationEngineTerminalResult:
        if self._context is None:
            self._context = self._build_context()
        assert self._context is not None
        self._lifecycle.set_shutdown_requested(True)
        if self._lifecycle.state is HostedApplicationLifecycleState.READY:
            self._lifecycle.transition_to(HostedApplicationLifecycleState.STOPPING, reason_code=reason_code)
        elif self._lifecycle.state is HostedApplicationLifecycleState.STARTING:
            self._lifecycle.transition_to(HostedApplicationLifecycleState.STOPPING, reason_code=reason_code)
        self._health.refresh_once()
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STOPPING)
        await self._hooks.execute_blocking(HostedApplicationHookPoint.BEFORE_STOP, self._context)
        await self._health.stop_polling()
        await self._components.stop_started(self._context)
        if self._runtime is not None:
            try:
                await self._runtime.stop(self._context)
            except Exception as exc:
                self._diagnostics.record_secondary_failure(
                    phase=HostedApplicationFailurePhase.RUNTIME_STOP,
                    source_kind="runtime",
                    source_id="stop",
                    exc=exc,
                    reason_code="runtime_stop_failed",
                )
        self._hooks.schedule_observers(HostedApplicationHookPoint.AFTER_STOP, self._context)
        drain_timeout = self.definition.profile.lifecycle.default_observer_hook_timeout_seconds
        await self._observer_tasks.drain(drain_timeout)
        self._observer_tasks.cancel_remaining()
        await self._release_lease()
        self._close_context()
        if self._lifecycle.state is HostedApplicationLifecycleState.STOPPING:
            self._lifecycle.transition_to(HostedApplicationLifecycleState.STOPPED, reason_code=reason_code)
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STOPPED)
        return self._terminal_result(reason_code)

    async def _release_lease(self) -> None:
        if self._lease_released or self._lease is None:
            return
        try:
            await self._lease.release()
        except Exception as exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.INSTANCE_ACQUIRE,
                source_kind="instance_lease",
                source_id="release",
                exc=exc,
                reason_code="lease_release_failed",
            )
        finally:
            self._lease_released = True
            self._diagnostics.mark_lease_released()

    def _close_context(self) -> None:
        if self._context_closed:
            return
        if self._context is not None:
            self._context.close()
        self._context_closed = True
        self._diagnostics.mark_context_closed()

    def _terminal_result(self, reason_code: str) -> HostedApplicationEngineTerminalResult:
        return HostedApplicationEngineTerminalResult(
            terminal_state=self._lifecycle.state,
            reason_code=reason_code,
            diagnostics=self.diagnostics_snapshot(),
        )

    async def _publish_lifecycle_event(self, event_type: HostedApplicationEventType) -> None:
        assert self._context is not None
        await self._event_dispatcher.publish(
            HostedApplicationEvent(
                event_type=event_type,
                application_id=self._context.application_id,
                instance_id=self._context.instance_id,
                lifecycle_state=self._lifecycle.state,
            )
        )
