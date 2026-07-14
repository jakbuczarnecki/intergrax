# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application engine orchestration (APP-HOST-2F)."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

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
    HostedApplicationOperationPhase,
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
    HostedApplicationShutdownError,
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
    failure_id_generator: object | None = None
    _operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _startup_aborted: bool = field(default=False, repr=False)
    _instance_acquire_failed: bool = field(default=False, repr=False)
    _reuse_blocked: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.health_poll_interval_seconds) or self.health_poll_interval_seconds <= 0:
            raise HostedApplicationEngineError(
                "health_poll_interval_seconds must be finite and positive"
            )
        self._lifecycle = HostedApplicationLifecycleController(self.clock)
        self._services = HostedApplicationServiceRegistry()
        self._diagnostics = DiagnosticsRecorder(
            clock=self.clock,
            application_id=self.definition.application_id,
            instance_id=self.instance_id,
            profile_digest=self.definition.profile_digest,
            definition_digest=self.definition.definition_digest,
            component_start_order=self.definition.component_start_order,
            hook_ids_by_point={
                point.value: tuple(hook.hook_id for hook in hooks)
                for point, hooks in self.definition.hook_registrations.items()
            },
            event_subscription_ids=tuple(
                resolved.subscription.subscription_id
                for resolved in self.definition.event_subscriptions
            ),
            failure_id_generator=self.failure_id_generator,  # type: ignore[arg-type]
        )
        self._observer_tasks = ObserverTaskRegistry(self._diagnostics)
        self._health = HostedApplicationHealthCoordinator(self._lifecycle, self.clock)
        self._health.configure_polling(
            interval_seconds=self.health_poll_interval_seconds,
            sleeper=self.health_poll_sleeper,
            on_poll_failure=self._handle_health_poll_failure,
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
            self.definition.lifecycle_policy,
            self._diagnostics,
            self._observer_tasks,
            self._event_dispatcher.publish,
        )
        self._components = ComponentCoordinator(
            definition=self.definition,
            lifecycle_policy=self.definition.lifecycle_policy,
            diagnostics=self._diagnostics,
            publish_event=self._event_dispatcher,
        )
        self._context: HostedApplicationContext | None = None
        self._runtime: HostedApplicationRuntime | None = None
        self._lease: HostedApplicationInstanceLeasePort | None = None
        self._lease_released = False
        self._context_closed = False

    @property
    def context(self) -> HostedApplicationContext:
        if self._context is None:
            raise HostedApplicationEngineError("hosted application context is not available")
        return self._context

    @property
    def accepts_new_work(self) -> bool:
        return self._health.accepts_new_work()

    @property
    def reuse_blocked(self) -> bool:
        """When true, a prior fatal failure blocked further start attempts."""
        return self._reuse_blocked

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
        async with self._operation_lock:
            if self._reuse_blocked:
                raise HostedApplicationStartupError(
                    "hosted application engine reuse is blocked after fatal failure"
                )
            state = self._lifecycle.state
            if state is not HostedApplicationLifecycleState.CREATED:
                if self._lifecycle.is_terminal:
                    raise HostedApplicationStartupError(
                        "cannot start hosted application engine from terminal state"
                    )
                raise HostedApplicationStartupError(
                    "hosted application engine has already started"
                )
            self._startup_aborted = False
            self._instance_acquire_failed = False
            self._diagnostics.clear_current_failure()
            self._diagnostics.clear_primary_exception()
            self._diagnostics.reset_attempt_local_state()
            self._health.reset_attempt_state()
            self._lease = None
            self._lease_released = False
            self._runtime = None
            self._observer_tasks = ObserverTaskRegistry(self._diagnostics)
            self._event_dispatcher = HostingEventDispatcher(
                self.event_publisher,
                self.definition.event_subscriptions,
                self._diagnostics,
                self._observer_tasks,
            )
            self._hooks = HookCoordinator(
                self.definition,
                self.definition.lifecycle_policy,
                self._diagnostics,
                self._observer_tasks,
                self._event_dispatcher.publish,
            )
            self._components = ComponentCoordinator(
                definition=self.definition,
                lifecycle_policy=self.definition.lifecycle_policy,
                diagnostics=self._diagnostics,
                publish_event=self._event_dispatcher,
            )
            self._context = self._build_context()
            self._services.register(HostedApplicationReadinessService, self._readiness)
            self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.STARTUP)
            try:
                await self._startup_sequence()
            except Exception as exc:
                if self._lifecycle.state is HostedApplicationLifecycleState.STARTING:
                    await self._startup_failure_cleanup(exc)
                    primary = self._diagnostics.primary_exception or exc
                    raise HostedApplicationStartupError("hosted application startup failed") from primary
                if self._instance_acquire_failed:
                    raise HostedApplicationStartupError("instance acquisition failed") from exc
                raise

    async def stop(self, *, reason_code: str = "engine.stop") -> HostedApplicationEngineTerminalResult:
        pre_state = self._lifecycle.state
        if pre_state is HostedApplicationLifecycleState.STARTING or (
            pre_state is HostedApplicationLifecycleState.CREATED and self._operation_lock.locked()
        ):
            self._startup_aborted = True
            self._lifecycle.set_shutdown_requested(True)
        async with self._operation_lock:
            state = self._lifecycle.state
            if state is HostedApplicationLifecycleState.CREATED:
                raise HostedApplicationShutdownError(
                    "cannot stop hosted application engine before start"
                )
            if state is HostedApplicationLifecycleState.STOPPED:
                return self._terminal_result(reason_code)
            if state is HostedApplicationLifecycleState.FAILED:
                return await self._failed_terminal_cleanup(reason_code)
            if state is HostedApplicationLifecycleState.STARTING:
                self._startup_aborted = True
                self._lifecycle.set_shutdown_requested(True)
            return await self._graceful_stop_sequence(reason_code=reason_code)

    async def run_until_stopped(self) -> HostedApplicationEngineTerminalResult:
        if self._lifecycle.state is HostedApplicationLifecycleState.CREATED:
            await self.start()
        request = await self.shutdown.wait_until_requested()
        return await self.stop(reason_code=request.reason_code)

    def _build_context(self) -> HostedApplicationContext:
        return HostedApplicationContext(
            application_id=self.definition.application_id,
            instance_id=self.instance_id,
            profile=self.definition.profile_public_snapshot,
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
            application_id=self.definition.application_id,
            instance_id=self.instance_id,
            profile_digest=self.definition.profile_digest,
            process_identity=self.process_identity,
        )
        try:
            self._lease = await self.instance_guard.acquire(identity)
        except Exception as exc:
            self._instance_acquire_failed = True
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.INSTANCE_ACQUIRE,
                source_kind="instance_guard",
                source_id="acquire",
                exc=exc,
                reason_code="instance_acquire_failed",
            )
            self._reset_context_after_pre_lifecycle_failure()
            raise HostedApplicationStartupError("instance acquisition failed") from exc

        self._health.set_lease(self._lease)
        self._diagnostics.mark_lease_acquired()
        self._lifecycle.transition_to(HostedApplicationLifecycleState.STARTING, reason_code="starting")
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STARTING)
        await self._hooks.execute_blocking(HostedApplicationHookPoint.BEFORE_START, self._context)
        if self._should_abort_startup():
            await self._abort_startup_to_stopping("startup_aborted")
            return
        await self._components.start_phase(
            self._context,
            self.definition.pre_runtime_component_ids,
        )
        if self._should_abort_startup():
            await self._abort_startup_to_stopping("startup_aborted")
            return
        try:
            self._runtime = await invoke_application_factory(
                self.definition.application_factory,
                self._context,
            )
        except Exception as exc:
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.RUNTIME_FACTORY,
                source_kind="runtime_factory",
                source_id="factory",
                exc=exc,
                reason_code="runtime_factory_failed",
            )
            raise
        self._diagnostics.mark_runtime_created()
        try:
            await self._runtime.start(self._context)
        except Exception as exc:
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.RUNTIME_START,
                source_kind="runtime",
                source_id="start",
                exc=exc,
                reason_code="runtime_start_failed",
            )
            raise
        self._diagnostics.mark_runtime_started()
        if self._should_abort_startup():
            await self._abort_startup_to_stopping("startup_aborted")
            return
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STARTED)
        self._hooks.schedule_observers(HostedApplicationHookPoint.AFTER_START, self._context)
        await self._components.start_phase(
            self._context,
            self.definition.post_runtime_component_ids,
        )
        if self._should_abort_startup():
            await self._abort_startup_to_stopping("startup_aborted")
            return
        await self._hooks.execute_blocking(HostedApplicationHookPoint.BEFORE_READY, self._context)
        if self._should_abort_startup():
            await self._abort_startup_to_stopping("startup_aborted")
            return
        runtime_ready = await self._runtime.ready(self._context)
        self._health.set_runtime_ready(runtime_ready)
        self._health.update_component_health(
            self._components.component_health(),
            mark_not_ready_failed=self._components.mark_not_ready_component_ids,
            degraded_component_ids=self._components.degraded_component_ids,
        )
        gate = self._health.evaluate_startup_readiness_gate()
        if not gate.passed:
            exc = RuntimeError(f"startup_readiness_gate_failed:{gate.reason_code}")
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.HEALTH_EVALUATION,
                source_kind="health",
                source_id="startup_gate",
                exc=exc,
                reason_code=gate.reason_code or "readiness_gate_failed",
            )
            raise HostedApplicationStartupError("startup readiness gate failed") from exc
        self._services.seal()
        self._lifecycle.transition_to(HostedApplicationLifecycleState.READY, reason_code="ready")
        aggregate = self._health.refresh_once()
        if not aggregate.ready or not aggregate.accepting_new_work:
            exc = RuntimeError("post_ready_aggregate_failed")
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.HEALTH_EVALUATION,
                source_kind="health",
                source_id="post_ready_aggregate",
                exc=exc,
                reason_code="post_ready_aggregate_failed",
            )
            raise HostedApplicationStartupError("post-ready aggregate evaluation failed") from exc
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_READY)
        self._hooks.schedule_observers(HostedApplicationHookPoint.AFTER_READY, self._context)
        await self._health.start_polling(self._refresh_health)
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.IDLE)

    def _should_abort_startup(self) -> bool:
        return self._startup_aborted or self.shutdown.is_shutdown_requested()

    async def _abort_startup_to_stopping(self, reason_code: str) -> None:
        await self._graceful_stop_sequence(reason_code=reason_code)

    async def _refresh_health(self) -> None:
        if self._context is None or self._runtime is None:
            return
        try:
            runtime_ready = await self._runtime.ready(self._context)
            await self._components.refresh_component_health(self._context)
            self._health.clear_health_evaluation_failed()
            self._health.set_runtime_ready(runtime_ready)
            self._health.update_component_health(
                self._components.component_health(),
                mark_not_ready_failed=self._components.mark_not_ready_component_ids,
                degraded_component_ids=self._components.degraded_component_ids,
            )
        except Exception as exc:
            self._handle_health_poll_failure(exc)

    def _handle_health_poll_failure(self, exc: BaseException) -> None:
        self._diagnostics.record_secondary_failure(
            phase=HostedApplicationFailurePhase.HEALTH_EVALUATION,
            source_kind="health_poll",
            source_id="refresh",
            exc=exc,
            reason_code="health_poll_failed",
        )
        self._health.mark_health_evaluation_failed()
        self._health.refresh_once()

    async def _startup_failure_cleanup(self, exc: Exception) -> None:
        assert self._context is not None
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.ROLLBACK)
        if self._diagnostics.primary_exception is None:
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.RUNTIME_START,
                source_kind="engine",
                source_id="startup",
                exc=exc,
                reason_code="startup_failed",
            )
        await self._safe_phase(self._hooks.schedule_on_failure, self._context)
        await self._safe_phase(self._health.stop_polling)
        if self._components.started_component_ids:
            await self._safe_phase(self._components.stop_started, self._context)
        if self._runtime is not None:
            await self._safe_phase(
                self._runtime.stop,
                self._context,
                phase=HostedApplicationFailurePhase.RUNTIME_STOP,
            )
        await self._drain_observer_tasks(close_first=True)
        await self._safe_phase(self._release_lease)
        self._close_context()
        if self._lifecycle.state is HostedApplicationLifecycleState.STARTING:
            self._lifecycle.transition_to(HostedApplicationLifecycleState.FAILED, reason_code="failed")
        self._reuse_blocked = True
        try:
            await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_FAILED)
        except Exception as publish_exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.EVENT_PUBLISH,
                source_kind="event_publisher",
                source_id=HostedApplicationEventType.APPLICATION_FAILED.value,
                exc=publish_exc,
                reason_code="terminal_event_publish_failed",
            )
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.IDLE)

    async def _graceful_stop_sequence(self, *, reason_code: str) -> HostedApplicationEngineTerminalResult:
        if self._context is None:
            self._context = self._build_context()
        assert self._context is not None
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.SHUTDOWN)
        self._lifecycle.set_shutdown_requested(True)
        self._health.refresh_once()
        state = self._lifecycle.state
        if state in {
            HostedApplicationLifecycleState.READY,
            HostedApplicationLifecycleState.STARTING,
        }:
            self._lifecycle.transition_to(
                HostedApplicationLifecycleState.STOPPING,
                reason_code=reason_code,
            )
        await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STOPPING)
        await self._safe_phase(
            self._hooks.execute_blocking,
            HostedApplicationHookPoint.BEFORE_STOP,
            self._context,
            phase=HostedApplicationFailurePhase.BEFORE_STOP_HOOK,
        )
        await self._safe_phase(self._health.stop_polling)
        await self._safe_phase(self._components.stop_started, self._context)
        if self._runtime is not None:
            await self._safe_phase(
                self._runtime.stop,
                self._context,
                phase=HostedApplicationFailurePhase.RUNTIME_STOP,
            )
        self._hooks.schedule_observers(HostedApplicationHookPoint.AFTER_STOP, self._context)
        drain_timeout = self.definition.lifecycle_policy.default_observer_hook_timeout_seconds
        await self._safe_phase(self._observer_tasks.drain, drain_timeout)
        await self._safe_phase(self._release_lease)
        self._close_context()
        if self._lifecycle.state is HostedApplicationLifecycleState.STOPPING:
            self._lifecycle.transition_to(
                HostedApplicationLifecycleState.STOPPED,
                reason_code=reason_code,
            )
        await self._publish_terminal_stopped_event()
        await self._safe_phase(self._observer_tasks.drain, drain_timeout)
        self._observer_tasks.close_to_new_tasks()
        self._observer_tasks.cancel_remaining()
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.IDLE)
        return self._terminal_result(reason_code)

    async def _failed_terminal_cleanup(self, reason_code: str) -> HostedApplicationEngineTerminalResult:
        await self._drain_observer_tasks(close_first=True)
        await self._safe_phase(self._release_lease)
        self._close_context()
        return self._terminal_result(reason_code)

    async def _publish_terminal_stopped_event(self) -> None:
        try:
            await self._publish_lifecycle_event(HostedApplicationEventType.APPLICATION_STOPPED)
        except Exception as publish_exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.EVENT_PUBLISH,
                source_kind="event_publisher",
                source_id=HostedApplicationEventType.APPLICATION_STOPPED.value,
                exc=publish_exc,
                reason_code="terminal_event_publish_failed",
            )

    async def _drain_observer_tasks(self, *, close_first: bool = False) -> None:
        drain_timeout = self.definition.lifecycle_policy.default_observer_hook_timeout_seconds
        if close_first:
            self._observer_tasks.close_to_new_tasks()
        await self._safe_phase(self._observer_tasks.drain, drain_timeout)
        self._observer_tasks.close_to_new_tasks()
        await self._safe_phase(self._observer_tasks.drain, drain_timeout)
        self._observer_tasks.cancel_remaining()

    async def _safe_phase(self, callback, *args, phase: HostedApplicationFailurePhase | None = None, **kwargs) -> None:
        try:
            result = callback(*args, **kwargs)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception as exc:
            self._diagnostics.record_secondary_failure(
                phase=phase or HostedApplicationFailurePhase.ROLLBACK,
                source_kind="cleanup",
                source_id=callback.__name__ if hasattr(callback, "__name__") else "phase",
                exc=exc,
                reason_code="cleanup_phase_failed",
            )

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

    def _reset_context_after_pre_lifecycle_failure(self) -> None:
        if self._context is not None:
            self._context.close()
        self._context = None
        self._context_closed = False
        self._services = HostedApplicationServiceRegistry()
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.IDLE)

    def _terminal_result(self, reason_code: str) -> HostedApplicationEngineTerminalResult:
        self._diagnostics.set_observer_task_count(self._observer_tasks.task_count)
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
