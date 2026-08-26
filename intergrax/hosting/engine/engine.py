# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application engine orchestration (APP-HOST-2F)."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

from intergrax.contracts.event_severity import EventSeverity
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
    HostedApplicationEffectiveControlRequest,
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
    hosted_failure_event_payload,
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
    HostedApplicationInstanceLeasePort,
    HostedApplicationRuntime,
)
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from intergrax.hosting.errors import (
    HostedApplicationEngineError,
    HostedApplicationInstanceConflictError,
    HostedApplicationShutdownError,
    HostedApplicationStartupError,
)
from intergrax.hosting.instance.contracts import InstanceAcquisitionClassification
from intergrax.hosting.engine.runtime import invoke_application_factory
from intergrax.hosting.shutdown import (
    HostedApplicationActiveWorkController,
    HostedApplicationFlushService,
    HostedApplicationGlobalShutdownBudget,
    HostedApplicationShutdownExecutor,
    HostedApplicationShutdownPhase,
    HostedApplicationShutdownPhaseOutcome,
    MonotonicClock,
    ShutdownPhaseRecorder,
    SystemMonotonicClock,
    build_shutdown_execution_snapshot,
    compute_shutdown_budget_seconds,
    run_bounded_phase,
)
from intergrax.hosting.eventing import HostingEventDispatcher
from intergrax.hosting.services import HostedApplicationServiceRegistry
from intergrax.utils import attribute_access


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
    active_work_controller: HostedApplicationActiveWorkController | None = None
    flush_services: tuple[HostedApplicationFlushService, ...] = ()
    health_poll_interval_seconds: float = 5.0
    health_poll_sleeper: object | None = None
    failure_id_generator: object | None = None
    monotonic_clock: MonotonicClock | None = None
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
        self._monotonic_clock = self.monotonic_clock or SystemMonotonicClock()
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
                state = self._lifecycle.state
                if state in {
                    HostedApplicationLifecycleState.STARTING,
                    HostedApplicationLifecycleState.READY,
                }:
                    await self._startup_failure_cleanup(exc)
                    primary = self._diagnostics.primary_exception or exc
                    raise HostedApplicationStartupError("hosted application startup failed") from primary
                if self._instance_acquire_failed:
                    raise HostedApplicationStartupError("instance acquisition failed") from exc
                raise

    async def stop(
        self,
        *,
        reason_code: str = "engine.stop",
        control_request: HostedApplicationEffectiveControlRequest | None = None,
    ) -> HostedApplicationEngineTerminalResult:
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
            return await self._graceful_stop_sequence(
                reason_code=reason_code,
                control_request=control_request,
            )

    async def run_until_stopped(self) -> HostedApplicationEngineTerminalResult:
        if self._lifecycle.state is HostedApplicationLifecycleState.CREATED:
            await self.start()
        request = await self.shutdown.wait_until_requested()
        return await self.stop(reason_code=request.reason_code, control_request=request)

    def _build_context(self) -> HostedApplicationContext:
        return HostedApplicationContext(
            application_id=self.definition.application_id,
            instance_id=self.instance_id,
            profile=self.definition.profile_public_view(),
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
            acquisition = await self.instance_guard.acquire(identity)
            self._lease = acquisition.lease
            acquisition_classification = acquisition.classification
        except HostedApplicationInstanceConflictError as exc:
            self._instance_acquire_failed = True
            self._diagnostics.record_primary_failure(
                phase=HostedApplicationFailurePhase.INSTANCE_ACQUIRE,
                source_kind="instance_guard",
                source_id="acquire",
                exc=exc,
                reason_code="instance_conflict",
            )
            await self._publish_instance_event(HostedApplicationEventType.INSTANCE_REJECTED)
            self._reset_context_after_pre_lifecycle_failure()
            raise HostedApplicationStartupError("instance acquisition failed") from exc
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

        if acquisition_classification in {
            InstanceAcquisitionClassification.STALE_OWNER,
            InstanceAcquisitionClassification.CORRUPTED_METADATA,
        }:
            await self._publish_instance_event(HostedApplicationEventType.INSTANCE_STALE_RECOVERED)
        await self._publish_instance_event(HostedApplicationEventType.INSTANCE_ACQUIRED)

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
        # Refresh after before_ready so hook-driven component readiness is visible to the gate.
        await self._components.refresh_component_health(self._context)
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
        self._diagnostics.mark_ready(ready_at=self.clock.now())
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
        if self._startup_aborted:
            return True
        if self.shutdown.is_shutdown_requested():
            return True
        if hasattr(self.shutdown, "is_restart_requested"):
            return bool(self.shutdown.is_restart_requested())  # type: ignore[attr-defined]
        return False

    async def _abort_startup_to_stopping(self, reason_code: str) -> None:
        control_request = self._resolve_effective_control_request()
        await self._graceful_stop_sequence(
            reason_code=control_request.reason_code if control_request is not None else reason_code,
            control_request=control_request,
        )

    def _resolve_effective_control_request(self) -> HostedApplicationEffectiveControlRequest | None:
        if hasattr(self.shutdown, "current_effective_request"):
            effective = self.shutdown.current_effective_request()  # type: ignore[attr-defined]
            if effective is not None:
                return effective
        current = self.shutdown.current_request()
        if current is None:
            return None
        if isinstance(current, HostedApplicationEffectiveControlRequest):
            return current
        source_id = attribute_access.optional(current, "source_id", "runtime")
        return HostedApplicationEffectiveControlRequest(
            intent="stop",
            reason_code=current.reason_code,
            requested_at=current.requested_at,
            deadline_at=current.deadline_at,
            source_id=source_id,
        )

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
        await self._execute_bounded_terminal_cleanup(
            record_lifecycle_failure_event=True,
            transition_to_failed=True,
        )
        self._reuse_blocked = True
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.IDLE)

    async def _graceful_stop_sequence(
        self,
        *,
        reason_code: str,
        control_request: HostedApplicationEffectiveControlRequest | None = None,
    ) -> HostedApplicationEngineTerminalResult:
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

        effective_request = control_request or self._resolve_effective_control_request()
        policy = self.definition.lifecycle_policy
        budget_seconds = compute_shutdown_budget_seconds(
            shutdown_policy=self.definition.shutdown_policy,
            blocking_hook_timeout=policy.default_blocking_hook_timeout_seconds,
            observer_drain_timeout=policy.default_observer_hook_timeout_seconds,
            component_stop_budget=policy.default_blocking_hook_timeout_seconds,
            runtime_stop_budget=policy.default_blocking_hook_timeout_seconds,
            lease_release_timeout=policy.default_observer_hook_timeout_seconds,
            explicit_deadline_at=effective_request.deadline_at if effective_request else None,
            clock=self.clock,
            requested_at=effective_request.requested_at if effective_request else self.clock.now(),
        )
        budget = HostedApplicationGlobalShutdownBudget(
            deadline_monotonic=self._monotonic_clock.monotonic() + budget_seconds,
            monotonic_clock=self._monotonic_clock,
        )
        recorder = ShutdownPhaseRecorder(clock=self.clock)
        active_before = (
            self.active_work_controller.active_work_count()
            if self.active_work_controller is not None
            else 0
        )
        context = self._context
        assert context is not None

        before_stop_started = self.clock.now()
        before_stop_outcome = await run_bounded_phase(
            budget,
            policy.default_blocking_hook_timeout_seconds,
            lambda: self._hooks.execute_blocking(
                HostedApplicationHookPoint.BEFORE_STOP,
                context,
            ),
        )
        if before_stop_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        recorder.record(
            phase=HostedApplicationShutdownPhase.BEFORE_STOP,
            outcome=before_stop_outcome,
            started_at=before_stop_started,
        )

        shutdown_executor = HostedApplicationShutdownExecutor(
            shutdown_policy=self.definition.shutdown_policy,
            clock=self.clock,
            monotonic_clock=self._monotonic_clock,
            active_work_controller=self.active_work_controller,
            flush_services=self.flush_services,
        )
        try:
            await shutdown_executor.execute(
                request=effective_request,
                budget=budget,
                recorder=recorder,
            )
        except Exception as exc:
            recorder.record(
                phase=HostedApplicationShutdownPhase.STOP_INTAKE,
                outcome=HostedApplicationShutdownPhaseOutcome.FAILED,
                started_at=self.clock.now(),
            )
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.RUNTIME_STOP,
                source_kind="shutdown_executor",
                source_id="execute",
                exc=exc,
                reason_code="shutdown_execution_failed",
            )

        health_started = self.clock.now()
        health_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            self._health.stop_polling,
        )
        if health_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        recorder.record(
            phase=HostedApplicationShutdownPhase.HEALTH_POLL_STOP,
            outcome=health_outcome,
            started_at=health_started,
        )

        component_started = self.clock.now()
        component_outcome = await run_bounded_phase(
            budget,
            policy.default_blocking_hook_timeout_seconds,
            lambda: self._components.stop_started(context),
        )
        if component_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        recorder.record(
            phase=HostedApplicationShutdownPhase.COMPONENT_STOP,
            outcome=component_outcome,
            started_at=component_started,
        )
        if component_outcome is HostedApplicationShutdownPhaseOutcome.FAILED:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.COMPONENT_STOP,
                source_kind="components",
                source_id="stop",
                exc=RuntimeError("component_stop_failed"),
                reason_code="component_stop_failed",
            )

        if self._runtime is not None:
            runtime_started = self.clock.now()
            runtime_outcome = await run_bounded_phase(
                budget,
                policy.default_blocking_hook_timeout_seconds,
                lambda: self._runtime.stop(context),  # type: ignore[union-attr]
            )
            if runtime_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
                recorder.timed_out = True
            recorder.record(
                phase=HostedApplicationShutdownPhase.RUNTIME_STOP,
                outcome=runtime_outcome,
                started_at=runtime_started,
            )

        observer_started = self.clock.now()
        self._hooks.schedule_observers(HostedApplicationHookPoint.AFTER_STOP, context)
        observer_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            lambda: self._observer_tasks.drain(policy.default_observer_hook_timeout_seconds),
        )
        if observer_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        recorder.record(
            phase=HostedApplicationShutdownPhase.AFTER_STOP_OBSERVER,
            outcome=observer_outcome,
            started_at=observer_started,
        )

        lease_started = self.clock.now()
        lease_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            self._bounded_release_lease,
        )
        lease_released = lease_outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED and self._lease_released
        if lease_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
            recorder.timed_out = True
        elif lease_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        elif lease_outcome is HostedApplicationShutdownPhaseOutcome.FAILED:
            pass
        elif not lease_released and lease_outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED:
            lease_outcome = HostedApplicationShutdownPhaseOutcome.FAILED
        recorder.record(
            phase=HostedApplicationShutdownPhase.LEASE_RELEASE,
            outcome=lease_outcome,
            started_at=lease_started,
        )
        if lease_released:
            instance_release_outcome = await run_bounded_phase(
                budget,
                policy.default_observer_hook_timeout_seconds,
                lambda: self._publish_instance_event(HostedApplicationEventType.INSTANCE_RELEASED),
            )
            if instance_release_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
                recorder.timed_out = True
            elif instance_release_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
                recorder.timed_out = True

        self._close_context()
        if self._lifecycle.state is HostedApplicationLifecycleState.STOPPING:
            self._lifecycle.transition_to(
                HostedApplicationLifecycleState.STOPPED,
                reason_code=reason_code,
            )
        self._health.refresh_once()

        stopped_publish_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            self._publish_terminal_stopped_event,
        )
        if stopped_publish_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
            recorder.timed_out = True
        elif stopped_publish_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True

        terminal_started = self.clock.now()
        self._observer_tasks.close_to_new_tasks()
        terminal_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            lambda: self._observer_tasks.drain(policy.default_observer_hook_timeout_seconds),
        )
        if terminal_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
            recorder.timed_out = True
        elif terminal_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        recorder.record(
            phase=HostedApplicationShutdownPhase.TERMINAL_SUBSCRIBER_DRAIN,
            outcome=terminal_outcome,
            started_at=terminal_started,
        )
        self._observer_tasks.cancel_remaining()

        active_after = (
            self.active_work_controller.active_work_count()
            if self.active_work_controller is not None
            else 0
        )
        shutdown_snapshot = build_shutdown_execution_snapshot(
            shutdown_policy=self.definition.shutdown_policy,
            request=effective_request,
            clock=self.clock,
            recorder=recorder,
            active_work_before=active_before,
            active_work_after=active_after,
        )
        self._diagnostics.set_shutdown_execution(shutdown_snapshot)
        self._diagnostics.set_observer_task_count(self._observer_tasks.task_count)
        self._diagnostics.set_operation_phase(HostedApplicationOperationPhase.IDLE)
        return self._terminal_result(reason_code)

    async def _failed_terminal_cleanup(self, reason_code: str) -> HostedApplicationEngineTerminalResult:
        await self._execute_bounded_terminal_cleanup(
            record_lifecycle_failure_event=False,
            transition_to_failed=False,
        )
        return self._terminal_result(reason_code)

    async def _execute_bounded_terminal_cleanup(
        self,
        *,
        record_lifecycle_failure_event: bool,
        transition_to_failed: bool,
    ) -> None:
        policy = self.definition.lifecycle_policy
        budget_seconds = compute_shutdown_budget_seconds(
            shutdown_policy=self.definition.shutdown_policy,
            blocking_hook_timeout=policy.default_blocking_hook_timeout_seconds,
            observer_drain_timeout=policy.default_observer_hook_timeout_seconds,
            component_stop_budget=policy.default_blocking_hook_timeout_seconds,
            runtime_stop_budget=policy.default_blocking_hook_timeout_seconds,
            lease_release_timeout=policy.default_observer_hook_timeout_seconds,
            explicit_deadline_at=None,
            clock=self.clock,
            requested_at=self.clock.now(),
        )
        budget = HostedApplicationGlobalShutdownBudget(
            deadline_monotonic=self._monotonic_clock.monotonic() + budget_seconds,
            monotonic_clock=self._monotonic_clock,
        )
        recorder = ShutdownPhaseRecorder(clock=self.clock)
        context = self._context

        health_started = self.clock.now()
        health_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            self._health.stop_polling,
        )
        if health_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        recorder.record(
            phase=HostedApplicationShutdownPhase.HEALTH_POLL_STOP,
            outcome=health_outcome,
            started_at=health_started,
        )

        if context is not None and self._components.started_component_ids:
            component_started = self.clock.now()
            component_outcome = await run_bounded_phase(
                budget,
                policy.default_blocking_hook_timeout_seconds,
                lambda: self._components.stop_started(context),
            )
            if component_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
                recorder.timed_out = True
            recorder.record(
                phase=HostedApplicationShutdownPhase.COMPONENT_STOP,
                outcome=component_outcome,
                started_at=component_started,
            )
            if component_outcome is HostedApplicationShutdownPhaseOutcome.FAILED:
                self._diagnostics.record_secondary_failure(
                    phase=HostedApplicationFailurePhase.COMPONENT_STOP,
                    source_kind="components",
                    source_id="stop",
                    exc=RuntimeError("component_stop_failed"),
                    reason_code="component_stop_failed",
                )

        if context is not None and self._runtime is not None:
            runtime_started = self.clock.now()
            runtime_outcome = await run_bounded_phase(
                budget,
                policy.default_blocking_hook_timeout_seconds,
                lambda: self._runtime.stop(context),  # type: ignore[union-attr]
            )
            if runtime_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
                recorder.timed_out = True
            recorder.record(
                phase=HostedApplicationShutdownPhase.RUNTIME_STOP,
                outcome=runtime_outcome,
                started_at=runtime_started,
            )
            if runtime_outcome is HostedApplicationShutdownPhaseOutcome.FAILED:
                self._diagnostics.record_secondary_failure(
                    phase=HostedApplicationFailurePhase.RUNTIME_STOP,
                    source_kind="runtime",
                    source_id="stop",
                    exc=RuntimeError("runtime_stop_failed"),
                    reason_code="runtime_stop_failed",
                )

        lease_started = self.clock.now()
        lease_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            self._bounded_release_lease,
        )
        lease_released = (
            lease_outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED and self._lease_released
        )
        if lease_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
            recorder.timed_out = True
        elif lease_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        elif lease_outcome is HostedApplicationShutdownPhaseOutcome.FAILED:
            pass
        elif not lease_released and lease_outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED:
            lease_outcome = HostedApplicationShutdownPhaseOutcome.FAILED
        recorder.record(
            phase=HostedApplicationShutdownPhase.LEASE_RELEASE,
            outcome=lease_outcome,
            started_at=lease_started,
        )
        if lease_released:
            instance_release_outcome = await run_bounded_phase(
                budget,
                policy.default_observer_hook_timeout_seconds,
                lambda: self._publish_instance_event(HostedApplicationEventType.INSTANCE_RELEASED),
            )
            if instance_release_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
                recorder.timed_out = True
            elif (
                instance_release_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED
                and budget.exhausted()
            ):
                recorder.timed_out = True

        self._close_context()

        if transition_to_failed:
            if self._lifecycle.state in {
                HostedApplicationLifecycleState.STARTING,
                HostedApplicationLifecycleState.READY,
            }:
                self._lifecycle.transition_to(
                    HostedApplicationLifecycleState.FAILED,
                    reason_code="failed",
                )
            self._health.refresh_once()

        if record_lifecycle_failure_event:
            await run_bounded_phase(
                budget,
                policy.default_observer_hook_timeout_seconds,
                lambda: self._observer_tasks.drain(policy.default_observer_hook_timeout_seconds),
            )
            failed_publish_outcome = await run_bounded_phase(
                budget,
                policy.default_observer_hook_timeout_seconds,
                self._publish_lifecycle_failed_event,
            )
            if failed_publish_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
                recorder.timed_out = True
            elif (
                failed_publish_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED
                and budget.exhausted()
            ):
                recorder.timed_out = True

        self._observer_tasks.close_to_new_tasks()
        terminal_started = self.clock.now()
        terminal_outcome = await run_bounded_phase(
            budget,
            policy.default_observer_hook_timeout_seconds,
            lambda: self._observer_tasks.drain(policy.default_observer_hook_timeout_seconds),
        )
        if terminal_outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
            recorder.timed_out = True
        elif terminal_outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
            recorder.timed_out = True
        recorder.record(
            phase=HostedApplicationShutdownPhase.TERMINAL_SUBSCRIBER_DRAIN,
            outcome=terminal_outcome,
            started_at=terminal_started,
        )
        self._observer_tasks.cancel_remaining()

        active_before = (
            self.active_work_controller.active_work_count()
            if self.active_work_controller is not None
            else 0
        )
        active_after = (
            self.active_work_controller.active_work_count()
            if self.active_work_controller is not None
            else 0
        )
        shutdown_snapshot = build_shutdown_execution_snapshot(
            shutdown_policy=self.definition.shutdown_policy,
            request=None,
            clock=self.clock,
            recorder=recorder,
            active_work_before=active_before,
            active_work_after=active_after,
        )
        self._diagnostics.set_shutdown_execution(shutdown_snapshot)
        self._diagnostics.set_observer_task_count(self._observer_tasks.task_count)
        if not transition_to_failed:
            self._health.refresh_once()

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

    async def _publish_lifecycle_failed_event(self) -> None:
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

    async def _quiescent_drain_failure_observers(self) -> None:
        """Drain failure hooks and subscriptions after APPLICATION_FAILED is published."""
        drain_timeout = self.definition.lifecycle_policy.default_observer_hook_timeout_seconds
        await self._safe_phase(self._observer_tasks.drain, drain_timeout)
        self._observer_tasks.close_to_new_tasks()
        await self._safe_phase(self._observer_tasks.drain, drain_timeout)
        self._observer_tasks.cancel_remaining()

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

    async def _bounded_release_lease(self) -> None:
        await self._release_lease_verified()

    async def _release_lease_verified(self) -> bool:
        if self._lease_released or self._lease is None:
            return self._lease_released
        try:
            await self._lease.release()
        except Exception as exc:
            self._diagnostics.record_secondary_failure(
                phase=HostedApplicationFailurePhase.INSTANCE_RELEASE,
                source_kind="instance_lease",
                source_id="release",
                exc=exc,
                reason_code="lease_release_failed",
            )
            return False
        self._lease_released = True
        self._diagnostics.mark_lease_released()
        return True

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
        diagnostics = self.diagnostics_snapshot()
        return HostedApplicationEngineTerminalResult(
            terminal_state=self._lifecycle.state,
            reason_code=reason_code,
            diagnostics=diagnostics,
            ready_at=diagnostics.ready_at,
            ready_duration_seconds=diagnostics.ready_duration_seconds,
        )

    async def _publish_instance_event(self, event_type: HostedApplicationEventType) -> None:
        assert self._context is not None
        payload: dict[str, object] = {}
        if self._lease is not None and hasattr(self._lease, "public_view"):
            try:
                public_view = self._lease.public_view()
                payload = {
                    "instance_id": public_view.instance_id,
                    "process_id": public_view.process_id,
                    "profile_digest": public_view.profile_digest,
                }
            except Exception:
                payload = {}
        from pydantic import JsonValue

        safe_payload: dict[str, JsonValue] = {key: value for key, value in payload.items()}  # type: ignore[misc]
        await self._event_dispatcher.publish(
            HostedApplicationEvent(
                event_type=event_type,
                application_id=self._context.application_id,
                instance_id=self._context.instance_id,
                lifecycle_state=self._lifecycle.state,
                payload=safe_payload,
            )
        )

    async def _publish_lifecycle_event(self, event_type: HostedApplicationEventType) -> None:
        assert self._context is not None
        from pydantic import JsonValue

        payload: dict[str, JsonValue] = {}
        severity = EventSeverity.INFO
        if event_type is HostedApplicationEventType.APPLICATION_FAILED:
            severity = EventSeverity.ERROR
            current_failure = self._diagnostics.current_failure
            if current_failure is not None:
                payload = {
                    key: value
                    for key, value in hosted_failure_event_payload(current_failure).items()
                }
        await self._event_dispatcher.publish(
            HostedApplicationEvent(
                event_type=event_type,
                application_id=self._context.application_id,
                instance_id=self._context.instance_id,
                lifecycle_state=self._lifecycle.state,
                severity=severity,
                payload=payload,
            )
        )
