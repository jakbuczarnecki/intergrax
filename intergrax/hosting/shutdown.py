# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graceful shutdown execution contracts and executor (APP-HOST-4D)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.hosting.contracts.context import HostedApplicationClock
from intergrax.hosting.contracts.lifecycle import HostedApplicationEffectiveControlRequest
from intergrax.hosting.contracts.policies import ShutdownPolicy, ShutdownStrategy


@runtime_checkable
class MonotonicClock(Protocol):
  """Injectable monotonic clock for bounded shutdown phases."""

  def monotonic(self) -> float: ...


class SystemMonotonicClock:
  """Default monotonic clock backed by time.monotonic()."""

  def monotonic(self) -> float:
    import time

    return time.monotonic()


@runtime_checkable
class HostedApplicationActiveWorkController(Protocol):
  """Port for bounded active-work drain and cancellation during shutdown."""

  async def stop_intake(self) -> None: ...

  def active_work_count(self) -> int: ...

  async def wait_for_idle(self) -> None: ...

  async def cancel_active_work(self) -> None: ...


@runtime_checkable
class HostedApplicationFlushService(Protocol):
  """Port for bounded flush of traces, events, or checkpoints."""

  @property
  def flush_id(self) -> str: ...

  async def flush(self) -> None: ...


class HostedApplicationShutdownPhase(str, Enum):
  BEFORE_STOP = "before_stop"
  STOP_INTAKE = "stop_intake"
  DRAIN = "drain"
  CANCEL = "cancel"
  FLUSH = "flush"
  HEALTH_POLL_STOP = "health_poll_stop"
  COMPONENT_STOP = "component_stop"
  RUNTIME_STOP = "runtime_stop"
  AFTER_STOP_OBSERVER = "after_stop_observer"
  LEASE_RELEASE = "lease_release"
  TERMINAL_SUBSCRIBER_DRAIN = "terminal_subscriber_drain"


class HostedApplicationShutdownPhaseOutcome(str, Enum):
  COMPLETED = "completed"
  TIMED_OUT = "timed_out"
  FAILED = "failed"
  SKIPPED = "skipped"


class HostedApplicationShutdownPhaseRecord(BaseModel):
  """Safe record for one shutdown phase execution."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  phase: HostedApplicationShutdownPhase
  outcome: HostedApplicationShutdownPhaseOutcome
  started_at: datetime
  completed_at: datetime | None = None
  flush_id: str = ""
  active_work_before: int | None = None
  active_work_after: int | None = None


class HostedApplicationShutdownExecutionSnapshot(BaseModel):
  """Safe shutdown execution snapshot without exception objects."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  strategy: ShutdownStrategy
  requested_at: datetime
  effective_deadline_at: datetime | None = None
  phase_records: tuple[HostedApplicationShutdownPhaseRecord, ...] = ()
  active_work_before: int = 0
  active_work_after: int = 0
  timed_out: bool = False
  forced: bool = False
  completed_at: datetime | None = None


@dataclass
class HostedApplicationGlobalShutdownBudget:
  """Shared monotonic shutdown deadline budget across all phases."""

  deadline_monotonic: float
  monotonic_clock: MonotonicClock

  def remaining_seconds(self) -> float:
    return max(0.0, self.deadline_monotonic - self.monotonic_clock.monotonic())

  def exhausted(self) -> bool:
    return self.remaining_seconds() <= 0.0

  def phase_timeout(self, configured_limit: float) -> float:
    if self.exhausted():
      return 0.0
    return min(configured_limit, self.remaining_seconds())


async def run_bounded_phase(
  budget: HostedApplicationGlobalShutdownBudget,
  configured_limit: float,
  coro_factory: Callable[[], Awaitable[None]],
) -> HostedApplicationShutdownPhaseOutcome:
  timeout = budget.phase_timeout(configured_limit)
  if timeout <= 0.0:
    return HostedApplicationShutdownPhaseOutcome.SKIPPED
  try:
    await asyncio.wait_for(coro_factory(), timeout=timeout)
    return HostedApplicationShutdownPhaseOutcome.COMPLETED
  except TimeoutError:
    return HostedApplicationShutdownPhaseOutcome.TIMED_OUT
  except Exception:
    return HostedApplicationShutdownPhaseOutcome.FAILED


def compute_shutdown_budget_seconds(
  *,
  shutdown_policy: ShutdownPolicy,
  blocking_hook_timeout: float,
  observer_drain_timeout: float,
  component_stop_budget: float,
  runtime_stop_budget: float,
  lease_release_timeout: float,
  explicit_deadline_at: datetime | None,
  clock: HostedApplicationClock,
  requested_at: datetime,
) -> float:
  policy_total = (
    shutdown_policy.drain_timeout_seconds
    + shutdown_policy.cancel_timeout_seconds
    + shutdown_policy.flush_timeout_seconds
  )
  internal = (
    blocking_hook_timeout
    + policy_total
    + component_stop_budget
    + runtime_stop_budget
    + observer_drain_timeout
    + lease_release_timeout
    + 5.0
  )
  if explicit_deadline_at is not None:
    wall_remaining = (explicit_deadline_at - clock.now()).total_seconds()
    return max(0.0, min(internal, wall_remaining))
  return max(0.0, internal)


@dataclass
class ShutdownPhaseRecorder:
  """Collects shutdown phase records and timeout/forced flags."""

  clock: HostedApplicationClock
  records: list[HostedApplicationShutdownPhaseRecord] = field(default_factory=list)
  timed_out: bool = False
  forced: bool = False

  def record(
    self,
    *,
    phase: HostedApplicationShutdownPhase,
    outcome: HostedApplicationShutdownPhaseOutcome,
    started_at: datetime,
    active_before: int | None = None,
    flush_id: str = "",
  ) -> None:
    if outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
      self.timed_out = True
    active_after = None
    self.records.append(
      HostedApplicationShutdownPhaseRecord(
        phase=phase,
        outcome=outcome,
        started_at=started_at,
        completed_at=self.clock.now(),
        flush_id=flush_id,
        active_work_before=active_before,
        active_work_after=active_after,
      )
    )


@dataclass
class HostedApplicationShutdownExecutor:
  """Executes bounded shutdown phases against active work and flush services."""

  shutdown_policy: ShutdownPolicy
  clock: HostedApplicationClock
  monotonic_clock: MonotonicClock = field(default_factory=SystemMonotonicClock)
  active_work_controller: HostedApplicationActiveWorkController | None = None
  flush_services: tuple[HostedApplicationFlushService, ...] = ()

  def __post_init__(self) -> None:
    seen: set[str] = set()
    for service in self.flush_services:
      flush_id = service.flush_id
      if flush_id in seen:
        raise ValueError(f"duplicate flush_id: {flush_id}")
      seen.add(flush_id)

  async def execute(
    self,
    *,
    request: HostedApplicationEffectiveControlRequest | None,
    budget: HostedApplicationGlobalShutdownBudget,
    recorder: ShutdownPhaseRecorder,
  ) -> None:
    active_before = self._active_count()
    strategy = self.shutdown_policy.strategy

    await self._phase_stop_intake(budget, recorder, active_before)
    if strategy is ShutdownStrategy.CANCEL_IMMEDIATELY:
      await self._phase_cancel(budget, recorder, active_before)
      if self._active_count() > 0:
        recorder.forced = True
    elif strategy is ShutdownStrategy.DRAIN_THEN_CANCEL:
      drained = await self._phase_drain(budget, recorder, active_before)
      if not drained and self._active_count() > 0:
        await self._phase_cancel(budget, recorder, active_before)
        cancel_outcome = recorder.records[-1].outcome
        if (
          self._active_count() > 0
          or cancel_outcome
          not in {
            HostedApplicationShutdownPhaseOutcome.COMPLETED,
            HostedApplicationShutdownPhaseOutcome.SKIPPED,
          }
        ):
          recorder.forced = True
    elif strategy is ShutdownStrategy.WAIT_UNTIL_COMPLETE:
      drained = await self._phase_drain(budget, recorder, active_before)
      if not drained and self._active_count() > 0:
        recorder.timed_out = True

    await self._phase_flush_all(budget, recorder)

  def _active_count(self) -> int:
    if self.active_work_controller is None:
      return 0
    return self.active_work_controller.active_work_count()

  async def _phase_stop_intake(
    self,
    budget: HostedApplicationGlobalShutdownBudget,
    recorder: ShutdownPhaseRecorder,
    active_before: int,
  ) -> None:
    started_at = self.clock.now()
    if self.active_work_controller is None:
      recorder.record(
        phase=HostedApplicationShutdownPhase.STOP_INTAKE,
        outcome=HostedApplicationShutdownPhaseOutcome.SKIPPED,
        started_at=started_at,
      )
      return
    outcome = await run_bounded_phase(
      budget,
      self.shutdown_policy.drain_timeout_seconds,
      self.active_work_controller.stop_intake,
    )
    recorder.record(
      phase=HostedApplicationShutdownPhase.STOP_INTAKE,
      outcome=outcome,
      started_at=started_at,
      active_before=active_before,
    )

  async def _phase_drain(
    self,
    budget: HostedApplicationGlobalShutdownBudget,
    recorder: ShutdownPhaseRecorder,
    active_before: int,
  ) -> bool:
    started_at = self.clock.now()
    if self.active_work_controller is None or self.shutdown_policy.drain_timeout_seconds <= 0:
      recorder.record(
        phase=HostedApplicationShutdownPhase.DRAIN,
        outcome=HostedApplicationShutdownPhaseOutcome.SKIPPED,
        started_at=started_at,
      )
      return True
    outcome = await run_bounded_phase(
      budget,
      self.shutdown_policy.drain_timeout_seconds,
      self.active_work_controller.wait_for_idle,
    )
    completed = outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED and self._active_count() == 0
    if outcome is HostedApplicationShutdownPhaseOutcome.TIMED_OUT:
      recorder.timed_out = True
    elif outcome is HostedApplicationShutdownPhaseOutcome.SKIPPED and budget.exhausted():
      recorder.timed_out = True
    record_outcome = outcome
    if outcome is HostedApplicationShutdownPhaseOutcome.COMPLETED and not completed:
      record_outcome = HostedApplicationShutdownPhaseOutcome.TIMED_OUT
      recorder.timed_out = True
    recorder.record(
      phase=HostedApplicationShutdownPhase.DRAIN,
      outcome=record_outcome,
      started_at=started_at,
      active_before=active_before,
    )
    return completed

  async def _phase_cancel(
    self,
    budget: HostedApplicationGlobalShutdownBudget,
    recorder: ShutdownPhaseRecorder,
    active_before: int,
  ) -> None:
    started_at = self.clock.now()
    if self.active_work_controller is None or self.shutdown_policy.cancel_timeout_seconds <= 0:
      recorder.record(
        phase=HostedApplicationShutdownPhase.CANCEL,
        outcome=HostedApplicationShutdownPhaseOutcome.SKIPPED,
        started_at=started_at,
      )
      return
    outcome = await run_bounded_phase(
      budget,
      self.shutdown_policy.cancel_timeout_seconds,
      self.active_work_controller.cancel_active_work,
    )
    if outcome is not HostedApplicationShutdownPhaseOutcome.COMPLETED:
      recorder.timed_out = True
      recorder.forced = True
    recorder.record(
      phase=HostedApplicationShutdownPhase.CANCEL,
      outcome=outcome,
      started_at=started_at,
      active_before=active_before,
    )

  async def _phase_flush_all(
    self,
    budget: HostedApplicationGlobalShutdownBudget,
    recorder: ShutdownPhaseRecorder,
  ) -> None:
    ordered = sorted(self.flush_services, key=lambda service: service.flush_id)
    for service in ordered:
      started_at = self.clock.now()
      outcome = await run_bounded_phase(
        budget,
        self.shutdown_policy.flush_timeout_seconds,
        service.flush,
      )
      recorder.record(
        phase=HostedApplicationShutdownPhase.FLUSH,
        outcome=outcome,
        started_at=started_at,
        flush_id=service.flush_id,
      )


def build_shutdown_execution_snapshot(
  *,
  shutdown_policy: ShutdownPolicy,
  request: HostedApplicationEffectiveControlRequest | None,
  clock: HostedApplicationClock,
  recorder: ShutdownPhaseRecorder,
  active_work_before: int,
  active_work_after: int,
) -> HostedApplicationShutdownExecutionSnapshot:
  requested_at = request.requested_at if request is not None else clock.now()
  effective_deadline_at = request.deadline_at if request is not None else None
  timed_out = recorder.timed_out
  if effective_deadline_at is not None and clock.now() >= effective_deadline_at:
    timed_out = True
  snapshot = HostedApplicationShutdownExecutionSnapshot(
    strategy=shutdown_policy.strategy,
    requested_at=requested_at,
    effective_deadline_at=effective_deadline_at,
    phase_records=tuple(recorder.records),
    active_work_before=active_work_before,
    active_work_after=active_work_after,
    timed_out=timed_out,
    forced=recorder.forced,
    completed_at=clock.now(),
  )
  return snapshot
