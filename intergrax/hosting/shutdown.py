# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graceful shutdown execution contracts and executor (APP-HOST-4D)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.hosting.contracts.context import HostedApplicationClock
from intergrax.hosting.contracts.lifecycle import HostedApplicationShutdownRequestSnapshot
from intergrax.hosting.contracts.policies import ShutdownPolicy, ShutdownStrategy
from intergrax.hosting.errors import HostedApplicationShutdownTimeoutError


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
  STOP_INTAKE = "stop_intake"
  DRAIN = "drain"
  CANCEL = "cancel"
  FLUSH = "flush"


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
class ShutdownBudget:
  """Monotonic shutdown deadline budget shared across phases."""

  deadline_monotonic: float

  def remaining_seconds(self) -> float:
    return max(0.0, self.deadline_monotonic - time.monotonic())

  def exhausted(self) -> bool:
    return self.remaining_seconds() <= 0.0


async def _run_bounded(
  budget: ShutdownBudget,
  coro_factory: Callable[[], Awaitable[None]],
) -> bool:
  remaining = budget.remaining_seconds()
  if remaining <= 0.0:
    return False
  try:
    await asyncio.wait_for(coro_factory(), timeout=remaining)
    return True
  except TimeoutError:
    return False


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
    return max(0.1, min(internal, wall_remaining))
  return max(0.1, internal)


@dataclass
class HostedApplicationShutdownExecutor:
  """Executes bounded shutdown phases against active work and flush services."""

  shutdown_policy: ShutdownPolicy
  clock: HostedApplicationClock
  active_work_controller: HostedApplicationActiveWorkController | None = None
  flush_services: tuple[HostedApplicationFlushService, ...] = ()
  phase_records: list[HostedApplicationShutdownPhaseRecord] = field(default_factory=list)
  timed_out: bool = False
  forced: bool = False

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
    request: HostedApplicationShutdownRequestSnapshot | None,
    budget_seconds: float,
  ) -> HostedApplicationShutdownExecutionSnapshot:
    requested_at = request.requested_at if request is not None else self.clock.now()
    effective_deadline_at = request.deadline_at if request is not None else None
    budget = ShutdownBudget(deadline_monotonic=time.monotonic() + budget_seconds)
    active_before = self._active_count()
    strategy = self.shutdown_policy.strategy

    await self._phase_stop_intake(budget, active_before)
    if strategy is ShutdownStrategy.CANCEL_IMMEDIATELY:
      await self._phase_cancel(budget, active_before)
    elif strategy is ShutdownStrategy.DRAIN_THEN_CANCEL:
      drained = await self._phase_drain(budget, active_before)
      if not drained and self._active_count() > 0:
        await self._phase_cancel(budget, active_before)
        self.forced = True
    elif strategy is ShutdownStrategy.WAIT_UNTIL_COMPLETE:
      drained = await self._phase_drain(budget, active_before)
      if not drained and self._active_count() > 0:
        self.timed_out = True

    await self._phase_flush_all(budget)
    active_after = self._active_count()
    completed_at = self.clock.now()
    snapshot = HostedApplicationShutdownExecutionSnapshot(
      strategy=strategy,
      requested_at=requested_at,
      effective_deadline_at=effective_deadline_at,
      phase_records=tuple(self.phase_records),
      active_work_before=active_before,
      active_work_after=active_after,
      timed_out=self.timed_out,
      forced=self.forced,
      completed_at=completed_at,
    )
    if self.timed_out and budget.exhausted():
      raise HostedApplicationShutdownTimeoutError("shutdown deadline exhausted")
    return snapshot

  def _active_count(self) -> int:
    if self.active_work_controller is None:
      return 0
    return self.active_work_controller.active_work_count()

  async def _record_phase(
    self,
    *,
    phase: HostedApplicationShutdownPhase,
    outcome: HostedApplicationShutdownPhaseOutcome,
    started_at: datetime,
    active_before: int | None = None,
    flush_id: str = "",
  ) -> None:
    active_after = self._active_count() if active_before is not None else None
    self.phase_records.append(
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

  async def _phase_stop_intake(self, budget: ShutdownBudget, active_before: int) -> None:
    started_at = self.clock.now()
    if self.active_work_controller is None:
      await self._record_phase(
        phase=HostedApplicationShutdownPhase.STOP_INTAKE,
        outcome=HostedApplicationShutdownPhaseOutcome.SKIPPED,
        started_at=started_at,
      )
      return
    ok = await _run_bounded(budget, self.active_work_controller.stop_intake)
    outcome = (
      HostedApplicationShutdownPhaseOutcome.COMPLETED
      if ok
      else HostedApplicationShutdownPhaseOutcome.TIMED_OUT
    )
    if not ok:
      self.timed_out = True
    await self._record_phase(
      phase=HostedApplicationShutdownPhase.STOP_INTAKE,
      outcome=outcome,
      started_at=started_at,
      active_before=active_before,
    )

  async def _phase_drain(self, budget: ShutdownBudget, active_before: int) -> bool:
    started_at = self.clock.now()
    if self.active_work_controller is None or self.shutdown_policy.drain_timeout_seconds <= 0:
      await self._record_phase(
        phase=HostedApplicationShutdownPhase.DRAIN,
        outcome=HostedApplicationShutdownPhaseOutcome.SKIPPED,
        started_at=started_at,
      )
      return True
    drain_budget = ShutdownBudget(
      deadline_monotonic=min(
        budget.deadline_monotonic,
        time.monotonic() + self.shutdown_policy.drain_timeout_seconds,
      )
    )
    ok = await _run_bounded(drain_budget, self.active_work_controller.wait_for_idle)
    outcome = (
      HostedApplicationShutdownPhaseOutcome.COMPLETED
      if ok and self._active_count() == 0
      else HostedApplicationShutdownPhaseOutcome.TIMED_OUT
    )
    if not ok or self._active_count() > 0:
      self.timed_out = True
    await self._record_phase(
      phase=HostedApplicationShutdownPhase.DRAIN,
      outcome=outcome,
      started_at=started_at,
      active_before=active_before,
    )
    return ok and self._active_count() == 0

  async def _phase_cancel(self, budget: ShutdownBudget, active_before: int) -> None:
    started_at = self.clock.now()
    if self.active_work_controller is None or self.shutdown_policy.cancel_timeout_seconds <= 0:
      await self._record_phase(
        phase=HostedApplicationShutdownPhase.CANCEL,
        outcome=HostedApplicationShutdownPhaseOutcome.SKIPPED,
        started_at=started_at,
      )
      return
    cancel_budget = ShutdownBudget(
      deadline_monotonic=min(
        budget.deadline_monotonic,
        time.monotonic() + self.shutdown_policy.cancel_timeout_seconds,
      )
    )
    ok = await _run_bounded(cancel_budget, self.active_work_controller.cancel_active_work)
    outcome = (
      HostedApplicationShutdownPhaseOutcome.COMPLETED
      if ok
      else HostedApplicationShutdownPhaseOutcome.TIMED_OUT
    )
    if not ok:
      self.timed_out = True
      self.forced = True
    await self._record_phase(
      phase=HostedApplicationShutdownPhase.CANCEL,
      outcome=outcome,
      started_at=started_at,
      active_before=active_before,
    )

  async def _phase_flush_all(self, budget: ShutdownBudget) -> None:
    ordered = sorted(self.flush_services, key=lambda service: service.flush_id)
    for service in ordered:
      started_at = self.clock.now()
      flush_budget = ShutdownBudget(
        deadline_monotonic=min(
          budget.deadline_monotonic,
          time.monotonic() + self.shutdown_policy.flush_timeout_seconds,
        )
      )
      try:
        ok = await _run_bounded(flush_budget, service.flush)
        outcome = (
          HostedApplicationShutdownPhaseOutcome.COMPLETED
          if ok
          else HostedApplicationShutdownPhaseOutcome.TIMED_OUT
        )
        if not ok:
          self.timed_out = True
      except Exception:
        outcome = HostedApplicationShutdownPhaseOutcome.FAILED
      await self._record_phase(
        phase=HostedApplicationShutdownPhase.FLUSH,
        outcome=outcome,
        started_at=started_at,
        flush_id=service.flush_id,
      )
