# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Restart policy evaluator and deterministic backoff (APP-HOST-5B)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.hosting.contracts.context import HostedApplicationClock
from intergrax.hosting.contracts.policies import RestartMode, RestartPolicy
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.errors import HostedApplicationRestartPolicyError
from intergrax.hosting.shutdown import MonotonicClock, SystemMonotonicClock
from intergrax.hosting.supervisor.classification import (
  HostedApplicationExitKind,
  HostedApplicationExitRecord,
)


@runtime_checkable
class HostedApplicationRandomSource(Protocol):
  def random(self) -> float: ...


@runtime_checkable
class HostedApplicationSleeper(Protocol):
  async def sleep(self, seconds: float) -> None: ...


class SystemRandomSource:
    def random(self) -> float:
        import random

        return random.random()


class AsyncioSleeper:
  async def sleep(self, seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


class HostedApplicationRestartDecision(BaseModel):
  """Deterministic restart decision for one terminal exit."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  should_restart: bool
  attempt_number: int
  delay_seconds: float = 0.0
  reason_code: str = ""
  exit_kind: HostedApplicationExitKind | None = None
  exhausted: bool = False


@dataclass
class HostedApplicationRestartAttempt:
  attempt_number: int
  started_at: datetime
  exit_record: HostedApplicationExitRecord | None = None


@dataclass
class HostedApplicationRestartPolicyEvaluator:
  """Evaluates restart policy, rolling windows, and bounded backoff."""

  policy: RestartPolicy
  clock: HostedApplicationClock
  random_source: HostedApplicationRandomSource = field(default_factory=SystemRandomSource)
  monotonic_clock: MonotonicClock = field(default_factory=SystemMonotonicClock)
  attempts: list[HostedApplicationRestartAttempt] = field(default_factory=list)
  _stable_since: datetime | None = None

  def record_launch(self, *, attempt_number: int, started_at: datetime) -> None:
    self.attempts.append(HostedApplicationRestartAttempt(attempt_number=attempt_number, started_at=started_at))

  def record_stable_runtime(self, *, stable_at: datetime, ready_duration_seconds: float) -> None:
    if ready_duration_seconds < self.policy.reset_after_stable_seconds:
      return
    self._stable_since = stable_at
    reset_after = timedelta(seconds=self.policy.reset_after_stable_seconds)
    self.attempts = [
      attempt
      for attempt in self.attempts
      if stable_at - attempt.started_at < reset_after
    ]

  def evaluate(
    self,
    exit_record: HostedApplicationExitRecord,
    *,
    attempt_number: int,
  ) -> HostedApplicationRestartDecision:
    self._prune_window()
    should_restart = self._should_restart(exit_record)
    if not should_restart:
      return HostedApplicationRestartDecision(
        should_restart=False,
        attempt_number=attempt_number,
        reason_code=exit_record.reason_code,
        exit_kind=exit_record.exit_kind,
      )
    replacement_count = sum(1 for attempt in self.attempts if attempt.attempt_number > 0)
    if replacement_count >= self.policy.max_attempts:
      return HostedApplicationRestartDecision(
        should_restart=False,
        attempt_number=attempt_number,
        reason_code="restart_exhausted",
        exit_kind=exit_record.exit_kind,
        exhausted=True,
      )
    delay = self._calculate_backoff(attempt_number=max(1, replacement_count + 1))
    return HostedApplicationRestartDecision(
      should_restart=True,
      attempt_number=attempt_number + 1,
      delay_seconds=delay,
      reason_code=exit_record.reason_code,
      exit_kind=exit_record.exit_kind,
    )

  def _prune_window(self) -> None:
    window = timedelta(seconds=self.policy.attempt_window_seconds)
    now = self.clock.now()
    self.attempts = [attempt for attempt in self.attempts if now - attempt.started_at <= window]

  def _should_restart(self, exit_record: HostedApplicationExitRecord) -> bool:
    mode = self.policy.mode
    if mode is RestartMode.NEVER:
      return False
    if mode is RestartMode.ALWAYS:
      return True
    if mode is RestartMode.ON_FAILURE:
      return exit_record.retryable or exit_record.exit_kind is HostedApplicationExitKind.RESTART_REQUESTED
    if mode is RestartMode.CUSTOM:
      classifier = self.policy.custom_classifier
      if classifier is None:
        raise HostedApplicationRestartPolicyError("custom restart mode requires classifier")
      return self._invoke_custom_classifier(classifier, exit_record)
    raise HostedApplicationRestartPolicyError(f"unsupported restart mode: {mode.value}")

  def _invoke_custom_classifier(
    self,
    classifier: Callable[..., bool],
    exit_record: HostedApplicationExitRecord,
  ) -> bool:
    signature = inspect.signature(classifier)
    params = list(signature.parameters.values())
    if len(params) != 1:
      raise HostedApplicationRestartPolicyError("custom classifier must accept exactly one parameter")
    if params[0].annotation not in (inspect.Parameter.empty, HostedApplicationExitRecord):
      annotation = params[0].annotation
      if annotation is not HostedApplicationExitRecord:
        raise HostedApplicationRestartPolicyError("custom classifier parameter must be HostedApplicationExitRecord")
    result = classifier(exit_record)
    if not isinstance(result, bool):
      raise HostedApplicationRestartPolicyError("custom classifier must return bool")
    return result

  def _calculate_backoff(self, *, attempt_number: int) -> float:
    base = min(
      self.policy.max_backoff_seconds,
      self.policy.initial_backoff_seconds * (self.policy.multiplier ** (attempt_number - 1)),
    )
    jitter_ratio = max(0.0, min(1.0, self.policy.jitter_ratio))
    factor = 1.0
    if jitter_ratio > 0.0:
      rnd = max(0.0, min(1.0, self.random_source.random()))
      factor = 1.0 - jitter_ratio + (2.0 * jitter_ratio * rnd)
    jittered = base * factor
    return max(0.0, min(self.policy.max_backoff_seconds, jittered))

  async def wait_backoff(
    self,
    delay_seconds: float,
    *,
    control: HostedApplicationControlCoordinator,
    sleeper: HostedApplicationSleeper,
    poll_interval: float = 0.05,
  ) -> bool:
    """Wait interruptibly; return False when persistent STOP prevents restart."""
    deadline = self.monotonic_clock.monotonic() + delay_seconds
    while self.monotonic_clock.monotonic() < deadline:
      if control.is_shutdown_requested():
        return False
      remaining = deadline - self.monotonic_clock.monotonic()
      await sleeper.sleep(min(poll_interval, max(0.0, remaining)))
    return not control.is_shutdown_requested()
