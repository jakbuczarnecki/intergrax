# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-process hosted application supervisor (APP-HOST-5C)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from intergrax.hosting.contracts.context import HostedApplicationClock, HostedApplicationEventPublisher
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.contracts.public_data import validate_instance_id
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.definition import HostedApplicationDefinition
from intergrax.hosting.engine.diagnostics import HostedApplicationEngineTerminalResult
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.errors import HostedApplicationSupervisorError
from intergrax.hosting.supervisor.classification import (
  HostedApplicationExitClassifier,
  HostedApplicationExitKind,
  HostedApplicationExitRecord,
)
from intergrax.hosting.supervisor.restart import (
  AsyncioSleeper,
  HostedApplicationRandomSource,
  HostedApplicationRestartPolicyEvaluator,
  HostedApplicationSleeper,
  SystemRandomSource,
)

InstanceIdGenerator = Callable[[], str]


@runtime_checkable
class HostedApplicationEngineFactory(Protocol):
  def __call__(
    self,
    launch: HostedApplicationSupervisorLaunchContext,
  ) -> HostedApplicationEngine | Awaitable[HostedApplicationEngine]: ...


class HostedApplicationSupervisorLaunchContext(BaseModel):
  """Immutable launch context for one supervised engine instance."""

  model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

  definition: HostedApplicationDefinition = Field(repr=False)
  instance_id: str
  attempt_number: int
  control: HostedApplicationControlCoordinator = Field(repr=False)

  def __init__(self, **data: object) -> None:
    super().__init__(**data)
    validate_instance_id(self.instance_id)


class HostedApplicationSupervisorAttemptRecord(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  attempt_number: int
  instance_id: str
  exit_record: HostedApplicationExitRecord | None = None
  terminal_result: HostedApplicationEngineTerminalResult | None = Field(default=None, repr=False)
  cleanup_verified: bool = True
  cleanup_issue: str = ""


class HostedApplicationSupervisorResult(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  application_id: str
  profile_digest: str
  definition_digest: str
  final_exit: HostedApplicationExitRecord
  attempts: tuple[HostedApplicationSupervisorAttemptRecord, ...] = ()
  restart_exhausted: bool = False


@dataclass
class HostedApplicationSupervisor:
  """Reference in-process supervisor for one hosted application definition."""

  definition: HostedApplicationDefinition
  engine_factory: HostedApplicationEngineFactory
  control: HostedApplicationControlCoordinator
  event_publisher: HostedApplicationEventPublisher
  clock: HostedApplicationClock
  sleeper: HostedApplicationSleeper = field(default_factory=AsyncioSleeper)
  random_source: HostedApplicationRandomSource = field(default_factory=SystemRandomSource)
  instance_id_generator: InstanceIdGenerator = field(default_factory=lambda: (lambda: str(uuid4())))

  async def run(self) -> HostedApplicationSupervisorResult:
    restart_evaluator = HostedApplicationRestartPolicyEvaluator(
      policy=self.definition.restart_policy,
      clock=self.clock,
      random_source=self.random_source,
    )
    attempt_number = 0
    attempt_records: list[HostedApplicationSupervisorAttemptRecord] = []
    last_exit: HostedApplicationExitRecord | None = None
    exhausted = False

    if self.control.is_shutdown_requested():
      return self._result_for_stop_before_launch(attempt_records)

    while True:
      if self.control.is_shutdown_requested():
        break
      instance_id = validate_instance_id(self.instance_id_generator())
      self.control.prepare_next_instance()
      launch = HostedApplicationSupervisorLaunchContext(
        definition=self.definition,
        instance_id=instance_id,
        attempt_number=attempt_number,
        control=self.control,
      )
      restart_evaluator.record_launch(
        attempt_number=attempt_number,
        started_at=self.clock.now(),
      )

      engine: HostedApplicationEngine | None = None
      terminal: HostedApplicationEngineTerminalResult | None = None
      exit_record: HostedApplicationExitRecord | None = None
      cleanup_verified = True
      cleanup_issue = ""

      try:
        engine = await self._build_engine(launch)
        self._verify_engine_contract(engine, launch)
      except HostedApplicationSupervisorError as exc:
        classifier = HostedApplicationExitClassifier()
        exit_record = classifier.classify_exception(
          exc,
          application_id=self.definition.application_id,
          instance_id=instance_id,
          profile_digest=self.definition.profile_digest,
          occurred_at=self.clock.now(),
        )
        attempt_records.append(
          HostedApplicationSupervisorAttemptRecord(
            attempt_number=attempt_number,
            instance_id=instance_id,
            exit_record=exit_record,
            cleanup_verified=True,
          )
        )
        last_exit = exit_record
        decision = restart_evaluator.evaluate(exit_record, attempt_number=attempt_number)
        if not decision.should_restart:
          break
        if not await self._schedule_restart(
          restart_evaluator,
          decision,
          last_exit=last_exit,
          attempt_number=attempt_number,
        ):
          break
        attempt_number = decision.attempt_number
        continue

      try:
        terminal = await engine.run_until_stopped()
      except Exception as exc:
        classifier = HostedApplicationExitClassifier(
          restart_requested=self.control.is_restart_requested(),
        )
        exit_record = classifier.classify_exception(
          exc,
          application_id=self.definition.application_id,
          instance_id=instance_id,
          profile_digest=self.definition.profile_digest,
          occurred_at=self.clock.now(),
        )
      finally:
        if engine is not None:
          cleanup_verified, cleanup_issue = self._verify_engine_cleanup(engine)

      if exit_record is None and terminal is not None:
        if terminal.ready_duration_seconds is not None:
          restart_evaluator.record_stable_runtime(
            stable_at=self.clock.now(),
            ready_duration_seconds=terminal.ready_duration_seconds,
          )
        classifier = HostedApplicationExitClassifier(
          restart_requested=self.control.is_restart_requested(),
        )
        exit_record = classifier.classify_terminal_result(
          terminal,
          application_id=self.definition.application_id,
          instance_id=instance_id,
          profile_digest=self.definition.profile_digest,
          occurred_at=self.clock.now(),
          shutdown_execution=terminal.diagnostics.shutdown_execution,
        )

      assert exit_record is not None
      attempt_records.append(
        HostedApplicationSupervisorAttemptRecord(
          attempt_number=attempt_number,
          instance_id=instance_id,
          exit_record=exit_record,
          terminal_result=terminal,
          cleanup_verified=cleanup_verified,
          cleanup_issue=cleanup_issue,
        )
      )
      last_exit = exit_record

      if self.control.is_shutdown_requested():
        break

      decision = restart_evaluator.evaluate(exit_record, attempt_number=attempt_number)
      if not decision.should_restart:
        if decision.exhausted:
          exhausted = True
          await self._publish_restart_event_safe(
            HostedApplicationEventType.RESTART_EXHAUSTED,
            attempt_number=attempt_number,
            delay=0.0,
            exit_kind=exit_record.exit_kind,
            reason_code=decision.reason_code,
          )
        break

      if not cleanup_verified:
        break

      if not await self._schedule_restart(
        restart_evaluator,
        decision,
        last_exit=exit_record,
        attempt_number=attempt_number,
      ):
        break
      attempt_number = decision.attempt_number

    if last_exit is None:
      return self._result_for_stop_before_launch(attempt_records)
    return HostedApplicationSupervisorResult(
      application_id=self.definition.application_id,
      profile_digest=self.definition.profile_digest,
      definition_digest=self.definition.definition_digest,
      final_exit=last_exit,
      attempts=tuple(attempt_records),
      restart_exhausted=exhausted,
    )

  def _result_for_stop_before_launch(
    self,
    attempt_records: list[HostedApplicationSupervisorAttemptRecord],
  ) -> HostedApplicationSupervisorResult:
    exit_record = HostedApplicationExitRecord(
      exit_kind=HostedApplicationExitKind.CLEAN_STOP,
      retryable=False,
      reason_code="stop_before_launch",
      application_id=self.definition.application_id,
      instance_id="supervisor",
      profile_digest=self.definition.profile_digest,
      terminal_lifecycle_state=HostedApplicationLifecycleState.STOPPED,
      occurred_at=self.clock.now(),
    )
    return HostedApplicationSupervisorResult(
      application_id=self.definition.application_id,
      profile_digest=self.definition.profile_digest,
      definition_digest=self.definition.definition_digest,
      final_exit=exit_record,
      attempts=tuple(attempt_records),
      restart_exhausted=False,
    )

  async def _schedule_restart(
    self,
    restart_evaluator: HostedApplicationRestartPolicyEvaluator,
    decision,
    *,
    last_exit: HostedApplicationExitRecord,
    attempt_number: int,
  ) -> bool:
    await self._publish_restart_event_safe(
      HostedApplicationEventType.RESTART_REQUESTED,
      attempt_number=decision.attempt_number,
      delay=decision.delay_seconds,
      exit_kind=last_exit.exit_kind,
      reason_code=last_exit.reason_code,
    )
    await self._publish_restart_event_safe(
      HostedApplicationEventType.RESTART_SCHEDULED,
      attempt_number=decision.attempt_number,
      delay=decision.delay_seconds,
      exit_kind=last_exit.exit_kind,
      reason_code=last_exit.reason_code,
    )
    if not await restart_evaluator.wait_backoff(
      decision.delay_seconds,
      control=self.control,
      sleeper=self.sleeper,
    ):
      return False
    await self._publish_restart_event_safe(
      HostedApplicationEventType.RESTART_STARTED,
      attempt_number=decision.attempt_number,
      delay=0.0,
      exit_kind=last_exit.exit_kind,
      reason_code=last_exit.reason_code,
    )
    return True

  async def _build_engine(self, launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
    try:
      produced = self.engine_factory(launch)
      if asyncio.iscoroutine(produced) or asyncio.isfuture(produced):
        engine = await produced
      else:
        engine = produced
    except Exception as exc:
      raise HostedApplicationSupervisorError("engine factory failed") from exc
    if not isinstance(engine, HostedApplicationEngine):
      raise HostedApplicationSupervisorError("engine factory returned invalid type")
    return engine

  def _verify_engine_contract(
    self,
    engine: HostedApplicationEngine,
    launch: HostedApplicationSupervisorLaunchContext,
  ) -> None:
    if engine.instance_id != launch.instance_id:
      raise HostedApplicationSupervisorError("engine instance_id mismatch")
    if engine.definition.profile_digest != self.definition.profile_digest:
      raise HostedApplicationSupervisorError("engine profile_digest mismatch")
    if engine.definition.definition_digest != self.definition.definition_digest:
      raise HostedApplicationSupervisorError("engine definition_digest mismatch")
    if engine.definition.application_id != self.definition.application_id:
      raise HostedApplicationSupervisorError("engine application_id mismatch")

  def _verify_engine_cleanup(self, engine: HostedApplicationEngine) -> tuple[bool, str]:
    diagnostics = engine.diagnostics_snapshot()
    if diagnostics.instance_lease_acquired and not diagnostics.instance_lease_released:
      return False, "prior_engine_lease_not_released"
    if not diagnostics.context_closed:
      return False, "prior_engine_context_not_closed"
    return True, ""

  async def _publish_restart_event_safe(
    self,
    event_type: HostedApplicationEventType,
    *,
    attempt_number: int,
    delay: float,
    exit_kind: HostedApplicationExitKind,
    reason_code: str,
  ) -> None:
    try:
      await self.event_publisher.publish(
        HostedApplicationEvent(
          event_type=event_type,
          application_id=self.definition.application_id,
          instance_id="supervisor",
          lifecycle_state=HostedApplicationLifecycleState.STOPPED,
          payload={
            "attempt_number": attempt_number,
            "delay_seconds": delay,
            "exit_kind": exit_kind.value,
            "reason_code": reason_code,
            "profile_digest": self.definition.profile_digest,
          },
        )
      )
    except Exception:
      return
