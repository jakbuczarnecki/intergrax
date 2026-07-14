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
      engine = await self._build_engine(launch)
      self._verify_engine_contract(engine, launch)
      try:
        terminal = await engine.run_until_stopped()
      except Exception as exc:
        classifier = HostedApplicationExitClassifier(
          restart_requested=self.control.is_restart_requested(),
        )
        last_exit = classifier.classify_exception(
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
            exit_record=last_exit,
          )
        )
        raise
      finally:
        await self._verify_engine_released(engine)

      classifier = HostedApplicationExitClassifier(
        restart_requested=self.control.is_restart_requested(),
      )
      shutdown_execution = terminal.diagnostics.shutdown_execution
      last_exit = classifier.classify_terminal_result(
        terminal,
        application_id=self.definition.application_id,
        instance_id=instance_id,
        profile_digest=self.definition.profile_digest,
        occurred_at=self.clock.now(),
        shutdown_execution=shutdown_execution,
      )
      attempt_records.append(
        HostedApplicationSupervisorAttemptRecord(
          attempt_number=attempt_number,
          instance_id=instance_id,
          exit_record=last_exit,
          terminal_result=terminal,
        )
      )

      if self.control.is_shutdown_requested():
        break

      decision = restart_evaluator.evaluate(last_exit, attempt_number=attempt_number)
      if not decision.should_restart:
        if decision.exhausted:
          exhausted = True
          await self._publish_restart_event(
            HostedApplicationEventType.RESTART_EXHAUSTED,
            attempt_number=attempt_number,
            delay=0.0,
            exit_kind=last_exit.exit_kind,
            reason_code=decision.reason_code,
          )
        break

      await self._publish_restart_event(
        HostedApplicationEventType.RESTART_REQUESTED,
        attempt_number=decision.attempt_number,
        delay=decision.delay_seconds,
        exit_kind=last_exit.exit_kind,
        reason_code=last_exit.reason_code,
      )
      await self._publish_restart_event(
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
        break
      await self._publish_restart_event(
        HostedApplicationEventType.RESTART_STARTED,
        attempt_number=decision.attempt_number,
        delay=0.0,
        exit_kind=last_exit.exit_kind,
        reason_code=last_exit.reason_code,
      )
      attempt_number = decision.attempt_number

    if last_exit is None:
      raise HostedApplicationSupervisorError("supervisor completed without terminal exit")
    return HostedApplicationSupervisorResult(
      application_id=self.definition.application_id,
      profile_digest=self.definition.profile_digest,
      definition_digest=self.definition.definition_digest,
      final_exit=last_exit,
      attempts=tuple(attempt_records),
      restart_exhausted=exhausted,
    )

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

  async def _verify_engine_released(self, engine: HostedApplicationEngine) -> None:
    diagnostics = engine.diagnostics_snapshot()
    if diagnostics.instance_lease_acquired and not diagnostics.instance_lease_released:
      raise HostedApplicationSupervisorError("prior engine lease not released")
    if not diagnostics.context_closed:
      raise HostedApplicationSupervisorError("prior engine context not closed")

  async def _publish_restart_event(
    self,
    event_type: HostedApplicationEventType,
    *,
    attempt_number: int,
    delay: float,
    exit_kind: HostedApplicationExitKind,
    reason_code: str,
  ) -> None:
    from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState

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
