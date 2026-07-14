# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed shutdown/restart control coordinator (APP-HOST-4C)."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from intergrax.hosting.contracts.context import HostedApplicationClock
from intergrax.hosting.contracts.lifecycle import HostedApplicationShutdownRequestSnapshot
from intergrax.hosting.contracts.public_data import validate_bounded_identifier
from intergrax.hosting.errors import HostedApplicationControlError


def _validate_timezone_aware(value: datetime, *, field_name: str) -> datetime:
  if value.tzinfo is None:
    raise ValueError(f"{field_name} must be timezone-aware")
  return value


class HostedApplicationControlIntent(str, Enum):
  STOP = "stop"
  RESTART = "restart"


class HostedApplicationRestartRequestSnapshot(BaseModel):
  """Immutable restart request snapshot."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  reason_code: str
  requested_at: datetime
  deadline_at: datetime | None = None
  source_id: str = "runtime"

  @field_validator("reason_code", "source_id")
  @classmethod
  def _validate_identifiers(cls, value: str, info) -> str:
    return validate_bounded_identifier(value, field_name=info.field_name)

  @field_validator("requested_at", "deadline_at")
  @classmethod
  def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
    if value is None:
      return None
    return _validate_timezone_aware(value, field_name="timestamp")

  @model_validator(mode="after")
  def _validate_deadline_order(self) -> HostedApplicationRestartRequestSnapshot:
    if self.deadline_at is not None and self.deadline_at < self.requested_at:
      raise ValueError("deadline_at must not be earlier than requested_at")
    return self


class HostedApplicationControlSnapshot(BaseModel):
  """Immutable combined control state snapshot."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  effective_intent: HostedApplicationControlIntent | None = None
  shutdown_request: HostedApplicationShutdownRequestSnapshot | None = None
  restart_request: HostedApplicationRestartRequestSnapshot | None = None


@dataclass
class HostedApplicationControlCoordinator:
  """Thread-safe idempotent shutdown/restart request coordinator."""

  clock: HostedApplicationClock
  _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
  _shutdown_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
  _shutdown_request: HostedApplicationShutdownRequestSnapshot | None = field(default=None, repr=False)
  _restart_request: HostedApplicationRestartRequestSnapshot | None = field(default=None, repr=False)
  _effective_intent: HostedApplicationControlIntent | None = field(default=None, repr=False)

  def request_shutdown(
    self,
    reason_code: str,
    *,
    deadline_at: datetime | None = None,
    source_id: str = "runtime",
  ) -> HostedApplicationShutdownRequestSnapshot:
    safe_reason = validate_bounded_identifier(reason_code, field_name="reason_code")
    validate_bounded_identifier(source_id, field_name="source_id")
    if deadline_at is not None:
      _validate_timezone_aware(deadline_at, field_name="deadline_at")
    with self._lock:
      now = self.clock.now()
      incoming = HostedApplicationShutdownRequestSnapshot(
        reason_code=safe_reason,
        requested_at=now,
        deadline_at=deadline_at,
      )
      if self._shutdown_request is None:
        self._shutdown_request = incoming
      else:
        self._shutdown_request = self._coalesce_shutdown(self._shutdown_request, incoming)
      self._effective_intent = HostedApplicationControlIntent.STOP
      self._shutdown_event.set()
      return self._shutdown_request

  def request_restart(
    self,
    reason_code: str,
    *,
    deadline_at: datetime | None = None,
    source_id: str = "runtime",
  ) -> HostedApplicationRestartRequestSnapshot:
    safe_reason = validate_bounded_identifier(reason_code, field_name="reason_code")
    safe_source = validate_bounded_identifier(source_id, field_name="source_id")
    if deadline_at is not None:
      _validate_timezone_aware(deadline_at, field_name="deadline_at")
    with self._lock:
      if self._effective_intent is HostedApplicationControlIntent.STOP:
        raise HostedApplicationControlError("restart cannot override persistent stop")
      now = self.clock.now()
      incoming = HostedApplicationRestartRequestSnapshot(
        reason_code=safe_reason,
        requested_at=now,
        deadline_at=deadline_at,
        source_id=safe_source,
      )
      if self._restart_request is None:
        self._restart_request = incoming
        if self._effective_intent is None:
          self._effective_intent = HostedApplicationControlIntent.RESTART
      else:
        self._restart_request = self._coalesce_restart(self._restart_request, incoming)
      self._shutdown_event.set()
      return self._restart_request

  def is_shutdown_requested(self) -> bool:
    with self._lock:
      return self._effective_intent is HostedApplicationControlIntent.STOP

  def is_restart_requested(self) -> bool:
    with self._lock:
      return (
        self._effective_intent is HostedApplicationControlIntent.RESTART
        and self._restart_request is not None
      )

  def current_request(self) -> HostedApplicationShutdownRequestSnapshot | None:
    with self._lock:
      return self._shutdown_request

  def current_restart_request(self) -> HostedApplicationRestartRequestSnapshot | None:
    with self._lock:
      return self._restart_request

  def snapshot(self) -> HostedApplicationControlSnapshot:
    with self._lock:
      return HostedApplicationControlSnapshot(
        effective_intent=self._effective_intent,
        shutdown_request=self._shutdown_request,
        restart_request=self._restart_request,
      )

  def health_probe(self) -> HostedApplicationControlSnapshot:
    return self.snapshot()

  async def wait_until_requested(self) -> HostedApplicationShutdownRequestSnapshot:
    await self._shutdown_event.wait()
    with self._lock:
      if self._shutdown_request is not None:
        return self._shutdown_request
      if self._restart_request is not None:
        return HostedApplicationShutdownRequestSnapshot(
          reason_code=self._restart_request.reason_code,
          requested_at=self._restart_request.requested_at,
          deadline_at=self._restart_request.deadline_at,
        )
      raise HostedApplicationControlError("control event set without request")

  def prepare_next_instance(self) -> None:
    with self._lock:
      if self._effective_intent is HostedApplicationControlIntent.STOP:
        return
      self._restart_request = None
      self._effective_intent = None
      self._shutdown_event.clear()

  @staticmethod
  def _deadline_rank(deadline_at: datetime | None) -> tuple[int, float]:
    if deadline_at is None:
      return (0, 0.0)
    return (1, -deadline_at.timestamp())

  def _coalesce_shutdown(
    self,
    current: HostedApplicationShutdownRequestSnapshot,
    incoming: HostedApplicationShutdownRequestSnapshot,
  ) -> HostedApplicationShutdownRequestSnapshot:
    if self._deadline_rank(incoming.deadline_at) > self._deadline_rank(current.deadline_at):
      return incoming
    return current

  def _coalesce_restart(
    self,
    current: HostedApplicationRestartRequestSnapshot,
    incoming: HostedApplicationRestartRequestSnapshot,
  ) -> HostedApplicationRestartRequestSnapshot:
    if self._deadline_rank(incoming.deadline_at) > self._deadline_rank(current.deadline_at):
      return incoming
    return current
