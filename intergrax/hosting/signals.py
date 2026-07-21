# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Portable foreground signal adapter (APP-HOST-4E)."""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.errors import HostedApplicationControlError, HostedApplicationSignalError


from intergrax.utils import attribute_access


@runtime_checkable
class HostedApplicationSignalBridge(Protocol):
  """Bridge translating OS signals into typed control requests."""

  def install(self) -> None: ...

  def restore(self) -> None: ...


@runtime_checkable
class SignalApi(Protocol):
  """Injectable signal API for tests."""

  def signal(self, signum: int, handler: Any) -> Any: ...

  def getsignal(self, signum: int) -> Any: ...


class _DefaultSignalApi:
  def signal(self, signum: int, handler: Any) -> Any:
    return signal.signal(signum, handler)

  def getsignal(self, signum: int) -> Any:
    return signal.getsignal(signum)


@dataclass
class PortableForegroundSignalAdapter:
  """Reference foreground signal adapter delegating to the control coordinator."""

  coordinator: HostedApplicationControlCoordinator
  signal_api: SignalApi = field(default_factory=_DefaultSignalApi)
  enable_sighup_restart: bool = False
  _installed: bool = field(default=False, init=False, repr=False)
  _previous_handlers: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

  def install(self) -> None:
    if self._installed:
      return
    if threading.current_thread() is not threading.main_thread():
      raise HostedApplicationSignalError("signal adapter must install from main thread")
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
      signals.append(signal.SIGBREAK)
    if self.enable_sighup_restart and hasattr(signal, "SIGHUP"):
      signals.append(attribute_access.optional(signal, "SIGHUP"))
    for signum in signals:
      self._previous_handlers[signum] = self.signal_api.getsignal(signum)
      self.signal_api.signal(signum, self._make_handler(signum))
    if self.enable_sighup_restart and not hasattr(signal, "SIGHUP"):
      self.enable_sighup_restart = False
    self._installed = True

  def restore(self) -> None:
    if not self._installed:
      return
    for signum, previous in self._previous_handlers.items():
      self.signal_api.signal(signum, previous)
    self._previous_handlers.clear()
    self._installed = False

  def _make_handler(self, signum: int):
    def _handler(_sig: int, _frame: Any) -> None:
      if signum == signal.SIGINT:
        self.coordinator.request_shutdown("signal.sigint")
      elif signum == signal.SIGTERM:
        self.coordinator.request_shutdown("signal.sigterm")
      elif hasattr(signal, "SIGBREAK") and signum == signal.SIGBREAK:
        self.coordinator.request_shutdown("signal.sigbreak")
      elif (
        self.enable_sighup_restart
        and hasattr(signal, "SIGHUP")
        and signum == attribute_access.optional(signal, "SIGHUP")
      ):
        try:
          self.coordinator.request_restart("signal.sighup")
        except HostedApplicationControlError:
          return
    return _handler
