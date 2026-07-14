# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import signal
import threading

import pytest

from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.errors import HostedApplicationSignalError
from intergrax.hosting.signals import PortableForegroundSignalAdapter
from tests.unit.hosting.engine._fakes import FixedClock

pytestmark = pytest.mark.unit


class FakeSignalApi:
    def __init__(self) -> None:
        self.handlers: dict[int, object] = {}
        self.previous: dict[int, object] = {}

    def signal(self, signum: int, handler: object) -> object:
        self.previous[signum] = self.handlers.get(signum, signal.SIG_DFL)
        self.handlers[signum] = handler
        return self.previous[signum]

    def getsignal(self, signum: int) -> object:
        return self.handlers.get(signum, signal.SIG_DFL)


def test_install_restore_and_sigint_mapping() -> None:
    coordinator = HostedApplicationControlCoordinator(clock=FixedClock())
    api = FakeSignalApi()
    adapter = PortableForegroundSignalAdapter(coordinator=coordinator, signal_api=api)
    adapter.install()
    handler = api.handlers[signal.SIGINT]
    assert callable(handler)
    handler(signal.SIGINT, None)
    assert coordinator.is_shutdown_requested()
    adapter.restore()
    adapter.restore()


def test_non_main_thread_rejected() -> None:
    coordinator = HostedApplicationControlCoordinator(clock=FixedClock())
    adapter = PortableForegroundSignalAdapter(coordinator=coordinator, signal_api=FakeSignalApi())
    error: list[Exception] = []

    def _worker() -> None:
        try:
            adapter.install()
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    assert error and isinstance(error[0], HostedApplicationSignalError)
