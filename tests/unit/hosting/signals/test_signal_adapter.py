# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import signal
import threading
from typing import Callable, cast

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


def test_sigterm_then_sighup_preserves_stop() -> None:
    coordinator = HostedApplicationControlCoordinator(clock=FixedClock())
    api = FakeSignalApi()
    adapter = PortableForegroundSignalAdapter(
        coordinator=coordinator,
        signal_api=api,
        enable_sighup_restart=True,
    )
    adapter.install()
    term_handler = cast(Callable[[int, object | None], None], api.handlers[signal.SIGTERM])
    term_handler(signal.SIGTERM, None)
    sighup = getattr(signal, "SIGHUP", None)
    if sighup is not None:
        hup_handler = cast(Callable[[int, object | None], None], api.handlers[sighup])
        hup_handler(sighup, None)
    else:
        coordinator.request_restart("signal.sighup")
    assert coordinator.is_shutdown_requested()
    effective = coordinator.current_effective_request()
    assert effective is not None
    assert effective.intent == "stop"
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
