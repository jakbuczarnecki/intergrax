# © Artur Czarnecki. All rights reserved.

"""Leak guard for the ``pdc_x`` suite: a module must not leave a non-daemon thread behind.

``threading._shutdown`` joins every non-daemon thread at interpreter exit, so one leaked
non-daemon thread keeps the pytest process alive after its summary line (R-05 of the P6
acceptance record). The guard fails the module at teardown and names the thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

_GRACE_SECONDS = 2.0
_POLL_SECONDS = 0.05


def live_non_daemon_threads() -> set[threading.Thread]:
    """The alive, non-daemon threads of the process right now."""
    return {thread for thread in threading.enumerate() if thread.is_alive() and not thread.daemon}


def describe_thread(thread: threading.Thread) -> str:
    """A report line for one thread: its name and the target it runs."""
    target = getattr(thread, "_target", None)
    target_name = (
        f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', repr(target))}"
        if target is not None
        else type(thread).__qualname__
    )
    return f"{thread.name!r} (target={target_name})"


def find_leaked_non_daemon_threads(
    before: set[threading.Thread], *, grace_seconds: float = _GRACE_SECONDS
) -> list[threading.Thread]:
    """Non-daemon threads started after ``before`` that are still alive after a short grace wait."""
    deadline = time.monotonic() + grace_seconds
    while True:
        leaked = sorted(live_non_daemon_threads() - before, key=lambda thread: thread.name)
        if not leaked or time.monotonic() >= deadline:
            return leaked
        time.sleep(_POLL_SECONDS)


@pytest.fixture(autouse=True, scope="module")
def _no_leaked_non_daemon_thread() -> Iterator[None]:
    before = live_non_daemon_threads()
    yield
    leaked = find_leaked_non_daemon_threads(before)
    if leaked:
        names = ", ".join(describe_thread(thread) for thread in leaked)
        pytest.fail(f"non-daemon thread(s) still alive after the module (the pytest process would not exit): {names}")
