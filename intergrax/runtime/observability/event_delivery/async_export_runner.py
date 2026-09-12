# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dedicated asyncio loop for sync drain thread → async export port calls."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


class AsyncExportRunner:
    """Runs coroutines from the bounded-event drain worker thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="w5c-export-async",
            daemon=True,
        )
        self._thread.start()

    def run(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        timeout_seconds: float = 30.0,
    ) -> _T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout_seconds)

    def shutdown(self, *, timeout_seconds: float = 10.0) -> None:
        if not self._loop.is_running():
            return

        def _stop() -> None:
            self._loop.stop()

        self._loop.call_soon_threadsafe(_stop)
        self._thread.join(timeout=timeout_seconds)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()
