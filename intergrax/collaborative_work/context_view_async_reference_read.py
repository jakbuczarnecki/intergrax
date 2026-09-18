# © Artur Czarnecki. All rights reserved.

"""Async reference-read runner seam for MP-5F-B5 (composition / wiring only).

Adapters depend on this protocol — they do not call ``asyncio.run`` or
``run_until_complete`` themselves.
"""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Protocol, TypeVar

T = TypeVar("T")


class ContextViewAsyncReferenceReadRunner(Protocol):
    """Execute an async source-domain ``read_references`` coroutine from sync MP-5D ports."""

    def run(self, coro: Coroutine[object, object, T]) -> T:
        """Run ``coro`` to completion and return its result."""
