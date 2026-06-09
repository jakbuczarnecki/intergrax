# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-flight task registry for mid-run cancel and autonomy changes (FLOW-CTL)."""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

from intergrax.runtime.task.task import Task

_ACTIVE: Dict[str, Task] = {}
_LOCK = asyncio.Lock()


class ActiveTaskRegistry:
    @staticmethod
    async def register(task: Task) -> None:
        async with _LOCK:
            _ACTIVE[task.task_id] = task

    @staticmethod
    async def unregister(task_id: str) -> None:
        async with _LOCK:
            _ACTIVE.pop(task_id, None)

    @staticmethod
    async def get(task_id: str) -> Optional[Task]:
        async with _LOCK:
            return _ACTIVE.get(task_id)

    @staticmethod
    async def list_ids() -> list[str]:
        async with _LOCK:
            return list(_ACTIVE.keys())

    @staticmethod
    def clear_for_tests() -> None:
        _ACTIVE.clear()
