# © Artur Czarnecki. All rights reserved.

"""Root-supervised catalog child work ingress (TLX-TL11 cross-loop)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RootChildWorkIngressRefusal(str, Enum):
    """Fail-closed refusal reasons for cross-loop catalog child submit."""

    SHUTDOWN = "shutdown"
    QUEUE_FULL = "queue_full"
    NOT_SUPERVISED = "not_supervised"
    WRONG_TASK = "wrong_task"
    SUBMIT_TIMEOUT = "submit_timeout"


@dataclass(frozen=True, slots=True)
class RootCatalogChildWorkTerminal:
    """Bounded terminal for one ingress ticket."""

    success: bool
    payload: dict[str, object]
    refusal: RootChildWorkIngressRefusal | None = None


@runtime_checkable
class RootExecutionChildWorkIngressPort(Protocol):
    """Queue catalog child work for cooperative drain on the supervised root asyncio.Task."""

    def bind_supervised_task(self, task_id: Any) -> None: ...

    def unbind_supervised_task(self, task_id: Any) -> None: ...

    def refuse_new_submissions(self) -> None: ...

    async def submit_catalog_child_work(
        self,
        ticket: Any,
        *,
        wait_timeout_s: float,
    ) -> RootCatalogChildWorkTerminal: ...

    def notify_ready(self) -> None: ...

    async def wait_ready(self) -> None: ...

    def has_ready_work(self) -> bool: ...

    async def drain_one(self) -> bool: ...


__all__ = [
    "RootCatalogChildWorkTerminal",
    "RootChildWorkIngressRefusal",
    "RootExecutionChildWorkIngressPort",
]
