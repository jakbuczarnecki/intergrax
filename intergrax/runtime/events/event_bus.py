# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event bus (architecture §42.2)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Awaitable, Callable, DefaultDict, List, Optional, Set, Union
from uuid import uuid4

from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    resolve_event_tenant_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

logger = logging.getLogger(__name__)

EventHandler = Callable[[RuntimeEvent], Union[None, Awaitable[None]]]


class RuntimeEventBus:
    """
    Synchronous-first pub/sub for runtime signals.

    Hooks and policy subscribe here; metrics sinks may use async fan-out later (§42.2).
    """

    def __init__(
        self,
        *,
        persistence: Optional[RuntimeEventPersistence] = None,
        record_history: bool = True,
    ) -> None:
        self._handlers: DefaultDict[RuntimeEventType, List[tuple[str, int, EventHandler]]] = (
            defaultdict(list)
        )
        self._wildcard: List[tuple[str, int, EventHandler]] = []
        self._history: List[RuntimeEvent] = []
        self._record_history: bool = record_history
        self._persistence: Optional[RuntimeEventPersistence] = persistence

    def attach_persistence(self, persistence: RuntimeEventPersistence) -> None:
        """Wire or replace the persistence adapter after construction."""
        self._persistence = persistence

    @property
    def persistence(self) -> Optional[RuntimeEventPersistence]:
        return self._persistence

    def subscribe(
        self,
        handler: EventHandler,
        *,
        event_types: Optional[Set[RuntimeEventType]] = None,
        priority: int = 100,
        subscription_id: Optional[str] = None,
    ) -> str:
        sid = subscription_id or f"sub_{uuid4().hex[:8]}"
        if event_types is None:
            self._wildcard.append((sid, priority, handler))
            self._wildcard.sort(key=lambda x: x[1])
            return sid
        for et in event_types:
            self._handlers[et].append((sid, priority, handler))
            self._handlers[et].sort(key=lambda x: x[1])
        return sid

    def unsubscribe(self, subscription_id: str) -> None:
        self._wildcard = [t for t in self._wildcard if t[0] != subscription_id]
        for et in list(self._handlers.keys()):
            self._handlers[et] = [t for t in self._handlers[et] if t[0] != subscription_id]

    async def publish(self, event: RuntimeEvent) -> None:
        self.record(event)
        handlers = list(self._wildcard)
        handlers.extend(self._handlers.get(event.event_type, []))
        handlers.sort(key=lambda x: x[1])
        for sid, _prio, handler in handlers:
            try:
                result = handler(event)
                if result is not None:
                    await result
            except Exception:
                logger.exception("RuntimeEventBus handler %s failed for %s", sid, event.event_type)

    @property
    def history(self) -> List[RuntimeEvent]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def record(self, event: RuntimeEvent, *, tenant_id: Optional[str] = None) -> None:
        """Synchronous append for callers that cannot await (e.g. TaskLifecycle)."""
        if self._record_history:
            self._history.append(event)
        if self._persistence is not None:
            try:
                self._persistence.append(
                    event,
                    tenant_id=resolve_event_tenant_id(event, tenant_id),
                )
            except Exception:
                logger.exception(
                    "RuntimeEvent persistence failed for %s",
                    event.event_type.value,
                )
