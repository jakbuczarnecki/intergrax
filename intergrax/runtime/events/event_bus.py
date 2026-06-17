# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event bus (architecture §42.2)."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable, DefaultDict, List, Optional, Set, Union
from uuid import uuid4

from intergrax.runtime.events.event_catalog import should_persist_event
from intergrax.runtime.events.event_taxonomy import EventCategory
from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    resolve_event_tenant_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

logger = logging.getLogger(__name__)

EventHandler = Callable[[RuntimeEvent], Union[None, Awaitable[None]]]


@dataclass(frozen=True, slots=True)
class _TaxonomySubscription:
    subscription_id: str
    priority: int
    handler: EventHandler
    event_types: frozenset[RuntimeEventType] | None = None
    categories: frozenset[EventCategory] | None = None
    kind_prefix: str | None = None
    ops_hints: frozenset[str] | None = None

    def matches(self, event: RuntimeEvent) -> bool:
        if self.event_types is not None and event.event_type not in self.event_types:
            return False
        if self.categories is not None:
            if event.event_category is None or event.event_category not in self.categories:
                return False
        if self.kind_prefix is not None and not (event.event_kind or "").startswith(
            self.kind_prefix
        ):
            return False
        if self.ops_hints is not None and event.ops_hint not in self.ops_hints:
            return False
        return True


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
        self._taxonomy: List[_TaxonomySubscription] = []
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
        categories: Optional[Set[EventCategory]] = None,
        kind_prefix: Optional[str] = None,
        ops_hints: Optional[Set[str]] = None,
        priority: int = 100,
        subscription_id: Optional[str] = None,
    ) -> str:
        sid = subscription_id or f"sub_{uuid4().hex[:8]}"
        if categories is not None or kind_prefix is not None or ops_hints is not None:
            self._taxonomy.append(
                _TaxonomySubscription(
                    subscription_id=sid,
                    priority=priority,
                    handler=handler,
                    event_types=frozenset(event_types) if event_types else None,
                    categories=frozenset(categories) if categories else None,
                    kind_prefix=kind_prefix,
                    ops_hints=frozenset(ops_hints) if ops_hints else None,
                )
            )
            self._taxonomy.sort(key=lambda item: item.priority)
            return sid
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
        self._taxonomy = [t for t in self._taxonomy if t.subscription_id != subscription_id]
        for et in list(self._handlers.keys()):
            self._handlers[et] = [t for t in self._handlers[et] if t[0] != subscription_id]

    def _collect_handlers(self, event: RuntimeEvent) -> List[tuple[str, int, EventHandler]]:
        handlers: List[tuple[str, int, EventHandler]] = list(self._wildcard)
        handlers.extend(self._handlers.get(event.event_type, []))
        for sub in self._taxonomy:
            if sub.matches(event):
                handlers.append((sub.subscription_id, sub.priority, sub.handler))
        handlers.sort(key=lambda x: x[1])
        return handlers

    async def publish(self, event: RuntimeEvent) -> None:
        self.record(event)
        for sid, _prio, handler in self._collect_handlers(event):
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
        if self._persistence is not None and should_persist_event(event):
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
        for sid, _prio, handler in self._collect_handlers(event):
            try:
                result = handler(event)
                if result is not None:
                    logger.warning(
                        "RuntimeEventBus async handler %s skipped on sync record for %s",
                        sid,
                        event.event_type,
                    )
            except Exception:
                logger.exception(
                    "RuntimeEventBus handler %s failed on record for %s",
                    sid,
                    event.event_type,
                )
