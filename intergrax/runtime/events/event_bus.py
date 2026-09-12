# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event bus (architecture §42.2)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable, DefaultDict, List, Optional, Set, Union
from uuid import uuid4

from intergrax.contracts.event_delivery import (
    CriticalEventDeliveryError,
    EventDeliveryDisposition,
    EventPriority,
    EventSinkPort,
)
from intergrax.runtime.events.event_taxonomy import EventCategory
from intergrax.runtime.observability.event_delivery.delivery_metrics import (
    InternalDeliveryMetrics,
)
from intergrax.runtime.observability.event_delivery.runtime_event_delivery import (
    delivery_priority_for_runtime_event,
    runtime_event_to_deliverable,
)
from intergrax.runtime.events.evidence_durability import (
    EvidencePersistenceRequirement,
    evidence_persistence_requirement,
)
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.runtime.events.evidence_persistence_adapter import as_evidence_persistence_port
from intergrax.runtime.events.persistence_contract import (
    MandatoryEvidencePersistenceError,
    resolve_event_tenant_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

logger = logging.getLogger(__name__)

EventHandler = Callable[[RuntimeEvent], Union[None, Awaitable[None]]]


def _dispatch_handler_result_sync(result: Awaitable[None]) -> None:
    """Run or schedule an async handler result from synchronous ``record`` dispatch."""
    if inspect.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(result)
            return
        loop.create_task(result)
        return

    async def _await_result() -> None:
        await result

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_await_result())
        return
    loop.create_task(_await_result())


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
        persistence: Optional[EvidencePersistencePort] = None,
        record_history: bool = True,
        event_sink: EventSinkPort | None = None,
        delivery_metrics: InternalDeliveryMetrics | None = None,
    ) -> None:
        self._handlers: DefaultDict[RuntimeEventType, List[tuple[str, int, EventHandler]]] = (
            defaultdict(list)
        )
        self._wildcard: List[tuple[str, int, EventHandler]] = []
        self._taxonomy: List[_TaxonomySubscription] = []
        self._history: List[RuntimeEvent] = []
        self._record_history: bool = record_history
        self._persistence: Optional[EvidencePersistencePort] = as_evidence_persistence_port(
            persistence,
        )
        self._event_sink: EventSinkPort | None = event_sink
        self._delivery_metrics: InternalDeliveryMetrics | None = (
            delivery_metrics
            if delivery_metrics is not None
            else (InternalDeliveryMetrics() if event_sink is not None else None)
        )
        self._closed = False

    def attach_persistence(self, persistence: EvidencePersistencePort) -> None:
        """Wire or replace the persistence adapter after construction."""
        self._persistence = as_evidence_persistence_port(persistence)

    @property
    def persistence(self) -> Optional[EvidencePersistencePort]:
        return self._persistence

    @property
    def delivery_metrics(self) -> InternalDeliveryMetrics | None:
        return self._delivery_metrics

    @property
    def event_sink(self) -> EventSinkPort | None:
        return self._event_sink

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Drain and stop an optional observability ``EventSinkPort`` (W5-B)."""
        if self._closed:
            return
        self._closed = True
        if self._event_sink is not None:
            self._event_sink.close()

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
        """Persist then notify subscribers once (async handlers are awaited)."""
        self._commit_durable_evidence(event)
        self._deliver_through_event_sink(event)
        await self._dispatch_handlers_async(event)

    @property
    def history(self) -> List[RuntimeEvent]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def record(self, event: RuntimeEvent, *, tenant_id: Optional[str] = None) -> None:
        """Synchronous append for callers that cannot await (e.g. TaskLifecycle)."""
        self._commit_durable_evidence(event, tenant_id=tenant_id)
        self._deliver_through_event_sink(event)
        self._dispatch_handlers_sync(event)

    def _deliver_through_event_sink(self, event: RuntimeEvent) -> None:
        sink = self._event_sink
        if sink is None:
            return
        from intergrax.runtime.observability.event_delivery.bounded_event_sink import (
            BoundedEventSink,
        )

        priority = delivery_priority_for_runtime_event(event)
        deliverable = runtime_event_to_deliverable(event)
        started = time.monotonic()
        if isinstance(sink, BoundedEventSink):
            result = sink.publish(
                deliverable,
                priority=priority,
                source_event=event,
            )
        else:
            result = sink.publish(deliverable, priority=priority)
        latency = time.monotonic() - started
        metrics = self._delivery_metrics
        if metrics is not None:
            metrics.record(result, latency_seconds=latency)
        if priority is EventPriority.CRITICAL:
            if result.disposition is EventDeliveryDisposition.DROPPED:
                raise CriticalEventDeliveryError(
                    f"critical runtime event {deliverable.event_id} was dropped at sink",
                )
            if result.disposition is EventDeliveryDisposition.REJECTED:
                raise CriticalEventDeliveryError(
                    f"critical runtime event {deliverable.event_id} was rejected at sink",
                )

    def _commit_durable_evidence(
        self,
        event: RuntimeEvent,
        *,
        tenant_id: Optional[str] = None,
    ) -> None:
        requirement = evidence_persistence_requirement(event)
        if self._persistence is not None and requirement is not EvidencePersistenceRequirement.NOT_PERSISTED:
            scoped_tenant = resolve_event_tenant_id(event, tenant_id)
            try:
                self._persistence.append(event, tenant_id=scoped_tenant)
            except MandatoryEvidencePersistenceError:
                raise
            except Exception as exc:
                if requirement is EvidencePersistenceRequirement.MANDATORY:
                    raise MandatoryEvidencePersistenceError(
                        "mandatory runtime event evidence persistence failed for "
                        f"{event.event_type.value}",
                    ) from exc
                logger.exception(
                    "RuntimeEvent persistence failed for %s",
                    event.event_type.value,
                )
        if self._record_history:
            self._history.append(event)

    async def _dispatch_handlers_async(self, event: RuntimeEvent) -> None:
        for sid, _prio, handler in self._collect_handlers(event):
            try:
                result = handler(event)
                if result is not None:
                    await result
            except Exception:
                logger.exception(
                    "RuntimeEventBus handler %s failed for %s",
                    sid,
                    event.event_type,
                )

    def _dispatch_handlers_sync(self, event: RuntimeEvent) -> None:
        for sid, _prio, handler in self._collect_handlers(event):
            try:
                result = handler(event)
                if result is not None:
                    _dispatch_handler_result_sync(result)
            except Exception:
                logger.exception(
                    "RuntimeEventBus handler %s failed on record for %s",
                    sid,
                    event.event_type,
                )
