# © Artur Czarnecki. All rights reserved.

"""Unit tests for ``QueueBackedEventDeliveryBuffer`` public contract."""

from __future__ import annotations

import threading

import pytest

from intergrax.contracts.event_delivery import EventDeliveryBufferCapacityExhausted
from intergrax.runtime.observability.event_delivery.queue_backed_event_delivery_buffer import (
    QueueBackedEventDeliveryBuffer,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_queue_backed_buffer_item_then_shutdown_ordering() -> None:
    buffer: QueueBackedEventDeliveryBuffer[str] = QueueBackedEventDeliveryBuffer(
        capacity=2
    )
    buffer.enqueue_item_nowait("user")
    buffer.enqueue_shutdown_nowait()
    first = buffer.take_next()
    second = buffer.take_next()
    assert first.kind == "item"
    assert first.item == "user"
    assert second.kind == "shutdown"
    buffer.acknowledge_processed()
    buffer.acknowledge_processed()


def test_queue_backed_buffer_nowait_capacity_exhausted() -> None:
    buffer: QueueBackedEventDeliveryBuffer[int] = QueueBackedEventDeliveryBuffer(
        capacity=1
    )
    buffer.enqueue_item_nowait(1)
    with pytest.raises(EventDeliveryBufferCapacityExhausted):
        buffer.enqueue_item_nowait(2)


def test_queue_backed_buffer_pending_depth_tracks_occupancy() -> None:
    buffer: QueueBackedEventDeliveryBuffer[str] = QueueBackedEventDeliveryBuffer(
        capacity=2
    )
    assert buffer.pending_depth == 0
    buffer.enqueue_item_nowait("a")
    assert buffer.pending_depth == 1
    taken = buffer.take_next()
    assert taken.kind == "item"
    buffer.acknowledge_processed()
    assert buffer.pending_depth == 0


def test_queue_backed_buffer_blocking_enqueue_waits_for_space() -> None:
    buffer: QueueBackedEventDeliveryBuffer[str] = QueueBackedEventDeliveryBuffer(
        capacity=1
    )
    buffer.enqueue_item_nowait("blocking")
    done = threading.Event()
    errors: list[BaseException] = []

    def _producer() -> None:
        try:
            buffer.enqueue_item("waiter", timeout=2.0)
            done.set()
        except BaseException as exc:  # noqa: BLE001 — capture for assertion
            errors.append(exc)

    thread = threading.Thread(target=_producer, name="buffer-waiter")
    thread.start()
    first = buffer.take_next()
    assert first.kind == "item"
    buffer.acknowledge_processed()
    assert done.wait(timeout=2.0)
    assert errors == []
    second = buffer.take_next()
    assert second.kind == "item"
    assert second.item == "waiter"
    buffer.acknowledge_processed()
    thread.join(timeout=2.0)
