# © Artur Czarnecki. All rights reserved.

"""Kafka lifecycle event publisher for reviewer-visible task timelines."""

from __future__ import annotations

import json

from intergrax.background_tasks.events import TaskEvent, TaskEventEmitter, TaskEventName
from intergrax.queueing.contracts.message_producer import MessageProducer


class KafkaTaskLifecycleEmitter:
    """Publishes TaskEvent records to Kafka observability topics."""

    def __init__(
        self,
        *,
        producer: MessageProducer,
        events_topic: str,
        status_topic: str,
        results_topic: str,
    ) -> None:
        self._producer = producer
        self._events_topic = events_topic
        self._status_topic = status_topic
        self._results_topic = results_topic

    def emit(self, event: TaskEvent) -> None:
        payload = json.dumps(event.to_record(), separators=(",", ":"), sort_keys=True).encode("utf-8")
        self._producer.publish(topic=self._events_topic, payload=payload)
        if event.name in {
            TaskEventName.STARTED,
            TaskEventName.SUCCEEDED,
            TaskEventName.FAILED,
        }:
            self._producer.publish(topic=self._status_topic, payload=payload)
        if event.name == TaskEventName.RESULT_STORED:
            self._producer.publish(topic=self._results_topic, payload=payload)
