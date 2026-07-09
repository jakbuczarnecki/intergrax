# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Kafka integration configuration (Phase M.4)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_KAFKA_BOOTSTRAP_SERVERS = "INTERGRAX_KAFKA_BOOTSTRAP_SERVERS"
ENV_KAFKA_TOPIC = "INTERGRAX_KAFKA_TOPIC"
ENV_KAFKA_EVENTS_TOPIC = "INTERGRAX_KAFKA_EVENTS_TOPIC"
ENV_KAFKA_STATUS_TOPIC = "INTERGRAX_KAFKA_STATUS_TOPIC"
ENV_KAFKA_RESULTS_TOPIC = "INTERGRAX_KAFKA_RESULTS_TOPIC"
ENV_KAFKA_CONSUMER_GROUP = "INTERGRAX_KAFKA_CONSUMER_GROUP"

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "intergrax.tasks"
DEFAULT_EVENTS_TOPIC = "intergrax.task-events"
DEFAULT_STATUS_TOPIC = "intergrax.task-status"
DEFAULT_RESULTS_TOPIC = "intergrax.task-results"
DEFAULT_CONSUMER_GROUP = "intergrax-default"


class KafkaIntegrationConfig(BaseIntegrationConfig):
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS
    topic: str = DEFAULT_TOPIC
    events_topic: str = DEFAULT_EVENTS_TOPIC
    status_topic: str = DEFAULT_STATUS_TOPIC
    results_topic: str = DEFAULT_RESULTS_TOPIC
    consumer_group: str = DEFAULT_CONSUMER_GROUP

    @classmethod
    def from_env(cls, **overrides: object) -> KafkaIntegrationConfig:
        bootstrap = (
            os.environ.get(ENV_KAFKA_BOOTSTRAP_SERVERS, DEFAULT_BOOTSTRAP_SERVERS).strip()
            or DEFAULT_BOOTSTRAP_SERVERS
        )
        topic = os.environ.get(ENV_KAFKA_TOPIC, DEFAULT_TOPIC).strip() or DEFAULT_TOPIC
        events_topic = (
            os.environ.get(ENV_KAFKA_EVENTS_TOPIC, DEFAULT_EVENTS_TOPIC).strip()
            or DEFAULT_EVENTS_TOPIC
        )
        status_topic = (
            os.environ.get(ENV_KAFKA_STATUS_TOPIC, DEFAULT_STATUS_TOPIC).strip()
            or DEFAULT_STATUS_TOPIC
        )
        results_topic = (
            os.environ.get(ENV_KAFKA_RESULTS_TOPIC, DEFAULT_RESULTS_TOPIC).strip()
            or DEFAULT_RESULTS_TOPIC
        )
        group = (
            os.environ.get(ENV_KAFKA_CONSUMER_GROUP, DEFAULT_CONSUMER_GROUP).strip()
            or DEFAULT_CONSUMER_GROUP
        )
        payload: dict[str, object] = {
            "bootstrap_servers": bootstrap,
            "topic": topic,
            "events_topic": events_topic,
            "status_topic": status_topic,
            "results_topic": results_topic,
            "consumer_group": group,
        }
        payload.update(overrides)
        return cls.model_validate(payload)
