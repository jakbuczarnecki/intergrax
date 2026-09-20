# © Artur Czarnecki. All rights reserved.

"""Kafka broker readiness probing for cross-process qualification."""

from __future__ import annotations

import time

from confluent_kafka.admin import AdminClient


def kafka_broker_ready(
    bootstrap_servers: str,
    *,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.25,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            client = AdminClient({"bootstrap.servers": bootstrap_servers})
            client.list_topics(timeout=5.0)
            return True
        except Exception as exc:  # noqa: BLE001 — qualification probe
            last_error = str(exc)
            time.sleep(poll_interval_seconds)
    raise TimeoutError(
        f"Kafka broker not ready at {bootstrap_servers!r} within {timeout_seconds}s: {last_error}",
    )


def ensure_kafka_topic(
    bootstrap_servers: str,
    topic: str,
    *,
    num_partitions: int = 1,
    replication_factor: int = 1,
) -> None:
    from confluent_kafka.admin import NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=num_partitions, replication_factor=replication_factor)],
    )
    for _, future in futures.items():
        try:
            future.result(timeout=15.0)
        except Exception:
            # Topic may already exist for repeated qualification runs.
            continue
