# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosting lifecycle metric events (APP-HOST-3D).

Lifecycle metrics are typed hosting events (``hosting.lifecycle.metric``) published on the
existing ``HostedApplicationEventPublisher`` port and projected by the existing
observability export bridge. This module only builds events; hosting owns no metrics
store, registry, or backend.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import JsonValue

from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState


class HostedApplicationLifecycleMetric(StrEnum):
    """Stable names of the hosting lifecycle metrics."""

    START = "start"
    READY = "ready"
    DEGRADED = "degraded"
    DRAIN_DURATION = "drain_duration"
    RESTART_COUNT = "restart_count"
    EXIT_CLASS = "exit_class"


_UNITS: dict[HostedApplicationLifecycleMetric, str] = {
    HostedApplicationLifecycleMetric.START: "count",
    HostedApplicationLifecycleMetric.READY: "seconds",
    HostedApplicationLifecycleMetric.DEGRADED: "count",
    HostedApplicationLifecycleMetric.DRAIN_DURATION: "seconds",
    HostedApplicationLifecycleMetric.RESTART_COUNT: "count",
    HostedApplicationLifecycleMetric.EXIT_CLASS: "count",
}


def lifecycle_metric_event(
    metric: HostedApplicationLifecycleMetric,
    *,
    application_id: str,
    instance_id: str,
    lifecycle_state: HostedApplicationLifecycleState,
    value: float,
    label: str = "",
    occurred_at: datetime | None = None,
) -> HostedApplicationEvent:
    """Build one lifecycle metric event.

    ``label`` carries the bounded classification of the sample (for example the exit class
    or the degraded component ids).
    """
    payload: dict[str, JsonValue] = {
        "metric": metric.value,
        "value": float(value),
        "unit": _UNITS[metric],
        "label": label,
    }
    fields: dict[str, object] = {}
    if occurred_at is not None:
        fields["occurred_at"] = occurred_at
    return HostedApplicationEvent(
        event_type=HostedApplicationEventType.LIFECYCLE_METRIC,
        application_id=application_id,
        instance_id=instance_id,
        lifecycle_state=lifecycle_state,
        payload=payload,
        **fields,  # type: ignore[arg-type]
    )
