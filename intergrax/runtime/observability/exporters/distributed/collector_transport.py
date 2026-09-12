# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP collector-boundary transport adapter for ``OtlpTransportPort`` (W5-F)."""

from __future__ import annotations

from intergrax.contracts.observability_export import (
    OtlpExportConfiguration,
    OtlpTransportError,
    OtlpTransportPort,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.observability.exporters.distributed.distributed_configuration import (
    DistributedTransportConfiguration,
    validate_distributed_transport_configuration,
)
from intergrax.runtime.observability.exporters.distributed.errors import (
    DistributedTransportError,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport


class CollectorTransport(OtlpTransportPort):
    """
    Maps ``RuntimeEvent`` → serialized telemetry → external OTLP collector.

    No queue, retry loop, circuit breaker, rate limit, or worker threads.
    """

    def __init__(self, config: DistributedTransportConfiguration) -> None:
        validated = validate_distributed_transport_configuration(config)
        self._config = validated
        self._inner = OtlpTransport(
            OtlpExportConfiguration(
                endpoint=validated.endpoint,
                protocol=validated.protocol,
                timeout_seconds=validated.timeout_seconds,
            ),
            service_name=validated.service_name,
        )

    def export(self, event: RuntimeEvent) -> None:
        try:
            self._inner.export(event)
        except OtlpTransportError as exc:
            raise DistributedTransportError(str(exc)) from exc

    def flush(self) -> None:
        self._inner.flush()

    def close(self) -> None:
        self._inner.close()
