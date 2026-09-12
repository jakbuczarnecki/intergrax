# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable distributed OTLP transport configuration (W5-F)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.observability_export import (
    ConfigurationError,
    OtlpProtocol,
)


@dataclass(frozen=True, slots=True)
class DistributedTransportConfiguration:
    endpoint: str
    protocol: OtlpProtocol
    service_name: str
    timeout_seconds: float


def validate_distributed_transport_configuration(
    config: DistributedTransportConfiguration,
) -> DistributedTransportConfiguration:
    endpoint = config.endpoint.strip()
    service_name = config.service_name.strip()
    if not endpoint:
        raise ConfigurationError("distributed transport endpoint must be non-empty")
    if not service_name:
        raise ConfigurationError("distributed transport service_name is required")
    if config.timeout_seconds <= 0:
        raise ConfigurationError("distributed transport timeout_seconds must be positive")
    if endpoint != config.endpoint or service_name != config.service_name:
        return DistributedTransportConfiguration(
            endpoint=endpoint,
            protocol=config.protocol,
            service_name=service_name,
            timeout_seconds=config.timeout_seconds,
        )
    return config
