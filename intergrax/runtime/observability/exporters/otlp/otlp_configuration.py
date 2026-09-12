# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable OTLP export configuration validation (composition boundary)."""

from __future__ import annotations

from intergrax.contracts.observability_export import (
    ConfigurationError,
    OtlpExportConfiguration,
)


def validate_otlp_export_configuration(config: OtlpExportConfiguration) -> OtlpExportConfiguration:
    endpoint = config.endpoint.strip()
    if not endpoint:
        raise ConfigurationError("otlp export endpoint must be non-empty")
    if config.timeout_seconds <= 0:
        raise ConfigurationError("otlp export timeout_seconds must be positive")
    if endpoint != config.endpoint:
        return OtlpExportConfiguration(
            endpoint=endpoint,
            protocol=config.protocol,
            timeout_seconds=config.timeout_seconds,
        )
    return config
