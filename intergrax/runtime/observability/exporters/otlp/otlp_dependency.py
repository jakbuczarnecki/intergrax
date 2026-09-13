# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional OTLP SDK dependency boundary (W5-H1)."""

from __future__ import annotations

from intergrax.contracts.observability_export import ConfigurationError, ExporterKind

OTLP_OBSERVABILITY_PROFILE_EXTRA = "observability-otlp"

_OTLP_DEPENDENCY_MESSAGE = (
    "DISTRIBUTED_OTLP requires the OTLP observability dependency profile."
)
_OTLP_EXPORT_DEPENDENCY_MESSAGE = (
    "OTLP export requires the OTLP observability dependency profile."
)


def require_otlp_observability_dependency_profile(
    *,
    exporter_kind: ExporterKind | None = None,
) -> None:
    """Fail closed when OTLP SDK packages are not installed (no silent fallback)."""
    message = _OTLP_DEPENDENCY_MESSAGE
    if exporter_kind is ExporterKind.OTLP:
        message = _OTLP_EXPORT_DEPENDENCY_MESSAGE
    try:
        import opentelemetry.sdk._logs  # noqa: F401
        import opentelemetry.exporter.otlp.proto.grpc._log_exporter  # noqa: F401
        import opentelemetry.exporter.otlp.proto.http._log_exporter  # noqa: F401
    except ImportError:
        raise ConfigurationError(message) from None
