# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP infrastructure transport adapter (W5-E)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport

__all__ = [
    "OtlpTransport",
    "validate_otlp_export_configuration",
]


def __getattr__(name: str) -> object:
    if name == "OtlpTransport":
        from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport

        return OtlpTransport
    if name == "validate_otlp_export_configuration":
        from intergrax.runtime.observability.exporters.otlp.otlp_configuration import (
            validate_otlp_export_configuration,
        )

        return validate_otlp_export_configuration
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
