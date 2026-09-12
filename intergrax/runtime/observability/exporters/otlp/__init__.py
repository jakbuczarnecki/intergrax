# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP infrastructure transport adapter (W5-E)."""

from intergrax.runtime.observability.exporters.otlp.otlp_configuration import (
    validate_otlp_export_configuration,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport

__all__ = [
    "OtlpTransport",
    "validate_otlp_export_configuration",
]
