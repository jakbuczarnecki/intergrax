# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.observability_backend.opentelemetry_collector.bundle import create_opentelemetry_collector_observability_backend
from intergrax.integrations.providers.observability_backend.opentelemetry_collector.register import register_opentelemetry_collector_integration

__all__ = ["create_opentelemetry_collector_observability_backend", "register_opentelemetry_collector_integration"]
