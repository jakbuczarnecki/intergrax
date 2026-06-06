# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.observability_backend.newrelic.bundle import create_newrelic_observability_backend
from intergrax.integrations.providers.observability_backend.newrelic.register import register_newrelic_integration

__all__ = ["create_newrelic_observability_backend", "register_newrelic_integration"]
