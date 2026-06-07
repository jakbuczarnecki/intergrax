# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.health.contracts import (
    HealthCheckIntegrationInput,
    HealthCheckIntegrationOutput,
    HealthCheckProfileInput,
    HealthCheckProfileOutput,
)
from intergrax.tools.providers.health.service import health_check_integration, health_check_profile


class HealthCheckIntegrationHandler(
    ServiceToolHandler[HealthCheckIntegrationInput, HealthCheckIntegrationOutput]
):
    _service = health_check_integration


class HealthCheckProfileHandler(ServiceToolHandler[HealthCheckProfileInput, HealthCheckProfileOutput]):
    _service = health_check_profile
