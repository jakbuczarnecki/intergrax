# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.providers.security_status.client import HttpxSecurityStatusReadClient
from intergrax.integrations.providers.security_status.config import SecurityStatusIntegrationConfig
from intergrax.integrations.providers.security_status.integration import SecurityStatusIntegration


def create_security_status_integration(
    *,
    base_url: str,
    timeout_seconds: float = 5.0,
    http_client_factory: Callable[[SecurityStatusIntegrationConfig], Any] | None = None,
) -> SecurityStatusIntegration:
    config = SecurityStatusIntegrationConfig(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    factory = http_client_factory or (
        lambda parsed: HttpxSecurityStatusReadClient(config=parsed)
    )
    return SecurityStatusIntegration.from_client(factory(config), config=config)
