# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.providers.project_status.client import HttpxProjectStatusReadClient
from intergrax.integrations.providers.project_status.config import ProjectStatusIntegrationConfig
from intergrax.integrations.providers.project_status.integration import ProjectStatusIntegration


def create_project_status_integration(
    *,
    base_url: str,
    timeout_seconds: float = 5.0,
    http_client_factory: Callable[[ProjectStatusIntegrationConfig], Any] | None = None,
) -> ProjectStatusIntegration:
    config = ProjectStatusIntegrationConfig(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    factory = http_client_factory or (
        lambda parsed: HttpxProjectStatusReadClient(config=parsed)
    )
    return ProjectStatusIntegration.from_client(factory(config), config=config)
