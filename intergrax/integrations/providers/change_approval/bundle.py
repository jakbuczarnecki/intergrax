# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.providers.change_approval.client import HttpxChangeApprovalReadClient
from intergrax.integrations.providers.change_approval.config import ChangeApprovalIntegrationConfig
from intergrax.integrations.providers.change_approval.integration import ChangeApprovalIntegration


def create_change_approval_integration(
    *,
    base_url: str,
    timeout_seconds: float = 5.0,
    http_client_factory: Callable[[ChangeApprovalIntegrationConfig], Any] | None = None,
) -> ChangeApprovalIntegration:
    config = ChangeApprovalIntegrationConfig(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    factory = http_client_factory or (
        lambda parsed: HttpxChangeApprovalReadClient(config=parsed)
    )
    return ChangeApprovalIntegration.from_client(factory(config), config=config)
