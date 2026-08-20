# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.providers.governance_approval.client import (
    HttpxGovernanceApprovalReadClient,
)
from intergrax.integrations.providers.governance_approval.config import (
    GovernanceApprovalIntegrationConfig,
)
from intergrax.integrations.providers.governance_approval.integration import (
    GovernanceApprovalIntegration,
)


def create_governance_approval_integration(
    *,
    base_url: str,
    timeout_seconds: float = 5.0,
    http_client_factory: Callable[[GovernanceApprovalIntegrationConfig], Any] | None = None,
) -> GovernanceApprovalIntegration:
    config = GovernanceApprovalIntegrationConfig(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    factory = http_client_factory or (
        lambda parsed: HttpxGovernanceApprovalReadClient(config=parsed)
    )
    return GovernanceApprovalIntegration.from_client(factory(config), config=config)
