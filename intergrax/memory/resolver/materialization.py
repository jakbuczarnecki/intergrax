# © Artur Czarnecki. All rights reserved.

"""Memory store materialization context (ENTERPRISE-5 / BLOCK D)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.registry.profile import IntegrationProfile


@dataclass(frozen=True, slots=True)
class MemoryStoreMaterializationContext:
    """Bounded factory inputs for Memory store plugin materialization."""

    tenant_id: str | None
    integration_profile: IntegrationProfile
