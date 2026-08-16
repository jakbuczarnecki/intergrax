# © Artur Czarnecki. All rights reserved.

"""Memory store materialization context (ENTERPRISE-5 / BLOCK D)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack


@dataclass(frozen=True, slots=True)
class MemoryStoreMaterializationContext:
    """Bounded factory inputs for Memory store plugin materialization."""

    env: ApplicationEnvironmentProfile
    tenant_id: str | None
    integration_profile: IntegrationProfile
    rag_stack: RagStack | None = None
