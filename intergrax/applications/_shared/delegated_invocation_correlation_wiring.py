# © Artur Czarnecki. All rights reserved.

"""Tier-3 composition wiring for delegated invocation correlation (P2.1-S2C1-C1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationDurabilityMode,
    DelegatedInvocationCorrelationDurabilityPolicy,
)
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.execution.delegated_execution.correlation_composition import (
    resolve_delegated_invocation_correlation_service,
)
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    wire_delegated_invocation_correlation_store,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationService,
)


def resolve_delegated_invocation_correlation_for_host(
    env: ApplicationEnvironmentProfile,
    *,
    document_store: ConditionalDocumentStore | None,
) -> DelegatedInvocationCorrelationService | None:
    """Resolve correlation service from canonical environment profile and document store."""
    mode = env.governance.reliability.delegated_invocation_correlation_durability
    policy = DelegatedInvocationCorrelationDurabilityPolicy(mode=mode)
    if mode is DelegatedInvocationCorrelationDurabilityMode.DISABLED:
        return resolve_delegated_invocation_correlation_service(policy)
    store = wire_delegated_invocation_correlation_store(
        durability_mode=mode,
        document_store=document_store,
    )
    return resolve_delegated_invocation_correlation_service(
        policy,
        correlation_store=store,
    )


__all__ = ["resolve_delegated_invocation_correlation_for_host"]
