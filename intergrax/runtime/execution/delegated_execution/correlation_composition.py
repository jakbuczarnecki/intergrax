# © Artur Czarnecki. All rights reserved.

"""Delegated invocation correlation composition (P2.1-S2C1-C1)."""

from __future__ import annotations

from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationCompositionError,
    DelegatedInvocationCorrelationDurabilityMode,
    DelegatedInvocationCorrelationDurabilityPolicy,
    DelegatedInvocationCorrelationStore,
)
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    InMemoryDelegatedInvocationCorrelationStore,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationService,
)


def resolve_delegated_invocation_correlation_service(
    policy: DelegatedInvocationCorrelationDurabilityPolicy,
    *,
    correlation_store: DelegatedInvocationCorrelationStore | None = None,
    correlation_service: DelegatedInvocationCorrelationService | None = None,
) -> DelegatedInvocationCorrelationService | None:
    """
    Resolve correlation service from explicit durability policy and optional store.

    Fails closed at composition when policy and store durability are inconsistent.
    """
    mode = policy.mode
    if correlation_service is not None and correlation_store is not None:
        raise DelegatedInvocationCorrelationCompositionError(
            "provide either correlation_service or correlation_store, not both",
        )
    if correlation_service is not None:
        _validate_store_for_mode(mode, correlation_service.store)
        if mode is DelegatedInvocationCorrelationDurabilityMode.DISABLED:
            raise DelegatedInvocationCorrelationCompositionError(
                "correlation service cannot be wired when durability mode is disabled",
            )
        return correlation_service
    if mode is DelegatedInvocationCorrelationDurabilityMode.DISABLED:
        if correlation_store is not None:
            raise DelegatedInvocationCorrelationCompositionError(
                "correlation store cannot be wired when durability mode is disabled",
            )
        return None
    if mode is DelegatedInvocationCorrelationDurabilityMode.REQUIRED:
        if correlation_store is None:
            raise DelegatedInvocationCorrelationCompositionError(
                "durable correlation store required but not configured",
            )
        _validate_store_for_mode(mode, correlation_store)
        return DelegatedInvocationCorrelationService(correlation_store)
    if correlation_store is None:
        correlation_store = InMemoryDelegatedInvocationCorrelationStore()
    _validate_store_for_mode(mode, correlation_store)
    return DelegatedInvocationCorrelationService(correlation_store)


def _validate_store_for_mode(
    mode: DelegatedInvocationCorrelationDurabilityMode,
    store: DelegatedInvocationCorrelationStore,
) -> None:
    if mode is DelegatedInvocationCorrelationDurabilityMode.REQUIRED and not store.is_durable:
        raise DelegatedInvocationCorrelationCompositionError(
            "durable correlation store required but store is not durable",
        )
    if (
        mode is DelegatedInvocationCorrelationDurabilityMode.NON_DURABLE_TEST
        and store.is_durable
    ):
        raise DelegatedInvocationCorrelationCompositionError(
            "non-durable test mode cannot use a durable correlation store",
        )


__all__ = ["resolve_delegated_invocation_correlation_service"]
