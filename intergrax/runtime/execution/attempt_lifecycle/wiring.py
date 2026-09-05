# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition resolver for durable attempt lifecycle provider selection (P0C-8C)."""

from __future__ import annotations

from intergrax.contracts.attempt_lifecycle import (
    AmbiguousAttemptLifecycleProviderError,
    AttemptLifecycleError,
    AttemptLifecyclePersistenceProvider,
    AttemptLifecycleStore,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.execution.attempt_lifecycle.persistence import wire_attempt_lifecycle_store

_AMBIGUOUS_PROVIDER_MSG = (
    "multiple durable attempt lifecycle providers are available; set "
    "attempt_lifecycle_persistence_provider explicitly"
)
_MISSING_SELECTED_PROVIDER_MSG = (
    "attempt_lifecycle_persistence_provider={provider} requires the selected capability"
)


def _available_attempt_providers(
    *,
    kv_store: DistributedKVStore | None,
    document_store: DocumentStore | None,
) -> tuple[AttemptLifecyclePersistenceProvider, ...]:
    available: list[AttemptLifecyclePersistenceProvider] = []
    if kv_store is not None:
        available.append(AttemptLifecyclePersistenceProvider.KV)
    if document_store is not None:
        available.append(AttemptLifecyclePersistenceProvider.DOCUMENT_STORE)
    return tuple(available)


def resolve_attempt_lifecycle_provider(
    *,
    provider: AttemptLifecyclePersistenceProvider | None,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> AttemptLifecyclePersistenceProvider | None:
    """Resolve the configured attempt provider or ``None`` when no durable provider exists."""
    available = _available_attempt_providers(
        kv_store=kv_store,
        document_store=document_store,
    )
    if not available:
        return None
    if provider is not None:
        return provider
    if len(available) == 1:
        return available[0]
    raise AmbiguousAttemptLifecycleProviderError(_AMBIGUOUS_PROVIDER_MSG)


def resolve_platform_store_for_attempt_lifecycle_provider(
    provider: AttemptLifecyclePersistenceProvider,
    *,
    kv_store: DistributedKVStore | None,
    document_store: DocumentStore | None,
) -> tuple[DistributedKVStore | None, DocumentStore | None]:
    """Return at most one platform store aligned with the selected attempt lifecycle provider."""
    if provider is AttemptLifecyclePersistenceProvider.KV:
        return kv_store, None
    if provider is AttemptLifecyclePersistenceProvider.DOCUMENT_STORE:
        return None, document_store
    raise AttemptLifecycleError(
        _MISSING_SELECTED_PROVIDER_MSG.format(provider=provider.value),
    )


def resolve_attempt_lifecycle_store(
    *,
    provider: AttemptLifecyclePersistenceProvider | None = None,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    explicit_store: AttemptLifecycleStore | None = None,
) -> AttemptLifecycleStore | None:
    """
    Select exactly one durable attempt lifecycle store from configured platform capabilities.

    Precedence inside this resolver:
    explicit ``explicit_store`` > configured provider > single available provider.
    """
    if explicit_store is not None:
        return explicit_store

    selected = resolve_attempt_lifecycle_provider(
        provider=provider,
        kv_store=kv_store,
        document_store=document_store,
    )
    if selected is None:
        return None

    attempt_kv_store, attempt_doc_store = resolve_platform_store_for_attempt_lifecycle_provider(
        selected,
        kv_store=kv_store,
        document_store=document_store,
    )
    if attempt_kv_store is None and attempt_doc_store is None:
        raise AttemptLifecycleError(
            _MISSING_SELECTED_PROVIDER_MSG.format(provider=selected.value),
        )
    return wire_attempt_lifecycle_store(
        kv_store=attempt_kv_store,
        document_store=attempt_doc_store,
    )
