# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition resolver for durable execution terminal provider selection (P0C-8B)."""

from __future__ import annotations

from intergrax.contracts.execution_terminal import (
    AmbiguousExecutionTerminalProviderError,
    ExecutionTerminalError,
    ExecutionTerminalPersistenceCapability,
    ExecutionTerminalPersistenceProvider,
    ExecutionTerminalStore,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.execution.execution_terminal.persistence import wire_execution_terminal_store

_AMBIGUOUS_PROVIDER_MSG = (
    "multiple durable execution terminal providers are available; set "
    "execution_terminal_persistence_provider explicitly"
)
_MISSING_SELECTED_PROVIDER_MSG = (
    "execution_terminal_persistence_provider={provider} requires the selected capability"
)


def _available_terminal_providers(
    *,
    kv_store: DistributedKVStore | None,
    document_store: DocumentStore | None,
    checkpoint_store: ExecutionTerminalPersistenceCapability | None,
) -> tuple[ExecutionTerminalPersistenceProvider, ...]:
    available: list[ExecutionTerminalPersistenceProvider] = []
    if kv_store is not None:
        available.append(ExecutionTerminalPersistenceProvider.KV)
    if document_store is not None:
        available.append(ExecutionTerminalPersistenceProvider.DOCUMENT_STORE)
    if checkpoint_store is not None and isinstance(
        checkpoint_store,
        ExecutionTerminalPersistenceCapability,
    ):
        available.append(ExecutionTerminalPersistenceProvider.CHECKPOINT)
    return tuple(available)


def resolve_execution_terminal_provider(
    *,
    provider: ExecutionTerminalPersistenceProvider | None,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    checkpoint_store: ExecutionTerminalPersistenceCapability | None = None,
) -> ExecutionTerminalPersistenceProvider | None:
    """Resolve the configured terminal provider or ``None`` when no durable provider exists."""
    available = _available_terminal_providers(
        kv_store=kv_store,
        document_store=document_store,
        checkpoint_store=checkpoint_store,
    )
    if not available:
        return None
    if provider is not None:
        return provider
    if len(available) == 1:
        return available[0]
    raise AmbiguousExecutionTerminalProviderError(_AMBIGUOUS_PROVIDER_MSG)


def resolve_execution_terminal_store(
    *,
    provider: ExecutionTerminalPersistenceProvider | None = None,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    checkpoint_store: ExecutionTerminalPersistenceCapability | None = None,
    execution_terminal_store: ExecutionTerminalStore | None = None,
) -> ExecutionTerminalStore:
    """
    Select exactly one durable terminal store from configured platform capabilities.

    Precedence inside this resolver:
    explicit ``execution_terminal_store`` > configured provider > single available provider.
    """
    if execution_terminal_store is not None:
        return execution_terminal_store

    selected = resolve_execution_terminal_provider(
        provider=provider,
        kv_store=kv_store,
        document_store=document_store,
        checkpoint_store=checkpoint_store,
    )
    if selected is None:
        return wire_execution_terminal_store()

    if selected is ExecutionTerminalPersistenceProvider.KV:
        if kv_store is None:
            raise ExecutionTerminalError(
                _MISSING_SELECTED_PROVIDER_MSG.format(
                    provider=ExecutionTerminalPersistenceProvider.KV.value,
                ),
            )
        return wire_execution_terminal_store(kv_store=kv_store)
    if selected is ExecutionTerminalPersistenceProvider.DOCUMENT_STORE:
        if document_store is None:
            raise ExecutionTerminalError(
                _MISSING_SELECTED_PROVIDER_MSG.format(
                    provider=ExecutionTerminalPersistenceProvider.DOCUMENT_STORE.value,
                ),
            )
        return wire_execution_terminal_store(document_store=document_store)
    if checkpoint_store is None or not isinstance(
        checkpoint_store,
        ExecutionTerminalPersistenceCapability,
    ):
        raise ExecutionTerminalError(
            _MISSING_SELECTED_PROVIDER_MSG.format(
                provider=ExecutionTerminalPersistenceProvider.CHECKPOINT.value,
            ),
        )
    return wire_execution_terminal_store(checkpoint_store=checkpoint_store)


def resolve_platform_store_for_terminal_provider(
    provider: ExecutionTerminalPersistenceProvider,
    *,
    kv_store: DistributedKVStore | None,
    document_store: DocumentStore | None,
) -> tuple[DistributedKVStore | None, DocumentStore | None]:
    """Return at most one platform store aligned with the selected terminal provider."""
    if provider is ExecutionTerminalPersistenceProvider.KV:
        return kv_store, None
    if provider is ExecutionTerminalPersistenceProvider.DOCUMENT_STORE:
        return None, document_store
    return None, None
