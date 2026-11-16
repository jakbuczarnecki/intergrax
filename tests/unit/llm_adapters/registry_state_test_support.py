# © Artur Czarnecki. All rights reserved.

"""Test helpers for full LLMAdapterRegistry logical state isolation."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from intergrax.contracts.external_operation_termination import ExternalOperationCapabilities
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterFactory,
    ProviderExternalOperationSeamFactory,
)


@dataclass(frozen=True, slots=True)
class RegistryStateSnapshot:
    factories: dict[str, LLMAdapterFactory]
    external_operation_seam_factories: dict[str, ProviderExternalOperationSeamFactory]
    external_operation_capabilities: dict[str, ExternalOperationCapabilities]


def snapshot_registry_state() -> RegistryStateSnapshot:
    return RegistryStateSnapshot(
        factories=dict(LLMAdapterRegistry._factories),
        external_operation_seam_factories=dict(LLMAdapterRegistry._external_operation_seam_factories),
        external_operation_capabilities=dict(LLMAdapterRegistry._external_operation_capabilities),
    )


def restore_registry_state(snapshot: RegistryStateSnapshot) -> None:
    LLMAdapterRegistry._factories.clear()
    LLMAdapterRegistry._factories.update(snapshot.factories)
    LLMAdapterRegistry._external_operation_seam_factories.clear()
    LLMAdapterRegistry._external_operation_seam_factories.update(snapshot.external_operation_seam_factories)
    LLMAdapterRegistry._external_operation_capabilities.clear()
    LLMAdapterRegistry._external_operation_capabilities.update(snapshot.external_operation_capabilities)


_PROVIDER_ADAPTER_PREFIX = "intergrax.llm_adapters.providers."


def evict_provider_adapter_modules() -> dict[str, object]:
    """Remove imported provider adapter modules so a blocked SDK is really re-imported.

    A provider adapter module that an earlier test imported (with its SDK installed) stays in
    ``sys.modules``; ``create`` would then succeed although the SDK is blocked. Callers restore the
    returned modules with ``sys.modules.update(...)`` in a ``finally`` block.
    """
    removed: dict[str, object] = {}
    for name in list(sys.modules):
        if name.startswith(_PROVIDER_ADAPTER_PREFIX) and name.endswith("_adapter"):
            removed[name] = sys.modules.pop(name)
    return removed


@pytest.fixture()
def restore_registry_state_fixture() -> Iterator[RegistryStateSnapshot]:
    snapshot = snapshot_registry_state()
    try:
        yield snapshot
    finally:
        restore_registry_state(snapshot)
