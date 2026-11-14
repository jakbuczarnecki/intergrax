# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contribute the openai embedding provider to the single ``openai`` catalog row.

The ``openai`` slug is owned by ``managed_retrieval/openai`` (one catalog row per slug). This module only
augments that row with the embedding category and contract spec, like the other embedding providers that
share a slug with another category. The factory of that row dispatches on ``integration_category``: the
embedding category builds the embedding provider integration, every other category keeps the factory
registered by the owning row.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.catalog_factory import IntegrationFactory
from intergrax.integrations.providers.embedding_provider.openai.bundle import (
    create_openai_embedding_provider_integration,
)
from intergrax.integrations.providers.embedding_provider.openai.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.catalog import augment_integration_contract_specs, register_integration

if TYPE_CHECKING:
    from intergrax.runtime.integrations.contracts import PlatformIntegrationContract


def _dispatching_factory(owning_factory: IntegrationFactory) -> IntegrationFactory:
    def _factory(**kwargs: object) -> PlatformIntegrationContract:
        if kwargs.get("integration_category") == IntegrationCategory.EMBEDDING_PROVIDER:
            return create_openai_embedding_provider_integration(enabled=bool(kwargs.get("enabled", False)))
        return owning_factory(**kwargs)

    return _factory


def register_openai_embedding_provider_integration(*, override: bool = False) -> None:
    del override
    entry = augment_integration_contract_specs(
        "openai",
        categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
        contract_specs=CONTRACT_SPECS,
    )
    register_integration(replace(entry, factory=_dispatching_factory(entry.factory)), override=True)
