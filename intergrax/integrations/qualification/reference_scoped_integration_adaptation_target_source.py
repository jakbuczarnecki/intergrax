# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference target catalog — constructed independently from adaptation requests (AW-7C-P4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationTarget,
    ScopedIntegrationAdaptationTargetLookupKey,
)
from intergrax.integrations.qualification.reference_scoped_integration_adaptation import (
    REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
)


def _reference_key(
    *,
    tenant_id: str = "tenant-a",
    resource_scope: str = "rs-1",
) -> ScopedIntegrationAdaptationTargetLookupKey:
    return ScopedIntegrationAdaptationTargetLookupKey(
        tenant_id=tenant_id,
        integration_category=IntegrationCategory.MESSAGE_BUS,
        provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
        resource_scope=resource_scope,
    )


def reference_scoped_integration_adaptation_target_source() -> (
    ReferenceScopedIntegrationAdaptationTargetSource
):
    """Default qualification catalog entry — revision truth is ``rev-1``."""
    entries = (
        ReferenceScopedIntegrationAdaptationTargetCatalogEntry(
            key=_reference_key(),
            target=ScopedIntegrationAdaptationTarget(
                tenant_id="tenant-a",
                integration_category=IntegrationCategory.MESSAGE_BUS,
                provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
                resource_scope="rs-1",
                current_revision="rev-1",
            ),
        ),
    )
    return ReferenceScopedIntegrationAdaptationTargetSource(entries)


class ReferenceScopedIntegrationAdaptationTargetCatalogEntry:
    __slots__ = ("key", "target")

    def __init__(
        self,
        *,
        key: ScopedIntegrationAdaptationTargetLookupKey,
        target: ScopedIntegrationAdaptationTarget,
    ) -> None:
        self.key = key
        self.target = target


class ReferenceScopedIntegrationAdaptationTargetSource:
    """Immutable in-memory target source for qualification — not production default."""

    def __init__(
        self,
        entries: tuple[ReferenceScopedIntegrationAdaptationTargetCatalogEntry, ...],
    ) -> None:
        self._by_key = {self._index_key(entry.key): entry.target for entry in entries}

    @staticmethod
    def _index_key(key: ScopedIntegrationAdaptationTargetLookupKey) -> tuple[str, str, str, str]:
        return (
            key.tenant_id,
            key.integration_category.value,
            key.provider_id,
            key.resource_scope,
        )

    def resolve(
        self,
        key: ScopedIntegrationAdaptationTargetLookupKey,
    ) -> ScopedIntegrationAdaptationTarget:
        indexed = self._index_key(key)
        try:
            return self._by_key[indexed]
        except KeyError:
            raise LookupError(f"no target for lookup key {indexed!r}") from None


__all__ = [
    "ReferenceScopedIntegrationAdaptationTargetCatalogEntry",
    "ReferenceScopedIntegrationAdaptationTargetSource",
    "reference_scoped_integration_adaptation_target_source",
]
