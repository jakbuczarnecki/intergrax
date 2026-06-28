# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Supabase relational store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SUPABASE_RELATIONAL_STORE_PROVIDER_ID = "supabase"


class SupabaseRelationalStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Supabase relational store integration."""

    pass


@runtime_checkable
class SupabaseRelationalStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SupabaseRelationalStoreIntegration(RelationalStoreIntegrationContract):
    """
    Supabase relational store integration.

    The legacy facade (create_supabase_relational_store) remains separate and backward-compatible.
    """

    config: SupabaseRelationalStoreIntegrationConfig = SupabaseRelationalStoreIntegrationConfig()
    _client: SupabaseRelationalStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: SupabaseRelationalStoreClient,
        *,
        enabled: bool = False,
    ) -> SupabaseRelationalStoreIntegration:
        integration = cls.for_provider(
            provider_id=SUPABASE_RELATIONAL_STORE_PROVIDER_ID,
            display_name="Supabase",
            config=SupabaseRelationalStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SupabaseRelationalStoreClient | None:
        return self._client
