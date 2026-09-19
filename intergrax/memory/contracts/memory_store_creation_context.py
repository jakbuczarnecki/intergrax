# © Artur Czarnecki. All rights reserved.

"""Typed creation contexts for Memory store plugins (MEM-HARDEN-FINAL-1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TenantScopedMemoryStoreCreationContext:
    """Vendor-neutral plugin materialization inputs for tenant-scoped memory stores."""

    tenant_id: str | None


UserProfileStoreCreationContext = TenantScopedMemoryStoreCreationContext
SessionStorageCreationContext = TenantScopedMemoryStoreCreationContext
EntityTemporalMemoryStoreCreationContext = TenantScopedMemoryStoreCreationContext
ProceduralMemoryStoreCreationContext = TenantScopedMemoryStoreCreationContext
LongHorizonMemoryStoreCreationContext = TenantScopedMemoryStoreCreationContext
