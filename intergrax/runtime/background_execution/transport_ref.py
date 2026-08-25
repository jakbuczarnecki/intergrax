# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed transport identity for background execution resolution (BG-EXEC-2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackgroundTransportExecutionRef:
    """Opaque provider transport identity scoped by tenant and provider."""

    tenant_id: str
    provider: str
    transport_task_id: str

    def __post_init__(self) -> None:
        tenant = self.tenant_id.strip()
        provider = self.provider.strip()
        transport_task_id = self.transport_task_id.strip()
        if not tenant:
            raise ValueError("tenant_id must be non-empty")
        if not provider:
            raise ValueError("provider must be non-empty")
        if not transport_task_id:
            raise ValueError("transport_task_id must be non-empty")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "transport_task_id", transport_task_id)
