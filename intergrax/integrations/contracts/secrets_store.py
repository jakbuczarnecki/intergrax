# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Secrets store integration contract (§7.1.2, Phase M.7)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SecretsStore(Protocol):
    """Tenant-scoped secret read/write facade (Vault, cloud secret managers, …)."""

    def get_secret(self, path: str, *, version: Optional[str] = None) -> str:
        """Return secret value at ``path`` (tenant prefix applied by adapter)."""

    def put_secret(self, path: str, value: str) -> None:
        """Create or update secret at ``path``."""

    def delete_secret(self, path: str) -> None:
        """Remove secret at ``path`` if present."""
