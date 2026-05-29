# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cloud platform facade contract — infrastructure only, no LLM (§7.1.2, Phase M.2)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from intergrax.integrations.contracts.base import HealthStatus


@runtime_checkable
class CloudPlatform(Protocol):
    """
    Unified auth + region for native cloud services.

    ``resolve(category)`` returns a provider slug for infrastructure categories
    (object_storage, message_bus, …). LLM wiring stays in ``llm_adapters/``.
    """

    @property
    def slug(self) -> str:
        """Platform identifier: aws, azure, gcp."""

    @property
    def default_region(self) -> Optional[str]:
        """Default region / location when configured."""

    def resolve(self, category: str) -> Optional[str]:
        """Return default integration slug for ``category`` on this platform."""

    def health(self) -> HealthStatus:
        """Optional startup probe."""
