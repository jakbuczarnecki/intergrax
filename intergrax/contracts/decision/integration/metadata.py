# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Operational metadata for decision integration production plugins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class DecisionIntegrationPluginDescriptor:
    """Identity surfaced to plugin admission before an adapter is wired."""

    plugin_id: str
    version: str
    source: str
    manifest_id: str | None

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("plugin_id must be non-empty")
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.manifest_id is not None and not self.manifest_id.strip():
            raise ValueError("manifest_id must be non-empty when set")


@dataclass(frozen=True, slots=True)
class IntegrationAuditProviderMetadata:
    """Metadata stamped by a recording audit provider on each sink append."""

    provider_id: str
    provider_version: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id must be non-empty")
        if not self.provider_version.strip():
            raise ValueError("provider_version must be non-empty")
        if self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")


__all__ = [
    "DecisionIntegrationPluginDescriptor",
    "IntegrationAuditProviderMetadata",
]
