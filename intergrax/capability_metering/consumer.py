# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability usage consumer port (CAPABILITY-CATALOG-1 Stage 13)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.capability_metering import CapabilityUsageEvent


class CapabilityUsageConsumer(Protocol):
    """Pluggable downstream sink — no global registry."""

    def consume(self, event: CapabilityUsageEvent) -> None: ...
