# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned Integrations-owned health composition surface (EBH-3-R1).

Cross-layer consumers (applications, tools) must import from this module — not
``intergrax.integrations._shared.health``.
"""

from __future__ import annotations

from intergrax.integrations._shared.health import (
    health_check,
    health_check_all,
    health_check_catalog_slugs,
    health_check_entry,
    http_ping_ok,
    probe_client_health,
)

__all__ = [
    "health_check",
    "health_check_all",
    "health_check_catalog_slugs",
    "health_check_entry",
    "http_ping_ok",
    "probe_client_health",
]
