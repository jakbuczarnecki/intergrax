# © Artur Czarnecki. All rights reserved.

"""Qualification-only configuration provenance discovery sentinel (not production authority)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ZetaScopedConfigurationIdentityTrace:
    """Structural configuration provenance probe — arbitrary symbol for negative sensitivity."""

    tenant_id: str
    request_id: str
    configuration_fingerprint: str
    configuration_version: str
    current_revision: str
