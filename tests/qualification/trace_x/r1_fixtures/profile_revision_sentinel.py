# © Artur Czarnecki. All rights reserved.

"""Qualification-only profile revision discovery sentinel (not production authority)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Qx7PinnedTenantExecutionRevisionEvidence:
    """Structural profile revision probe — arbitrary symbol for negative sensitivity."""

    tenant_id: str
    execution_id: str
    revision_id: str
    fingerprint: str
