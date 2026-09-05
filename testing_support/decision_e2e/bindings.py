# © Artur Czarnecki. All rights reserved.

"""Safe provider binding evidence for DS-E2E qualification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderBindingEvidence:
    """Safe provider identity without secrets."""

    profile_id: str
    provider: str
    model: str | None
    host: str | None = None
