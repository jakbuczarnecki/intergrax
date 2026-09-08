# © Artur Czarnecki. All rights reserved.

"""Governance permission preset contract — configuration shorthand only (P1.11)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GovernancePermissionPreset(StrEnum):
    """
    Ergonomic security posture selector for Tier-3 hosts.

    Presets expand into canonical profile/policy fields only. They are not
    authority, policy engines, or runtime enforcement surfaces.
    """

    RESTRICTED = "restricted"
    BALANCED = "balanced"
    TRUSTED = "trusted"


class GovernancePermissionPresetProvenance(BaseModel):
    """Redaction-safe expansion evidence attached to domain policy fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_preset: GovernancePermissionPreset
    effective_require_human_on_critical: bool
    clamped_fields: tuple[str, ...] = Field(default_factory=tuple)
