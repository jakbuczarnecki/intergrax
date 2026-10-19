# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed offline targets for the preferred Spanner and Azure pair.

These records carry no secret, no retention duration, and no qualification.
Protected append and the combined hold stay unqualified.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_NON_BLANK = Field(min_length=1)


class SpannerPrimaryTarget(BaseModel):
    """Primary register location. Not a live client and not a capacity choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = _NON_BLANK
    instance_id: str = _NON_BLANK
    database_id: str = _NON_BLANK
    edition: Literal["enterprise_plus"]
    instance_configuration: Literal["eur3"]
    processing_units: None = None


class AzureWitnessTarget(BaseModel):
    """Witness location. Immutability duration stays unset.

    ``protected_append`` and ``combined_hold`` cannot be set to a qualified
    value. This record does not apply a container policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    region: Literal["northeurope"]
    account_sku: Literal["Standard_RAGZRS"]
    account_kind: Literal["StorageV2"]
    access_tier: Literal["Hot"]
    immutability_period_days: None = None
    protected_append: Literal["unqualified"] = "unqualified"
    combined_hold: Literal["unqualified"] = "unqualified"
