# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Governed discovery partition result (CAPABILITY-CATALOG-1 Stage 5)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.capability_catalog.governed_candidate import (
    BlockedCapabilityCandidate,
    GovernedCapabilityCandidate,
)

SCHEMA_GOVERNED_DISCOVERY_RESULT_V1: Final = "governed_discovery_result.v1"


class GovernedDiscoveryResult(BaseModel):
    """Total partition of ranked candidates into allowed and blocked sets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_discovery_result.v1"] = (
        SCHEMA_GOVERNED_DISCOVERY_RESULT_V1
    )
    allowed: tuple[GovernedCapabilityCandidate, ...]
    blocked: tuple[BlockedCapabilityCandidate, ...]
