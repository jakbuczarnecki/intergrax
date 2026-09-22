# © Artur Czarnecki. All rights reserved.

"""Operator contracts for governed integration catalog hot reload."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
)
from intergrax.contracts.integration_catalog_revision import CatalogRevision

SCHEMA_CATALOG_HOT_RELOAD_OPERATOR_REQUEST_V1 = "integration_catalog_hot_reload_operator_request.v1"
SCHEMA_CATALOG_HOT_RELOAD_RESULT_V1 = "integration_catalog_hot_reload_result.v1"

CatalogHotReloadPreset = Literal["core", "full"]


class CatalogHotReloadOperatorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["integration_catalog_hot_reload_operator_request.v1"] = (
        SCHEMA_CATALOG_HOT_RELOAD_OPERATOR_REQUEST_V1
    )
    mutation_id: str = Field(min_length=1)
    preset: CatalogHotReloadPreset = "core"
    expected_revision: CatalogRevision


class CatalogHotReloadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["integration_catalog_hot_reload_result.v1"] = (
        SCHEMA_CATALOG_HOT_RELOAD_RESULT_V1
    )
    mutation_id: str = Field(min_length=1)
    before_revision: CatalogRevision
    after_revision: CatalogRevision
    changed: bool
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None
    blocker_code: str | None = None
    policy_action: str | None = None
