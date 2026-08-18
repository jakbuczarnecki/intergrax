# © Artur Czarnecki. All rights reserved.

"""Typed Policy Catalog definition contracts (Governed Execution G2B).

Represents *what policy capability exists* — catalog metadata and configuration
contract identity only. Does not resolve handlers, validate configuration, or
implement a runtime catalog.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class PolicyDefinitionSource(StrEnum):
    """Origin category for a policy definition (metadata only)."""

    BUILT_IN = "built_in"
    PLUGIN = "plugin"


class PolicyDefinition(BaseModel):
    """Immutable catalog entry describing one selectable policy capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["policy_definition.v1"] = "policy_definition.v1"

    policy_id: str
    version: str

    display_name: str
    description: str = ""

    handler_id: str
    configuration_contract_id: str

    source: PolicyDefinitionSource

    @field_validator(
        "policy_id",
        "version",
        "display_name",
        "handler_id",
        "configuration_contract_id",
    )
    @classmethod
    def _strip_required_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        return value.strip()
