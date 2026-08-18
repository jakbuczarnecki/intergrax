# © Artur Czarnecki. All rights reserved.

"""Meaningful external side-effect policy request (GEC-5).

Generic description of a proposed external action that may create commitments,
mutations, disclosures, or other irreversible consequences. Reuses
``PolicyDecision`` / ``PolicyAction`` for evaluation outcomes.

Not a quote model, payment model, or provider authorization layer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_MEANINGFUL_SIDE_EFFECT_REQUEST_V1: Final = "meaningful_side_effect_request.v1"

_NON_EMPTY = Field(min_length=1)


class MeaningfulSideEffectKind(StrEnum):
    """Coarse impact classes — not an action catalog."""

    COMMITMENT = "commitment"
    MUTATION = "mutation"
    DISCLOSURE = "disclosure"
    ACCESS = "access"


class MeaningfulSideEffectRequest(BaseModel):
    """Proposed external side effect for policy evaluation before execution.

    ``action`` is a consumer-defined identifier (e.g. domain ``ACCEPT_QUOTE``).
    Domain-specific payloads belong in ``context`` / ``correlation`` — not as
    quote- or provider-SDK-typed fields on this model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["meaningful_side_effect_request.v1"] = (
        SCHEMA_MEANINGFUL_SIDE_EFFECT_REQUEST_V1
    )
    action: str = _NON_EMPTY
    kinds: tuple[MeaningfulSideEffectKind, ...] = Field(min_length=1)
    side_effect_scope_id: str = _NON_EMPTY
    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    principal_id: str | None = None
    tenant_id: str | None = None
    resource: str | None = None
    external_target: str | None = None
    correlation: Mapping[str, Any] = Field(default_factory=dict)
    context: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("action", "side_effect_scope_id", "task_id", "run_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("principal_id", "tenant_id", "resource", "external_target")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
